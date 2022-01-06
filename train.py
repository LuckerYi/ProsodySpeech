import argparse
import math
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import time

import torch
import torch.distributed as dist
from numpy import finfo
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from distributed import apply_gradient_allreduce
from hparams import create_hparams
from models import load_model
from utils.data_reader import TextMelCollate, TextMelLoader, DynamicBatchSampler
from utils.data_reader_refine import TextMelLoader_refine
from utils.logger import Tacotron2Logger
from utils.utils import get_checkpoint_path, learning_rate_decay, print_rank, _plot_and_save
from utils import ValueWindow
from weightlist import Weightlist
import numpy as np


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt*(1/float(n_gpus))
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams) if not hparams.is_partial_refine else TextMelLoader_refine(hparams.training_files, hparams)
    collate_fn = TextMelCollate(hparams)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    if hparams.batch_criterion == 'frame':
        batch_sampler = DynamicBatchSampler(train_sampler, frames_threshold=hparams.batch_size)
        train_loader = DataLoader(trainset,
                                batch_sampler=batch_sampler,
                                num_workers=hparams.numberworkers,
                                pin_memory=True,
                                collate_fn=collate_fn)
    elif hparams.batch_criterion == 'utterance':
        train_loader = DataLoader(trainset,
                                sampler=train_sampler, batch_size=hparams.batch_size,
                                num_workers=hparams.numberworkers, shuffle=shuffle,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn)
    else:
        raise ValueError("batch criterion not supported: %s." % hparams.batch_criterion)

    return train_loader, collate_fn, trainset


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(log_directory)
    else:
        logger = None
    return logger

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint_dict['state_dict'], strict=False)
    try:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    except:
        print("Can't use old optimizer, maybe some parameters are not in old optimizer")

    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def load_pretrain_part_checkpoint(checkpoint_path, model, load_list):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    model_dict = checkpoint_dict['state_dict']
    model_dict = {k: v for k, v in model_dict.items()
                  if k in load_list}
    for k, v in model_dict.items():
        print(k)
    dummy_dict = model.state_dict()
    dummy_dict.update(model_dict)
    model_dict = dummy_dict
    model.load_state_dict(model_dict)
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded pretrain_encoder checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    
    return model, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams, refine_from):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    weightlist = Weightlist()

    model = load_model(hparams)
    learning_rate = hparams.initial_learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.use_GAN and hparams.GAN_type=='lsgan':
        from discriminator import Lsgan_Loss, Calculate_Discrim
        model_D = Calculate_Discrim(hparams).cuda() if torch.cuda.is_available() else Calculate_Discrim(hparams)
        lsgan_loss = Lsgan_Loss(hparams)
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)
    
    if hparams.use_GAN and hparams.GAN_type=='wgan-gp':
        from discriminator import Wgan_GP, GP
        model_D = Wgan_GP(hparams).cuda() if torch.cuda.is_available() else Wgan_GP(hparams)
        calc_gradient_penalty = GP(hparams).cuda() if torch.cuda.is_available() else GP(hparams)
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.use_mutual_information:
        from mutual_information import ma_et, Mine, learn_mine, ma
        mine_net = Mine(hparams).cuda()
        optimizer_mine_net = torch.optim.Adam(mine_net.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

# Parameters
    MIfrozen_list = weightlist.MIfrozen_list #if hparams.use_mutual_information else None

    choosestl_list = weightlist.choosestl_list

    if hparams.is_partial_refine:
        refine_list = weightlist.refine_list if not hparams.full_refine else weightlist.full_weight_list
        if hparams.use_gaussian_upsampling:
            refine_list += weightlist.refine_list_gaussian
        if hparams.use_f0:
            refine_list += weightlist.refine_list_f0

    open_choosestl = False if hparams.use_mutual_information else True

    gst_reference_encoder_list = weightlist.gst_reference_encoder_list if hparams.gst_reference_encoder == 'multiheadattention' else weightlist.conv_gst_reference_encoder_list

    for name, param in model.named_parameters():
        if hparams.is_partial_refine:
            if name in refine_list:
                param.requires_grad = True 
            else:
                param.requires_grad = False
        print(name, param.requires_grad, param.shape)

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
        if hparams.use_GAN:
            model_D = apply_gradient_allreduce(model_D)

    logger = prepare_directories_and_logger(output_directory, log_directory, rank)

    train_loader, collate_fn, trainset = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if not checkpoint_path:
        checkpoint_path = get_checkpoint_path(output_directory) if not hparams.is_partial_refine else refine_from#if file_list and file_list[-1].startswith("checkpoint_"):
                                                                                                                                        #     return os.path.join(output_directory, file_list[-1])
                                                                                                                                        # else:
                                                                                                                                        #     return None
    if checkpoint_path is not None:
        if not hparams.use_mutual_information:
            if hparams.training_stage == 'train_text_encoder':#also normal tts
                model, optimizer, _learning_rate, iteration = load_checkpoint(checkpoint_path, model, optimizer)
                if hparams.gst_train_att:
                    print('Configuration error!!!')
                    print('gst_train_att=True, if you want to train a normal tts model, pls set gst_train_att=False')
            elif hparams.training_stage == 'train_style_attention':
                model, _, iteration = load_pretrain_part_checkpoint(checkpoint_path, model, load_list=MIfrozen_list + gst_reference_encoder_list)
                if hparams.style_extractor_presteps < iteration:
                    print('Configuration error!!!')
                    print('style_extractor_presteps must bigger than current iteration')
                    raise
                hparams.use_saved_learning_rate = False
            elif hparams.training_stage == 'train_refine_layernorm' and hparams.is_partial_refine:
                model, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer)
                hparams.use_saved_learning_rate = False
            else:
                print('Configuration error!!!')
                print('use_mutual_information=False, so training_stage are either train_text_encoder or train_style_attention')
                raise
        else:
            if hparams.training_stage == 'train_style_extractor':
                model, _, _ = load_pretrain_part_checkpoint(checkpoint_path, model, load_list=MIfrozen_list)
                hparams.use_saved_learning_rate = False
            else:
                print('Configuration error!!!')
                print('use_mutual_information=True, so training_stage must be train_style_extractor')
                raise                
        if hparams.use_saved_learning_rate:
            learning_rate = _learning_rate
        iteration = (iteration + 1)  if not hparams.is_partial_refine else 1# next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader))) if not hparams.is_partial_refine else 0

    model.train()
    if hparams.use_GAN:
        model_D.train()
    else:
        hparams.use_GAN = True
        hparams.Generator_pretrain_step = hparams.iters#3k
    is_overflow = False
    epoch = epoch_offset
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    check_paramaters = {}
    Print_Function = [True,True,True,True,True,True,True]
    mi_lb = None
    Wasserstein_D = None
    att = None
    # ================ MAIN TRAINNIG LOOP! ===================
    while iteration <= hparams.iters:
        # print("Epoch: {}".format(epoch))
        if hparams.distributed_run and hparams.batch_criterion == 'utterance':
            train_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            learning_rate = learning_rate_decay(iteration, hparams)*0.1 if hparams.is_partial_refine and iteration >= int(hparams.iters/2) else learning_rate_decay(iteration, hparams)
            
            if hparams.use_GAN:
                # Discriminator turn
                if iteration > hparams.Generator_pretrain_step + 1:
                    for param_group in optimizer_D.param_groups:
                        param_group['lr'] = learning_rate*0.1 if hparams.is_partial_refine else learning_rate
                    optimizer.zero_grad()
                    optimizer_D.zero_grad()
                    for name, param in model.named_parameters():
                        param.requires_grad = False
                    for name, param in model_D.named_parameters():
                        param.requires_grad = True 

                    loss, loss_dict, weight, pred_outs, ys, olens, _, _, _= model(*model._parse_batch(batch,hparams,utt_mels=trainset.utt_mels if hparams.is_refine_style else None), open_choosestl=open_choosestl)
                    if hparams.GAN_type=='lsgan':
                        discrim_gen_output, discrim_target_output = model_D(pred_outs + (torch.randn(pred_outs.size()).cuda() if hparams.add_noise else 0), ys + (torch.randn(pred_outs.size()).cuda() if hparams.add_noise else 0), olens)
                        loss_D = lsgan_loss(discrim_gen_output, discrim_target_output, train_object='D')
                        loss_G = lsgan_loss(discrim_gen_output, discrim_target_output, train_object='G')                
                        loss_D.backward(retain_graph=True)
                    if hparams.GAN_type=='wgan-gp':
                        D_real = model_D(ys, olens)
                        D_real = -D_real.mean()
                        D_real.backward(retain_graph=True)
                        D_fake = model_D(pred_outs, olens)
                        D_fake = D_fake.mean()
                        D_fake.backward()
                        gradient_penalty = calc_gradient_penalty(model_D, ys.data, pred_outs.data, olens.data)
                        gradient_penalty.backward()
                        D_cost = D_real + D_fake + gradient_penalty
                        Wasserstein_D = -D_real - D_fake
                    grad_norm_D = torch.nn.utils.clip_grad_norm_(model_D.parameters(), hparams.grad_clip_thresh)
                    optimizer_D.step()
                    print('\n')
                    if hparams.GAN_type=='lsgan':
                        print("Epoch:{} step:{} loss_D: {:>9.6f}, loss_G: {:>9.6f}, Grad Norm: {:>9.6f}".format(epoch, iteration, loss_D, loss_G, grad_norm_D))
                    if hparams.GAN_type=='wgan-gp':
                        print("Epoch:{} step:{} D_cost: {:>9.6f}, Wasserstein_D: {:>9.6f}, GP: {:>9.6f}, Grad Norm: {:>9.6f}".format(epoch, iteration, D_cost, Wasserstein_D, gradient_penalty, grad_norm_D))

                # Generator turn
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate 
                optimizer.zero_grad()
                
                if iteration > hparams.Generator_pretrain_step + 1:
                    for name, param in model.named_parameters():
                        if hparams.is_partial_refine:
                            if name in refine_list:
                                param.requires_grad = True
                        else:
                            param.requires_grad = True                                 
                    for name, param in model_D.named_parameters():
                        param.requires_grad = False
                    optimizer_D.zero_grad()

                if not hparams.is_partial_refine and iteration > hparams.style_extractor_presteps:
                    if hparams.use_mutual_information:
                        print('use_mutual_information=True, pls set hparams.style_extractor_presteps > iters')
                        raise
                    open_choosestl = True
                    if iteration <= (hparams.style_extractor_presteps + hparams.choosestl_steps):
                        for name, param in model.named_parameters():
                            if name in choosestl_list:
                                param.requires_grad = True
                            else:
                                param.requires_grad = False
                            if Print_Function[0]:
                                print(name, param.requires_grad, param.shape)
                        Print_Function[0] = False
                    else:
                        for name, param in model.named_parameters():
                            if name in gst_reference_encoder_list + MIfrozen_list:
                                param.requires_grad = False
                            else:
                                param.requires_grad = True
                            if Print_Function[1]:
                                print(name, param.requires_grad, param.shape)
                        Print_Function[1] = False                                         
                
                if hparams.use_mutual_information and (iteration-1) % 2 == 0:
                    for name, param in model.named_parameters():
                        param.requires_grad = True                    
                    for name, param in mine_net.named_parameters():
                        param.requires_grad = True# to ensure the grad calculation for the next operation
                    hs_random_choosed, style_embs = model(*model._parse_batch(batch,hparams,utt_mels=trainset.utt_mels if hparams.is_refine_style else None), open_choosestl=open_choosestl, is_MI_step=True)
                    mi_lb, ma_et, loss_mi = learn_mine(batch=(hs_random_choosed, style_embs), mine_net=mine_net, ma_et=ma_et, hparams=hparams)
                    for name, param in model.named_parameters():
                        param.requires_grad = False
                    for param_group in optimizer_mine_net.param_groups:
                        param_group['lr'] = learning_rate*0.01
                    optimizer_mine_net.zero_grad()
                    loss_mi.backward()
                    grad_norm_mine_net = torch.nn.utils.clip_grad_norm_(mine_net.parameters(), hparams.grad_clip_thresh)
                    optimizer_mine_net.step()
                    optimizer.zero_grad()
                    for name, param in model.named_parameters():
                        param.requires_grad = True
                    print('mi_loss=', loss_mi.item())
                    print('mi_lb_loss=', hparams.mutual_information_lambda*mi_lb.item())
                    print('grad_norm_mine_net=', grad_norm_mine_net)
                    iteration += 1
                    torch.cuda.empty_cache()
                    continue

                loss, loss_dict, weight, pred_outs, ys, olens, att, hs_random_choosed, style_embs = model(*model._parse_batch(batch,hparams,utt_mels=trainset.utt_mels if hparams.is_refine_style else None), open_choosestl=open_choosestl)
                
                if att is not None and iteration % 100 == 0:
                    file_dir = hparams.mel_dir + hparams.training_files.split('/')[-2].split('_')[-1] + hparams.att_name
                    os.makedirs(file_dir, exist_ok=True)
                    image_path = os.path.join(file_dir, "{}_att_{}.png".format(iteration,hparams.style_query_level))
                    txt_path = os.path.join(file_dir, "{}_att_{}.txt".format(iteration,hparams.style_query_level))
                    _plot_and_save(att[-1].unsqueeze(0).float().data.cpu().numpy(), image_path)
                    print("att.shape={},\n, att[-1].unsqueeze(0).float().data.cpu().numpy()={},\n".format(att.shape, att[-1].unsqueeze(0).float().data.cpu().numpy()), file=open(txt_path,'w+'))
                
                if iteration > hparams.Generator_pretrain_step + 1:
                    if hparams.GAN_type=='lsgan':
                        discrim_gen_output, discrim_target_output = model_D(pred_outs, ys, olens)
                        loss_D = lsgan_loss(discrim_gen_output, discrim_target_output, train_object='D')
                        loss_G = lsgan_loss(discrim_gen_output, discrim_target_output, train_object='G')
                    if hparams.GAN_type=='wgan-gp':
                        loss_G = model_D(pred_outs, olens)
                        loss_G = -loss_G.mean()
                    loss = loss + loss_G*hparams.GAN_alpha*abs(loss.item()/loss_G.item())
                if hparams.distributed_run:
                    reduced_loss = reduce_tensor(loss.data, n_gpus).item()
                    if loss_dict:
                        for key in loss_dict:
                            loss_dict[key] = reduce_tensor(loss_dict[key].data, n_gpus).item()
                else:
                    reduced_loss = loss.item()
                    if loss_dict:
                        for key in loss_dict:
                            loss_dict[key] = loss_dict[key].item()
                if hparams.use_mutual_information:
                    mi_lb, ma_et, loss_mi = learn_mine(batch=(hs_random_choosed, style_embs), mine_net=mine_net, ma_et=ma_et, hparams=hparams)
                    for name, param in mine_net.named_parameters():
                        param.requires_grad = False
                    for name, param in model.named_parameters():
                        if name in MIfrozen_list:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                    loss = loss + hparams.mutual_information_lambda*mi_lb
                    print('mi_loss=', loss_mi.item())
                    print('mi_lb_loss=', hparams.mutual_information_lambda*mi_lb.item())                                     

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
                optimizer.step()
                duration = time.perf_counter() - start
                time_window.append(duration)
                loss_window.append(reduced_loss)
                if not is_overflow and (rank == 0):
                    if iteration % hparams.log_per_checkpoint == 0:
                        if hparams.GAN_type=='lsgan':
                            print("Epoch:{} step:{} Train loss: {:>9.6f}, avg loss: {:>9.6f}, Grad Norm: {:>9.6f}, {:>5.2f}s/it, {:s} loss: {:>9.6f}, D_loss: {:>9.6f}, G_loss: {:>9.6f}, duration loss: {:>9.6f}, f0 loss: {:>9.6f}, ssim loss: {:>9.6f}, lr: {:>4}".format(
                            epoch, iteration, reduced_loss, loss_window.average, grad_norm, time_window.average, hparams.loss_type, loss_dict[hparams.loss_type], loss_D.item() if iteration > hparams.Generator_pretrain_step + 1 else 0, loss_G.item() if iteration > hparams.Generator_pretrain_step else 0, loss_dict["duration_loss"], loss_dict["pitch_loss"], loss_dict["ssim_loss"], learning_rate))
                        if hparams.GAN_type=='wgan-gp':
                            print("Epoch:{} step:{} Train loss: {:>9.6f}, avg loss: {:>9.6f}, Grad Norm: {:>9.6f}, {:>5.2f}s/it, {:s} loss: {:>9.6f}, G_loss: {:>9.6f}, duration loss: {:>9.6f}, f0 loss: {:>9.6f}, ssim loss: {:>9.6f}, lr: {:>4}".format(
                            epoch, iteration, reduced_loss, loss_window.average, grad_norm, time_window.average, hparams.loss_type, loss_dict[hparams.loss_type], loss_G.item() if iteration > hparams.Generator_pretrain_step + 1 else 0, loss_dict["duration_loss"], loss_dict["pitch_loss"], loss_dict["ssim_loss"], learning_rate))
                        if Wasserstein_D is not None:
                            loss_dict['Wasserstein_D']=Wasserstein_D.item()
                        logger.log_training(
                            reduced_loss, grad_norm, learning_rate, duration, iteration, loss_dict, mi_lb.item() if hparams.use_mutual_information else None)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}_refine_{}".format(iteration, hparams.training_files.split('/')[-2].split('_')[-1]) if hparams.is_partial_refine else "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)
            if iteration > hparams.iters:
                continue            
            iteration += 1
            torch.cuda.empty_cache()
        epoch += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-model-path', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log-dir', type=str, required=False,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument("--input-training-data-path", default=None, type=str, 
                        required=False, help="philly input data path")
    parser.add_argument('--refine_from', dest='refine_from', type=str, default=None, required=False,
                        help='load model to be refined')
    parser.add_argument('--warm_start', action='store_true', 
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--hparams_json', type=str,
                        required=False, help='hparams json file')
    
    args, _ = parser.parse_known_args()

    if args.output_model_path:
      if not os.path.isdir(args.output_model_path):
        os.makedirs(args.output_model_path, exist_ok=True)
        os.makedirs(os.path.join(args.output_model_path,'logdir'), exist_ok=True)
    if args.input_training_data_path:
      if not os.path.isdir(args.input_training_data_path):
        raise Exception("input training data path %s does not exist!" % args.input_training_data_path)

    hparams = create_hparams(args.hparams, args.hparams_json)
    output_directory = args.output_model_path
    log_directory = os.path.join(args.output_model_path,'logdir')

    hparams.training_files = os.path.join(args.input_training_data_path,'training_with_mel_frame_refine.txt')
    hparams.mel_dir = args.input_training_data_path + os.sep

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

#==============================prepare_data========================================
    input_folder = args.input_training_data_path
    refine_dir = args.input_training_data_path
    fi = open(os.path.join(input_folder, 'metadata_phone.csv'))
    uttlines_fi = fi.readlines()
    fi.close()
    encdec_attn_weights = np.load(os.path.join(input_folder, 'encdec_attn_weights.npz'))
    outputs = np.load(os.path.join(input_folder, 'outputs.npz'))
    fo = open(os.path.join(refine_dir, 'training_with_mel_frame_refine.txt'), 'w', encoding='utf-8')
    cnt = 0
    for i in range(len(uttlines_fi)-1):
        mel_id = uttlines_fi[i+1].strip().split('|')[3]
        cnt_id_1 = str(cnt)
        style_id = '0'
        cnt_id_2 = str(cnt)
        spk_id = uttlines_fi[i+1].strip().split('|')[4]
        text = uttlines_fi[i+1].strip().split('|')[-1] + ' / <EOS>'
        mel_numbers = str(outputs[outputs.files[i]].shape[0])
        final_text = mel_id + '|' + cnt_id_1 + '|' + style_id + '|' + cnt_id_2 + '|' + spk_id + '|' + text + '|' + text + '|' + text + '|' + text + '|' + mel_numbers + '\n'
        fo.write(final_text)
        cnt += 1
        np.save(os.path.join(refine_dir, 'encdec_' + str(mel_id) + '.npy'), encdec_attn_weights[encdec_attn_weights.files[i]])
        np.save(os.path.join(refine_dir, 'out_' + str(mel_id) + '.npy'), (outputs[outputs.files[i]]+4.0)/8.0)
    fo.close()
#==============================prepare_data_done====================================
    
    train(output_directory, log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams, args.refine_from)

#python train.py -o=./checkpoint -l=./logdir --refine_from=./checkpoint/ --hparams='use_gst=True,is_refine_style=True,distributed_run=False,fp16_run=False,cudnn_benchmark=False,iters=40000,batch_criterion=utterance,batch_size=4,is_multi_styles=False,is_multi_speakers=True,is_spk_layer_norm=True,use_ssim_loss=True,loss_type=L1,is_partial_refine=True,use_GAN=True,GAN_alpha=0.1'