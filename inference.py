import argparse
import os, io
import shutil
import sys, json
sys.path.append(os.path.dirname(sys.path[0]))
import zipfile
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch
import numpy as np
import soundfile
from scipy.io.wavfile import read as wavread
from hparams import create_hparams
from models import load_model
from utils.utils import _plot_and_save, save_htk_data
from utils.audio import Audio
from utils.data_reader import _read_meta_yyh
from utils.data_reader_refine import TextMelLoader_refine
from utils.utils import get_checkpoint_path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(model_path, hparams):
    # Load model from checkpoint
    model = load_model(hparams)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    for name, param in model.named_parameters():
        print(str(name), param.requires_grad, param.shape) 
    model.to(device).eval()
    return model

class Synthesizer():
    def __init__(self, model_path, out_dir, text_file, sil_file, use_griffin_lim, hparams):
        self.model_path = model_path
        self.out_dir = out_dir
        self.text_file = text_file
        self.sil_file = sil_file
        self.use_griffin_lim = use_griffin_lim
        self.hparams = hparams

        self.model = get_model(model_path, hparams)
        self.audio_class = Audio(hparams)

        if hparams.use_phone:
            from text.phones import Phones
            phone_class = Phones(hparams.phone_set_file)
            self.text_to_sequence = phone_class.text_to_sequence
        else:
            from text import text_to_sequence
            self.text_to_sequence = text_to_sequence


        self.out_wav_dir = self.out_dir
        # self.out_wav_dir = os.path.join(self.out_dir, 'wav')
        # os.makedirs(self.out_wav_dir, exist_ok=True)
        
        # self.out_att_dir = os.path.join(self.out_dir, 'att')
        # os.makedirs(self.out_att_dir, exist_ok=True)
    def get_inputs(self, meta_data):
        hparams = self.hparams
        SEQUENCE_ = []
        SPEAKERID_ = []
        STYLE_ID_ = []
        FILENAME_ = []
        # Prepare text input
        for i in range(len(meta_data)):
            filename = meta_data[i].strip().split('|')[1]
            print('Filename=', filename)

            phone_text = meta_data[i].strip().split('|')[-1]
            print('Text=', phone_text)

            speaker_id = int(meta_data[i].strip().split('|')[-2])
            print('SpeakerID=', speaker_id)
            
            sequence = np.array(self.text_to_sequence(meta_data[i].strip().split('|')[-1], ['english_cleaners']))   # [None, :]
            print(sequence)
            
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()
            
            speaker_id = torch.LongTensor([speaker_id]).to(device) if hparams.is_multi_speakers else None

            style_id = torch.LongTensor([int(meta_data[i].strip().split('|')[1])]).to(device) if hparams.is_multi_styles else None

            SEQUENCE_.append(sequence)
            SPEAKERID_.append(speaker_id)
            STYLE_ID_.append(style_id)
            FILENAME_.append(filename)

        return SEQUENCE_, SPEAKERID_, STYLE_ID_, FILENAME_

    def gen_mel(self, meta_data):
        SEQUENCE_, SPEAKERID_, STYLE_ID_, FILENAME_ = self.get_inputs(meta_data)
        MEL_OUTOUTS_ = []
        FILENAME_NEW_ = []
        style_embs_generated = None
        # Decode text input and plot results
        with torch.no_grad():
            for i in range(len(SEQUENCE_)):
                mel_outputs, _, _, att, _, _, style_embs_generated = self.model.inference(text=SEQUENCE_[i], spk_ids=SPEAKERID_[i], utt_mels=TextMelLoader_refine(hparams.training_files, hparams, is_inference=True).utt_mels, style_embs_generated=style_embs_generated) if hparams.use_gst else self.model.inference(text=SEQUENCE_[i], spk_ids=SPEAKERID_[i])
                MEL_OUTOUTS_.append(mel_outputs.transpose(0, 1).float().data.cpu().numpy())  # (dim, length)
                FILENAME_NEW_.append('spkid_' + str(SPEAKERID_[i].item()) + '_filenum_' + FILENAME_[i])
                print('mel_outputs.shape=',mel_outputs.shape)
                # if att is not None:
                #     image_path = os.path.join(self.out_att_dir, FILENAME_NEW_[-1] + '.png')
                #     txt_path = os.path.join(self.out_att_dir, FILENAME_NEW_[-1] + '.txt')
                #     _plot_and_save(att[-1].unsqueeze(0).float().data.cpu().numpy(), image_path)
                # print("att.shape={},\n, att[-1].unsqueeze(0).float().data.cpu().numpy()={},\n".format(att.shape, att[-1].unsqueeze(0).float().data.cpu().numpy()), file=open(txt_path,'w+'))
        return MEL_OUTOUTS_, FILENAME_NEW_

    def gen_wav_griffin_lim(self, mel_outputs, filename):
        grf_wav = self.audio_class.inv_mel_spectrogram(mel_outputs)
        grf_wav = self.audio_class.inv_preemphasize(grf_wav)
        wav_path = os.path.join(self.out_wav_dir, "{}-gl.wav".format(filename))
        self.audio_class.save_wav(grf_wav, wav_path)

    def inference_f(self):
        # print(meta_data['n'])
        meta_data = _read_meta_yyh(self.text_file)
        MEL_OUTOUTS_, FILENAME_NEW_ = self.gen_mel(meta_data)
        for i in range(len(MEL_OUTOUTS_)):
            np.save(os.path.join(self.out_wav_dir, "{}.npy".format(FILENAME_NEW_[i] + '_' + self.model_path.split('/')[-1])), MEL_OUTOUTS_[i].transpose(1, 0))
            if self.use_griffin_lim:
                self.gen_wav_griffin_lim(MEL_OUTOUTS_[i], FILENAME_NEW_[i] + '_' + self.model_path.split('/')[-1])
            
        return print('finished')

if __name__ == '__main__':
    def str2bool(s):
        s = s.lower()
        assert s in ["true", "false"]
        return {'t': True, 'f': False}[s[0]]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-training-data-path", default=None, type=str, 
                        required=False, help="philly input data path")
    parser.add_argument('-o', '--output-model-path', type=str,
                        help='directory to save mel npy files')#automaticly has following output-model-path documents
    parser.add_argument("--input-previous-model-path", default=None, type=str, 
                        required=False, help="philly input model path")    
    parser.add_argument("--input-validation-data-path", default=None, type=str, 
                        required=False, help="philly input data path")
    parser.add_argument('-s', '--sil_file', type=str, default="utils/Seed_16k.wav",
                        required=False, help='silence audio')
    parser.add_argument('--use_griffin_lim', type=str2bool, default=False,
                        help='whether generate wav using grifflin lim')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--hparams_json', type=str,
                        required=False, help='hparams json file')

    args, _ = parser.parse_known_args()
    hparams = create_hparams(args.hparams, args.hparams_json)
    
    if args.input_training_data_path:
        if not os.path.isdir(args.input_training_data_path):
            raise Exception("input training data path %s does not exist!" % args.input_training_data_path)
        model_path_dir = os.path.join(args.input_training_data_path, "output-model-path")
        model_path = get_checkpoint_path(model_path_dir)
    
    if args.output_model_path:
        if not os.path.isdir(args.output_model_path):
            os.makedirs(args.output_model_path, exist_ok=True)

    if args.input_previous_model_path:
        hparams.training_files = os.path.join(args.input_previous_model_path,'training_with_mel_frame_refine.txt')
        hparams.mel_dir = args.input_previous_model_path + os.sep       

    if args.input_validation_data_path:
        input_folder = args.input_validation_data_path
        test_file_before = os.path.join(input_folder, 'metadata_phone_test.csv')
        test_file_after = os.path.join(input_folder, 'test_txt_others.txt')
        fi = open(test_file_before)
        uttlines_fi = fi.readlines()
        fi.close()
        fo = open(test_file_after, 'w', encoding='utf-8')
        cnt = 0
        style_id = '0'
        spk_id = '182'
        for i in range(len(uttlines_fi)-1):
            text = uttlines_fi[i].strip() + ' / <EOS>'
            final_text = str(cnt) + '|' + str(cnt) + '|' + style_id + '|' +str(cnt) + '|' + spk_id + '|' + text + '\n'
            fo.write(final_text)
            cnt += 1
        fo.close()
        text_file = test_file_after

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

    synthesizer = Synthesizer(model_path, args.output_model_path, text_file, args.sil_file,
                              args.use_griffin_lim, hparams)
    synthesizer.inference_f()

# export NPY_MKL_FORCE_INTEL=1 && python inference.py --hparams='use_batch_norm=False,use_f0=True,log_f0=True,distributed_run=False,fp16_run=False,cudnn_benchmark=False,numberworkers=4,use_gaussian_upsampling=True,use_gst=True,use_mutual_information=False,gst_train_att=True,is_partial_refine=True,is_refine_style=True,style_embed_integration_type=concat,gst_reference_encoder=multiheadattention,style_vector_type=mha,gst_reference_encoder_mha_layers=4,is_multi_styles=False,is_multi_speakers=True,is_spk_layer_norm=True,training_files=./refine_data_SDIM02_20/training_with_mel_frame_refine.txt,mel_dir=./refine_data_SDIM02_20/,phone_set_file=./phone_set.json'
#gst:
# export NPY_MKL_FORCE_INTEL=1 && python inference.py --hparams='use_batch_norm=False,use_f0=True,log_f0=True,distributed_run=False,fp16_run=False,cudnn_benchmark=False,numberworkers=4,use_gaussian_upsampling=True,use_gst=True,use_mutual_information=False,gst_train_att=True,is_partial_refine=True,is_refine_style=True,style_embed_integration_type=concat,gst_reference_encoder=convs,style_vector_type=gru,is_multi_styles=False,is_multi_speakers=True,is_spk_layer_norm=True,training_files=./refine_data_Eva_20/training_with_mel_frame_refine.txt,mel_dir=./refine_data_Eva_20/,phone_set_file=./phone_set.json'
#valinna:
# export NPY_MKL_FORCE_INTEL=1 && python inference.py --hparams='use_batch_norm=True,use_f0=False,log_f0=False,distributed_run=False,fp16_run=False,cudnn_benchmark=False,numberworkers=4,use_gaussian_upsampling=False,use_gst=False,use_mutual_information=False,gst_train_att=False,is_partial_refine=True,is_refine_style=True,is_multi_styles=False,is_multi_speakers=True,is_spk_layer_norm=True,training_files=./refine_data_Eva_20/training_with_mel_frame_refine.txt,mel_dir=./refine_data_Eva_20/,phone_set_file=./phone_set.json'