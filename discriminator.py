import torch
import torch.autograd as autograd
from torch.autograd import Variable
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention


class GP(torch.nn.Module):
	def __init__(self, hparams):
		super(GP, self).__init__()
		self.hparams = hparams

	def forward(self, netD, real_data, fake_data, olens):
		alpha = torch.rand(real_data.size(0), 1, 1)
		alpha = alpha.expand(real_data.size())
		alpha = alpha.cuda()

		interpolates = alpha * real_data + ((1 - alpha) * fake_data)

		interpolates = interpolates.cuda()
		interpolates = Variable(interpolates, requires_grad=True)

		disc_interpolates = netD(interpolates, olens)

		gradients = autograd.grad(outputs=disc_interpolates,inputs=interpolates,grad_outputs=torch.ones(disc_interpolates.size()).cuda(),create_graph=True, retain_graph=True, only_inputs=True)[0]
		gradient_penalty = (torch.sqrt(torch.sum(torch.sum(torch.pow(gradients,2),dim=2),dim=1))-1.)**2

		return gradient_penalty.mean()* self.hparams.GP_beata

class Wgan_GP(torch.nn.Module):
	"""docstring for WGAN_GP"""
	def __init__(self, hparams, window_sizes=[100, 50], channels=[128, 64, 32], dropout_rate=0.3):
		super(Wgan_GP, self).__init__()
		self.hparams = hparams
		self.window_sizes = window_sizes
		self.channels = channels
		self.convs = torch.nn.ModuleList()
		self.smooth_dense_layer = torch.nn.ModuleList()

		for k in range(len(channels)):
			self.convs_k = torch.nn.Sequential(Conv2Norm(in_channels=1,
													out_channels=channels[k],
													kernel_size=(3,3),
													bias=False,
													w_init_gain='leaky_relu'
													),
										torch.nn.BatchNorm2d(channels[k]),
										torch.nn.LeakyReLU(),
										Conv2Norm(in_channels=channels[k],
													out_channels=channels[k],
													kernel_size=(3,3),
													bias=False,
													w_init_gain='leaky_relu'
													),
										torch.nn.BatchNorm2d(channels[k]),
										torch.nn.LeakyReLU(),
										Conv2Norm(in_channels=channels[k],
													out_channels=channels[k],
													kernel_size=(3,3),
													bias=False,
													w_init_gain='leaky_relu'
													),
										torch.nn.BatchNorm2d(channels[k]),
										torch.nn.LeakyReLU(),
										torch.nn.Dropout(dropout_rate))
			self.dense_k = torch.nn.Linear(channels[k]*hparams.num_mels, 32)
			self.convs.append(self.convs_k)
			self.smooth_dense_layer.append(self.dense_k)
		
		self.multihead_attention = MultiHeadedAttention(hparams.aheads, 32, hparams.transformer_enc_dropout_rate)
		self.smooth_dense_layer_final = torch.nn.Linear(32, 1)

	def discriminator_factory(self, convs, input, layer_number):
		discrim_final_output = None
		discrim_output = self.convs[layer_number](input)#(B,128,T,80)
		discrim_output = self.smooth_dense_layer[layer_number](discrim_output.transpose(1,2).contiguous().view(discrim_output.size(0),discrim_output.size(2),-1))#(B,T,32)
		discrim_output = self.multihead_attention(query=discrim_output,key=discrim_output,value=discrim_output,mask=None)#(B,T,32)
		discrim_output = self.smooth_dense_layer_final(discrim_output).squeeze(-1)#(B,T)
		discrim_final_output = torch.mean(discrim_output,-1)#(B,)
		
		return discrim_final_output


	def forward(self, gen_output, olens):#(B,T,80)
		out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(gen_output.device)
		gen_output = gen_output.masked_fill(out_masks.eq(0), 0.0)
		window_frame = []

		for win_size in self.window_sizes:
			maxv = 1 if (gen_output.size(1) - win_size) < 1 else (gen_output.size(1) - win_size)
			start_idx = int(torch.distributions.uniform.Uniform(0,maxv).sample())
			end_idx = start_idx + win_size if (start_idx + win_size) < gen_output.size(1) -1 else gen_output.size(1) -1
			window_frame.append((start_idx, end_idx))

		gen_output = gen_output.unsqueeze(1)#(B,1,T,80)
		discrim_gen_output = self.discriminator_factory(convs=self.convs, input=gen_output, layer_number=0)
		layer_num = 1
		for frame, channel in zip(window_frame, self.channels[1:]):
			discrim_gen_output_new = self.discriminator_factory(convs=self.convs, input=gen_output[:, :, frame[0]: frame[1], :], layer_number=layer_num)
			discrim_gen_output += discrim_gen_output_new
			layer_num += 1

		return 	discrim_gen_output
		

class Conv2Norm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,1), stride=(1,1),
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(Conv2Norm, self).__init__()
        if padding is None:
            assert(kernel_size[0] % 2 == 1)
            assert(kernel_size[1] % 2 == 1)
            padding = (int(dilation * (kernel_size[0] - 1) / 2),int(dilation * (kernel_size[1] - 1) / 2))

        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(self.conv.weight,gain=torch.nn.init.calculate_gain(w_init_gain)) if w_init_gain != 'leaky_relu' \
        else  torch.nn.init.xavier_uniform_(self.conv.weight,gain=torch.nn.init.calculate_gain('leaky_relu', 0.1))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Lsgan_Loss(torch.nn.Module):

	def __init__(self, hparams):
		super(Lsgan_Loss, self).__init__()
		self.hparams = hparams
		self.l2_criterion = torch.nn.MSELoss()
	
	def forward(self, discrim_gen_output, discrim_target_output, train_object='G'):
		self.zeros = torch.zeros(discrim_gen_output[0].size()).cuda(non_blocking=True).to(discrim_gen_output[0].device) \
						if torch.cuda.is_available() else torch.zeros(discrim_gen_output[0].size()).to(discrim_gen_output[0].device)
		
		self.ones = torch.ones(discrim_gen_output[0].size()).cuda(non_blocking=True).to(discrim_gen_output[0].device) \
						if torch.cuda.is_available() else torch.ones(discrim_gen_output[0].size()).to(discrim_gen_output[0].device)

		if train_object == 'D':
			lsgan_loss_D = self.l2_criterion(discrim_gen_output[0], self.zeros) + self.l2_criterion(discrim_target_output[0], self.ones) \
							+ (self.l2_criterion(discrim_gen_output[1], self.zeros) + self.l2_criterion(discrim_target_output[1], self.ones)) if self.hparams.is_partial_refine else 0
			return lsgan_loss_D
		
		if train_object == 'G':
			lsgan_loss_G = self.l2_criterion(discrim_gen_output[0], self.ones) \
							+ self.l2_criterion(discrim_gen_output[1], self.ones) if self.hparams.is_partial_refine else 0
			return lsgan_loss_G

class Calculate_Discrim(torch.nn.Module):

	def __init__(self, hparams, window_sizes=[100, 50], channels=[128, 64, 32], dropout_rate=0.3):
		super(Calculate_Discrim, self).__init__()
		self.hparams = hparams
		self.window_sizes = window_sizes
		self.channels = channels
		self.convs = torch.nn.ModuleList()
		self.smooth_dense_layer = torch.nn.ModuleList()

		for k in range(len(channels)):
			self.convs_k = torch.nn.Sequential(Conv2Norm(in_channels=1,
													out_channels=channels[k],
													kernel_size=(3,3),
													bias=False,
													w_init_gain='leaky_relu'
													),
										torch.nn.BatchNorm2d(channels[k]),
										torch.nn.ReLU(),
										Conv2Norm(in_channels=channels[k],
													out_channels=channels[k],
													kernel_size=(3,3),
													bias=False,
													w_init_gain='leaky_relu'
													),
										torch.nn.BatchNorm2d(channels[k]),
										torch.nn.ReLU(),
										Conv2Norm(in_channels=channels[k],
													out_channels=channels[k],
													kernel_size=(3,3),
													bias=False,
													w_init_gain='leaky_relu'
													),
										torch.nn.BatchNorm2d(channels[k]),
										torch.nn.ReLU(),
										torch.nn.Dropout(dropout_rate))
			self.dense_k = torch.nn.Linear(channels[k]*hparams.num_mels, 32)
			self.convs.append(self.convs_k)
			self.smooth_dense_layer.append(self.dense_k)
		
		self.multihead_attention = MultiHeadedAttention(hparams.aheads, 32, hparams.transformer_enc_dropout_rate)
		self.smooth_dense_layer_final = torch.nn.Linear(32, 1)

	def discriminator_factory(self, convs, input, layer_number, is_target_mel=False):
		discrim_final_output = None
		discrim_output = self.convs[layer_number](input)#(B,128,T,80)
		discrim_output = self.smooth_dense_layer[layer_number](discrim_output.transpose(1,2).contiguous().view(discrim_output.size(0),discrim_output.size(2),-1))#(B,T,32)
		discrim_output = self.multihead_attention(query=discrim_output,key=discrim_output,value=discrim_output,mask=None)#(B,T,32)
		discrim_output_temp = discrim_output#(B,T,32)
		discrim_output_temp, input_indexes = torch.max(discrim_output_temp,-1) if not is_target_mel else torch.min(discrim_output_temp,-1)#(B,T)
		discrim_output_minmax = torch.mean(discrim_output_temp,-1)#(B,)
		discrim_output = self.smooth_dense_layer_final(discrim_output).squeeze(-1)#(B,T)
		discrim_output_final = torch.mean(discrim_output,-1)#(B,)
		if self.hparams.is_partial_refine:
			discrim_final_output = [discrim_output_final] + [discrim_output_minmax]
		else:
			discrim_final_output = [discrim_output_final]
		
		return discrim_final_output


	def forward(self, gen_output, target_output, olens):#(B,T,80)
		out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(target_output.device)
		gen_output = gen_output.masked_fill(out_masks.eq(0), 0.0)
		target_output = target_output.masked_fill(out_masks.eq(0), 0.0)

		window_frame = []

		for win_size in self.window_sizes:
			maxv = 1 if (gen_output.size(1) - win_size) < 1 else (gen_output.size(1) - win_size)
			start_idx = int(torch.distributions.uniform.Uniform(0,maxv).sample())
			end_idx = start_idx + win_size if (start_idx + win_size) < gen_output.size(1) -1 else gen_output.size(1) -1
			window_frame.append((start_idx, end_idx))

		gen_output = gen_output.unsqueeze(1)#(B,1,T,80)
		target_output = target_output.unsqueeze(1)#(B,1,T,80)
		discrim_gen_output = self.discriminator_factory(convs=self.convs, input=gen_output, layer_number=0, is_target_mel=False)
		layer_num = 1
		for frame, channel in zip(window_frame, self.channels[1:]):
			discrim_gen_output_new = self.discriminator_factory(convs=self.convs, input=gen_output[:, :, frame[0]: frame[1], :], layer_number=layer_num, is_target_mel=False)
			discrim_gen_output[0] += discrim_gen_output_new[0]
			if self.hparams.is_partial_refine:
				discrim_gen_output[1] += discrim_gen_output_new[1]
			layer_num += 1

		discrim_target_output = self.discriminator_factory(convs=self.convs, input=target_output, layer_number=0, is_target_mel=True)
		layer_num = 1
		for frame, channel in zip(window_frame, self.channels[1:]):
			discrim_target_output_new = self.discriminator_factory(convs=self.convs, input=target_output[:, :, frame[0]: frame[1], :], layer_number=layer_num, is_target_mel=True)
			discrim_target_output[0] += discrim_target_output_new[0]
			if self.hparams.is_partial_refine:
				discrim_target_output[1] += discrim_target_output_new[1]
			layer_num += 1

		return 	discrim_gen_output, discrim_target_output








		