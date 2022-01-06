import torch
import torch.nn as nn
import torch.nn.functional as F

ma_et = 1.

class Mine(nn.Module):
	def __init__(self, hparams, hidden_size=512):
		super().__init__()
		input_size = hparams.adim
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, 1)
		nn.init.normal_(self.fc1.weight,std=0.02)
		nn.init.constant_(self.fc1.bias, 0)
		nn.init.normal_(self.fc2.weight,std=0.02)
		nn.init.constant_(self.fc2.bias, 0)
		nn.init.normal_(self.fc3.weight,std=0.02)
		nn.init.constant_(self.fc3.bias, 0)

	def forward(self, input):
		output = F.elu(self.fc1(input))
		output = F.elu(self.fc2(output))
		output = self.fc3(output)
		return output

def mutual_information(joint, marginal, mine_net):
	t = mine_net(joint)
	et = torch.exp(mine_net(marginal))
	print('torch.mean(t)=',torch.mean(t))
	print('torch.mean(et)=',torch.mean(et))
	mi_lb = torch.mean(t) - torch.log(torch.mean(et) + 1e-39)
	return mi_lb, t, et

def learn_mine(batch, mine_net, ma_et, ma_rate=0.01, hparams=None):
	# batch is a tuple of (joint, marginal)
	joint , marginal = batch
	# joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
	# marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
	mi_lb , t, et = mutual_information(joint, marginal, mine_net)
	ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)

	# unbiasing use moving average
	if hparams.mi_loss_type == 'unbias':
		loss = -(torch.mean(t) - (1/(ma_et.mean() + 1e-39)).detach()*torch.mean(et))
	# use biased estimator
	elif hparams.mi_loss_type == 'bias':
		loss = - mi_lb
	else:
		raise
	return F.relu(mi_lb), ma_et, loss

def ma(a, window_size=100):
	return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]
