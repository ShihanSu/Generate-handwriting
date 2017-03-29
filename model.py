import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn

class SynNN(torch.nn.Module):
	def __init__(self, input_size, hidden_size, output_size, max_charlen, n_alphabet, numMixture, batch_size):
		'''
		Constructor that specify layers of the network for each time step
		The same module will be used to process every point in sequence.
		'''
		super(SynNN, self).__init__()     ## Use super for class inheritence
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.output_size = output_size
		self.numMixture = numMixture
		self.max_charlen = max_charlen    # length of longest input sequence
		self.n_alphabet = n_alphabet      # 77
		
		# architecture
		self.hidden1 = nn.GRUCell(self.input_size + self.n_alphabet, self.hidden_size, bias = True)
		self.window = Window(self.hidden_size, self.max_charlen, self.numMixture)
		self.hidden2 = nn.GRUCell(self.n_alphabet, self.output_size, bias = True)


	def forward(self, points, chars, hidden):
		'''
		Define forward pass of the network to calculate the output of single step
		Arguments:
			Points: (1, batch_size, 3)
			chars: (batch_size, seq_u, 77)
			Hidden: initial hidden state (n_layers, batch_size, output_size for each layer)
		Returns:
			Output: the output of multi layer for each time step (seq_length, batch, output_size)
			Hidden: num_layers, batch, output_size of each layer)
		'''
		# extract initial states
		hidden1 = hidden[0]
		window = hidden[1]
		hidden2 = hidden[2]
		lastk = hidden[3]                   # dim(char_len, 10)
	
		# concatenate sequence of input tensor
		input1 = torch.cat([points, window], 1)
		output1 = self.hidden1(input1, hidden1)
		window, lastk = self.window(output1, chars, lastk)
		output2 = self.hidden2(window, hidden2)  # dim(batch, 121)
		
		hidden = [output1, window, output2, lastk]
		if output2.size()[0] != 0:
			output2.unsqueeze(0)                   # dim(1, batch, 121)
		return output2, hidden
		
	def initialize(self, char_len):
		''' 
		Initialize for each batch
		char_len is length of char batch
		'''
		hidden1_ini = Variable(torch.zeros(self.batch_size, self.hidden_size))
		window_ini = Variable(torch.zeros(self.batch_size, self.n_alphabet))
		hidden2_ini = Variable(torch.zeros(self.batch_size, self.output_size))
		k_ini = Variable(torch.zeros(1, char_len, self.numMixture))

		return (hidden1_ini, window_ini, hidden2_ini, k_ini)


class Window(nn.Module):
	''' 
	Define the calculation for window layer.
	'''
	def __init__(self, hidden_size, max_charlen, numMixture):
		super(Window, self).__init__()
		self.u = max_charlen   # the maximum of char sequence length in the data
		self.k = numMixture
		self.weight = nn.Parameter(torch.FloatTensor(hidden_size, 3*self.u*self.k)) # 3 for a,b,k
		self.bias = nn.Parameter(torch.FloatTensor(1, 3*self.u*self.k))

	def forward(self, hidden, chars, lastk):
		'''
		Implementing eq 46 - 51
		Dim:
			chars: (batch, char_len, 77)
			hidden: (batch, hidden_size)
		'''
		initial = torch.mm(hidden, self.weight)                        # eq 48 dim(batch_size, 3ku)
		bias = self.bias.expand_as(initial)
		initial = initial + bias
		
		# unpack a, b, k
		alpha_h, beta_h, k_h = torch.split(initial, self.u*self.k, dim = 1)  # dim(batch_size, u*k)
		batch_size = initial.size()[0]
				
		# reshape
		alpha_ = alpha_h.contiguous().view(batch_size, self.u, self.k)        # dim(batch, u, k)
		beta_ = beta_h.contiguous().view(batch_size, self.u, self.k)
		k_ = k_h.contiguous().view(batch_size, self.u, self.k)
		
		# slice parameters according to chars length and exp
		char_len = chars.size()[1]
		alpha = alpha_[:, 0:char_len, :].exp()
		beta = beta_[:, 0:char_len, :].exp()
		k = k_[:, 0:char_len, :].exp()

		lastk = lastk.expand_as(k)                                     # lastk(1, char_len, k)
		k = k + lastk                                                  # dim(batch, char_len, k)
		
		# phi

		u = Variable(torch.range(0, char_len - 1).view(char_len, 1).expand_as(k))
		exp_part = torch.exp(-torch.mul(beta, torch.pow(k - u, 2)))    # dim(batch, char_len, k)
		phi = torch.sum(torch.mul(alpha,exp_part), dim = 2)            # sum over k, dim(batch, char_len, 1)
		phi = torch.transpose(phi, 1, 2)                               # dim(batch, 1, char_len)
		
		# w
		w = torch.bmm(phi, chars)                                      # dim(batch, 1, 77)
		w = torch.squeeze(w)                                           # dim(batch, 77)
		return w, k


class Loss(torch.nn.Module):
	'''
	Implement the loss function from output from RNN.
	Ref paper: https://arxiv.org/abs/1308.0850
	'''
	def __init__(self):
		'''
		x is sequence of coordinates with dim (batch, seq_length, 3).
		Parameters are sequence of output from rnn with dim (batch, seq_length, 128).
		'''
		super(Loss,self).__init__()
		self.e = []      	  # predicted end of stroke probability scalar
		self.m1 = []        # vector of means for x1 with len 20
		self.m2 = []        # vector of means for x2 with len 20
		self.pi = []        # vector of mixture density network coefficients with len 20
		self.rho = []       # vector of correlation with len 20
		self.s1 = []        # vector of standard deviation for x1 with len 20
		self.s2 = []        # vector of standard deviation for x2 with len 20
		self.x1 = []        # x1 coordinate at t+1
		self.x2 = []        # x2 coordinates at t + 1
		self.et = []        # end of probability indicator from ground truth
		self.batch = 0      # batch size
		self.seq_length = 0 # reduce by 1 because loss is caculated at t+1 timestamp
		self.parameters = []

	
	def forward(self, x, para):
		''' 
		Implement eq 26 of ref paper for each batch.
		Input:
			para: dim(seq_len, batch, 121)
			x:    dim(seq_len, batch, 3)
		'''
		if x.size()[0] == para.size()[0]:
			self.seq_length = x.size()[0] - 1
			total_loss = 0
			for i in range(self.seq_length):
				# prepare parameters
				self.__get_para(i, x, para)
				normalpdf = self.__para2normal(self.x1, self.x2, self.m1, self.m2, self.s1, self.s2, self.rho) #dim (n_batch, 20)
				single_loss = self.__singleLoss(normalpdf)
				total_loss += single_loss
			return total_loss
		else:
			raise Exception("x and para don't match")

	
	def __get_para(self, i, x, para):
		'''
		Slice and process parameters to the right form.
		Implementing eq 18-23 of ref paper.
		'''
		self.batch = x.size()[1]
		self.e = torch.sigmoid(-para[i,:,0])  # eq 18
		self.parameters = para
	
		# slice remaining parameters and training inputs
		self.pi, self.m1, self.m2, self.s1, self.s2, self.rho = torch.split(self.parameters[i,:,1:], 20, dim = 1) # dim(batch, 20)
		self.x1 = x[i+1,:,0].contiguous().view(self.batch, 1).expand_as(self.m1) # dim(batch, 20)
		self.x2 = x[i+1,:,1].contiguous().view(self.batch, 1).expand_as(self.m2)
		self.et = x[i+1,:,2].contiguous().view(self.batch, 1)
		
		## process parameters
		# pi
		max_pi = torch.max(self.pi, dim = 1)[0]
		max_pi = max_pi.expand_as(self.pi)
		diff = self.pi - max_pi
		red_sum = torch.sum(diff, dim = 1).expand_as(self.pi)
		self.pi = diff.div(red_sum)
	
		# sd
		self.s1 = self.s1.exp()
		self.s2 = self.s2.exp()
	
		# rho
		self.rho = self.rho.tanh()

	
	def __para2normal(self, x1, x2, m1, m2, s1, s2, rho):
		'''
		Implement eq 24, 25 of ref paper.
		All input with dim(1, batch, 20)
		'''
		norm1 = x1.sub(m1)
		norm2 = x2.sub(m2)
		s1s2 = torch.mul(s1, s2)
		z = torch.pow(torch.div(norm1, s1), 2) + torch.pow(torch.div(norm2, s2), 2) - \
			2*torch.div(torch.mul(rho, torch.mul(norm1, norm2)), s1s2)
		negRho = 1 - torch.pow(rho, 2)
		expPart = torch.exp(torch.div(-z, torch.mul(negRho, 2)))
		coef = 2*np.pi*torch.mul(s1s2, torch.sqrt(negRho))
		result = torch.div(expPart, coef)
		return result

		
	def __singleLoss(self, normalpdf):
		'''
		Calculate loss for single time stamp. eq 26
		Input: normalpdf (1,n_batch, 20).
		'''
		epsilon = 1e-20  # floor of loss from mixture density component since initial loss could be zero
		mix_den_loss = torch.mul(self.pi, normalpdf)
		red_sum_loss = torch.sum(torch.log1p(mix_den_loss))  # sum for all batch
		end_loss = torch.sum(torch.log1p(torch.mul(self.e, self.et) + torch.mul(1-self.e, 1 - self.et)))
		total_loss = -red_sum_loss - end_loss
		
		return total_loss/self.batch
		
		
