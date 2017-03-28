import datahp
import model
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils as utils



# parameters
n_epoch = 30
batch_size = 50
print_every = 10
input_size = 3
n_Mixture = 20     # for points
hidden_size = 400
output_size = 6*n_Mixture + 1
numMixture = 10    # for character sequence


# Load Data

loader = datahp.Loader(batch_size)
loader.load()

# Build Model
model_ = model.SynNN(input_size, hidden_size, output_size, loader.max_charlen(), loader.n_alphabet(), numMixture, batch_size)

# specify optimizer
optimizer = torch.optim.RMSprop(model_.parameters(), alpha = 0.95, eps = 0.0001)
	#momentum = 0.9, centered = True)

# initialize weights
for para in list(model_.parameters()):
	init.normal(para.data, 0, 0.075)

# track of time spent
start = time.time()
for epoch in range(n_epoch):
	# Reset data loader pointer
	loader.reset()

	for batch in range(loader.n_batch()):
		input = loader.next_batch()
		points, chars = loader.padded(input) # dim(batch, seq_len, *)
		
		# pack ?
		loss = train(points, chars, 0)
		

		# Print out progress information
		if batch % print_every == 0:
			print (
				"epoch{} {} batch{} {} {} loss = {:.3f}" \
				.format(
					epoch, epoch/n_epoch * 100, batch, batch/batch_size*100,
					timesince(start), loss)
				)

	torch.save(model,'synNN.pt' )




def train(points, chars, generate = 0):
	'''
	Gradient update for every batch of data
	'''
	# transpose points
	points.transpose_(0, 1)
	seq_len = points.size()[0]
	char_len = chars.size()[1]   # dim(char) = batch, seq_len, *
	hidden = model.initialize(char_len)
	
	outputs = []
	for i in range(seq_len):
		output, hidden = model(points[i], chars, hidden)
		outputs.append(output)
	
	# create variable from list
	outputs = Variable(torch.cat(tensor for tensor in outputs))
	
	if generate == 0:
		criterion = model.Loss()
		loss = criterion(points, outputs)

		# update gradient
		optimizer.zero_grad()
		loss.backward()
		
		# clip gradients  ?
		#utils.clip_grad_norm(model.parameters(), max_norm = 20, norm_type = 2)

		optimizer.step()
		return loss

	return outputs




def timesince(since):
	'''
	Calculate total training time since initialization
	'''
	now = time.time()
	s = now - since
	h = 60*60      # hour = seconds
	m = 60          # minimute = secondes
	hour = s//h
	mimute = (s - hour*h)//m
	s -= hour*h + minute*model

	return '%dh %dm %ds' % (hour, minute, s)
