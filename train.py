import datahp
import model
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils as utils

###### Things to change in real training mode ########
# print_every
# save_every
# loader.load()
# print message batch, add % completion for batch/n_batch in one epoch

def train(points, chars, generate = 0):
	'''
	Gradient update for every batch of data
	'''
	# transpose points
	points_t = points.transpose(0, 1)
	seq_len = points_t.size()[0]
	char_len = chars.size()[1]   # dim(char) = batch, seq_len, *
	hidden = model_.initialize()
	
	outputs = []
	for i in range(seq_len):
		output, hidden = model_(points_t[i], chars, hidden)
		outputs.append(output)
	
	# create output tensor
	outputs2 = torch.cat([x.unsqueeze(0) for x in outputs])
	
	if generate == 0:
		criterion = model.Loss()
		loss = criterion(points_t, outputs2)

		# update gradient
		optimizer.zero_grad()
		loss.backward()
		
		# clip gradients
		utils.clip_grad_norm(model_.parameters(), max_norm = 0.1, norm_type = 2)

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
	minute = (s - hour*h)//m
	s -= hour*h + minute*m

	return '%dh %dm %ds' % (hour, minute, s)


# parameters
n_epoch = 30
batch_size = 50
input_size = 3
n_Mixture = 20     # for points
hidden_size = 400
output_size = 6*n_Mixture + 1
numMixture = 10    # for character sequence
print_every = 1
save_every = 1


# Load Data

loader = datahp.Loader(batch_size)
loader.load('test')

# Build Model
model_ = model.SynNN(input_size, hidden_size, output_size, loader.n_alphabet, numMixture, batch_size)

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
	total_loss = 0

	for batch in range(2):
		input = loader.next_batch()
		points, chars = loader.padded(input) # dim(batch, seq_len, *)
		
		# pack ?
		loss = train(points, chars, 0)
		total_loss += loss
		

		# Print out progress information
		if batch % print_every == 0:
			print (
				"epoch{} batch{} time since start {} batch loss = {:.3f}" \
				.format(
					epoch, batch,
					timesince(start),
					loss.data[0])
				)
	print ('epoch{} epoch loss = {:.3f}'.format(epoch, total_loss.data[0]))
	
	if epoch % save_every == 0:
		print ('saving model epoch ', epoch )
		torch.save(model_.state_dict(),'synNN.pt' )




