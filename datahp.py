import glob
import math
import os
import pickle
import torch
import xml.etree.ElementTree as ET
from torch.autograd import Variable
import torch.nn.utils as utils

class Loader():
	def __init__(self, batch_size = 50):
		self.batch_size = batch_size
		self.data_dir = './data'       # Root of data directory
		self.pointer = 0               # The index of starting index of current batch input
		self.char_dict = {}            # key: filename; value: char string for the file
		self.alphabet = {}             # key: alphabet; value: index number
		self.data = []                 # Entire training data set
		self.reset()
		self.max_charlen = 0           # max len of char sequence
		self.n_alphabet = 0            # num of unique alphabet in chars
		self.n_batch = 0               # num of total batches in the data
	
	def load(self, mode):
		''' 
		A private method to Load entire dataset and save it into binary form
		'''
		binaryf_path = os.path.join(self.data_dir, 'training_data.cpkl')
		alphabet_path = os.path.join(self.data_dir, 'alphabet.cpkl')
		test_path = os.path.join(self.data_dir, 'test_data.cpkl')
		
		if mode == 'test':
			self.data = self.__reload(test_path)
		
		else:
			if not os.path.exists(binaryf_path):
				print ('Creating training data pkl file from raw source')
				#specify data and text path
				data_path = self.data_dir + '/lineStrokes'
				text_path = self.data_dir + '/original'
				
				# load seq-character dictionary
				textlist = self.__listxml(text_path)
				self.__loaddict(textlist)
				
				# load data
				datalist = self.__listxml(data_path)
				points = self.__readxml(datalist)
				data = self.__points2array(points)
				
				# sort data according to sequence length
				data = self.__sort(data)
				
				# save data
				self.__save2binary(data, self.alphabet, binaryf_path, alphabet_path)
		
			self.data = self.__reload(binaryf_path)
			
		self.alphabet = self.__reload(alphabet_path)
		self.max_charlen = max([len(x[1]) for x in self.data]) + 1       # match with padding size
		self.n_alphabet = len(self.alphabet)
		self.n_batch = math.ceil(len(self.data)/self.batch_size)
		self.reset()


	def __listxml(self, path):
		'''
		Extract all xml files from subdirectories of self.data_dir into a list
		'''
		filelist = glob.glob(path + '/*/*/*.xml')
		return filelist
		

	def __loaddict(self, textlist):
		'''
		Method to build the seq-character and character index dictionary.
		'''
		alphabet = []           # store all characters occured in data file
		for file in textlist:
			tree = ET.parse(file)
			root = tree.getroot()
			transcript = root[1]

			for line in transcript.findall('TextLine'):
				text = line.attrib['text']
				self.char_dict[line.attrib['id']] = list(text)      # match sequence to text
				alphabet.extend(list(text))
		
		
		alphabet = set(alphabet)
		for index, value in enumerate(alphabet):
			self.alphabet[value] = index           # match alphabet to index


	def __readxml(self, filelist):
		'''
		Extract points element from each file in filelist.
		Output:
			A list of tuple ([points], [index]) of length = len(filelist that has corresponding chars)
				where [points] is a sequence of all x, y coordinates of all strokes in a single xml file
							[index] is the index of end of stroke points in the sequence.
		'''
		total_seq = []
		for file in filelist:
		
			# characters of the sequence
			filename = file.split('/')[-1]
			chars = self.char_dict.get(filename[:-4], None) # some sequence don't have corresponding character sequence
			if chars:
				tree = ET.parse(file)
				root = tree.getroot()
				board = root[0]      # white board set up
			
				# creating offseting points
				x_offset = 1e20
				y_offset = 1e20
				y_height = 0
			
				for i in range(1, len(board)):
					x_offset = min(x_offset, float(board[i].attrib['x']))
					y_offset = min(y_offset, float(board[i].attrib['y']))
					y_height = max(y_height, float(board[i].attrib['y']))
				
				y_height -= y_offset
				x_offset -= 100
				y_offset -= 100
				
				# get all points in a strokeSet(sequence of strokes) into one list
				strokeSet = root[1]
				points = []
				
				# keep track of end of stroke
				index = []
				i = -1
				
				for stroke in strokeSet.findall('Stroke'):
					for point in stroke.findall('Point'):
						i += 1
						points.append([float(point.attrib['x']) - x_offset, float(point.attrib['y']) - y_offset])
					index.append(i)
				total_seq.append((points, index, chars))
		return total_seq
		

	def __points2array(self, total_seq):
		'''
		Convert raw points data into trainable format
		Output format example:
		[[[[x11, y11, I11],[x12, y12, I12]], char_data1 ]
		 [[[x21, y21, I21],[x22, y22, I22]], char_data2]
		 	x, y is the change from previous x, y, I indicate whether this point is end of stroke
			char_data is the variable with size (seq_len, 77)
		 	output dimensiton is (n_files, 2, n_points, 3)
		'''
		seq_len = len(total_seq)
		output = torch.zeros(len(total_seq), 2)
		for tuple in total_seq:
			points = tuple[0]
			index = tuple[1]
			chars = tuple[2]
			
			# chars array
			len_seq = len(chars)
			char_data = torch.zeros(len_seq, self.n_alphabet)

			for index, char in enumerate(chars):
				char_data[index, self.alphabet[char]] = 1
			
			# points array
			n_points = len(points)
			data = torch.zeros((n_points, 3))
			
			# inialize starting x, y coordinates
			prev_x = 0
			prev_y = 0

			for i in range(n_points):
				data[i, 0] = points[i][0] - prev_x
				data[i, 1] = points[i][1] - prev_y
				prev_x = int(points[i][0])
				prev_y = int(points[i][1])
			
			# inject indicator of end points
			data[:, 2][index] = 1
			output.append((data, char_data))
		return output


	def __sort(self, data):
		return sorted(data, key = lambda x: len(x[0]), reverse = True)
	

	def __save2binary(self, data, alphabet, data_path, alphabet_path):
		f = open(data_path, 'wb')
		pickle.dump(data, f)
		f.close()
		f = open(alphabet_path, 'wb')
		pickle.dump(alphabet, f)
		f.close()


	def __reload(self, file_path):
		'''
		Load processed binary file
		'''
		print ('Loading binary data file')
		f = open(file_path, 'rb')
		data = pickle.load(f)
		f.close()
		return data
	
	
	def next_batch(self):
		'''
		Load next batch of input data to feed into the network
		'''
		data = self.data[self.pointer: self.pointer + self.batch_size]
		self.pointer += self.batch_size
		return data
	

	def padded(self, input):
		points = [instance[0] for instance in input]
		chars = [instance[1] for instance in input]
		
		# create length list
		#points_len = [len(tensor)+1 for tensor in points]
		#chars_len = [len(tensor)+1 for tensor in chars]

		
		# max length
		max_len = points[0].size()[0] + 1      # +1 so that the first sequence can also be padded
		max_charlen = max([char.size()[0] for char in chars]) + 1
		
		# create batch dimension
		
		if points[0].size()[0] != 1:
			points = [tensor.unsqueeze(0) for tensor in points]
			chars = [tensor.unsqueeze(0) for tensor in chars]
		
		
		# pad points and chars
		padded_points = Variable(torch.cat([self.__pad(tensor, max_len) for tensor in points]))
		padded_chars = Variable(torch.cat([self.__pad(tensor, max_charlen) for tensor in chars]))
		
		# pack sequence
		#pack_points = utils.rnn.pack_padded_sequence(padded_points, points_len, batch_first = True)
		#pack_chars = utils.rnn.pack_padded_sequence(padded_chars, chars_len, batch_first = True)
		return padded_points, padded_chars


	def __pad(self, tensor, length):
		return torch.cat([tensor, tensor.new(1, (length - tensor.size()[1]), *tensor.size()[2:]).zero_()], 1)


	def reset(self):
		'''
		Reset the pointer position and shuffle index before each new epoch of training
		'''
		# reshuffle the index of data instance
		
		# reset the pointer index to 0
		self.pointer = 0
	
