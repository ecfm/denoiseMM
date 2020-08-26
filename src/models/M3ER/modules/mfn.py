import numpy as np
seed = 123
np.random.seed(seed)
import random
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau

import h5py
import time
import data_loader as loader
from collections import defaultdict, OrderedDict
import argparse
import cPickle as pickle
import time
import json, os, ast, h5py

from keras.models import Model
from keras.layers import Input
from keras.layers.embeddings import Embedding

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import sys


class MFN(nn.Module):
	def __init__(self,d_l, d_a, d_v, h_l, h_a, h_v, memsize, windowsize, h_att1, h_att2, h_gamma1, h_gamma2, h_out, d_fusion, beta, M=3):
		super(MFN, self).__init__()
		[self.d_l,self.d_a,self.d_v] = [d_l, d_a, d_v]
		[self.dh_l,self.dh_a,self.dh_v] = [h_l, h_a, h_v]
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		self.mem_dim = memsize
		window_dim = windowsize
		output_dim = 1
		attInShape = total_h_dim*window_dim
		gammaInShape = attInShape+self.mem_dim
                self.d_fusion = d_fusion
		final_out = self.d_fusion+self.mem_dim
		att1_dropout = 0
		att2_dropout = 0
		gamma1_dropout = 0
		gamma2_dropout = 0
		out_dropout = 0

		self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
		self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
		self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

		self.att1_fc1 = nn.Linear(attInShape, h_att1)
		self.att1_fc2 = nn.Linear(h_att1, attInShape)
		self.att1_dropout = nn.Dropout(att1_dropout)

		self.att2_fc1 = nn.Linear(attInShape, h_att2)
		self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
		self.att2_dropout = nn.Dropout(att2_dropout)

		self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
		self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
		self.gamma1_dropout = nn.Dropout(gamma1_dropout)

		self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
		self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
		self.gamma2_dropout = nn.Dropout(gamma2_dropout)
                
                self.fusion_l = nn.Linear(self.dh_l, self.d_fusion)
                self.fusion_a = nn.Linear(self.dh_a, self.d_fusion)
                self.fusion_v = nn.Linear(self.dh_v, self.d_fusion)		

		self.out_fc1 = nn.Linear(final_out, h_out)
		self.out_fc2 = nn.Linear(h_out, output_dim)
		self.out_dropout = nn.Dropout(out_dropout)
		self.beta = beta
                self.M = M
	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]
		self.h_l = torch.zeros(n, self.dh_l).cuda()
		self.h_a = torch.zeros(n, self.dh_a).cuda()
		self.h_v = torch.zeros(n, self.dh_v).cuda()
		self.c_l = torch.zeros(n, self.dh_l).cuda()
		self.c_a = torch.zeros(n, self.dh_a).cuda()
		self.c_v = torch.zeros(n, self.dh_v).cuda()
		self.mem = torch.zeros(n, self.mem_dim).cuda()
		all_h_ls = []
		all_h_as = []
		all_h_vs = []
		all_c_ls = []
		all_c_as = []
		all_c_vs = []
		all_mems = []
		for i in range(t):
			# prev time step
			prev_c_l = self.c_l
			prev_c_a = self.c_a
			prev_c_v = self.c_v
			# curr time step
			new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
			new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
			new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
			# concatenate
			prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
			new_cs = torch.cat([new_c_l,new_c_a,new_c_v], dim=1)
			cStar = torch.cat([prev_cs,new_cs], dim=1)
			attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
			attended = attention*cStar
			cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
			both = torch.cat([attended,self.mem], dim=1)
			gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
			gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
			self.mem = gamma1*self.mem + gamma2*cHat
			all_mems.append(self.mem)
			# update
			self.h_l, self.c_l = new_h_l, new_c_l
			self.h_a, self.c_a = new_h_a, new_c_a
			self.h_v, self.c_v = new_h_v, new_c_v
			all_h_ls.append(self.h_l)
			all_h_as.append(self.h_a)
			all_h_vs.append(self.h_v)
			all_c_ls.append(self.c_l)
			all_c_as.append(self.c_a)
			all_c_vs.append(self.c_v)

		# last hidden layer last_hs is n x h
		last_h_l = all_h_ls[-1]
		last_h_a = all_h_as[-1]
		last_h_v = all_h_vs[-1]
		last_mem = all_mems[-1]
                
                last_hf_l = self.fusion_l(last_h_l)
                last_hf_a = self.fusion_a(last_h_a)
                last_hf_v = self.fusion_v(last_h_v)
                last_hf_lav = torch.pow(last_hf_l, self.beta/(M-1)) * torch.log(last_hf_l) + torch.pow(last_hf_a, self.beta/(M-1)) * torch.log(last_hf_a) + torch.pow(last_hf_v, self.beta/(M-1)) * torch.log(last_hf_v)
		last_hs = torch.cat([last_hf_lav,last_mem], dim=1)
		output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
		return output

def train_mfn(X_train, y_train, X_valid, y_valid, X_test, y_test, configs):
	p = np.random.permutation(X_train.shape[0])
	X_train = X_train[p]
	y_train = y_train[p]

	X_train = X_train.swapaxes(0,1)
	X_valid = X_valid.swapaxes(0,1)
	X_test = X_test.swapaxes(0,1)

	d = X_train.shape[2]
	h = 128
	t = X_train.shape[0]
	output_dim = 1
	dropout = 0.5

	[config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs

	model = MFN(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)

	optimizer = optim.Adam(model.parameters(),lr=config["lr"])

	criterion = nn.L1Loss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	criterion = criterion.to(device)
	scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)

	def train(model, batchsize, X_train, y_train, optimizer, criterion):
		epoch_loss = 0
		model.train()
		total_n = X_train.shape[1]
		num_batches = total_n / batchsize
		for batch in xrange(num_batches):
			start = batch*batchsize
			end = (batch+1)*batchsize
			optimizer.zero_grad()
			batch_X = torch.Tensor(X_train[:,start:end]).cuda()
			batch_y = torch.Tensor(y_train[start:end]).cuda()
			predictions = model.forward(batch_X).squeeze(1)
			loss = criterion(predictions, batch_y)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
		return epoch_loss / num_batches

	def evaluate(model, X_valid, y_valid, criterion):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_valid).cuda()
			batch_y = torch.Tensor(y_valid).cuda()
			predictions = model.forward(batch_X).squeeze(1)
			epoch_loss = criterion(predictions, batch_y).item()
		return epoch_loss

	def predict(model, X_test):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			predictions = model.forward(batch_X).squeeze(1)
			predictions = predictions.cpu().data.numpy()
		return predictions

	best_valid = 999999.0
	rand = random.randint(0,100000)
	for epoch in range(config["num_epochs"]):
		train_loss = train(model, config["batchsize"], X_train, y_train, optimizer, criterion)
		valid_loss = evaluate(model, X_valid, y_valid, criterion)
		scheduler.step(valid_loss)
		if valid_loss <= best_valid:
			# save model
			best_valid = valid_loss
			print epoch, train_loss, valid_loss, 'saving model'
			torch.save(model, 'temp_models/mfn_%d.pt' %rand)
		else:
			print epoch, train_loss, valid_loss

	print 'model number is:', rand
	model = torch.load('temp_models/mfn_%d.pt' %rand)

	predictions = predict(model, X_test)
	mae = np.mean(np.absolute(predictions-y_test))
	print "mae: ", mae
	corr = np.corrcoef(predictions,y_test)[0][1]
	print "corr: ", corr
	mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
	print "mult_acc: ", mult
	f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
	print "mult f_score: ", f_score
	true_label = (y_test >= 0)
	predicted_label = (predictions >= 0)
	print "Confusion Matrix :"
	print confusion_matrix(true_label, predicted_label)
	print "Classification Report :"
	print classification_report(true_label, predicted_label, digits=5)
	print "Accuracy ", accuracy_score(true_label, predicted_label)
	sys.stdout.flush()

def test(X_test, y_test, metric):
	X_test = X_test.swapaxes(0,1)
	def predict(model, X_test):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			predictions = model.forward(batch_X).squeeze(1)
			predictions = predictions.cpu().data.numpy()
		return predictions
	if metric == 'mae':
		model = torch.load('best/mfn_mae.pt')
	if metric == 'acc':
		model = torch.load('best/mfn_acc.pt')
	model = model.cpu().cuda()
	
	predictions = predict(model, X_test)
	print predictions.shape
	print y_test.shape
	mae = np.mean(np.absolute(predictions-y_test))
	print "mae: ", mae
	corr = np.corrcoef(predictions,y_test)[0][1]
	print "corr: ", corr
	mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
	print "mult_acc: ", mult
	f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
	print "mult f_score: ", f_score
	true_label = (y_test >= 0)
	predicted_label = (predictions >= 0)
	print "Confusion Matrix :"
	print confusion_matrix(true_label, predicted_label)
	print "Classification Report :"
	print classification_report(true_label, predicted_label, digits=5)
	print "Accuracy ", accuracy_score(true_label, predicted_label)
	sys.stdout.flush()

local = False

if local:
	X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(args,config)

	h5f = h5py.File('data/X_train.h5', 'w')
	h5f.create_dataset('data', data=X_train)
	h5f = h5py.File('data/y_train.h5', 'w')
	h5f.create_dataset('data', data=y_train)
	h5f = h5py.File('data/X_valid.h5', 'w')
	h5f.create_dataset('data', data=X_valid)
	h5f = h5py.File('data/y_valid.h5', 'w')
	h5f.create_dataset('data', data=y_valid)
	h5f = h5py.File('data/X_test.h5', 'w')
	h5f.create_dataset('data', data=X_test)
	h5f = h5py.File('data/y_test.h5', 'w')
	h5f.create_dataset('data', data=y_test)

	sys.stdout.flush()

X_train, y_train, X_valid, y_valid, X_test, y_test = load_saved_data()

test(X_test, y_test, 'mae')
test(X_test, y_test, 'acc')
assert False

#config = dict()
#config["batchsize"] = 32
#config["num_epochs"] = 100
#config["lr"] = 0.01
#config["h"] = 128
#config["drop"] = 0.5
#train_ef(X_train, y_train, X_valid, y_valid, X_test, y_test, config)
#assert False

while True:
	# mae 0.993 [{'input_dims': [300, 5, 20], 'batchsize': 128, 'memsize': 128, 
	#'windowsize': 2, 'lr': 0.01, 'num_epochs': 100, 'h_dims': [88, 48, 16], 'momentum': 0.9}, 
	#{'shapes': 128, 'drop': 0.0}, {'shapes': 64, 'drop': 0.2}, 
	#{'shapes': 256, 'drop': 0.0}, {'shapes': 64, 'drop': 0.2}, 
	#{'shapes': 64, 'drop': 0.5}]

	# acc 77.0 [{'input_dims': [300, 5, 20], 'batchsize': 128, 'memsize': 400, 
	#'windowsize': 2, 'lr': 0.005, 'num_epochs': 100, 'h_dims': [64, 8, 80], 'momentum': 0.9}, 
	#{'shapes': 128, 'drop': 0.5}, {'shapes': 128, 'drop': 0.2}, 
	#{'shapes': 128, 'drop': 0.5}, {'shapes': 128, 'drop': 0.5}, 
	#{'shapes': 256, 'drop': 0.5}]

	config = dict()
	config["input_dims"] = [300,5,20]
	hl = random.choice([32,64,88,128,156,256])
	ha = random.choice([8,16,32,48,64,80])
	hv = random.choice([8,16,32,48,64,80])
	config["h_dims"] = [hl,ha,hv]
	config["memsize"] = random.choice([64,128,256,300,400])
	config["windowsize"] = 2
	config["batchsize"] = random.choice([32,64,128,256])
	config["num_epochs"] = 50
	config["lr"] = random.choice([0.001,0.002,0.005,0.008,0.01])
	config["momentum"] = random.choice([0.1,0.3,0.5,0.6,0.8,0.9])
	NN1Config = dict()
	NN1Config["shapes"] = random.choice([32,64,128,256])
	NN1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	NN2Config = dict()
	NN2Config["shapes"] = random.choice([32,64,128,256])
	NN2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	gamma1Config = dict()
	gamma1Config["shapes"] = random.choice([32,64,128,256])
	gamma1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	gamma2Config = dict()
	gamma2Config["shapes"] = random.choice([32,64,128,256])
	gamma2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
	outConfig = dict()
	outConfig["shapes"] = random.choice([32,64,128,256])
	outConfig["drop"] = random.choice([0.0,0.2,0.5,0.7])
	configs = [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig]
	print configs
	train_mfn(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)


