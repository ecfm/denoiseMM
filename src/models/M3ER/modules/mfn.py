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
import argparse
import time
import json, os, h5py

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
		p_h_l = self.compute_exp(last_hf_l)
		p_h_a = self.compute_exp(last_hf_a)
		p_h_v = self.compute_exp(last_hf_v)
		last_hf_lav = torch.pow(p_h_l, self.beta/(self.M-1)) * torch.log(p_h_l) + torch.pow(p_h_a, self.beta/(self.M-1)) * torch.log(p_h_a) + torch.pow(p_h_v, self.beta/(self.M-1)) * torch.log(p_h_v)
		last_hs = torch.cat([last_hf_lav,last_mem], dim=1)
		output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs)))).flatten()
		return output
	
	def compute_exp(self, logits):
		logits_max = torch.max(logits)
		off_logits = logits - logits_max
		p = torch.exp(off_logits)
		return p

