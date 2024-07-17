import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch import nn 

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle 
torch.manual_seed(2)
np.random.seed(3)
import copy

import os

from utils import *  
import wandb

class CNN(nn.Sequential):
	def __init__(self, encoding, **config):
		super(CNN, self).__init__()
		if encoding == 'drug':
			in_ch = [63] + config['cnn_drug_filters']
			kernels = config['cnn_drug_kernels']
			layer_size = len(config['cnn_drug_filters'])
			self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
													out_channels = in_ch[i+1], 
													kernel_size = kernels[i]) for i in range(layer_size)])
			self.conv = self.conv.double()
			n_size_d = self._get_conv_output((63, 100))
			#n_size_d = 1000
			self.fc1 = nn.Linear(n_size_d, config['hidden_dim_drug'])

		if encoding == 'protein':
			in_ch = [26] + config['cnn_target_filters']
			kernels = config['cnn_target_kernels']
			layer_size = len(config['cnn_target_filters'])
			self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
													out_channels = in_ch[i+1], 
													kernel_size = kernels[i]) for i in range(layer_size)])
			self.conv = self.conv.double()
			n_size_p = self._get_conv_output((26, 1000))

			self.fc1 = nn.Linear(n_size_p, config['hidden_dim_protein'])

	def _get_conv_output(self, shape):
		bs = 1
		input = Variable(torch.rand(bs, *shape))
		output_feat = self._forward_features(input.double())
		n_size = output_feat.data.view(bs, -1).size(1)
		return n_size

	def _forward_features(self, x):
		for l in self.conv:
			x = F.relu(l(x))
		x = F.adaptive_max_pool1d(x, output_size=1)
		return x

	def forward(self, v):
		v = self._forward_features(v.double())
		v = v.view(v.size(0), -1)
		v = self.fc1(v.float())
		return v


class Classifier(nn.Sequential):
	def __init__(self, model_drug, model_protein, **config):
		super(Classifier, self).__init__()
		self.input_dim_drug = config['hidden_dim_drug']
		self.input_dim_protein = config['hidden_dim_protein']

		self.model_drug = model_drug
		self.model_protein = model_protein

		self.dropout = nn.Dropout(0.1)

		self.hidden_dims = config['cls_hidden_dims']
		layer_size = len(self.hidden_dims) + 1
		dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [1]
		
		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v_D, v_P, get_chem_embedding = False):
		# each encoding
		v_D = self.model_drug(v_D)
		v_P = self.model_protein(v_P)
		if get_chem_embedding:
			return v_D
		# concatenate and classify
		v_f = torch.cat((v_D, v_P), 1)
		for i, l in enumerate(self.predictor):
			if i==(len(self.predictor)-1):
				v_f = l(v_f)
			else:
				v_f = F.relu(self.dropout(l(v_f)))
		return v_f

def model_initialize(**config):
	model = DBTA(**config)
	return model

def model_pretrained(path_dir = None, model = None):
	if model is not None:
		path_dir = download_pretrained_model(model)
	config = load_dict(path_dir)
	model = DBTA(**config)
	model.load_pretrained(path_dir + '/model.pt')    
	return model


class DBTA:
	'''
		Drug Target Binding Affinity 
	'''

	def __init__(self, **config):
		drug_encoding = config['drug_encoding']
		target_encoding = config['target_encoding']

		if drug_encoding == 'Morgan' or drug_encoding == 'ErG' or drug_encoding=='Pubchem' or drug_encoding=='Daylight' or drug_encoding=='rdkit_2d_normalized' or drug_encoding == 'ESPF':
			# Future TODO: support multiple encoding scheme for static input 
			self.model_drug = MLP(config['input_dim_drug'], config['hidden_dim_drug'], config['mlp_hidden_dims_drug'])
		elif drug_encoding == 'CNN':
			self.model_drug = CNN('drug', **config)
		else:
			raise AttributeError('Please use one of the available encoding method.')

		if target_encoding == 'AAC' or target_encoding == 'PseudoAAC' or  target_encoding == 'Conjoint_triad' or target_encoding == 'Quasi-seq' or target_encoding == 'ESPF':
			self.model_protein = MLP(config['input_dim_protein'], config['hidden_dim_protein'], config['mlp_hidden_dims_target'])
		elif target_encoding == 'CNN':
			self.model_protein = CNN('protein', **config)
		
		else:
			raise AttributeError('Please use one of the available encoding method.')

		self.model = Classifier(self.model_drug, self.model_protein, **config)
		self.config = config

		if 'cuda_id' in self.config:
			if self.config['cuda_id'] is None:
				self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			else:
				self.device = torch.device('cuda:' + str(self.config['cuda_id']) if torch.cuda.is_available() else 'cpu')
		else:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		self.drug_encoding = drug_encoding
		self.target_encoding = target_encoding
		self.result_folder = config['result_folder']
		if not os.path.exists(self.result_folder):
			os.mkdir(self.result_folder)            
		self.binary = False
		if 'num_workers' not in self.config.keys():
			self.config['num_workers'] = 0
		if 'decay' not in self.config.keys():
			self.config['decay'] = 0

	def test_(self, data_generator, model, repurposing_mode = False, test = False, get_chem_embedding = False):
		y_pred = []
		y_label = []
		model.eval()
		for i, (v_d, v_p, label) in enumerate(data_generator):
			if self.drug_encoding in ["MPNN", 'Transformer', 'DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
				v_d = v_d
			else:
				v_d = v_d.float().to(self.device)                
			if self.target_encoding == 'Transformer':
				v_p = v_p
			else:
				v_p = v_p.float().to(self.device)                
			score = self.model(v_d, v_p, get_chem_embedding = get_chem_embedding)
			if self.binary:
				m = torch.nn.Sigmoid()
				logits = torch.squeeze(m(score)).detach().cpu().numpy()
			elif get_chem_embedding:
				logits = score.detach().cpu().numpy()
			else:
				loss_fct = torch.nn.MSELoss()
				n = torch.squeeze(score, 1)
				loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(self.device))
				logits = torch.squeeze(score).detach().cpu().numpy()
			if get_chem_embedding:
				y_pred = y_pred + logits.tolist()
			else:
				label_ids = label.to('cpu').numpy()
				y_label = y_label + label_ids.flatten().tolist()
				y_pred = y_pred + logits.flatten().tolist()
				outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
		if get_chem_embedding:
			return y_pred
		model.train()
		if self.binary:
			if repurposing_mode:
				return y_pred
			## ROC-AUC curve
			if test:
				roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
				plt.figure(0)
				roc_curve(y_pred, y_label, roc_auc_file, self.drug_encoding + '_' + self.target_encoding)
				plt.figure(1)
				pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
				prauc_curve(y_pred, y_label, pr_auc_file, self.drug_encoding + '_' + self.target_encoding)

			return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), log_loss(y_label, outputs), y_pred
		else:
			if repurposing_mode:
				return y_pred
			return mean_squared_error(y_label, y_pred), pearsonr(y_label, y_pred)[0], pearsonr(y_label, y_pred)[1], concordance_index(y_label, y_pred), y_pred, loss



	def train(self, train, val = None, test = None, verbose = True):
		if len(train.Label.unique()) == 2:
			self.binary = True
			self.config['binary'] = True

		lr = self.config['LR']
		decay = self.config['decay']
		BATCH_SIZE = self.config['batch_size']
		train_epoch = self.config['train_epoch']
		if 'test_every_X_epoch' in self.config.keys():
			test_every_X_epoch = self.config['test_every_X_epoch']
		else:     
			test_every_X_epoch = 40
		loss_history = []

		self.model = self.model.to(self.device)

		# support multiple GPUs
		if torch.cuda.device_count() > 1:
			if verbose:
				print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
			self.model = nn.DataParallel(self.model, dim = 0)
		elif torch.cuda.device_count() == 1:
			if verbose:
				print("Let's use " + str(torch.cuda.device_count()) + " GPU!")
		else:
			if verbose:
				print("Let's use CPU/s!")
		# Future TODO: support multiple optimizers with parameters
		opt = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = decay)
		if verbose:
			print('--- Data Preparation ---')

		params = {'batch_size': BATCH_SIZE,
	    		'shuffle': True,
	    		'num_workers': self.config['num_workers'],
	    		'drop_last': True}


		training_generator = data.DataLoader(data_process_loader(train.index.values, train.Label.values, train, **self.config), **params)
		if val is not None:
			validation_generator = data.DataLoader(data_process_loader(val.index.values, val.Label.values, val, **self.config), **params)
		
		if test is not None:
			info = data_process_loader(test.index.values, test.Label.values, test, **self.config)
			params_test = {'batch_size': BATCH_SIZE,
					'shuffle': False,
					'num_workers': self.config['num_workers'],
					'drop_last': False,
					'sampler':SequentialSampler(info)}
        
			testing_generator = data.DataLoader(data_process_loader(test.index.values, test.Label.values, test, **self.config), **params_test)

		# early stopping
		if self.binary:
			max_auc = 0
		else:
			max_MSE = 10000
		model_max = copy.deepcopy(self.model)

		valid_metric_record = []
		valid_metric_header = ["# epoch"] 
		if self.binary:
			valid_metric_header.extend(["AUROC", "AUPRC", "F1"])
		else:
			valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
	
		if verbose:
			print('--- Go for Training ---')

		t_start = time() 
		iteration_loss = 0
		for epo in range(train_epoch):
			for i, (v_d, v_p, label) in enumerate(training_generator):
				
				v_p = v_p.float().to(self.device) 
				v_d = v_d.float().to(self.device)                
				#score = self.model(v_d, v_p.float().to(self.device))
               
				score = self.model(v_d, v_p)
			
				label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)

				if self.binary:
					loss_fct = torch.nn.BCELoss()
					m = torch.nn.Sigmoid()
					n = torch.squeeze(m(score), 1)
					loss = loss_fct(n, label)
				else:
					loss_fct = torch.nn.MSELoss()
					n = torch.squeeze(score, 1)
					loss = loss_fct(n, label)
				loss_history.append(loss.item())
				
				iteration_loss += 1

				opt.zero_grad()
				loss.backward()
				opt.step()

				if verbose:
					if (i % 100 == 0):
						t_now = time()
						print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
							' with loss ' + str(loss.cpu().detach().numpy())[:7] +\
							". Total time " + str(int(t_now - t_start)/3600)[:7] + " hours") 
						### record total run time
			
			if val is not None:
				##### validate, select the best model up to now 
				with torch.set_grad_enabled(False):
					if self.binary:
						## binary: ROC-AUC, PR-AUC, F1, cross-entropy loss
						auc, auprc, f1, loss, logits = self.test_(validation_generator, self.model)
						if auc > max_auc:
							self.save_model(self.config['model_save_path'])
							model_max = copy.deepcopy(self.model)
							max_auc = auc
						wandb.log({"val_auc": auc, "val_auprc": auprc, "val_f1": f1, "val_loss": loss})
						# lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc, f1]))
						# valid_metric_record.append(lst)
						
						if verbose:
							print('Validation at Epoch '+ str(epo + 1) + ', AUROC: ' + str(auc)[:7] + \
								' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
								str(loss)[:7])
						test_auc, test_auprc, test_f1, test_loss, test_logits = self.test_(testing_generator, self.model)
				
						wandb.log({"test_auc": test_auc, "test_auprc": test_auprc, "test_f1": test_f1, "test_loss": test_loss})
						if verbose:
							print('Testing at Epoch '+ str(epo + 1) + ', AUROC: ' + str(test_auc)[:7] + \
								' , AUPRC: ' + str(test_auprc)[:7] + ' , F1: '+str(test_f1)[:7] + ' , Cross-entropy Loss: ' + \
								str(test_loss)[:7])
					else:
						## regression: MSE, Pearson Correlation, with p-value, Concordance Index
						MSE, r2, p_val, CI, logits, valid_loss = self.test_(validation_generator, self.model)
						if MSE < max_MSE:
							self.save_model(self.config['model_save_path'])
							model_max = copy.deepcopy(self.model)
							max_MSE = MSE
						wandb.log({"val_MSE": MSE, "val_r2": r2, "val_p_val": p_val, "val_CI": CI})
						# lst = ["epoch " + str(epo)] + list(map(float2str,[MSE, r2, p_val, CI]))
						# valid_metric_record.append(lst)
						if verbose:
							print('Validation at Epoch '+ str(epo + 1) + ', MSE: ' + str(MSE)[:7] + ' , Pearson Correlation: '\
								+ str(r2)[:7] + ' with p-value: ' + str(p_val)[:7] + ' , Concordance Index: '+str(CI)[:7])
						test_MSE, test_r2, test_p_val, test_CI, test_logits, test_loss = self.test_(testing_generator, self.model)
						wandb.log({"test_MSE": test_MSE, "test_r2": test_r2, "test_p_val": test_p_val, "test_CI": test_CI})
						if verbose:
							print('Testing at Epoch '+ str(epo + 1) + ', MSE: ' + str(test_MSE)[:7] + ' , Pearson Correlation: '\
								+ str(test_r2)[:7] + ' with p-value: ' + str(test_p_val)[:7] + ' , Concordance Index: '+str(test_CI)[:7])
				

	def save_model(self, path_dir):
		if not os.path.exists(path_dir):
			os.makedirs(path_dir)
		torch.save(self.model.state_dict(), path_dir + '/model.pt')
		print('The model has been saved in ' + path_dir + '/model.pt')
		save_dict(path_dir, self.config)

	def load_pretrained(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

		state_dict = torch.load(path, map_location = torch.device('cpu'))
		# to support training from multi-gpus data-parallel:
        
		if next(iter(state_dict))[:7] == 'module.':
			# the pretrained model is from data-parallel module
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				name = k[7:] # remove `module.`
				new_state_dict[name] = v
			state_dict = new_state_dict

		self.model.load_state_dict(state_dict)

		self.binary = self.config['binary']


