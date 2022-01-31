import logging
from typing import Tuple, Optional

import numpy as np
import torch
#from pytorch_transformers import XLNetForSequenceClassification, XLNetTokenizer
from transformers import (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig,
						  RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig,
						  GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer,
						  TransfoXLConfig, TransfoXLForSequenceClassification, TransfoXLTokenizer,
						  XLMConfig, XLMForSequenceClassification, XLMTokenizer,
						  BertConfig, BertForSequenceClassification, BertTokenizer,
						  DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,
						  AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
						  BartConfig, BartForSequenceClassification, BartTokenizer)



from transformers import AutoTokenizer, AutoConfig
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

#from miscellaneous.typing import Documents, Classes, Tag2Idx, Tag2Name, Tags, DocumentsEmbedding
from src.main.python.utils.miscellaneous import Documents, Classes, Tag2Idx, Tag2Name, Tags, DocumentsEmbedding

np.random.seed(1608637542)
torch.manual_seed(1608637542)

from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import copy
from tqdm import tqdm
from src.main.python.utils.general import createPath
from sklearn.preprocessing import LabelEncoder


def duplicate_if_necessary(X, y):
	#q = []
	q1 = []

	y_set = list(sorted(list(set(y))))
	mydict = Counter(y)

	for i in mydict.keys():
		if mydict[i] == 1:
			q1.append(i)

	X_res = copy.copy(X)
	y_res = copy.copy(y)

	if len(q1)>0:
		print("duplicando")
		for q in q1:
			print("duplicando", q)
			#print(np.where(y_res == q))
			q2 = np.where(y_res == q)[0][0]
			#print(q2)
			y_res = np.append(y_res,y_res[q2])
			#print(len(y_res))
	
			#X_res.data = np.hstack((X_res.data,X_res[q2].data))
			#X_res.indices = np.hstack((X_res.indices,X_res[q2].indices))
			#X_res.indptr = np.hstack((X_res.indptr,(X_res[q2].indptr + X_res.nnz)[1:]))
			#X_res._shape = (X_res.shape[0]+1,X_res[q2].shape[1])
			X_res.append(X_res[q2])

	return X_res, y_res

def prep_data(X_train, y_train):
	#if not os.path.exists(working_dir):
	#    os.makedirs(working_dir)
	X_train, y_train = duplicate_if_necessary(X_train, y_train)
	#X_train, y_train = create_if_necessary(X_train, y_train)

	#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)
	sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=2018)
	for train_index, val_index in sss.split(X_train, y_train):
		continue

	X_train_new = [X_train[x] for x in train_index]
	y_train_new = [y_train[x] for x in train_index]
	X_val   = [X_train[x] for x in val_index]
	y_val   = [y_train[x] for x in val_index]

	return X_train_new, y_train_new, X_val, y_val



MODEL_CLASSES = {
	'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
	
	'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
	'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
	'gpt2': (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
	'transfoxl': (TransfoXLConfig, TransfoXLForSequenceClassification, TransfoXLTokenizer),
	'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
	#'distilroberta': (),
	'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
	'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
	'bart': (BartConfig, BartForSequenceClassification, BartTokenizer),
}

SPECIFIC_MODEL = {
	'bert-base': 'bert-base-uncased',
	'xlnet-base': 'xlnet-base-cased',
	'xlnet-large': 'xlnet-large-cased',
	'gpt2-base': 'gpt2',
	'transfoxl-base': 'transfo-xl-wt103',
	'roberta-base': 'roberta-base',
	'xlm-base': 'xlm-mlm-en-2048',
	'distilroberta-base': 'distilroberta-base',
	'distilbert-base': 'distilbert-base-cased',
	'albert-base': 'albert-base-v2',
	'albert-large': 'albert-large-v2',
	'albert-xlarge': 'albert-xxlarge-v2',
	'bart-base': 'facebook/bart-base',
}

class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels=None, n_test=0):
		self.encodings = encodings
		self.labels = labels
		if self.labels:
			self.lenght = len(self.labels)
		else:
			self.lenght = n_test

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		if self.labels:
			item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return self.lenght



class DeepClassifier(BaseEstimator, ClassifierMixin):
	"""XLNetClassifier build in sklearn interface.

	Attributes
	----------
	See pytorch_transformers.XLNetForSequenceClassification for more information about all parameters.

	:arg pretrained_model: XLNetForSequenceClassification
		XLNet pre-trained model.
	:arg tokenizer: XLNetTokenizer
		XLNet tokenizer.
	:arg device: torch.device
		Device (CPU or GPU) to running.
	:arg epochs: int
		Number of epochs for fine tuning.
	:arg batch_num: int
		Batch size.
	:arg max_len: int
		Document max lenght.
	:arg max_grad_norm: float
		Max clips norm of the overall gradient.

	:arg full_finetuning: bool, default=True
		Where to fine tuning all layers (full_finetuning=True) or just the
		0LD_classification layer (full_finetuning=False).
	:arg verbose: bool, default=True
		Where to show verbose or not (training only).
	:arg weight_decay_rate: float, default=0.01
		Weight decay rate (optimization).
	:arg learning_rate: float, default=5e-5
		Learning rate (optimization).
	"""

	def __init__(
			self, 
			deepmethod: str = 'bert', 
			epochs: int = 5, 
			batch_num: int = 32, 
			max_len: int = 150, 
			max_grad_norm: float = 1.0, 
			full_finetuning: bool = True, 
			verbose: bool = True, 
			weight_decay_rate: float = 0.01, 
			learning_rate: float = 5e-5, 
			max_patience: int = 5, 
			max_iter: int = 20, 
			base_or_large: str = "base", 
			load_model: int = 0, 
			save_model: int = 0, 
			out_dir: str = "", 
			pretrained_models_path: str = "resources/pretrained_models/",
	):

		self.epochs = epochs
		self.batch_num = batch_num
		self.max_len = max_len
		self.max_grad_norm = max_grad_norm
		self.full_finetuning = full_finetuning
		self.weight_decay_rate = weight_decay_rate
		self.learning_rate = learning_rate
		self.verbose = verbose
		self.max_patience = max_patience
		self.epoch_id = 0
		self.max_iter = max_iter

		self.base_or_large = base_or_large
		self.deepmethod = deepmethod

		self.load_model = load_model
		self.save_model = save_model
		self.out_dir = out_dir

		self.pretrained_models_path = pretrained_models_path


		if self.save_model:
			createPath(self.out_dir)
			

	def set_model(self):

		pretrained_folder = f"{self.pretrained_models_path}/{self.deepmethod}/{self.base_or_large}/"

		self.config_model, self.transformers_model, self.tokenizer_model = MODEL_CLASSES[self.deepmethod]
		
		self.tokenizer = self.tokenizer_model.from_pretrained(SPECIFIC_MODEL[f'{self.deepmethod}-{self.base_or_large}'],
											  do_lower_case=False,
											  max_length = self.max_len)

		config = self.config_model.from_pretrained(pretrained_folder)
		config.num_labels = self.num_classes
		config.max_length = self.max_len

		self.pretrained_model = self.transformers_model.from_pretrained(pretrained_folder, config = config)

		#if self.deepmethod == 'xlnet':
		if self.deepmethod == 'roberta' or self.deepmethod == 'bart':
			self.tokenizer.add_prefix_space = True

		if self.deepmethod == 'gpt2':
			self.tokenizer.padding_side = "left"
			self.tokenizer.pad_token = self.tokenizer.eos_token
			self.pretrained_model.resize_token_embeddings(len(self.tokenizer))
			self.pretrained_model.config.pad_token_id = self.pretrained_model.config.eos_token_id
	   
		if self.deepmethod == 'transfoxl':
			self.tokenizer.pad_token = self.tokenizer.eos_token
			self.pretrained_model.config.pad_token_id = self.pretrained_model.config.eos_token_id        
			#self.pretrained_model.resize_token_embeddings(len(self.tokenizer))
		
		#if self.deepmethod == 'xlm':
		#if self.deepmethod == 'distilbert':
		#if self.deepmethod == 'albert':
		#if self.deepmethod == 'bart':


	def load_model(self):
		config = self.config_model.from_pretrained(self.out_dir)
		config.num_labels = self.num_classes
		config.max_length = self.max_len
		self.pretrained_model = self.transformers_model.from_pretrained(self.out_dir, config = config)
		self.pretrained_model.resize_token_embeddings(len(self.tokenizer))
		self.pretrained_model.config.pad_token_id = self.pretrained_model.config.eos_token_id
		self.pretrained_model.to(self.device)
		self.training_loss = []
		self.validation_loss = []

	def fit(self, X: Documents, y: Classes):
		"""Fine-tuning of the pre-trained XLNet model.

		Parameters
		----------
		:param X: Documents
			Documents for training.
		:param y: Classes
			Classes for each document in training.
		"""
		gpu_id = 0
		self.device = torch.device(f'cuda:{gpu_id}')

		self.num_classes = len(list(set(y)))

		self.set_model() 

		if self.load_model: #gpt2
			self.load_model()
			return self

		# Send pre-trained model to GPU
		self.pretrained_model.to(self.device)

		self.training_loss = []
		self.validation_loss = []
		patience = 0
		best_loss = None



		X_train, y_train, X_val, y_val = prep_data(X, y)

		train_encodings = self.tokenizer(X_train, truncation=True, padding='max_length', max_length=self.max_len)
		val_encodings = self.tokenizer(X_val, truncation=True, padding='max_length', max_length=self.max_len)

		train_dataset = CustomDataset(train_encodings, y_train)
		val_dataset   = CustomDataset(val_encodings, y_val)

		sampler_train = SequentialSampler(train_dataset)
		data_loader_train = DataLoader(train_dataset, sampler = sampler_train, batch_size=self.batch_num, drop_last=False)
		sampler_val = SequentialSampler(val_dataset)
		data_loader_val   = DataLoader(val_dataset, sampler = sampler_val, batch_size=self.batch_num, drop_last=False)

		# Training the pretrained_model
		optimizer = self._set_optimizer()

		#for epoch_id in range(self.epochs):
		#for epoch_id in range(10):
		#while True:
		while self.epoch_id < self.max_iter:
			# Info
			self.epoch_id += 1

			self.pretrained_model.train()
			#self.pretrained_model.train(False)

			logging.info(f'epoch: {self.epoch_id}')
			print(f'epoch: {self.epoch_id}')

			tr_loss = 0
			nb_tr_steps = 0
			#for step, batch in tqdm(enumerate(data_loader_train), desc="Train"):
			#for step, batch in tqdm(enumerate(data_loader_train)):
			for batch in tqdm(data_loader_train, desc="Train"):
				# Add batch to GPU
				#batch = tuple(t.to(self.device) for t in batch)
				#b_input_ids, b_input_mask, b_segs, b_labels = batch

				# Forward pass
				'''
				all labels=labels
				'xlnet':input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask
				'roberta' or 'gpt2': input_ids=input_ids, attention_mask=attention_mask
				'transfoxl': input_ids=input_ids
				'''
				batch = {k:v.type(torch.long).to(self.device) for k,v in batch.items()}
				outputs = self.pretrained_model(**batch)
				loss, logits = outputs[:2]

				# Backward pass
				loss.backward()

				# Track train loss
				tr_loss += loss.item()
				nb_tr_steps += 1

				# Gradient clipping
				torch.nn.utils.clip_grad_norm_(parameters=self.pretrained_model.parameters(),
											   max_norm=self.max_grad_norm)

				# Update parameters
				optimizer.step()
				optimizer.zero_grad()

			# Print train loss per epoch
			self.training_loss.append(tr_loss / nb_tr_steps)
			logging.info(f'train loss: {tr_loss / nb_tr_steps:.4E}')
			print(f'train loss: {tr_loss / nb_tr_steps:.4E}')

			#Calcular loss na validacao aqui
			self.pretrained_model.eval()

			vl_loss = 0
			nb_vl_steps = 0
			#for step, batch in tqdm(enumerate(data_loader_val), desc = "Val"):
			for batch in tqdm(data_loader_val, desc = "Val"):
				# Add batch to GPU
				#batch = tuple(t.to(self.device) for t in batch)
				#b_input_ids, b_input_mask, b_segs, b_labels = batch

				with torch.no_grad():
					# Forward pass
					'''
					all labels=labels
					'xlnet':input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask
					'roberta' or 'gpt2': input_ids=input_ids, attention_mask=attention_mask
					'transfoxl': input_ids=input_ids
					'''
					batch = {k:v.type(torch.long).to(self.device) for k,v in batch.items()}
					outputs = self.pretrained_model(**batch)
					loss, logits = outputs[:2]

				vl_loss += loss.item()
				nb_vl_steps += 1

			dev_loss = vl_loss / nb_vl_steps
			
			if best_loss is None or dev_loss < best_loss:
				best_loss = dev_loss
				print('val best loss updated: {:.4f}'.format(best_loss))
				patience = 0
			else:
				for param_group in optimizer.param_groups:
					print(param_group['lr'])
				
				new_lr = optimizer.param_groups[0]['lr']/2
				#optimizer.set_learning_rate(new_lr)
				for g in optimizer.param_groups:
					g['lr'] = new_lr
								
				print('patience #{}: reducing the lr to {}'.format(patience, new_lr))
				if patience == self.max_patience:
					break
				patience+=1

			# Print train loss per epoch
			logging.info(f'val loss: {dev_loss:.4E}')
			print(f'val loss: {dev_loss:.4E}')
			self.validation_loss.append(dev_loss)

		if self.save_model: 
			self.pretrained_model.save_pretrained(f'{self.out_dir}')



		return self

	def predict(self, X: Documents):
		"""Prediction for new documents (using the fitted model).

		Parameters
		----------
		:param X: Documents
			Documents for prediction.

		Returns
		----------
		:return y_pred: numpy.ndarray
			Predictions for each document in X.
		"""
		# Generating DataLoader
		#data_loader = self._generate_data_loader(X=X, partition="test") #training=False)

		test_encodings = self.tokenizer(X, truncation=True, padding='max_length', max_length=self.max_len)
		test_dataset = CustomDataset(test_encodings, n_test = len(X))
		sampler_test = SequentialSampler(test_dataset)
		data_loader_test = DataLoader(test_dataset, sampler = sampler_test, batch_size=self.batch_num, drop_last=False)
		
		# Evalue loop
		self.pretrained_model.eval()

		y_pred = []
		#for step, batch in tqdm(enumerate(data_loader), desc="Test"):
		for batch in tqdm(data_loader_test, desc="Test"):
			#batch = tuple(t.to(self.device) for t in batch)
			#b_input_ids, b_input_mask, b_segs = batch

			with torch.no_grad():
				'''
				'xlnet':input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask
				'roberta' or 'gpt2': input_ids=input_ids, attention_mask=attention_mask
				'transfoxl': input_ids=input_ids
				'''
				batch = {k:v.type(torch.long).to(self.device) for k,v in batch.items()}
				outputs = self.pretrained_model(**batch)
				logits = outputs[0]


			# Predictions
			logits = logits.detach().cpu().numpy()
			for predict in np.argmax(logits, axis=1):
				y_pred.append(predict)

		y_pred = np.array(y_pred)

		return y_pred

	def score(self, X, y, sample_weight=None):
		pass

	def _set_optimizer(self) -> Adam:
		"""Setting the optimizer for the fit method.

		Returns
		----------
		:return optimizer: Adam
			The Adam optimizer from torch.optim.
		"""
		if self.full_finetuning:
			# Fine tune all layer parameters of the pre-trained model
			param_optimizer = list(self.pretrained_model.named_parameters())
			no_decay = ['bias', 'gamma', 'beta']
			optimizer_grouped_parameters = [
				{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
				 'weight_decay_rate': self.weight_decay_rate},
				{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
				 'weight_decay_rate': 0.0}
			]
		else:
			# Only fine tune classifier parameters
			param_optimizer = list(self.pretrained_model.classifier.named_parameters())
			optimizer_grouped_parameters = [
				{'params': [p for n, p in param_optimizer]}
			]

		optimizer = Adam(optimizer_grouped_parameters, lr=self.learning_rate)

		return optimizer

	
	def predict_proba(self, X: Documents):
		"""Class probability prediction for new documents (using the fitted model).

		Parameters
		----------
		:param X: Documents
			Documents for prediction.

		Returns
		----------
		:return y_pred: numpy.ndarray
			Class probability predictions for each document in X.
		"""
		# Generating test DataLoader
		#data_loader = self._generate_data_loader(X=X, partition='test') #training=False)
		test_encodings = self.tokenizer(X, truncation=True, padding='max_length', max_length=self.max_len)
		test_dataset = CustomDataset(test_encodings, n_test = len(X))
		sampler_test = SequentialSampler(test_dataset)
		data_loader_test = DataLoader(test_dataset, sampler = sampler_test, batch_size=self.batch_num, drop_last=False)

		# Evalue loop
		self.pretrained_model.eval()

		y_pred = []
		#for step, batch in tqdm(enumerate(data_loader), desc="Test"):
		for batch in tqdm(data_loader_test, desc="Test"):
			#batch = tuple(t.to(self.device) for t in batch)
			#b_input_ids, b_input_mask, b_segs = batch

			with torch.no_grad():
				'''
				'xlnet':input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask
				'roberta' or 'gpt2': input_ids=input_ids, attention_mask=attention_mask
				'transfoxl': input_ids=input_ids
				'''
				batch = {k:v.type(torch.long).to(self.device) for k,v in batch.items()}
				outputs = self.pretrained_model(**batch)
				logits = outputs[0]


			# Predictions
			logits = logits.detach().cpu().numpy()
			y_pred.append(logits)

		y_pred = np.concatenate(y_pred)

		return y_pred
	

	#vocabulary_file_name = f"{self.deepmethod}-{self.base_or_large}-cased-spiece.model"
	'''
	if self.base_or_large == 'base':
		vocabulary_file_name = 'xlnet-base-cased-spiece.model'
		pretrained_folder = 'pretrained_models_xlnet/'
	else:
		vocabulary_file_name = 'xlnet-large-cased-spiece.model'
		pretrained_folder = 'pretrained_models_xlnet_large/'
	'''
	#vocabulary_file = f'{pretrained_folder}/{vocabulary_file_name}'
	#self.pretrained_model = XLNetForSequenceClassification.from_pretrained(pretrained_folder, num_labels=num_classes)
	#self.tokenizer = XLNetTokenizer(vocab_file=vocabulary_file, do_lower_case=False)
		
	#self.pretrained_model = self.transformers_model.from_pretrained(vocab_file = pretrained_folder, merges_file = pretrained_folder, num_labels=num_classes)
	#self.tokenizer = self.tokenizer_model(vocab_file=vocabulary_file, do_lower_case=False)
	#self.tokenizer = self.tokenizer_model(pretrained_folder, do_lower_case=False)
	#self.tokenizer = self.tokenizer_model.from_pretrained("roberta-base")
	
	#tokenizer = AutoTokenizer.from_pretrained(f'{self.deepmethod}-base')
	#config = AutoConfig.from_pretrained(f'{self.deepmethod}-base')
	#tokenizer.save_pretrained(f'{pretrained_folder}')
	#config.save_pretrained(f'{pretrained_folder}')

	#self.tokenizer = self.tokenizer_model(f"{self.deepmethod}-{self.base_or_large}-cased")