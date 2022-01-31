
"""
ENN 
"""

from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier

class ENN(InstanceSelectionMixin):
	""" Edition Nearest Neighbors (ENN)
	Descrição:
	==========


	Parametros:
	===========


	Atributos:
	==========


	Ref.
	====
	
	"""

	def __init__(self, n_neighbors=3):
		#random.seed(1608637542)
		self.n_neighbors = n_neighbors
		self.classifier = None
		self.sample_indices_ = []

	def select_data(self, X, y):
		
		X, y = check_X_y(X, y, accept_sparse="csr")

		# 
		if self.classifier == None:
			self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
		if self.classifier.n_neighbors != self.n_neighbors:
			self.classifier.n_neighbors = self.n_neighbors

		classes = np.unique(y)
		self.classes_ = classes

		"""
		# So se o numero de vizinhos for maior que a quantidade de instancias
		# Nao é nosso caso em nenhum dadtaset
		if self.n_neighbors >= X.shape[0]:
			self.X_ = np.array(X)
			self.y_ = np.array(y)
			self.reduction_ = 0.0
			return self.X_, self.y_
		"""

		# Inicialmente todo mundo é selecionado
		mask = np.ones(y.size, dtype=bool)
		tmp_m = np.ones(y.size, dtype=bool)

		train_idx = list(range(len(y)))
		train_idx = random.sample(train_idx, len(train_idx))
		#print(train_idx[:10])

		# Treinamos com todas intancias
		self.classifier.fit(X, y)
		#for i in xrange(y.size):
		for i in train_idx:
			# Treinamos com todas menos a atual, para evitar overfftiong
			# Se a instancia tiver no conjunto ela é seu vizinho mais proximo
			tmp_m[i] = not tmp_m[i]
			self.classifier.fit(X[tmp_m], y[tmp_m])
			#sample, label = X[i], y[i]

			#print(self.classifier.kneighbors(sample))
			#print(self.classifier.kneighbors(sample)[1][0][1:])
			#print(y[self.classifier.kneighbors(sample)[1][0][1:]])
			#exit()

			# Se for igual devemos manter na solução
			# Se for diferente devemos remover da solução
			# Iniciamnete mask é 1
			if self.classifier.predict(X[i]) != [y[i]]:
				mask[i] = not mask[i]

			#Volta com instancia ao conjunto
			tmp_m[i] = not tmp_m[i]

		self.X_ = np.asarray(X[mask])
		self.y_ = np.asarray(y[mask])
		#self.sample_indices_ = np.where(mask == True)[0]
		self.sample_indices_ = np.where(mask == True)[0]
	   
		#print(sorted(idx_prots_s))
		#print(float(len(self.y_))/len(y))

		self.reduction_ = 1.0 - float(len(self.y_))/len(y)
		return self.X_, self.y_