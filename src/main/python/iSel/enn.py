
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

		mask = np.ones(y.size, dtype=bool)
		tmp_m = np.ones(y.size, dtype=bool)

		train_idx = list(range(len(y)))
		train_idx = random.sample(train_idx, len(train_idx))

		self.classifier.fit(X, y)

		for i in train_idx:

			tmp_m[i] = not tmp_m[i]
			self.classifier.fit(X[tmp_m], y[tmp_m])

			if self.classifier.predict(X[i]) != [y[i]]:
				mask[i] = not mask[i]

			tmp_m[i] = not tmp_m[i]

		self.X_ = np.asarray(X[mask])
		self.y_ = np.asarray(y[mask])
		self.sample_indices_ = np.where(mask == True)[0]
		self.reduction_ = 1.0 - float(len(self.y_))/len(y)
		return self.X_, self.y_