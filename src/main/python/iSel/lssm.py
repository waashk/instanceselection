
"""
LSSm 
"""

from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances
import copy

class LSSm(InstanceSelectionMixin):
	""" Local Set-based Smoother (LSSm)
	Descrição:
	==========


	Parametros:
	===========


	Atributos:
	==========


	Ref.
	====
	
	"""

	def __init__(self, n_neighbors=1):
		self.n_neighbors = n_neighbors
		self.classifier = None
		self.sample_indices_ = []

	def setMinEnemyDist(self,y):
		self.mindist_ne = np.zeros(y.size)
		self.ne = np.zeros(y.size) 
		for i in self.S:
			self.mindist_ne[i] = np.inf
			for j in self.S:
				if y[i] != y[j]:
					if self.pairwise_distances[i][j] < self.mindist_ne[i]:
						self.mindist_ne[i] = self.pairwise_distances[i][j]
						self.ne[i] = copy.copy(j)

	def isLessThanMinEnemyDist(self, idx_i, idx_j):
		if self.pairwise_distances[idx_i][idx_j] < self.mindist_ne[idx_i]:
			return 1
		return 0

	def setLs(self):
		for x in self.S:
			for j in self.S:
				if x != j and self.isLessThanMinEnemyDist(x,j): 
					self.ls[x].add(j)

	def getu(self, x):
		lenu = 0
		for i in self.S:
			if x != i and x in self.ls[i]:
				lenu+=1
		return lenu

	def geth(self, x):
		return len(np.where(self.ne == x)[0])

	def select_data(self, X, y):
		
		X, y = check_X_y(X, y, accept_sparse="csr")

		#randomize
		self.S = list(range(len(y)))

		self.mask = np.zeros(y.size, dtype=bool)                #mask = mascars
		
		self.pairwise_distances = euclidean_distances(X)

		self.setMinEnemyDist(y)

		# Set LS
		self.ls = [set() for i in self.S]
		self.setLs()

		u = np.zeros(y.size)
		h = np.zeros(y.size)
		for x in self.S:
			u[x] = self.getu(x)
			h[x] = self.geth(x)

		for x in self.S:
			if u[x] >= h[x]:
				self.mask[x] = True

		self.X_ = np.asarray(X[self.mask])
		self.y_ = np.asarray(y[self.mask])
		print(X[self.mask].shape)
		self.sample_indices_ = list(sorted(np.where(self.mask == True)[0]))
		self.reduction_ = 1.0 - float(len(self.y_))/len(y)
		return self.X_, self.y_