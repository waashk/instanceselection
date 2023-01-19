"""
EGDIS : enhanced global density-based instance selection algorithm 
"""

from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import copy

from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

class EGDIS(InstanceSelectionMixin):
    """ EGDIS : enhanced global density-based instance selection algorithm 
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

    def get_irrelevance(self, i, neighbors, y):

        count = 0
        for neigh in neighbors:
            if y[neigh] != y[i]:
                count+=1
        return count

    def get_relevance(self, i, neighbors, y):

        count = 0
        for neigh in neighbors:
            if y[neigh] == y[i]:
                count += 1
        return int(count)

    

    def set_density(self, X):
        self.pairwise_distances = euclidean_distances(X, X)
        self.density = - (1.0/X.shape[0]) * self.pairwise_distances.sum(axis=1)

    def select_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")

        self.classifier = KNeighborsClassifier(
            n_neighbors=self.n_neighbors+1, metric= "cosine")
        self.classifier.fit(X,y)

        self.mask = np.zeros(y.size, dtype=bool)
        self.classes_ = np.unique(y)
        self.set_density(X)
        
        for i, inst in enumerate(X):

            neighbors = self.classifier.kneighbors(inst.toarray(), return_distance=False)[0]
            neighbors = [neigh for neigh in neighbors if neigh != i]

            x = self.get_irrelevance(i, neighbors, y)
            #x = self.get_relevance(i, neighbors, y)
           
            if x == 0:
            #if x == self.n_neighbors:
                densest = False
                for neighbor in neighbors:
                    if self.density[i] < self.density[neighbor]:
                        densest = True

                if not densest:
                    self.mask[i] = True

            else:
                if x >= (self.n_neighbors/2.):
                    self.mask[i] = True

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = list(sorted(np.where(self.mask == True)[0]))
        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_
