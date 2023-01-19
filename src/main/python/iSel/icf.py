
"""
ICF 
"""

from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier

from src.main.python.iSel.enn import ENN
from src.main.python.iSel.cnn import CNN

from src.main.python.utils.general import load_splits_ids
from sklearn.metrics.pairwise import euclidean_distances

from collections import Counter
import copy


class ICF(InstanceSelectionMixin):
    """ Iterative case filtering (ICF)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====

    """

    def __init__(self, args, fold, n_neighbors=3):
        self.fold = fold
        self.outputdir = args.outputdir
        self.n_neighbors = n_neighbors
        self.classifier = None
        self.sample_indices_ = []

    def setMinEnemyDist(self, y):
        self.mindist_ne = np.zeros(y.size)
        self.ne = np.zeros(y.size)
        for i in range(len(self.S)):
            self.mindist_ne[i] = np.inf
            for j in range(len(self.S)):
                if y[i] != y[j]:
                    if self.pairwise_distances[i][j] < self.mindist_ne[i]:
                        self.mindist_ne[i] = self.pairwise_distances[i][j]
                        self.ne[i] = copy.copy(j)


    def getCoverage(self, idx):
        lenCoverage = 0
        for i in range(len(self.S)):
            if i != idx and self.mask[i]:
                lenCoverage += self.isLessThanMinEnemyDist(idx, i)
        return lenCoverage

    def getRecheable(self, idx):
        lenRecheable = 0
        for i in range(len(self.S)):
            if i != idx and self.mask[i]:
                lenRecheable += self.isLessThanMinEnemyDist(i, idx)
        return lenRecheable

    def isLessThanMinEnemyDist(self, idx_i, idx_j):
        if self.pairwise_distances[idx_i][idx_j] < self.mindist_ne[idx_i]:
            return 1
        return 0

    def ennpadrao(self, X, y):

        len_original_y = len(y)

        mask = np.ones(y.size, dtype=bool)

        classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        for i in range(X.shape[0]):
            classifier.fit(X[mask], y[mask])

            if classifier.predict(X[i]) != [y[i]]:
                mask[i] = not mask[i]

        S = np.asarray([x for x in range(X.shape[0])])
        S = S[mask]

        print("ENN ", round(1.0 - float(len(S))/len_original_y, 2))
        return S


    def select_data(self, X, y):

        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)


        S = self.ennpadrao(X, y)

        self.S = S

        X = X[self.S]
        y = y[self.S]

        nSel = len(y)

        self.mask = np.ones(y.size, dtype=bool)  # mask = mascars

        rechable, coverage = np.zeros(
            y.size), np.zeros(y.size)  # ls = rechable

        self.pairwise_distances = euclidean_distances(X)

        self.setMinEnemyDist(y)

        progress = True

        while progress:
            for x in range(len(self.S)):
                if self.mask[x]:
                    coverage[x] = self.getCoverage(x)
                    rechable[x] = self.getRecheable(x)

            progress = False
            for x in range(len(self.S)):
                if self.mask[x] and rechable[x] > coverage[x]:
                    self.mask[x] = False
                    nSel -= 1
                    progress = True

            if progress == False:
                break


        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = np.asarray(self.S)[self.mask]

        self.reduction_ = 1.0 - float(len(self.y_))/len_original_y
        return self.X_, self.y_
