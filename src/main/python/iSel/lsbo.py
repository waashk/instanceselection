
"""
LSBo
"""

from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier

from src.main.python.iSel.lssm import LSSm

from src.main.python.utils.general import get_splits
from sklearn.metrics.pairwise import euclidean_distances

from collections import Counter
import copy


class LSBo(InstanceSelectionMixin):
    """ Local Set Border Selector (LSBo)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====

    """

    def __init__(self, args, fold):
        self.outputdir = args.outputdir
        self.fold = fold
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

    def isLessThanMinEnemyDist(self, idx_i, idx_j):
        if self.pairwise_distances[idx_i][idx_j] < self.mindist_ne[idx_i]:
            self.ls[idx_i].add(idx_j)
            return 1
        return 0

    def getLs(self, idx):
        lenLs = 0
        for i in range(len(self.S)):
            if i != idx:  # and self.mask[i]:
                lenLs += self.isLessThanMinEnemyDist(idx, i)

        return lenLs

    def select_data(self, X, y):

        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)

        splits_df = get_splits(
            f"{self.outputdir}/split_10_lssm_idxinfold.pkl")
        S = splits_df.loc[self.fold].train_idxs


        self.S = S
        self.S_new = set()

        X = X[self.S]
        y = y[self.S]

        self.mask = np.zeros(y.size, dtype=bool)

        self.pairwise_distances = euclidean_distances(X)

        self.setMinEnemyDist(y)

        self.ls = [set() for i in range(len(self.S))]
        lenLs = np.zeros(y.size)
        for x in range(len(self.S)):
            lenLs[x] = self.getLs(x)  # ls = rechable


        sorded_idx_by_lenLs = np.argsort(lenLs)
        for x in sorded_idx_by_lenLs:
            inter = self.ls[x].intersection(self.S_new)
            if len(inter) == 0:
                self.S_new.add(x)
                self.mask[x] = True

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = np.asarray(self.S)[self.mask]
        self.reduction_ = 1.0 - float(len(self.y_))/len_original_y
        return self.X_, self.y_
