
"""
PSDSP 
"""

from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier

#from inout import load_splits_ids
from sklearn.metrics.pairwise import euclidean_distances

from collections import Counter
import copy
import time


class PSDSP(InstanceSelectionMixin):
    """ Local density-based instance selection (PSDSP)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====

    """

    def __init__(self):
        self.n = 4
        self.p = 0.1
        self.sample_indices_ = []

    def c(self, X, y, l):

        indice_mapeado = np.where(y == l)[0]
        X_tmp = copy.copy(X[indice_mapeado])
        y_tmp = copy.copy(y[indice_mapeado])
        return X_tmp, y_tmp, indice_mapeado

    def partitioning(self, X, y, n):

        X = X.toarray()
        idx_to_remove = np.argwhere(np.all(X[..., :] == 0, axis=0))
        X = np.delete(X, idx_to_remove, axis=1)

        n_features = X.shape[1]
        max_di = X.max(axis=0)
        min_di = X.min(axis=0)

        DRange = np.absolute(max_di - min_di)

        range_i = DRange/float(n)

        region = {}

        t0 = time.time()

        for idx, o in enumerate(X):
            x = np.divide((o - min_di), range_i)
            x = x.astype(np.int8)
            x[x == n] = n-1
            hash_x = "".join(map(str, x))
            try:
                region[hash_x].append(idx)
            except:
                region[hash_x] = [idx]
        tamanhos = [len(x) for x in region.values()]
        R = [region[x] for x in region.keys()]
        return R

    def extractPrototype(self, ri, X_tmp):

        c = np.mean(X_tmp[ri], axis=0)[0]

        dist = np.inf
        idx = -1
        for x in ri:
            if euclidean_distances(X_tmp[x], c)[0][0] < dist:
                idx = x
        return idx

    def select_data(self, X, y):

        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)

        self.S = [x for x in range(len_original_y)]

        self.mask = np.zeros(y.size, dtype=bool)  # mask = mascars

        labels = list(sorted(list(set(y))))

        P = set()

        nSel = 0

        for l in labels:

            X_tmp, y_tmp, indice_mapeado = self.c(X, y, l)

            k = int(self.p * y_tmp.shape[0])

            R = self.partitioning(X_tmp, y_tmp, self.n)

            R = sorted(R, key=len, reverse=True)

            i = 0
            while i <= len(R) and i <= k:
                if len(R[i]) == 1:
                    prof = R[i][0]
                else:
                    prof = self.extractPrototype(R[i], X_tmp)
                P.add(indice_mapeado[prof])
                self.mask[indice_mapeado[prof]] = True
                i += 1

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = np.asarray(self.S)[self.mask]
        self.reduction_ = 1.0 - float(len(self.y_))/len_original_y
        return self.X_, self.y_
