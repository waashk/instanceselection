
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

    # def __init__(self, dataset, fold, n_neighbors=3):
    def __init__(self):
        #self.n_neighbors = n_neighbors
        self.n = 4
        self.p = 0.1
        #self.fold = fold
        #self.dataset = dataset
        self.sample_indices_ = []

    def c(self, X, y, l):

        indice_mapeado = np.where(y == l)[0]
        X_tmp = copy.copy(X[indice_mapeado])
        y_tmp = copy.copy(y[indice_mapeado])
        return X_tmp, y_tmp, indice_mapeado

    def partitioning(self, X, y, n):

        X = X.toarray()
        # print(X)
        # print()
        idx_to_remove = np.argwhere(np.all(X[..., :] == 0, axis=0))
        X = np.delete(X, idx_to_remove, axis=1)

        n_features = X.shape[1]

        # exit()
        # maximo por coluna
        # print(X.max(axis=0))
        # print(X.max(axis=0).toarray()[0])
        # print(X.min(axis=0).toarray()[0])

        #max_di = X.max(axis=0).toarray()[0]
        max_di = X.max(axis=0)
        #min_di = X.min(axis=0).toarray()[0]
        min_di = X.min(axis=0)
        # print(min_di)

        # print(min_di)
        # exit()

        DRange = np.absolute(max_di - min_di)
        # print(DRange[0:5])

        range_i = DRange/float(n)
        # print(range_i)

        # exit()

        region = {}

        t0 = time.time()

        for idx, o in enumerate(X):
            # print(idx)
            #x = np.zeros(n_features, dtype=int)

            #x = (o - min_di) / range_i
            x = np.divide((o - min_di), range_i)
            # print(x)
            x = x.astype(np.int8)
            # print(x)
            # caso seja o maximo, pra ficar com a quantidade de slotes certo
            x[x == n] = n-1
            #print(x == 5)
            #x = np.nan_to_num(x, copy=False)
            #hash_x = x.tostring()
            hash_x = "".join(map(str, x))
            # print(hash_x)
            try:
                region[hash_x].append(idx)
            except:
                region[hash_x] = [idx]

        #print("como ser mais rapido?")
        #print(time.time()- t0, "seconds")
        # print(x)
        # for x in region.values():
        tamanhos = [len(x) for x in region.values()]
        # print(tamanhos)
        # print("regions",len(tamanhos))
        # print(sorted(tamanhos,reverse=True))
        # no original da 17 16 15 15 13 13 ; 7 6 5 4 4 4

        R = [region[x] for x in region.keys()]
        # print(R)
        return R

    def extractPrototype(self, ri, X_tmp):

        c = np.mean(X_tmp[ri], axis=0)[0]

        dist = np.inf
        idx = -1
        for x in ri:
            # if euclidean_distances(X_tmp[x],c)
            # print(euclidean_distances(X_tmp[x],c)[0][0])
            if euclidean_distances(X_tmp[x], c)[0][0] < dist:
                idx = x

        # print(c.shape)
        # print(c)
        # exit()
        return idx

    def select_data(self, X, y):

        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)

        self.S = [x for x in range(len_original_y)]

        # mask contem os elementos escolhidos, inicialmente todos são True
        self.mask = np.zeros(y.size, dtype=bool)  # mask = mascars

        # seta os n vizinhos mais proximos de cada instancia
        #self.set_nn(X, y)

        labels = list(sorted(list(set(y))))

        P = set()

        nSel = 0
        # print(X.shape)
        # print(labels)

        for l in labels:
            # for l in [1]:
            # print("Classe",l)

            X_tmp, y_tmp, indice_mapeado = self.c(X, y, l)

            # print(y_tmp.shape[0])
            k = int(self.p * y_tmp.shape[0])
            # print(k)

            R = self.partitioning(X_tmp, y_tmp, self.n)

            R = sorted(R, key=len, reverse=True)

            i = 0
            while i <= len(R) and i <= k:
                # while i <= 3:
                if len(R[i]) == 1:
                    prof = R[i][0]
                else:
                    prof = self.extractPrototype(R[i], X_tmp)
                P.add(indice_mapeado[prof])
                self.mask[indice_mapeado[prof]] = True
                i += 1

        # print(self.mask)
        # exit()
        # tem que mudar daqui pra baixo

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        # print(X[self.mask].shape)
        self.sample_indices_ = np.asarray(self.S)[self.mask]

        # print(sorted(idx_prots_s))
        # print(float(len(self.y_))/len(y))

        self.reduction_ = 1.0 - float(len(self.y_))/len_original_y
        return self.X_, self.y_
