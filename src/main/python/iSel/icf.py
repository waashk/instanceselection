
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

    def __init__(self, args, fold):  # , n_neighbors=3):
        # def __init__(self, n_neighbors=3):
        #self.n_neighbors = n_neighbors
        self.fold = fold
        #self.dataset = dataset
        #self.dataset = args.dataset
        self.outputdir = args.outputdir
        self.classifier = None
        self.sample_indices_ = []

    def setMinEnemyDist(self, y):
        # mindist_ne = distancia até primeira instancia de outra classe
        self.mindist_ne = np.zeros(y.size)
        # ne = idx of nearest enemy
        self.ne = np.zeros(y.size)
        # Para cada instancia, setamos a menor distancia até intancia de outra classe
        # for i in range(len(self.S)):
        for i in range(len(self.S)):
            self.mindist_ne[i] = np.inf
            for j in range(len(self.S)):
                if y[i] != y[j]:
                    if self.pairwise_distances[i][j] < self.mindist_ne[i]:
                        self.mindist_ne[i] = self.pairwise_distances[i][j]
                        self.ne[i] = copy.copy(j)

    # se i está no raio de idx

    def getCoverage(self, idx):
        lenCoverage = 0
        for i in range(len(self.S)):
            if i != idx and self.mask[i]:
                lenCoverage += self.isLessThanMinEnemyDist(idx, i)
        return lenCoverage

    # se idx esta dentro do raio de i
    # definicao de local set
    # quantidade de i's cuja distancia é menor que do inimigo mais proximo de idx
    def getRecheable(self, idx):
        lenRecheable = 0
        for i in range(len(self.S)):
            if i != idx and self.mask[i]:
                lenRecheable += self.isLessThanMinEnemyDist(i, idx)
        return lenRecheable

    # se j esta dentro do raio de i
    def isLessThanMinEnemyDist(self, idx_i, idx_j):
        if self.pairwise_distances[idx_i][idx_j] < self.mindist_ne[idx_i]:
            return 1
        return 0

    def select_data(self, X, y):

        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)

        # Executa ENN primeiro
        #pre_selector = ENN()
        #X_copy = copy.copy(X)
        #y_copy = copy.copy(y)
        # pre_selector.fit(X_copy,y_copy)
        #S = pre_selector.sample_indices_
        #print(len(S), S[:10])

        splits = load_splits_ids(
            f'{self.outputdir}/split_10_enn_idxinfold.csv')
        S, _ = splits[self.fold]

        self.S = S

        # Seleciona os elementos de X que pertencem a seleção do ENN
        X = X[self.S]
        y = y[self.S]

        nSel = len(y)

        # mask contem os elementos escolhidos, inicialmente todos são True
        self.mask = np.ones(y.size, dtype=bool)  # mask = mascars

        rechable, coverage = np.zeros(
            y.size), np.zeros(y.size)  # ls = rechable

        # primeiro calculamos os pares de distancias
        self.pairwise_distances = euclidean_distances(X)

        # Seta a distancia de cada instancia ao inimigo mais proximo
        self.setMinEnemyDist(y)

        # Ls
        #self.ls = [set() for i in range(len(self.S))]

        # Até aqui é definido fora do while
        progress = True

        while progress:
            for x in range(len(self.S)):
                if self.mask[x]:
                    coverage[x] = self.getCoverage(x)
                    rechable[x] = self.getRecheable(x)
                # Self. marks ta com todo mundo true, equivale a REM = False

            progress = False
            # print(coverage)
            # print(ls)
            for x in range(len(self.S)):
                #print(ls[x], coverage[x])
                # if ls[x] >= coverage[x]:
                if self.mask[x] and rechable[x] > coverage[x]:
                    self.mask[x] = False
                    nSel -= 1
                    progress = True
            # print(nSel)

            if progress == False:
                break

        #print(f'original pos enn {len(y)}')
        #print(f'nsel {nSel}')
        # print(self.mask)
        # print(len(self.S))
        # print(np.asarray(self.S)[self.mask])
        #print(np.where(self.mask == True)[0])
        # print(len(np.asarray(self.S)[self.mask]))

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        # print(X[self.mask].shape)
        self.sample_indices_ = np.asarray(self.S)[self.mask]

        # print(sorted(idx_prots_s))
        # print(float(len(self.y_))/len(y))

        self.reduction_ = 1.0 - float(len(self.y_))/len_original_y
        return self.X_, self.y_
