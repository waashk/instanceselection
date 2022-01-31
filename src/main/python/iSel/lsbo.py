
"""
LSBo
"""

from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier

from src.main.python.iSel.lssm import LSSm

from src.main.python.utils.general import load_splits_ids
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

    # def __init__(self, dataset, fold):
    def __init__(self, args, fold):
        #self.n_neighbors = n_neighbors
        #self.dataset = dataset
        #self.dataset = args.dataset
        self.outputdir = args.outputdir
        self.fold = fold
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

    def isLessThanMinEnemyDist(self, idx_i, idx_j):
        if self.pairwise_distances[idx_i][idx_j] < self.mindist_ne[idx_i]:
            self.ls[idx_i].add(idx_j)
            return 1
        return 0

    # se idx esta dentro do raio de i
    # definicao de local set
    # quantidade de i's cuja distancia é menor que do inimigo mais proximo de idx
    def getLs(self, idx):
        lenLs = 0
        #self.ls[idx] = set()
        for i in range(len(self.S)):
            if i != idx:  # and self.mask[i]:
                lenLs += self.isLessThanMinEnemyDist(idx, i)

        return lenLs

    def select_data(self, X, y):

        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)

        # Executa LSSm primeiro
        #pre_selector = LSSm()
        #X_copy = copy.copy(X)
        #y_copy = copy.copy(y)
        # pre_selector.fit(X_copy,y_copy)
        #S = pre_selector.sample_indices_
        #print(len(S), S[:10])

        splits = load_splits_ids(
            f'{self.outputdir}/split_10_lssm_idxinfold.csv')
        S, _ = splits[self.fold]

        #print(len(S2), S2[:10])
        #print("Result S:", Counter(y[S]) )
        #print("Result S2:", Counter(y[S2]) )
        #print(S == S2)

        #self.mask = None
        #self.S = None
        #self.mindist_ne = None
        #self.pairwise_distances = None

        # print(S)

        # Aqui self.s é o T do algoritmos original
        self.S = S
        self.S_new = set()

        # Seleciona os elementos de X que pertencem a seleção do ENN
        X = X[self.S]
        y = y[self.S]

        # mask contem os elementos escolhidos, inicialmente todos são True
        self.mask = np.zeros(y.size, dtype=bool)

        # Calculando o LS
        # primeiro calculamos os pares de distancias
        self.pairwise_distances = euclidean_distances(X)

        # Seta a distancia de cada instancia ao inimigo mais proximo
        self.setMinEnemyDist(y)

        # getting LS = rechable
        self.ls = [set() for i in range(len(self.S))]
        lenLs = np.zeros(y.size)
        for x in range(len(self.S)):
            #coverage[x] = self.getCoverage(x)
            lenLs[x] = self.getLs(x)  # ls = rechable

        # print(self.ls[0:10])
        # print(lenLs[0:10])

        sorded_idx_by_lenLs = np.argsort(lenLs)
        # print(sorded_idx_by_lenLs)
        # print(np.asarray(self.ls)[sorded_idx_by_lenLs[:10]])

        # As explained in original paper, PS: no review ele adiciona a intersecao completa
        for x in sorded_idx_by_lenLs:
            inter = self.ls[x].intersection(self.S_new)
            if len(inter) == 0:
                self.S_new.add(x)
                self.mask[x] = True

        # Implementado assim como no review
        # for x in sorded_idx_by_lenLs:
        #	inter = self.ls[x].intersection(self.S_new)
        #	#if len() == 0:
        #	for i in inter:
        #		self.S_new.add(i)
        #		self.mask[i] = True

        # print(self.S_new)
        # print(len(self.S_new))

        # print(self.ne[0])
        # print(self.pairwise_distances[0][306])
        # print(self.mindist_ne[0])
        # print(self.ls)
        # print(rechable)
        # print(Counter(rechable))

        # print(len_original_y)
        # print(len(y))
        #print(len(np.where(self.mask == True)[0]))

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        # print(X[self.mask].shape)
        self.sample_indices_ = np.asarray(self.S)[self.mask]
        #self.sample_indices_ = list(sorted(np.asarray(self.S)[self.mask]))
        #self.sample_indices_ = list(sorted(np.where(self.mask == True)[0]))

        # print(sorted(idx_prots_s))
        # print(float(len(self.y_))/len(y))

        self.reduction_ = 1.0 - float(len(self.y_))/len_original_y
        return self.X_, self.y_
