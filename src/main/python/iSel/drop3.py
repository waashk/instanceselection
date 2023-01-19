
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
import collections


class DROP3(InstanceSelectionMixin):
    """  (DROP3)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====

    """

    def __init__(self, args, fold, n_neighbors=3, loadenn=True):
        self.n_neighbors = n_neighbors
        self.fold = fold
        self.outputdir = args.outputdir
        self.classifier = None
        self.sample_indices_ = []
        self.loadenn = loadenn

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

    def skip_diag_masking(self, A):
        return A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)

    def set_nn(self, X, y):

        self.pairwise_distances = euclidean_distances(X)

        for i in range(self.pairwise_distances.shape[0]):
            self.pairwise_distances[i][i] = -1.0

        self.nn_complete = np.argsort(self.pairwise_distances)[:, 1:]
        self.nn = [x[:self.n_neighbors] for x in self.nn_complete]

    def set_A(self):

        self.A = [set() for x in self.S]
        for x in range(len(self.S)):
            for y in self.nn[x]:
                self.A[y].add(x)


    def most_common(self, lst):

        try:
            counts = np.bincount(lst)
            most_c = np.argmax(counts)
        except:
            most_c = collections.Counter(lst).most_common()[0][0]

        return most_c

    def get_nn(self, x):
        n_neighbors = self.n_neighbors

        nn = []

        for i in range(len(self.S) - 1):
       
            idx = self.nn_complete[x][i]

            if self.mask[idx]:
                nn.append(idx)

            if len(nn) == n_neighbors:
                return nn

        return nn

    def find_new_nn(self, y):

        for i in range(len(self.S)-1):
            idx = self.nn_complete[y][i]
            if self.mask[idx]:
                if idx not in self.nn[y]:
                    self.nn[y].append(idx)
                    break

    def classify(self, Ax_list, y, y_of_ax_list):
        predicited = []
        for x in Ax_list:
            nn_of_x = self.get_nn(x)
            predicited.append(self.most_common(y[nn_of_x]))
        aux = 0
        for i in range(len(Ax_list)):
            if y_of_ax_list[i] == predicited[i]:
                aux += 1
        return aux

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
        print(X.shape)
        y = y[self.S].astype(int)

        nSel = len(y)

        self.mask = np.ones(y.size, dtype=bool)  # mask = mascars
        self.set_nn(X, y)
        self.setMinEnemyDist(y)
        sorted_elements_by_ne = np.argsort(self.mindist_ne)
        self.set_A()

        for x in sorted_elements_by_ne:
            Ax_list = list(self.A[x])
            y_of_ax_list = y[Ax_list]

            self.mask[x] = False
            len_A_without = self.classify(Ax_list, y, y_of_ax_list)

            self.mask[x] = True
            len_A_with = self.classify(Ax_list, y, y_of_ax_list)

            if len_A_without >= len_A_with:
                nSel -= 1
                self.mask[x] = False

                for y1 in self.A[x]:
                    self.nn[y1] = [a for a in self.nn[y1] if a != x]

                    self.find_new_nn(y1)
                    nn_y1 = self.nn[y1]

                    for z1 in nn_y1:

                        self.A[z1].add(y1)


        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = np.asarray(self.S)[self.mask]
        print(self.sample_indices_)

        self.reduction_ = 1.0 - float(len(self.y_))/len_original_y

        return self.X_, self.y_
