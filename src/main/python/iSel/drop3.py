
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

    # def __init__(self, dataset, fold, n_neighbors=3, loadenn=True):
    def __init__(self, args, fold, n_neighbors=3, loadenn=True):
        # def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.fold = fold
        #self.dataset = dataset
        #self.dataset = args.dataset
        self.outputdir = args.outputdir
        self.classifier = None
        self.sample_indices_ = []
        self.loadenn = loadenn

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

    def skip_diag_masking(self, A):
        return A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)

    def set_nn(self, X, y):
        """#setando o classificador
        #if self.classifier == None:

        #Coloquei um a mais pq o vizinho mais proximo de x é ele mesmo
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors+1)
        #if self.classifier.n_neighbors != self.n_neighbors:
        #	self.classifier.n_neighbors = self.n_neighbors

        self.classifier.fit(X, y)
        #print(self.classifier.kneighbors(X,return_distance=False)[0])
        #print(self.classifier.kneighbors(X,return_distance=False)[1])

        #[:,1:] Faz slice de 1 a ultima coluna, ou seja, remove o proprio x
        #self.nn = self.classifier.kneighbors(X,return_distance=False)[:,1:]
        result = self.classifier.kneighbors(X,return_distance=False)[:,1:]
        self.nn = [set(result[x]) for x in range(len(result))]
        #print(self.nn)"""
        # Primeiro computa a distancia de todo mundo pra todo mundo
        self.pairwise_distances = euclidean_distances(X)

        # Depois setamos a distancia pra ele mesmo (diagonal) como -1, pra garantir que cada instancia seja o mais proximo de si memso
        for i in range(self.pairwise_distances.shape[0]):
            self.pairwise_distances[i][i] = -1.0

        # A seguir ordenamos a distancia e pegamos da coluna 1 em diante (excluindo a propria isntancia)
        self.nn_complete = np.argsort(self.pairwise_distances)[:, 1:]
        #print(len(self.nn_complete))
        #print(self.nn_complete[567])
        #print(len(self.nn_complete[567]))

        self.nn = [x[:self.n_neighbors] for x in self.nn_complete]
        # print(self.nn_complete[0][:10],self.nn[0])
        # exit()

    def set_A(self):
        # print(self.nn[0])

        self.A = [set() for x in self.S]
        #self.A = [[] for x in self.S]
        # for x in range(len(self.S)):
        # for x in [0]:#range(len(self.S)):
        #	for y in range(len(self.S)):
        #		if x in self.nn[y]:
        #			print(y)
        # exit()
        #			self.A[x].append(y)
        # for idx, x in enumerate(self.nn):
        for x in range(len(self.S)):
            for y in self.nn[x]:
                self.A[y].add(x)
        # print(self.A[0])
        # print(self.nn[767])

        # for x in range(len(self.S)):
        #	if x in self.A[x]:
        #		print("ok")

        # exit()

        # print(self.A)

    def set_N(self):
        # print(self.nn[0])
        #nncop = copy.copy(self.nn)

        #print(self.nn == nncop)
        """for x in range(len(self.S)):
        #for x in [2]:
                #print('x',x)
                #print('Ax',self.A[x])
                for y in self.A[x]:
                        #print('y',y)
                        #print('nny',self.nn[y])
                        self.nn[y].add(x)"""

        # antigo

        #aux= 0
        # for y in range(len(self.S)):
        #	for x in self.A[y]:
        #		if y not in self.A[x]:
        #			#aux+=1
        #			self.A[x].add(y)
        #			print(y, self.A[x])

        #aux = 0
        # for x in range(len(self.S)):
        #	for y in self.A[x]:
        #		if x not in self.nn[y]:
        #			print(self.nn_complete[y,:self.n_neighbors], x)
        #			aux+=1

        # print(aux)
        # exit()

        # for x in range(len(self.S)):
        #	for y in self.A[x]:
        #		self.A[y].add(x)

        #print(self.nn == nncop)

        # print(self.nn[0])

    def most_common(self, lst):

        #if len(lst) == 0:
        #    return None
        
        try:
            counts = np.bincount(lst)
            most_c = np.argmax(counts)
        except:
            most_c = collections.Counter(lst).most_common()[0][0]

        return most_c
        # print(collections.Counter(lst).most_common()[0][0])
        # exit()
        # return max(set(lst), key=lst.count)

    def get_nn(self, x):
        n_neighbors = self.n_neighbors

        nn = []
        #removido porque chegava noo nn_complete = len(self.S)
        #erros em sentistrength_bbc_2L, sarcasm_2L, 
        #for i in range(len(self.S)):
        for i in range(len(self.S) - 1):
            
            #try:           
            idx = self.nn_complete[x][i]
            #except Exception as e:
            #    print(e)
            #    print("continuing")
            #    exit()

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
            #nn_of_x = self.nn[x]
            nn_of_x = self.get_nn(x)
            # print(x,nn_of_x,y[nn_of_x],self.most_common(y[nn_of_x]))
            predicited.append(self.most_common(y[nn_of_x]))
            # exit()
        #predicted = [y[a] for a in nn_of_x]
        # print(predicited)
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
            #tmp_m[i] = not tmp_m[i]
            #classifier.fit(X[tmp_m], y[tmp_m])
            classifier.fit(X[mask], y[mask])

            if classifier.predict(X[i]) != [y[i]]:
                mask[i] = not mask[i]

            #tmp_m[i] = not tmp_m[i]

        S = np.asarray([x for x in range(X.shape[0])])
        S = S[mask]

        print("ENN ", round(1.0 - float(len(S))/len_original_y, 2))
        return S

    def select_data(self, X, y):

        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)
        #print("ll", len_original_y)
        #exit()

        # Executa ENN primeiro
        #pre_selector = ENN()
        #X_copy = copy.copy(X)
        #y_copy = copy.copy(y)
        # pre_selector.fit(X_copy,y_copy)
        #S = pre_selector.sample_indices_
        #print(len(S), S[:10])

        if self.loadenn:
            #splits = load_splits_ids(f'outselection/{self.dataset}/split_10_enn_idxinfold.csv')
            splits = load_splits_ids(
                f'{self.outputdir}/split_10_enn_idxinfold.csv')
            S, _ = splits[self.fold]
        else:
            S = self.ennpadrao(X, y)

        self.S = S

        # Seleciona os elementos de X que pertencem a seleção do ENN
        X = X[self.S]
        print(X.shape)
        #exit()
        y = y[self.S].astype(int)

        nSel = len(y)

        # mask contem os elementos escolhidos, inicialmente todos são True
        self.mask = np.ones(y.size, dtype=bool)  # mask = mascars

        # seta os n vizinhos mais proximos de cada instancia
        self.set_nn(X, y)

        # Seta a distancia de cada instancia ao inimigo mais proximo
        self.setMinEnemyDist(y)

        # ordem a ser seguida pelo drop3
        sorted_elements_by_ne = np.argsort(self.mindist_ne)
        #sorted_elements_by_ne = list(reversed(sorted_elements_by_ne))

        #print(np.where(self.mindist_ne == np.min(self.mindist_ne)), sorted_elements_by_ne[0:10])

        self.set_A()
        # print(self.A[0])
        self.set_N()
        # print(self.A[0])

        # for x in self.A[0]:
        #	print(self.nn[x])

        # for x in range(len(self.S)):
        # print('ok')
        # for x in sorted_elements_by_ne[:10]:
        #print(len(self.S))
        #print(sorted_elements_by_ne[sorted_elements_by_ne == len(self.S) - 1])
        #print(sorted_elements_by_ne)

        for x in sorted_elements_by_ne:

            # print(x)
            Ax_list = list(self.A[x])
            # print(Ax_list)

            # print(self.nn[Ax_list[0]])

            # print(Ax_list)
            y_of_ax_list = y[Ax_list]

            # print(y_of_ax_list)

            self.mask[x] = False
            len_A_without = self.classify(Ax_list, y, y_of_ax_list)

            self.mask[x] = True
            len_A_with = self.classify(Ax_list, y, y_of_ax_list)

            #print(len_A_with, len_A_without)

            if len_A_without >= len_A_with:
                # if len_A_without > len_A_with:
                #maioria = (self.n_neighbors+1)/2
                # if len_A_without > maioria:
                # print(x)
                nSel -= 1
                # print("Entrou")
                self.mask[x] = False

                # for y1 in self.A[x]:
                #	print(self.A[y1])

                # exit()

                for y1 in self.A[x]:

                    # testendo remover o x
                    # if x in self.A[y1]:
                    #	self.A[y1].remove(x)
                    # print(x)
                    # removendo x da lista de nn de y
                    self.nn[y1] = [a for a in self.nn[y1] if a != x]

                    # encontra novo vizinho para y1
                    # u =
                    self.find_new_nn(y1)

                    # print(self.A[y1])
                    # self.A[y1].remove(x)
                    #print(y, self.get_nn(y))
                    #print("ok", y)
                    # print(y)
                    nn_y1 = self.nn[y1]

                    for z1 in nn_y1:
                        # print(z,self.A[z])

                        # print(self.A[z1])

                        # self.A[z1].remove(x)
                        # if x in self.A[z1]:
                        #	self.A[z1].remove(x)

                        self.A[z1].add(y1)

        # print(nSel)

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        # print(X[self.mask].shape)
        self.sample_indices_ = np.asarray(self.S)[self.mask]
        print(self.sample_indices_)

        # print(sorted(idx_prots_s))
        # print(float(len(self.y_))/len(y))

        self.reduction_ = 1.0 - float(len(self.y_))/len_original_y

        # print(self.reduction_)
        return self.X_, self.y_
