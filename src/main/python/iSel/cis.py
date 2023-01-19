
"""
Curious Instance Selection 
"""

from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import copy
from sklearn.model_selection import train_test_split
from numpy.random import uniform
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from collections import Counter
from sklearn.metrics import accuracy_score

class CIS(InstanceSelectionMixin):
    """ Method (CIS)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====
    
    """

    def __init__(self, task=None):
        self.s_rate = None        # the fraction of data to sample from each cluster
        self.sample_indices_ = []
        self.task = task

    def proba_random_action(self, i, nIter):
        
        p1 = nIter*0.25    #nIter/4
        p2 = nIter*0.50    #nIter/2
        p3 = nIter*0.75    #nIter*(3/4)
        
        if i < p1:
            return 0.09
        elif i >= p1 and i < p2:
            return 0.05
        elif i >= p2 and i < p3:
            return 0.01
        elif i >= p3:
            return 0.005
    
    def get_lr(self, i, nIter):
        p1 = nIter*0.25  # nIter/4
        p2 = nIter*0.50  # nIter/2
        p3 = nIter*0.75  # nIter*(3/4)

       
        if i < p1:
            return 0.9
        elif i >= p1 and i < p2:
            return 0.5
        elif i >= p2 and i < p3:
            return 0.3
        elif i >= p3:
            return 0.1

    def get_s_rate(self, N):
        if N < 10_000:
            return 0.5
        elif (N >= 10_000 and N < 100_000):
            return 0.4
        else: # >= 100_000
            return 0.1


    def sample_clusters(self, yPredictedClusters):
        #self.s_rate
        idx_c = []
        classes = list(sorted(np.unique(yPredictedClusters)))
        for k in classes:
            idxs = np.where([yPredictedClusters == k])[1].tolist()
            sel = int(len(idxs)*self.s_rate) + 1 
            idxs = random.sample(idxs, k=sel)
            idx_c += list(idxs)
        return idx_c

    def split_data(self, X, y):
        for train_index, val_index in StratifiedKFold(n_splits=10, shuffle=True).split(X, y): 
            return train_index, val_index


    def get_clusters_order_to_considerate(self, Q):
        s0 = 0 
        order_to_considerate = [s0]

        considerated = 0
        while considerated < Q.shape[0] -1:

            order = np.argsort(Q[s0])[::-1]           
            
            for o in order: 
                if o not in order_to_considerate:
                    st = copy.copy(o)
                    break

            order_to_considerate.append(st)
            s0 = copy.copy(st)          
            considerated+=1

        return order_to_considerate[1:]

    def get_clusters_to_considerate(self, Q, X_train, y_train, y_train_PredictedClusters, X_val, y_val):

        order_to_considerate = self.get_clusters_order_to_considerate(Q)     

        consider_iter = []

        acc = []
        red = []
        accred = []

        besteff = - np.inf
        numberOfClusters = 0

        for i, c in enumerate(order_to_considerate):
            consider_iter.append(c)

            idx_to_consider = [x for x in range(X_train.shape[0]) if y_train_PredictedClusters[x] in consider_iter]

            reduction = 1. - len(idx_to_consider) / X_train.shape[0]

            learner = DecisionTreeClassifier(random_state=0)
            learner.fit(X_train[idx_to_consider], y_train[idx_to_consider])

            y_pred = learner.predict(X_val)

            acc.append(accuracy_score(y_val, y_pred))
            red.append(reduction)
            accred.append(accuracy_score(y_val, y_pred) * reduction)

            eff = accuracy_score(y_val, y_pred) * reduction
            if eff > besteff:
                besteff = copy.copy(eff)
                numberOfClusters = i+1

        return order_to_considerate[:numberOfClusters]


    def select_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")

        self.s_rate = self.get_s_rate(X.shape[0]) 

        if self.task == "atc":
            self.k_clusters = int((X.shape[0] / X.shape[1]) * (10.0/self.s_rate))  # pseudocode
        else:
            self.k_clusters = int(X.shape[0] / (X.shape[1]*(10.0/self.s_rate)))  # validated manuscript explanation ( Formula 3 ) 

        print(self.k_clusters)

        Q = np.zeros((self.k_clusters+1, self.k_clusters+1), dtype=float) 

        nIter = self.k_clusters * 100  
        gamma = 0.01                    

        kmeansEst = KMeans(n_clusters=self.k_clusters, random_state=0).fit(X) 
        yPredictedClusters = kmeansEst.labels_ + 1

        train_index, val_index = self.split_data(X, y)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        y_train_PredictedClusters = yPredictedClusters[train_index]

        for i in tqdm(range(nIter)):

            idx_x_epi = self.sample_clusters(y_train_PredictedClusters)
            X_epi, y_epi = X_train[idx_x_epi], y_train_PredictedClusters[idx_x_epi]

            X_epi_train, X_epi_val, y_epi_train, y_epi_val = train_test_split(X_epi, y_epi,
                                                                stratify=y_epi,
                                                                test_size=0.2,
                                                                shuffle=True)
            #except: 
            if X_epi_train.shape[0] == 0:
                    continue

            e0 = 0.5 
            st = 0      # initial stat where no cluster has been selected
            B_sel = list()                               # all clusters thate were selected during the episode
            
            B_available = list(set(y_train_PredictedClusters))

            epsilon = self.proba_random_action(i, nIter)   
            alpha = self.get_lr(i, nIter)               # learning rate

            while len(B_available) > 0:

                if uniform(0.0, 1.0) < epsilon:
                    at = random.sample(B_available, 1)[0]
                else:
                    max_value = -np.inf
                    for a in B_available:
                        if Q[st][a] > max_value:
                            at = copy.copy(a)
                            max_value = copy.copy(Q[st][a])

                B_sel.append(at)

                _idx_list = []
                for _idx, _y in enumerate(y_epi_train):
                    if _y in B_sel:
                        _idx_list.append(_idx)
                
                x_epi_train_st = X_epi_train[_idx_list]
                y_epi_train_st = y_epi_train[_idx_list]


                try:
                    O_learner = DecisionTreeClassifier(random_state=42, 
                                                        max_depth=10,
                                                        min_samples_leaf=5,
                                                        max_features=0.7)
                    O_learner.fit(x_epi_train_st, y_epi_train_st)
                except Exception as e:
                    print(e)
                    print(B_sel)
                    return

                y_pred_x_epi_val = O_learner.predict(X_epi_val)

                et = 1.0 - sum(y_pred_x_epi_val == y_epi_val) / y_epi_val.size

                rt = copy.copy(e0 - et)

                st1 = copy.copy(at)
                Q[st][at] = Q[st][at] + alpha *  (rt + gamma * Q[st1].max() - Q[st][at])  

                st = copy.copy(at)
                e0 = copy.copy(et)
                B_available.remove(at)

        final_clusters = self.get_clusters_to_considerate(Q, X_train, y_train, y_train_PredictedClusters, X_val, y_val)

        S = [x for x in range(X.shape[0])
             if yPredictedClusters[x] in final_clusters]
        self.sample_indices_ = S
        
        self.X_ = np.asarray(X[S])
        self.y_ = np.asarray(y[S])

        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_
