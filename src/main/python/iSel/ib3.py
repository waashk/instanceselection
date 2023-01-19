
from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from scipy.spatial.distance import cdist

import copy
from sklearn.metrics.pairwise import pairwise_distances

class IB3(InstanceSelectionMixin):
    """ Instance Base 3 (IB3)
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
        self.sample_indices_ = []
        self.nAccept = 0.9
        self.nDrop = 0.7

    def isAcceptable(self, c, cClass, recordClass, freqClass):
        if recordClass[c].sum() != 0.0:
            minAc = (recordClass[c][0] / recordClass[c].sum()) * self.nAccept
        else:
            minAc = 0.
        maxFr = freqClass[c][cClass] / freqClass[c].sum()

        if (minAc > maxFr):
            return True

        return False

    def isDrop(self, c, cClass, recordClass, freqClass):

        maxAc = recordClass[c][0] / recordClass[c].sum()
        minFr = (freqClass[c][cClass] / freqClass[c].sum()) * self.nDrop

        if (maxAc < minFr):
            return True

        return False
    


    def select_data(self, X, y):

        X, y = check_X_y(X, y, accept_sparse="csr")


        PDMatrix = pairwise_distances(X=X, metric='euclidean')


        classes = np.unique(y)
        self.classes_ = classes
        self.mask = np.zeros(y.size, dtype=bool)

        #recordmatrix
        recordClass = np.zeros(shape=(y.size, 2))

        #Class frequency
        nclasses = len(set(self.classes_))
        freqClass = np.zeros(shape=(y.size, nclasses))

        #similarity vector
        sim = np.zeros(y.size) - 1.0
        CD = []


        xidx_random_list = list(range(X.shape[0]))
        random.shuffle(xidx_random_list)

        for xidx in xidx_random_list:

            if len(CD) == 0:
                CD.append(xidx)
                self.mask[xidx] = True

                freqClass[xidx][y[xidx]]+=1
            
                recordClass[xidx][0] +=1
            else:
                sim[CD] = PDMatrix[xidx][CD]
                
                acceptable = False
                bestsim = np.inf

                for c in CD:
                    if self.isAcceptable(c, y[c], recordClass,freqClass):
                        if sim[c] < bestsim:
                            acceptable = True
                            cmax = copy.copy(c)
                            bestsim = copy.copy(sim[c])

                if not acceptable:
                    cmax = random.randint(0, len(CD)-1)

                if y[xidx] != y[cmax]:
                    CD.append(xidx)
                    self.mask[xidx] = True
                    sim[xidx] = 0



                for c in CD:
                    freqClass[c][y[xidx]] +=1

                for c in CD:
                    if sim[c] <= sim[cmax]:
                        if y[c] == y[xidx]:
                            recordClass[c][0]+=1
                        else:
                            recordClass[c][1]+=1

                        if self.isDrop(c, y[c], recordClass, freqClass):
                            CD.remove(c)
                            self.mask[c] = False

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = list(sorted(np.where(self.mask == True)[0]))
        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_

