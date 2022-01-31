
"""
Method 
"""

from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier

class Method(InstanceSelectionMixin):
    """ Method (MMM)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====
    
    """

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.classifier = None
        self.sample_indices_ = []

    def select_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")

        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        idx_s = []
        """
        Implementar aqui
        """

        self.X_ = np.asarray(X[idx_s])
        self.y_ = np.asarray(y[idx_s])
        self.sample_indices_ = list(sorted(idx_s))
       
        #print(sorted(idx_prots_s))
        #print(float(len(self.y_))/len(y))

        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_