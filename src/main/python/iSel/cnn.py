"""
CNN 
"""

from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier

class CNN(InstanceSelectionMixin):
    """ Condensed Nearest Neighbors (CNN)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====
    P. E. Hart, The condensed nearest neighbor rule (1968).
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

        classes = np.unique(y)
        self.classes_ = classes

        for cur_class in classes:
            mask = np.where(y == cur_class)[0]
            choose_id = random.choice(mask)
            idx_s = idx_s + [choose_id]

        train_idx = list(range(len(y)))
        train_idx = random.sample(train_idx, len(train_idx))

        flag = True
        while flag:
            flag = False
            self.classifier.fit(X[idx_s], y[idx_s])

            for idx in train_idx:
                if idx not in idx_s and self.classifier.predict(X[idx]) != [y[idx]]:
                    idx_s = idx_s + [idx]
                    self.classifier.fit(X[idx_s], y[idx_s])
                    flag = True
       
        self.X_ = np.asarray(X[idx_s])
        self.y_ = np.asarray(y[idx_s])
        self.sample_indices_ = list(sorted(idx_s))
        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_