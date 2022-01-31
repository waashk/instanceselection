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
        
        #Chega dimensões de X e y
        X, y = check_X_y(X, y, accept_sparse="csr")

        #Define classificador como KNN e numero de vizinhos
        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        idx_s = []

        #Note que classes só depende do treino
        classes = np.unique(y)
        self.classes_ = classes

        #Escolhe um de cada classe
        for cur_class in classes:
            mask = np.where(y == cur_class)[0]
            choose_id = random.choice(mask)
            idx_s = idx_s + [choose_id]

        #print(idx_s)
        #print(X[idx_s])
        #print(y[idx_s])

        train_idx = list(range(len(y)))
        train_idx = random.sample(train_idx, len(train_idx))

        flag = True
        # Realuza o processo passando por todo o dataset enquanto pelo menos uma instancia for 
        # adicionada a S
        while flag:
            flag = False
            # Inicialmente so com uma instancia de cada classe
            self.classifier.fit(X[idx_s], y[idx_s])

            for idx in train_idx:
                # Verifica se já não tá na solução
                # E se a predição é errada
                # Se positivo, adiciona a instancia a S e treina novamente o classificador
                if idx not in idx_s and self.classifier.predict(X[idx]) != [y[idx]]:
                    idx_s = idx_s + [idx]
                    self.classifier.fit(X[idx_s], y[idx_s])
                    flag = True
       
        self.X_ = np.asarray(X[idx_s])
        self.y_ = np.asarray(y[idx_s])
        self.sample_indices_ = list(sorted(idx_s))
       
        # (t-s)/t = 1 - s/t
        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_