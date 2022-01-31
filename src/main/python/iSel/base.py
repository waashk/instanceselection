"""Base para metodos de redução"""
import warnings
class InstanceReductionWarning(UserWarning):
    pass
warnings.simplefilter("always", InstanceReductionWarning)

from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
from sklearn.externals import six

class InstanceSelectionBase(six.with_metaclass(ABCMeta, BaseEstimator)):
    @abstractmethod
    def __init__(self):
        pass


class InstanceSelectionMixin(InstanceSelectionBase):

    """Mixin class for all instance reduction techniques"""
    def select_data(self, X, y):
        """
        Procedimento para redução dos dados. 
        # Entrada X e y
        # X = 
        # y =
        """
        pass

    def fit(self, X, y):
        """
        Call reduce data procedure
        """
        self.X = X
        self.y = y

        self.select_data(X, y)

        return self

