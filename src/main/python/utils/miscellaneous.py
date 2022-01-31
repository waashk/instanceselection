from pathlib import Path
from typing import Tuple, Union, Dict, Any, List, Optional

from numpy import ndarray
from numpy.random import RandomState

Documents = List[str]
Classes = List[int]
Indices = List[int]

FileOrFolder = Union[Path, str]
Folds = Tuple[ndarray, ndarray, ndarray, ndarray]
JsonResults = Dict[str, Any]

RandomState = Union[RandomState, int]
BestParameters = Dict[str, Any]
Hyperparameters = Optional[Dict[str, Any]]
Estimator = Any  # WIP

# XLNet Types
Tag2Idx = Dict[str, int]
Tag2Name = Dict[int, str]
Tags = List[int]
DocumentsEmbedding = Tuple[List, List, List]
