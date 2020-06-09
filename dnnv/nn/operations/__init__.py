from .base import *
from .generator import *
from .math import *
from .nn import *
from .patterns import *
from .tensor import *

Activation: OperationPattern = Relu | Sigmoid | Tanh  # TODO: enable isinstance checks
