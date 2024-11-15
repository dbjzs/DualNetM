from .import datasets
from .import utils
from .import eval_utils
from DualNetM.utils import data_preparation
from DualNetM.GRN_model import NetModel

__version__ = '0.1.0'
__url__ = 'https://github.com/dbjzs/DualNetM'
__author__ = 'Peizhuo Wang'
__author_email__ = '17516970902@163.com'

__all__ = [
    'datasets',
    'utils',
    'eval_utils',
    'NetModel',
    'data_preparation',
]
