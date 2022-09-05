"""
Extended versions of `scikit-learn` clusterers with enhanced support for data
frames.
"""

from .. import __sklearn_1_1__, __sklearn_version__
from ._clustering import *

if __sklearn_version__ >= __sklearn_1_1__:
    from ._clustering_v1_1 import *
