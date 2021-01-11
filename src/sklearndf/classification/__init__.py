"""
Extended versions of all Scikit-Learn classifiers with enhanced E2E support for data
frames.
"""
from .. import __sklearn_0_22__, __sklearn_0_23__, __sklearn_version__
from ._classification import *

if __sklearn_version__ >= __sklearn_0_22__:
    from ._classification_v0_22 import *

if __sklearn_version__ >= __sklearn_0_23__:
    from ._classification_v0_23 import *
