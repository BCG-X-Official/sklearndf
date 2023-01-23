"""
Extended versions of all `scikit-learn` regressors with enhanced support for data
frames.
"""
from .. import __sklearn_1_0__, __sklearn_version__
from ._regression import *
from ._regression_v0_22 import *
from ._regression_v0_23 import *

if __sklearn_version__ >= __sklearn_1_0__:
    from ._regression_v1_0 import *
