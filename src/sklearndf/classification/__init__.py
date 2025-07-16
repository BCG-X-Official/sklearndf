"""
Extended versions of all `scikit-learn` classifiers with enhanced support for data
frames.
"""

from .. import __sklearn_1_5__, __sklearn_1_6__, __sklearn_version__
from ._classification import *
from ._classification_v0_22 import *
from ._classification_v0_23 import *
from ._classification_v1_0 import *

if __sklearn_version__ >= __sklearn_1_5__:
    from ._classification_v1_5 import *

if __sklearn_version__ >= __sklearn_1_6__:
    from ._classification_v1_6 import *
