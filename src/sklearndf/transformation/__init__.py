"""
Extended versions of all `scikit-learn` transformers with enhanced support for data
frames.
"""

from .. import (
    __sklearn_0_22__,
    __sklearn_0_24__,
    __sklearn_1_0__,
    __sklearn_1_1__,
    __sklearn_version__,
)
from ._transformation import *

if __sklearn_version__ >= __sklearn_0_22__:
    from ._transformation_v0_22 import *

if __sklearn_version__ >= __sklearn_0_24__:
    from ._transformation_v0_24 import *

if __sklearn_version__ >= __sklearn_1_0__:
    from ._transformation_v1_0 import *

if __sklearn_version__ >= __sklearn_1_1__:
    from ._transformation_v1_1 import *
