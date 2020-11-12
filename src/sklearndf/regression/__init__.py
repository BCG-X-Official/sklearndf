"""
Extended versions of all Scikit-Learn regressors with enhanced E2E support for data
frames.
"""
from packaging.version import parse as __parse_version
from sklearn import __version__ as __sklearn_version__

from ._regression import *

__sklearn_version = __parse_version(__sklearn_version)

if __sklearn_version >= __parse_version("0.22"):
    from ._regression_v0_22 import *

if __sklearn_version >= __parse_version("0.23"):
    from ._regression_v0_23 import *
