"""
Data frame support and feature traceability for `scikit-learn`.

`sklearndf` augments more than 150 `scikit-learn` estimators for
native support of data frames, while leaving the original API intact.
"""

from packaging.version import parse as __parse_version
from sklearn import __version__ as __sklearn_version__

from ._sklearndf import *

__version__ = "2.0.dev3"

__sklearn_version__ = __parse_version(__sklearn_version__)
__sklearn_0_22__ = __parse_version("0.22")
__sklearn_0_23__ = __parse_version("0.23")
__sklearn_0_24__ = __parse_version("0.24")
__sklearn_1_0__ = __parse_version("1.0")

del __parse_version
