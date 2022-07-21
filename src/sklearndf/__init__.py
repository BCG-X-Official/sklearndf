"""
Data frame support and feature traceability for `scikit-learn`.

`sklearndf` augments more than 160 `scikit-learn` estimators for
native support of data frames, while leaving the original API intact.
"""

from packaging.version import Version as __Version
from sklearn import __version__ as __sklearn_version_str

from ._sklearndf import *

__version__ = "2.0.dev4"

__sklearn_version__ = __Version(__sklearn_version_str)
__sklearn_0_22__ = __Version("0.22")
__sklearn_0_23__ = __Version("0.23")
__sklearn_0_24__ = __Version("0.24")
__sklearn_1_0__ = __Version("1.0")

del __Version, __sklearn_version_str
