"""
Special constants for version checks for scikit-learn.
"""

from packaging.version import Version
from sklearn import __version__ as sklearn_version

__all__ = [
    "__sklearn_version__",
    "__sklearn_1_4__",
    "__sklearn_1_5__",
    "__sklearn_1_6__",
    "__sklearn_1_7__",
    "__sklearn_1_8__",
]

__sklearn_version__ = Version(sklearn_version)
__sklearn_1_4__ = Version("1.4")
__sklearn_1_5__ = Version("1.5")
__sklearn_1_6__ = Version("1.6")
__sklearn_1_7__ = Version("1.7")
__sklearn_1_8__ = Version("1.8")
