"""
Special constants for version checks for scikit-learn.
"""

from packaging.version import Version
from sklearn import __version__ as sklearn_version

__all__ = [
    "__sklearn_version__",
    "__sklearn_0_22__",
    "__sklearn_0_23__",
    "__sklearn_0_24__",
    "__sklearn_1_0__",
    "__sklearn_1_1__",
    "__sklearn_1_2__",
]

__sklearn_version__ = Version(sklearn_version)
__sklearn_0_22__ = Version("0.22")
__sklearn_0_23__ = Version("0.23")
__sklearn_0_24__ = Version("0.24")
__sklearn_1_0__ = Version("1.0")
__sklearn_1_1__ = Version("1.1")
__sklearn_1_2__ = Version("1.2")
