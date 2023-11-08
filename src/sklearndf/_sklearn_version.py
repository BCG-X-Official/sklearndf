"""
Special constants for version checks for scikit-learn.
"""

from packaging.version import Version
from sklearn import __version__ as sklearn_version

__all__ = [
    "__sklearn_version__",
    "__sklearn_1_1__",
    "__sklearn_1_2__",
    "__sklearn_1_3__",
    "__sklearn_1_4__",
]

__sklearn_version__ = Version(sklearn_version)
__sklearn_1_1__ = Version("1.1")
__sklearn_1_2__ = Version("1.2")
__sklearn_1_3__ = Version("1.3")
__sklearn_1_4__ = Version("1.4")
