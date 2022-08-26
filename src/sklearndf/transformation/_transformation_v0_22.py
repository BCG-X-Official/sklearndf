"""
Core implementation of :mod:`sklearndf.transformation` loaded
from sklearn 0.22 onwards
"""


import logging

from sklearn.impute import KNNImputer

from pytools.api import AllTracker

from .wrapper import ImputerWrapperDF

log = logging.getLogger(__name__)

__all__ = ["KNNImputerDF"]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# impute
#


class KNNImputerDF(ImputerWrapperDF[KNNImputer], native=KNNImputer):
    """Stub for DF wrapper of class ``KNNImputer``"""


#
# validate __all__
#

__tracker.validate()


#
# validate that __all__ comprises all symbols ending in "DF", and no others
#

__estimators = [
    sym
    for sym in dir()
    if sym.endswith("DF")
    and sym not in __imported_estimators
    and not sym.startswith("_")
]
if set(__estimators) != set(__all__):
    raise RuntimeError(
        "__all__ does not contain exactly all DF estimators; expected value is:\n"
        f"{__estimators}"
    )
