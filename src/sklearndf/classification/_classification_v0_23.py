"""
Additional implementation of :mod:`sklearndf.classification` loaded
from sklearn 0.23 onwards
"""

import logging
from typing import List

from pytools.api import AllTracker

log = logging.getLogger(__name__)

__all__: List[str] = []

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


# todo: add classification implementations for sklearn 0.23


__tracker.validate()

#
# validate that __all__ comprises all symbols ending in "DF", and no others
#

__estimators = {
    sym
    for sym in dir()
    if sym.endswith("DF")
    and sym not in __imported_estimators
    and not sym.startswith("_")
}
if __estimators != set(__all__):
    raise RuntimeError(
        "__all__ does not contain exactly all DF estimators; expected value is:\n"
        f"{__estimators}"
    )
