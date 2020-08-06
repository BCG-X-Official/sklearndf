"""
Additional implementation of :mod:`sklearndf.classification` loaded
from sklearn 0.23 onwards
"""

import logging

log = logging.getLogger(__name__)

__all__ = []

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

# todo: add here classification implementations for sklearn 0.23

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
