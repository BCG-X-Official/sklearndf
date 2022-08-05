"""
Core implementation of :mod:`sklearndf.classification.extra`
"""
import logging

from pytools.api import AllTracker

from ...wrapper import RegressorWrapperDF

log = logging.getLogger(__name__)

__all__ = ["LGBMClassifierDF", "XGBClassifierDF"]

try:
    # import lightgbm classes only if installed
    from lightgbm.sklearn import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    # import xgboost classes only if installed
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#

if LGBMClassifier:

    class LGBMClassifierDF(
        RegressorWrapperDF[LGBMClassifier],
        native=LGBMClassifier,
    ):
        """Stub for DF wrapper of class ``LGBMClassifierDF``"""

else:
    __all__.remove("LGBMClassifierDF")

if XGBClassifier:

    class XGBClassifierDF(
        RegressorWrapperDF[XGBClassifier],
        native=XGBClassifier,
    ):
        """Stub for DF wrapper of class ``XGBClassifierDF``"""

else:
    __all__.remove("XGBClassifierDF")

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


__tracker.validate()
