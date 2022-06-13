"""
Core implementation of :mod:`sklearndf.classification.extra`
"""
import logging
import warnings

from pytools.api import AllTracker

from ...wrapper import RegressorWrapperDF

# since we install LGBM via conda, the warning about the Clang compiler is irrelevant
warnings.filterwarnings("ignore", message=r"Starting from version 2\.2\.1")
# cross-validation will invariably generate sliced subsets, so the following warning
# is not helpful
warnings.filterwarnings(
    "ignore", message=r"Usage of np\.ndarray subset \(sliced data\) is not recommended"
)

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

__tracker = AllTracker(globals(), allow_imported_definitions=True)


#
# Class definitions
#

if LGBMClassifier:

    class LGBMClassifierDF(RegressorWrapperDF, LGBMClassifier, native=LGBMClassifier):
        """Stub for DF wrapper of class ``LGBMClassifierDF``"""

else:
    __all__.remove("LGBMClassifierDF")

if XGBClassifier:

    class XGBClassifierDF(RegressorWrapperDF, XGBClassifier, native=XGBClassifier):
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
