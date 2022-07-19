"""
Core implementation of :mod:`sklearndf.regression.extra`
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

__all__ = ["LGBMRegressorDF", "XGBRegressorDF"]

try:
    # import lightgbm classes only if installed
    from lightgbm.sklearn import LGBMRegressor

except ImportError:
    LGBMRegressor = None

try:
    # import xgboost classes only if installed
    from xgboost import XGBRegressor

except ImportError:
    XGBRegressor = None

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals(), allow_imported_definitions=True)


#
# Class definitions
#

if LGBMRegressor:

    class LGBMRegressorDF(
        RegressorWrapperDF[LGBMRegressor],
        LGBMRegressor,  # type: ignore
        native=LGBMRegressor,
    ):
        """Stub for DF wrapper of class ``LGBMRegressorDF``"""

else:
    __all__.remove("LGBMRegressorDF")

if XGBRegressor:

    class XGBRegressorDF(
        RegressorWrapperDF[XGBRegressor],
        XGBRegressor,  # type: ignore
        native=XGBRegressor,
    ):
        """Stub for DF wrapper of class ``XGBRegressorDF``"""

else:
    __all__.remove("XGBRegressorDF")

#
# validate __all__
#

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
