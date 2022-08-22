"""
Core implementation of :mod:`sklearndf.regression.extra`
"""
import logging
import warnings

from pytools.api import AllTracker

from ...wrapper import MissingEstimator, RegressorWrapperDF

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

    class LGBMRegressor(  # type: ignore
        MissingEstimator,
    ):
        """Mock-up for missing estimator."""


try:
    # import xgboost classes only if installed
    from xgboost import XGBRegressor

except ImportError:

    class XGBRegressor(  # type: ignore
        MissingEstimator,
    ):
        """Mock-up for missing estimator."""


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class LGBMRegressorDF(
    RegressorWrapperDF[LGBMRegressor],
    native=LGBMRegressor,
):
    """Stub for DF wrapper of class ``LGBMRegressorDF``"""


class XGBRegressorDF(
    RegressorWrapperDF[XGBRegressor],
    native=XGBRegressor,
):
    """Stub for DF wrapper of class ``XGBRegressorDF``"""


#
# validate __all__
#

__tracker.validate()
