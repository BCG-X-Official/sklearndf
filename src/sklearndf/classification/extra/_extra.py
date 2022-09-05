"""
Core implementation of :mod:`sklearndf.classification.extra`
"""
import logging
import warnings

from pytools.api import AllTracker

from ...wrapper import ClassifierWrapperDF, MissingEstimator

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

    class LGBMClassifier(  # type: ignore
        MissingEstimator,
    ):
        """Mock-up for missing estimator."""


try:
    # import xgboost classes only if installed
    from xgboost import XGBClassifier
except ImportError:

    class XGBClassifier(  # type: ignore
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


class LGBMClassifierDF(
    ClassifierWrapperDF[LGBMClassifier],
    native=LGBMClassifier,
):
    """Stub for DF wrapper of class ``LGBMClassifierDF``"""


class XGBClassifierDF(
    ClassifierWrapperDF[XGBClassifier],
    native=XGBClassifier,
):
    """Stub for DF wrapper of class ``XGBClassifierDF``"""


#
# validate that __all__
#

__tracker.validate()
