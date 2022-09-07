"""
Core implementation of :mod:`sklearndf.classification.extra`
"""
import logging

from sklearn.base import ClassifierMixin

from pytools.api import AllTracker

from ...wrapper import ClassifierWrapperDF, MissingEstimator

log = logging.getLogger(__name__)

__all__ = ["LGBMClassifierDF", "XGBClassifierDF"]

try:
    # import lightgbm classes only if installed
    from lightgbm.sklearn import LGBMClassifier
except ImportError:

    class LGBMClassifier(  # type: ignore
        MissingEstimator,
        ClassifierMixin,  # type: ignore
    ):
        """Mock-up for missing estimator."""


try:
    # import xgboost classes only if installed
    from xgboost import XGBClassifier
except ImportError:

    class XGBClassifier(  # type: ignore
        MissingEstimator,
        ClassifierMixin,  # type: ignore
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
