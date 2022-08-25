"""
Core implementation of :mod:`sklearndf.classification.extra`
"""
import logging

from pytools.api import AllTracker

from ...wrapper import MissingEstimator, RegressorWrapperDF

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
# validate that __all__
#

__tracker.validate()
