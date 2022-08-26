"""
Additional implementation of :mod:`sklearndf.regression` loaded
from sklearn 1.0 onwards
"""
import logging

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor

from pytools.api import AllTracker

from ..wrapper import RegressorWrapperDF

log = logging.getLogger(__name__)

__all__ = ["HistGradientBoostingRegressorDF", "QuantileRegressorDF"]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# ensemble
#


class HistGradientBoostingRegressorDF(
    RegressorWrapperDF[HistGradientBoostingRegressor],
    native=HistGradientBoostingRegressor,
):
    """Stub for DF wrapper of class ``HistGradientBoostingRegressor``"""


#
# linear model
#


class QuantileRegressorDF(
    RegressorWrapperDF[QuantileRegressor],
    native=QuantileRegressor,
):
    """Stub for DF wrapper of class ``QuantileRegressor``"""


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
