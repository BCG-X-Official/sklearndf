"""
Core implementation of :mod:`sklearndf.regression` loaded
from sklearn 0.23 onwards
"""
import logging

from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor

from pytools.api import AllTracker

from ..wrapper import RegressorWrapperDF

log = logging.getLogger(__name__)

__all__ = [
    "GammaRegressorDF",
    "PoissonRegressorDF",
    "TweedieRegressorDF",
]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

#
# type variables
#


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class PoissonRegressorDF(
    RegressorWrapperDF[PoissonRegressor],
    native=PoissonRegressor,
):
    """Stub for DF wrapper of class ``PoissonRegressor``"""


class GammaRegressorDF(
    RegressorWrapperDF[GammaRegressor],
    native=GammaRegressor,
):
    """Stub for DF wrapper of class ``GammaRegressor``"""


class TweedieRegressorDF(
    RegressorWrapperDF[TweedieRegressor],
    native=TweedieRegressor,
):
    """Stub for DF wrapper of class ``TweedieRegressor``"""


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
