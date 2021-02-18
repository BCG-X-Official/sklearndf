"""
Core implementation of :mod:`sklearndf.regression` loaded
from sklearn 0.22 onwards
"""
import logging
from typing import TypeVar

from sklearn.base import RegressorMixin
from sklearn.ensemble import StackingRegressor

from pytools.api import AllTracker

from .. import RegressorDF
from .._wrapper import _StackingRegressorWrapperDF, df_estimator

# noinspection PyProtectedMember

log = logging.getLogger(__name__)

__all__ = ["StackingRegressorDF"]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

#
# type variables
#

T_Regressor = TypeVar("T_Regressor", bound=RegressorMixin)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_StackingRegressorWrapperDF)
class StackingRegressorDF(RegressorDF, StackingRegressor):
    """
    Wraps :class:`sklearn.ensemble._stacking.StackingRegressor`; accepts and
     returns data frames.
    """

    pass


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
