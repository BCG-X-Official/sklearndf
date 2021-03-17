"""
Core implementation of :mod:`sklearndf.regression` loaded
from sklearn 0.22 onwards
"""
import logging
from typing import TypeVar

from sklearn.base import RegressorMixin
from sklearn.ensemble import StackingRegressor

from pytools.api import AllTracker

from ..wrapper import make_df_regressor
from .wrapper import StackingRegressorWrapperDF

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

__tracker = AllTracker(globals(), allow_imported_definitions=True)


#
# Class definitions
#

StackingRegressorDF = make_df_regressor(
    StackingRegressor, base_wrapper=StackingRegressorWrapperDF
)


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
