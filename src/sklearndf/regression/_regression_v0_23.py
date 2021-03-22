"""
Core implementation of :mod:`sklearndf.regression` loaded
from sklearn 0.23 onwards
"""
import logging
from typing import TypeVar

from sklearn.base import RegressorMixin
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.linear_model._glm import GeneralizedLinearRegressor

from pytools.api import AllTracker

from ..wrapper import make_df_regressor

# noinspection PyProtectedMember

log = logging.getLogger(__name__)

__all__ = [
    "GammaRegressorDF",
    "GeneralizedLinearRegressorDF",
    "PoissonRegressorDF",
    "TweedieRegressorDF",
]

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

PoissonRegressorDF = make_df_regressor(PoissonRegressor)
GammaRegressorDF = make_df_regressor(GammaRegressor)
TweedieRegressorDF = make_df_regressor(TweedieRegressor)
GeneralizedLinearRegressorDF = make_df_regressor(GeneralizedLinearRegressor)


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
