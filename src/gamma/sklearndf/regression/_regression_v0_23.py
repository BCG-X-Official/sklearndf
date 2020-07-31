"""
Core implementation of :mod:`gamma.sklearndf.regression` loaded
from sklearn 0.23 onwards
"""

import logging
from abc import ABCMeta
from typing import *

from sklearn.base import RegressorMixin
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.linear_model._glm import GeneralizedLinearRegressor

from gamma.sklearndf import RegressorDF
from gamma.sklearndf._wrapper import RegressorWrapperDF, df_estimator

# noinspection PyProtectedMember
from gamma.sklearndf.transformation._wrapper import (
    _ColumnPreservingTransformerWrapperDF,
)

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
# GLM regressors added with v0.23
#
# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=RegressorWrapperDF)
class PoissonRegressorDF(RegressorDF, PoissonRegressor):
    """
    Wraps :class:`sklearn.linear_model._glm.glm.PoissonRegressor`; accepts and
     returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=RegressorWrapperDF)
class GammaRegressorDF(RegressorDF, GammaRegressor):
    """
    Wraps :class:`sklearn.linear_model._glm.glm.GammaRegressor`; accepts and
     returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=RegressorWrapperDF)
class TweedieRegressorDF(RegressorDF, TweedieRegressor):
    """
    Wraps :class:`sklearn.linear_model._glm.glm.TweedieRegressor`; accepts and
     returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=RegressorWrapperDF)
class GeneralizedLinearRegressorDF(RegressorDF, GeneralizedLinearRegressor):
    """
    Wraps :class:`sklearn.linear_model._glm.glm.GeneralizedLinearRegressor`; accepts and
     returns data frames.
    """

    pass


#
# validate that __all__ comprises all symbols ending in "DF", and no others
#

__estimators = [
    sym
    for sym in dir()
    if sym.endswith("DF")
    and sym not in __imported_estimators
    and not sym.startswith("_")
]
if set(__estimators) != set(__all__):
    raise RuntimeError(
        "__all__ does not contain exactly all DF estimators; expected value is:\n"
        f"{__estimators}"
    )
