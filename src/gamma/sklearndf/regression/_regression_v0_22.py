"""
Core implementation of :mod:`gamma.sklearndf.regression` loaded
from sklearn 0.22 onwards
"""
import logging
from abc import ABCMeta
from typing import *

from sklearn.base import RegressorMixin
from sklearn.ensemble import StackingRegressor

from gamma.sklearndf import RegressorDF
from gamma.sklearndf._wrapper import (
    RegressorWrapperDF,
    StackingRegressorWrapperDF,
    df_estimator,
)

# noinspection PyProtectedMember
from gamma.sklearndf.transformation._wrapper import (
    _ColumnPreservingTransformerWrapperDF,
)

log = logging.getLogger(__name__)

__all__ = ["StackingRegressorDF"]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

#
# type variables
#

T_Regressor = TypeVar("T_Regressor", bound=RegressorMixin)

#
# wrapper for hybrid regressor/transformer classes
#


class _RegressorTransformerWrapperDF(
    RegressorWrapperDF[T_Regressor],
    _ColumnPreservingTransformerWrapperDF[T_Regressor],
    Generic[T_Regressor],
    metaclass=ABCMeta,
):
    """
    Wraps a combined regressor and constant column transformer
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=StackingRegressorWrapperDF)
class StackingRegressorDF(RegressorDF, StackingRegressor):
    """
    Wraps :class:`sklearn.ensemble._stacking.StackingRegressor`; accepts and
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
