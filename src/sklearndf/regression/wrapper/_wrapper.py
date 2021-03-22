"""
Core implementation of :mod:`sklearndf.regression.wrapper`
"""

import logging
from abc import ABCMeta
from typing import Any, Generic, Optional, TypeVar, Union

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.isotonic import IsotonicRegression

from pytools.api import AllTracker

from sklearndf.transformation.wrapper import ColumnPreservingTransformerWrapperDF
from sklearndf.wrapper import (
    MetaEstimatorWrapperDF,
    RegressorWrapperDF,
    StackingEstimatorWrapperDF,
)

log = logging.getLogger(__name__)

__all__ = [
    "IsotonicRegressionWrapperDF",
    "MetaRegressorWrapperDF",
    "RegressorTransformerWrapperDF",
    "StackingRegressorWrapperDF",
]


#
# type variables
#

T_Regressor = TypeVar("T_Regressor", bound=RegressorMixin)
T_NativeRegressor = TypeVar("T_NativeRegressor", bound=RegressorMixin)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class MetaRegressorWrapperDF(
    MetaEstimatorWrapperDF[T_NativeRegressor],
    RegressorWrapperDF,
    Generic[T_NativeRegressor],
    metaclass=ABCMeta,
):
    """
    Abstract base class of DF wrappers for regressors implementing
    :class:`sklearn.base.MetaEstimatorMixin`.
    """

    pass


class StackingRegressorWrapperDF(
    StackingEstimatorWrapperDF[T_NativeRegressor],
    RegressorWrapperDF,
    Generic[T_NativeRegressor],
    metaclass=ABCMeta,
):
    """
    Abstract base class of DF wrappers for regressors implementing
    :class:`sklearn.ensemble._stacking._BaseStacking`.
    """

    pass


class RegressorTransformerWrapperDF(
    RegressorWrapperDF[T_Regressor],
    ColumnPreservingTransformerWrapperDF[T_Regressor],
    Generic[T_Regressor],
    metaclass=ABCMeta,
):
    """
    DF wrapper for combined regressors and column preserving transformers.
    """

    pass


class IsotonicRegressionWrapperDF(
    RegressorTransformerWrapperDF[IsotonicRegression], metaclass=ABCMeta
):
    """
    DF wrapper for :class:`sklearn.isotonic.IsotonicRegression`.
    """

    # noinspection PyPep8Naming
    def _check_parameter_types(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        super()._check_parameter_types(X=X, y=y)
        if X.shape[1] != 1:
            raise ValueError(
                f"arg X expected to have exactly 1 column but has {X.shape[1]} columns"
            )

    # noinspection PyPep8Naming
    def _convert_X_for_delegate(self, X: pd.DataFrame) -> Any:
        return super()._convert_X_for_delegate(X).iloc[:, 0].values

    def _convert_y_for_delegate(
        self, y: Optional[Union[pd.Series, pd.DataFrame]]
    ) -> Any:
        y = super()._convert_y_for_delegate(y)
        return None if y is None else y.values


#
# Validate __all__
#

__tracker.validate()
