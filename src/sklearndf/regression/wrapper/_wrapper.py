"""
Core implementation of :mod:`sklearndf.regression.wrapper`
"""

import logging
from abc import ABCMeta
from typing import Any, Generic, Optional, TypeVar, Union, cast

import numpy.typing as npt
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.multioutput import MultiOutputRegressor

from pytools.api import AllTracker

from ...transformation.wrapper import (
    ColumnPreservingTransformerWrapperDF,
    NumpyTransformerWrapperDF,
)
from ...wrapper import MetaEstimatorWrapperDF, RegressorWrapperDF

log = logging.getLogger(__name__)

__all__ = [
    "IsotonicRegressionWrapperDF",
    "MetaRegressorWrapperDF",
    "RegressorTransformerWrapperDF",
    "PartialFitRegressorWrapperDF",
    "MultiOutputRegressorWrapperDF",
]


#
# type variables
#

T_PartialFitRegressorWrapperDF = TypeVar(
    "T_PartialFitRegressorWrapperDF",
    bound="PartialFitRegressorWrapperDF[RegressorMixin]",
)
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
    RegressorWrapperDF[T_NativeRegressor],
    Generic[T_NativeRegressor],
    metaclass=ABCMeta,
):
    """
    Abstract base class of DF wrappers for regressors implementing
    :class:`sklearn.base.MetaEstimatorMixin`.
    """

    pass


class PartialFitRegressorWrapperDF(
    RegressorWrapperDF[T_NativeRegressor],
    Generic[T_NativeRegressor],
    metaclass=ABCMeta,
):
    """
    Abstract base class of DF wrappers for regressors implementing
    method ``partial_fit()``.
    """

    # noinspection PyPep8Naming
    def partial_fit(
        self: T_PartialFitRegressorWrapperDF,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        sample_weight: Optional[pd.Series] = None,
    ) -> T_PartialFitRegressorWrapperDF:
        """
        Perform incremental fit on a batch of samples.

        This method is meant to be called multiple times for subsets of training
        data which, e.g., couldn't fit in the required memory in full. It can be
        also used for online learning.

        :param X: data frame with observations as rows and features as columns
        :param y: a series or data frame with one or more outputs per observation
        :param sample_weight: optional weights applied to individual samples
        :return: ``self``
        """
        self._check_parameter_types(X, y)
        self._partial_fit(X, y, sample_weight=sample_weight)

        return self

    # noinspection PyPep8Naming
    def _partial_fit(
        self: T_PartialFitRegressorWrapperDF,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        **partial_fit_params: Optional[Any],
    ) -> T_PartialFitRegressorWrapperDF:
        return cast(
            T_PartialFitRegressorWrapperDF,
            self._native_estimator.partial_fit(
                self._prepare_X_for_delegate(X),
                self._prepare_y_for_delegate(y),
                **{
                    arg: value
                    for arg, value in partial_fit_params.items()
                    if value is not None
                },
            ),
        )


class MultiOutputRegressorWrapperDF(
    MetaRegressorWrapperDF[MultiOutputRegressor],
    PartialFitRegressorWrapperDF[MultiOutputRegressor],
):
    """
    Abstract base class of DF wrappers for multi-output regressors.
    """

    pass


class RegressorTransformerWrapperDF(
    RegressorWrapperDF[T_NativeRegressor],
    ColumnPreservingTransformerWrapperDF[T_NativeRegressor],
    Generic[T_NativeRegressor],
    metaclass=ABCMeta,
):
    """
    DF wrapper for combined regressors and column preserving transformers.
    """

    pass


class IsotonicRegressionWrapperDF(
    RegressorTransformerWrapperDF[IsotonicRegression],
    NumpyTransformerWrapperDF[IsotonicRegression],
    metaclass=ABCMeta,
):
    """
    DF wrapper for :class:`sklearn.isotonic.IsotonicRegression`.
    """

    # noinspection PyPep8Naming
    def _check_parameter_types(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        *,
        expected_columns: pd.Index = None,
    ) -> None:
        super()._check_parameter_types(X, y, expected_columns=expected_columns)
        if X.shape[1] != 1:
            raise ValueError(
                f"arg X expected to have exactly 1 column but has {X.shape[1]} columns"
            )

    # noinspection PyPep8Naming
    def _adjust_X_type_for_delegate(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        return super()._adjust_X_type_for_delegate(X).ravel()


#
# Validate __all__
#

__tracker.validate()
