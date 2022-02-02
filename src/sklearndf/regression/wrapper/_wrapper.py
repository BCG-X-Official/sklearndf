"""
Core implementation of :mod:`sklearndf.regression.wrapper`
"""

import logging
from abc import ABCMeta
from typing import Callable, Generic, Optional, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.isotonic import IsotonicRegression

from pytools.api import AllTracker

from sklearndf import RegressorDF, SupervisedLearnerDF
from sklearndf.transformation.wrapper import (
    ColumnPreservingTransformerWrapperDF,
    NumpyTransformerWrapperDF,
)
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
    "PartialFitRegressorWrapperDF",
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


# noinspection PyProtectedMember
from ...wrapper._adapter import RegressorNPDF as _RegressorNPDF

# noinspection PyProtectedMember
from ...wrapper._wrapper import _StackableRegressorDF


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

    @staticmethod
    def _make_default_final_estimator() -> SupervisedLearnerDF:
        from sklearndf.regression import RidgeCVDF

        return RidgeCVDF()

    def _make_stackable_learner_df(
        self, learner: SupervisedLearnerDF
    ) -> _StackableRegressorDF:
        return _StackableRegressorDF(learner)

    def _make_learner_np_df(
        self, delegate: RegressorDF, column_names: Callable[[], Sequence[str]]
    ) -> _RegressorNPDF:
        return _RegressorNPDF(delegate, column_names)


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
    RegressorTransformerWrapperDF[IsotonicRegression],
    NumpyTransformerWrapperDF,
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
    def _adjust_X_type_for_delegate(
        self, X: pd.DataFrame, *, to_numpy: Optional[bool] = None
    ) -> np.ndarray:
        return super()._adjust_X_type_for_delegate(X).ravel()


class PartialFitRegressorWrapperDF(
    RegressorWrapperDF,
    Generic[T_NativeRegressor],
    metaclass=ABCMeta,
):
    """
    Abstract base class of DF wrappers for regressors implementing
    :func:`sklearn.linear_model.BaseSGDRegressor.partial_fit`.
    """

    def partial_fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        sample_weight: Optional[pd.Series] = None,
    ):
        """
        Perform incremental fit on a batch of samples.
        This method is meant to be called multiple times for subsets of training
        data, which e.g. couldn't fit in the required memory in full. It can be
        also used for online learning.

        :param X: data frame with observations as rows and features as columns
        :param y: a series or data frame with one or more outputs per observation
        :param sample_weight: optional weights applied to individual samples
        :return: ``self``
        """
        self._check_parameter_types(X, y)
        self._partial_fit(X, y, sample_weight)

        return self

    def _partial_fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        sample_weight: Optional[pd.Series],
    ):
        return self._native_estimator.partial_fit(
            self._prepare_X_for_delegate(X),
            self._prepare_y_for_delegate(y),
            sample_weight,
        )


#
# Validate __all__
#

__tracker.validate()
