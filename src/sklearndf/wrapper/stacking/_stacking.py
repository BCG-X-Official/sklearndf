"""
DF wrapper classes for stacking estimators.
"""
from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
)

from pytools.api import AllTracker, inheritdoc, subsdoc

from ... import ClassifierDF, LearnerDF, RegressorDF, SupervisedLearnerDF
from .. import ClassifierWrapperDF, RegressorWrapperDF, SupervisedLearnerWrapperDF
from ..numpy import ClassifierNPDF, RegressorNPDF, SupervisedLearnerNPDF

log = logging.getLogger(__name__)

__all__ = [
    "StackingEstimatorWrapperDF",
    "StackingClassifierWrapperDF",
    "StackingRegressorWrapperDF",
]


#
# Type variables
#

T_DelegateClassifierDF = TypeVar("T_DelegateClassifierDF", bound=ClassifierDF)
T_DelegateRegressorDF = TypeVar("T_DelegateRegressorDF", bound=RegressorDF)

T_NativeSupervisedLearner = TypeVar(
    "T_NativeSupervisedLearner", bound=Union[RegressorMixin, ClassifierMixin]
)
T_NativeRegressor = TypeVar("T_NativeRegressor", bound=RegressorMixin)
T_NativeClassifier = TypeVar("T_NativeClassifier", bound=ClassifierMixin)

T_SupervisedLearnerDF = TypeVar("T_SupervisedLearnerDF", bound="SupervisedLearnerDF")
T_StackableSupervisedLearnerDF = TypeVar(
    "T_StackableSupervisedLearnerDF",
    bound="_StackableSupervisedLearnerDF[SupervisedLearnerDF]",
)
T_StackingEstimatorWrapperDF = TypeVar(
    "T_StackingEstimatorWrapperDF",
    bound="StackingEstimatorWrapperDF[Union[RegressorMixin, ClassifierMixin]]",
)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Stacking Estimator wrappers
#

# noinspection PyPep8Naming
@inheritdoc(match="""[see superclass]""")
class StackingEstimatorWrapperDF(
    # note: MetaEstimatorMixin is the first public class in the mro of _BaseStacking
    # MetaEstimatorMixin <-- _BaseHeterogeneousEnsemble <-- _BaseStacking
    MetaEstimatorMixin,  # type: ignore
    SupervisedLearnerWrapperDF[T_NativeSupervisedLearner],
    Generic[T_NativeSupervisedLearner],
    metaclass=ABCMeta,
):
    """
    Abstract base class of wrappers for estimators implementing
    :class:`sklearn.ensemble._stacking._BaseStacking`.

    The stacking estimator will delegate to embedded estimators; this wrapper ensures
    the required conversions from and to numpy arrays as the native stacking estimator
    invokes the embedded estimators.
    """

    def fit(
        self: T_StackingEstimatorWrapperDF,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params: Any,
    ) -> T_StackingEstimatorWrapperDF:
        """[see superclass]"""

        class _ColumnNameFn:
            # noinspection PyMethodParameters
            def __call__(self_) -> Sequence[str]:
                return self._get_final_estimator_features_in()

            def __deepcopy__(self, memo: Any = None) -> Any:
                # prevent a deep copy of this callable, to preserve reference to
                # stacking estimator being fitted
                return self

        native: T_NativeSupervisedLearner = self.native_estimator
        estimators: Sequence[Tuple[str, BaseEstimator]] = native.estimators
        final_estimator: BaseEstimator = native.final_estimator

        try:
            native.estimators = [
                (
                    name,
                    self._make_stackable_learner_df(estimator)
                    if isinstance(estimator, SupervisedLearnerDF)
                    else estimator,
                )
                for name, estimator in native.estimators
            ]
            native.final_estimator = self._make_learner_np_df(
                delegate=native.final_estimator or self._make_default_final_estimator(),
                column_names=_ColumnNameFn(),
            )

            # suppress a false warning from PyCharm's type checker
            # noinspection PyTypeChecker
            return super().fit(X, y, **fit_params)

        finally:
            native.estimators = estimators
            native.final_estimator = final_estimator

    @abstractmethod
    def _make_stackable_learner_df(
        self, learner: T_SupervisedLearnerDF
    ) -> _StackableSupervisedLearnerDF[T_SupervisedLearnerDF]:
        pass

    @abstractmethod
    def _make_learner_np_df(
        self, delegate: T_SupervisedLearnerDF, column_names: Callable[[], Sequence[str]]
    ) -> SupervisedLearnerNPDF[T_SupervisedLearnerDF]:
        pass

    def _get_estimators_features_out(self) -> List[str]:
        return [name for name, estimator in self.estimators if estimator != "drop"]

    def _get_final_estimator_features_in(self) -> List[str]:
        names = self._get_estimators_features_out()
        if self.passthrough:
            return [*names, *self.estimators_[0].feature_names_in_]
        else:
            return names


class StackingClassifierWrapperDF(
    ClassifierWrapperDF[T_NativeClassifier],
    StackingEstimatorWrapperDF[T_NativeClassifier],
    Generic[T_NativeClassifier],
    metaclass=ABCMeta,
):
    """
    DF wrapper class for :class:`sklearn.classifier.StackingClassifierDF`.
    """

    @staticmethod
    def _make_default_final_estimator() -> LearnerDF:
        from sklearndf.classification import LogisticRegressionDF

        return LogisticRegressionDF()

    def _get_estimators_features_out(self) -> List[str]:
        classes = self.native_estimator.classes_
        names = super()._get_estimators_features_out()
        if len(classes) > 2:
            return [f"{name}_{c}" for name in names for c in classes]
        else:
            return names

    def _make_stackable_learner_df(
        self, learner: ClassifierDF
    ) -> _StackableClassifierDF:
        return _StackableClassifierDF(learner)

    def _make_learner_np_df(
        self,
        delegate: T_DelegateClassifierDF,
        column_names: Callable[[], Sequence[str]],
    ) -> ClassifierNPDF[T_DelegateClassifierDF]:
        return ClassifierNPDF(delegate, column_names)


class StackingRegressorWrapperDF(
    StackingEstimatorWrapperDF[T_NativeRegressor],
    RegressorWrapperDF[T_NativeRegressor],
    Generic[T_NativeRegressor],
    metaclass=ABCMeta,
):
    """
    DF wrapper class for :class:`sklearn.regression.StackingRegressorDF`.
    """

    @staticmethod
    def _make_default_final_estimator() -> SupervisedLearnerDF:
        from sklearndf.regression import RidgeCVDF

        return RidgeCVDF()

    def _make_stackable_learner_df(self, learner: RegressorDF) -> _StackableRegressorDF:
        return _StackableRegressorDF(learner)

    def _make_learner_np_df(
        self, delegate: T_DelegateRegressorDF, column_names: Callable[[], Sequence[str]]
    ) -> RegressorNPDF[T_DelegateRegressorDF]:
        return RegressorNPDF(delegate, column_names)


#
# Supporting classes
#


class _StackableSupervisedLearnerDF(
    BaseEstimator,  # type: ignore
    Generic[T_SupervisedLearnerDF],
):
    """
    Returns numpy arrays from all prediction functions, instead of pandas series or
    data frames.

    For use in stacking estimators that forward the predictions of multiple learners to
    one final learner.
    """

    def __init__(self, delegate: T_SupervisedLearnerDF) -> None:
        super().__init__()
        self.delegate = delegate

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.delegate.is_fitted

    # noinspection PyPep8Naming
    @subsdoc(pattern="", replacement="", using=SupervisedLearnerDF.fit)
    def fit(
        self: T_StackableSupervisedLearnerDF,
        X: pd.DataFrame,
        y: Optional[npt.NDArray[Any]] = None,
        **fit_params: Any,
    ) -> T_StackableSupervisedLearnerDF:
        """[see SupervisedLearnerDF.fit]"""
        self.delegate.fit(X, self._convert_y_to_series(X, y), **fit_params)
        return self

    # noinspection PyPep8Naming
    @subsdoc(
        pattern="predictions per observation as a series, or as a data frame",
        replacement="predictions as a numpy array",
        using=SupervisedLearnerDF.predict,
    )
    def predict(self, X: pd.DataFrame, **predict_params: Any) -> npt.NDArray[Any]:
        """[see SupervisedLearnerDF.predict]"""
        return cast(
            npt.NDArray[Any],
            self.delegate.predict(X, **predict_params).values,
        )

    # noinspection PyPep8Naming
    @subsdoc(pattern="", replacement="", using=SupervisedLearnerDF.score)
    def score(
        self,
        X: pd.DataFrame,
        y: npt.NDArray["np.floating[Any]"],
        sample_weight: Optional[pd.Series] = None,
    ) -> float:
        """[see SupervisedLearnerDF.score]"""
        return self.delegate.score(X, self._convert_y_to_series(X, y), sample_weight)

    def _get_features_in(self) -> pd.Index:
        # noinspection PyProtectedMember
        return self.delegate._get_features_in()

    def _get_n_features_in(self) -> int:
        # noinspection PyProtectedMember
        return self.delegate._get_n_features_in()

    def _get_n_outputs(self) -> int:
        # noinspection PyProtectedMember
        return self.delegate._get_n_outputs()

    # noinspection PyPep8Naming
    @staticmethod
    def _convert_y_to_series(
        X: pd.DataFrame, y: Optional[npt.NDArray[Any]]
    ) -> Optional[pd.Series]:
        if y is None:
            return y
        if not isinstance(y, np.ndarray):
            raise TypeError(
                f"expected numpy array for arg y but got a {type(y).__name__}"
            )
        if y.ndim != 1:
            raise TypeError(
                f"expected 1-d numpy array for arg y but got a {y.ndim}-d array"
            )
        if len(y) != len(X):
            raise ValueError(
                "args X and y have different lengths: "
                f"len(X)={len(X)} and len(y)={len(y)}"
            )
        return pd.Series(y, index=X.index)

    @staticmethod
    def _convert_prediction_to_numpy(
        prediction: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> Union[npt.NDArray[Any], List[npt.NDArray[Any]]]:
        if isinstance(prediction, list):
            return [proba.values for proba in prediction]
        else:
            return cast(npt.NDArray[Any], prediction.values)


# noinspection PyPep8Naming
@inheritdoc(match="""[see superclass]""")
class _StackableClassifierDF(_StackableSupervisedLearnerDF[ClassifierDF], ClassifierDF):
    """[see superclass]"""

    @property
    def classes_(self) -> Union[npt.NDArray[Any], List[npt.NDArray[Any]]]:
        """[see superclass]"""
        return self.delegate.classes_

    def predict_proba(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[npt.NDArray[Any], List[npt.NDArray[Any]]]:
        """[see superclass]"""
        return self._convert_prediction_to_numpy(
            self.delegate.predict_proba(X, **predict_params)
        )

    def predict_log_proba(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[npt.NDArray[Any], List[npt.NDArray[Any]]]:
        """[see superclass]"""
        return self._convert_prediction_to_numpy(
            self.delegate.predict_log_proba(X, **predict_params)
        )

    def decision_function(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> npt.NDArray[np.floating[Any]]:
        """[see superclass]"""
        return cast(
            npt.NDArray[np.floating[Any]],
            self.delegate.decision_function(X, **predict_params).values,
        )


@inheritdoc(match="""[see superclass]""")
class _StackableRegressorDF(_StackableSupervisedLearnerDF[RegressorDF], RegressorDF):
    """[see superclass]"""


#
# validate __all__
#

__tracker.validate()
