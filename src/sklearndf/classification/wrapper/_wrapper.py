"""
Core implementation of :mod:`sklearndf.classification.wrapper`
"""

import logging
from abc import ABCMeta
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

from pytools.api import AllTracker

from sklearndf import ClassifierDF, LearnerDF
from sklearndf.transformation.wrapper import NComponentsDimensionalityReductionWrapperDF
from sklearndf.wrapper import (
    ClassifierWrapperDF,
    MetaEstimatorWrapperDF,
    StackingEstimatorWrapperDF,
)

log = logging.getLogger(__name__)

__all__ = [
    "ClassifierChainWrapperDF",
    "LinearDiscriminantAnalysisWrapperDF",
    "MetaClassifierWrapperDF",
    "MultiOutputClassifierWrapperDF",
    "StackingClassifierWrapperDF",
    "PartialFitClassifierWrapperDF",
]

#
# Type variables
#

T_PartialFitClassifierWrapperDF = TypeVar(
    "T_PartialFitClassifierWrapperDF", bound="PartialFitClassifierWrapperDF"
)
T_NativeClassifier = TypeVar("T_NativeClassifier", bound=ClassifierMixin)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Wrapper classes
#


class LinearDiscriminantAnalysisWrapperDF(
    NComponentsDimensionalityReductionWrapperDF[LinearDiscriminantAnalysis],
    ClassifierWrapperDF[LinearDiscriminantAnalysis],
    metaclass=ABCMeta,
):
    """
    DF wrapper for
    :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.
    """

    pass


class MetaClassifierWrapperDF(
    MetaEstimatorWrapperDF[T_NativeClassifier],
    ClassifierWrapperDF,
    Generic[T_NativeClassifier],
    metaclass=ABCMeta,
):
    """
    Abstract base class of DF wrappers for classifiers implementing
    :class:`sklearn.base.MetaEstimatorMixin`.
    """

    pass


class PartialFitClassifierWrapperDF(
    ClassifierWrapperDF,
    Generic[T_NativeClassifier],
    metaclass=ABCMeta,
):
    """
    Abstract base class of DF wrappers for classifiers implementing
    method ``partial_fit()``.
    """

    def partial_fit(
        self: T_PartialFitClassifierWrapperDF,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        classes: Optional[Sequence[Any]] = None,
        sample_weight: Optional[pd.Series] = None,
    ) -> T_PartialFitClassifierWrapperDF:
        """
        Perform incremental fit on a batch of samples.

        This method is meant to be called multiple times for subsets of training
        data which, e.g., couldn't fit in the required memory in full. It can be
        also used for online learning.

        :param X: data frame with observations as rows and features as columns
        :param y: a series or data frame with one or more outputs per observation
        :param classes: all classes present across all calls to ``partial_fit``;
            only required for the first call of this method
        :param sample_weight: optional weights applied to individual samples
        :return: ``self``
        """
        self._check_parameter_types(X, y)
        self._partial_fit(X, y, classes=classes, sample_weight=sample_weight)

        return self

    def _partial_fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        **partial_fit_params: Optional[Any],
    ):
        return self._native_estimator.partial_fit(
            self._prepare_X_for_delegate(X),
            self._prepare_y_for_delegate(y),
            **{
                arg: value
                for arg, value in partial_fit_params.items()
                if value is not None
            },
        )


class MultiOutputClassifierWrapperDF(
    MetaClassifierWrapperDF[MultiOutputClassifier],
    PartialFitClassifierWrapperDF[MultiOutputClassifier],
    metaclass=ABCMeta,
):
    """
    DF wrapper for :class:`sklearn.multioutput.MultiOutputClassifier`.
    """

    # noinspection PyPep8Naming
    def _prediction_with_class_labels(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame, list, np.ndarray],
        classes: Optional[Sequence[Any]] = None,
    ) -> Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]:

        # if we have a multi-output classifier, prediction of probabilities
        # yields a list of NumPy arrays
        if not isinstance(y, list):
            raise ValueError(
                "prediction of multi-output classifier expected to be a list of NumPy "
                f"arrays, but got type {type(y)}"
            )

        delegate_estimator = self.native_estimator

        # store the super() object as this is not available within a generator
        _super = cast(ClassifierWrapperDF, super())

        # estimators attribute of abstract class MultiOutputEstimator
        # usually the delegate estimator will provide a list of estimators used
        # to predict each output. If present, use these estimators to get
        # individual class labels for each output; otherwise we cannot assign class
        # labels
        estimators = getattr(delegate_estimator, "estimators_", None)
        if estimators is None:
            return [_super._prediction_with_class_labels(X=X, y=output) for output in y]
        else:
            return [
                _super._prediction_with_class_labels(
                    X=X, y=output, classes=getattr(estimator, "classes_", None)
                )
                for estimator, output in zip(estimators, y)
            ]


class ClassifierChainWrapperDF(
    MetaClassifierWrapperDF[ClassifierChain], metaclass=ABCMeta
):
    """
    DF wrapper for :class:`sklearn.multioutput.ClassifierChain`.
    """

    # noinspection PyPep8Naming
    def _prediction_with_class_labels(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame, list, np.ndarray],
        classes: Optional[Sequence[Any]] = None,
    ) -> Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]:
        # todo: infer actual class names
        return super()._prediction_with_class_labels(
            X=X, y=y, classes=range(self.n_outputs_)
        )


# noinspection PyProtectedMember
from ...wrapper._adapter import ClassifierNPDF as _ClassifierNPDF

# noinspection PyProtectedMember
from ...wrapper._wrapper import _StackableClassifierDF


class StackingClassifierWrapperDF(
    StackingEstimatorWrapperDF[T_NativeClassifier],
    ClassifierWrapperDF,
    Generic[T_NativeClassifier],
    metaclass=ABCMeta,
):
    """
    Abstract base class of DF wrappers for classifiers implementing
    :class:`sklearn.ensemble._stacking._BaseStacking`.
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
        self, delegate: ClassifierDF, column_names: Callable[[], Sequence[str]]
    ) -> _ClassifierNPDF:
        return _ClassifierNPDF(delegate, column_names)


#
# Validate __all__
#

__tracker.validate()
