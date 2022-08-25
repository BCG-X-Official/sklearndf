"""
Core implementation of :mod:`sklearndf.classification.wrapper`
"""

import logging
from abc import ABCMeta
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union, cast

import numpy.typing as npt
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

from pytools.api import AllTracker

from ...transformation.wrapper import NComponentsDimensionalityReductionWrapperDF
from ...wrapper import ClassifierWrapperDF, MetaEstimatorWrapperDF

log = logging.getLogger(__name__)

__all__ = [
    "ClassifierChainWrapperDF",
    "LinearDiscriminantAnalysisWrapperDF",
    "MetaClassifierWrapperDF",
    "MultiOutputClassifierWrapperDF",
    "PartialFitClassifierWrapperDF",
]

#
# Type variables
#

T_PartialFitClassifierWrapperDF = TypeVar(
    "T_PartialFitClassifierWrapperDF",
    bound="PartialFitClassifierWrapperDF[ClassifierMixin]",
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
    ClassifierWrapperDF[LinearDiscriminantAnalysis],
    NComponentsDimensionalityReductionWrapperDF[LinearDiscriminantAnalysis],
    metaclass=ABCMeta,
):
    """
    DF wrapper for
    :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.
    """

    pass


class MetaClassifierWrapperDF(
    ClassifierWrapperDF[T_NativeClassifier],
    MetaEstimatorWrapperDF[T_NativeClassifier],
    Generic[T_NativeClassifier],
    metaclass=ABCMeta,
):
    """
    Abstract base class of DF wrappers for classifiers implementing
    :class:`sklearn.base.MetaEstimatorMixin`.
    """

    pass


class PartialFitClassifierWrapperDF(
    ClassifierWrapperDF[T_NativeClassifier],
    Generic[T_NativeClassifier],
    metaclass=ABCMeta,
):
    """
    Abstract base class of DF wrappers for classifiers implementing
    method ``partial_fit()``.
    """

    # noinspection PyPep8Naming
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

    # noinspection PyPep8Naming
    def _partial_fit(
        self: T_PartialFitClassifierWrapperDF,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        **partial_fit_params: Optional[Any],
    ) -> T_PartialFitClassifierWrapperDF:
        return cast(
            T_PartialFitClassifierWrapperDF,
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
        prediction: Union[
            pd.Series, pd.DataFrame, List[npt.NDArray[Any]], npt.NDArray[Any]
        ],
        classes: Optional[Sequence[Any]] = None,
    ) -> Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]:

        # if we have a multi-output classifier, prediction of probabilities
        # yields a list of NumPy arrays
        if not isinstance(prediction, list):
            raise ValueError(
                "prediction of multi-output classifier expected to be a list of NumPy "
                f"arrays, but got type {type(prediction)}"
            )

        delegate_estimator = self.native_estimator

        # store the super() object as this is not available within a generator
        _super = cast(ClassifierWrapperDF[MultiOutputClassifier], super())

        # estimators attribute of abstract class MultiOutputEstimator
        # usually the delegate estimator will provide a list of estimators used
        # to predict each output. If present, use these estimators to get
        # individual class labels for each output; otherwise we cannot assign class
        # labels
        estimators = getattr(delegate_estimator, "estimators_", None)
        if estimators is None:
            return [
                _super._prediction_with_class_labels(X=X, prediction=output)
                for output in prediction
            ]
        else:
            return [
                _super._prediction_with_class_labels(
                    X=X, prediction=output, classes=getattr(estimator, "classes_", None)
                )
                for estimator, output in zip(estimators, prediction)
            ]


class ClassifierChainWrapperDF(
    MetaEstimatorWrapperDF[ClassifierChain],
    ClassifierWrapperDF[ClassifierChain],
    metaclass=ABCMeta,
):
    """
    DF wrapper for :class:`sklearn.multioutput.ClassifierChain`.
    """

    # noinspection PyPep8Naming
    def _prediction_with_class_labels(
        self,
        X: pd.DataFrame,
        prediction: Union[
            pd.Series, pd.DataFrame, List[npt.NDArray[Any]], npt.NDArray[Any]
        ],
        classes: Optional[Sequence[Any]] = None,
    ) -> Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]:
        # todo: infer actual class names
        return super()._prediction_with_class_labels(
            X, prediction, classes=range(self.n_outputs_)
        )


#
# Validate __all__
#

__tracker.validate()
