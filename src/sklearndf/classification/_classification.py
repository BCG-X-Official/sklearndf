"""
Core implementation of :mod:`sklearndf.classification`
"""
import logging
from abc import ABCMeta
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
)
from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from pytools.api import AllTracker

from .. import ClassifierDF
from .._wrapper import _ClassifierWrapperDF, _MetaClassifierWrapperDF, df_estimator

log = logging.getLogger(__name__)

__all__ = [
    "AdaBoostClassifierDF",
    "BaggingClassifierDF",
    "BernoulliNBDF",
    "CalibratedClassifierCVDF",
    "ClassifierChainDF",
    "ComplementNBDF",
    "DecisionTreeClassifierDF",
    "DummyClassifierDF",
    "ExtraTreeClassifierDF",
    "ExtraTreesClassifierDF",
    "GaussianNBDF",
    "GaussianProcessClassifierDF",
    "GradientBoostingClassifierDF",
    "KNeighborsClassifierDF",
    "LabelPropagationDF",
    "LabelSpreadingDF",
    "LinearDiscriminantAnalysisDF",
    "LinearSVCDF",
    "LogisticRegressionCVDF",
    "LogisticRegressionDF",
    "MLPClassifierDF",
    "MultinomialNBDF",
    "MultiOutputClassifierDF",
    "NearestCentroidDF",
    "NuSVCDF",
    "OneVsOneClassifierDF",
    "OneVsRestClassifierDF",
    "OutputCodeClassifierDF",
    "PassiveAggressiveClassifierDF",
    "PerceptronDF",
    "QuadraticDiscriminantAnalysisDF",
    "RadiusNeighborsClassifierDF",
    "RandomForestClassifierDF",
    "RidgeClassifierCVDF",
    "RidgeClassifierDF",
    "SGDClassifierDF",
    "SVCDF",
    "VotingClassifierDF",
]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


#
# Dummy
#

# noinspection PyAbstractClass


@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class DummyClassifierDF(ClassifierDF, DummyClassifier):
    """
    Wraps :class:`sklearn.dummy.DummyClassifier`; accepts and
    returns data frames.
    """

    pass


#
# neighbors
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class NearestCentroidDF(ClassifierDF, NearestCentroid):
    """
    Wraps :class:`sklearn.neighbors.nearest_centroid.NearestCentroid`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class KNeighborsClassifierDF(ClassifierDF, KNeighborsClassifier):
    """
    Wraps :class:`sklearn.neighbors.classification.KNeighborsClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class RadiusNeighborsClassifierDF(ClassifierDF, RadiusNeighborsClassifier):
    """
    Wraps :class:`sklearn.neighbors.classification.RadiusNeighborsClassifier`; accepts
    and returns data frames.
    """

    pass


#
# voting
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_MetaClassifierWrapperDF)
class VotingClassifierDF(ClassifierDF, VotingClassifier):
    """
    Wraps :class:`sklearn.ensemble.voting.VotingClassifier`; accepts and returns data
    frames.
    """

    pass


#
# ensemble
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class RandomForestClassifierDF(ClassifierDF, RandomForestClassifier):
    """
    Wraps :class:`sklearn.ensemble.forest.RandomForestClassifier`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class ExtraTreesClassifierDF(ClassifierDF, ExtraTreesClassifier):
    """
    Wraps :class:`sklearn.ensemble.forest.ExtraTreesClassifier`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class GradientBoostingClassifierDF(ClassifierDF, GradientBoostingClassifier):
    """
    Wraps :class:`sklearn.ensemble.gradient_boosting.GradientBoostingClassifier`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class AdaBoostClassifierDF(ClassifierDF, AdaBoostClassifier):
    """
    Wraps :class:`sklearn.ensemble.weight_boosting.AdaBoostClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class BaggingClassifierDF(ClassifierDF, BaggingClassifier):
    """
    Wraps :class:`sklearn.ensemble.bagging.BaggingClassifier`; accepts and returns data
    frames.
    """

    pass


#
# tree
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class DecisionTreeClassifierDF(ClassifierDF, DecisionTreeClassifier):
    """
    Wraps :class:`sklearn.tree.tree.DecisionTreeClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class ExtraTreeClassifierDF(ClassifierDF, ExtraTreeClassifier):
    """
    Wraps :class:`sklearn.tree.tree.ExtraTreeClassifier`; accepts and returns data
    frames.
    """

    pass


#
# discriminant analysis
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class QuadraticDiscriminantAnalysisDF(ClassifierDF, QuadraticDiscriminantAnalysis):
    """
    Wraps :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class LinearDiscriminantAnalysisDF(ClassifierDF, LinearDiscriminantAnalysis):
    """
    Wraps :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`; accepts and
    returns data frames.
    """

    pass


#
# naive bayes
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class GaussianNBDF(ClassifierDF, GaussianNB):
    """
    Wraps :class:`sklearn.naive_bayes.GaussianNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class MultinomialNBDF(ClassifierDF, MultinomialNB):
    """
    Wraps :class:`sklearn.naive_bayes.MultinomialNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class ComplementNBDF(ClassifierDF, ComplementNB):
    """
    Wraps :class:`sklearn.naive_bayes.ComplementNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class BernoulliNBDF(ClassifierDF, BernoulliNB):
    """
    Wraps :class:`sklearn.naive_bayes.BernoulliNB`; accepts and returns data frames.
    """

    pass


#
# calibration
#

# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_MetaClassifierWrapperDF)
class CalibratedClassifierCVDF(ClassifierDF, CalibratedClassifierCV):
    """
    Wraps :class:`sklearn.calibration.CalibratedClassifierCV`; accepts and returns data
    frames.
    """

    pass


#
# SVM
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class SVCDF(ClassifierDF, SVC):
    """
    Wraps :class:`sklearn.svm.classes.SVC`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class NuSVCDF(ClassifierDF, NuSVC):
    """
    Wraps :class:`sklearn.svm.classes.NuSVC`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class LinearSVCDF(ClassifierDF, LinearSVC):
    """
    Wraps :class:`sklearn.svm.classes.LinearSVC`; accepts and returns data frames.
    """

    pass


#
# gaussian process
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class GaussianProcessClassifierDF(ClassifierDF, GaussianProcessClassifier):
    """
    Wraps :class:`sklearn.gaussian_process.gpc.GaussianProcessClassifier`; accepts and
    returns data frames.
    """

    pass


#
# linear model
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class LogisticRegressionDF(ClassifierDF, LogisticRegression):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegression`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class LogisticRegressionCVDF(ClassifierDF, LogisticRegressionCV):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegressionCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class PassiveAggressiveClassifierDF(ClassifierDF, PassiveAggressiveClassifier):
    """
    Wraps :class:`sklearn.linear_model.passive_aggressive.PassiveAggressiveClassifier`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class PerceptronDF(ClassifierDF, Perceptron):
    """
    Wraps :class:`sklearn.linear_model.perceptron.Perceptron`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class SGDClassifierDF(ClassifierDF, SGDClassifier):
    """
    Wraps :class:`sklearn.linear_model.stochastic_gradient.SGDClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class RidgeClassifierDF(ClassifierDF, RidgeClassifier):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class RidgeClassifierCVDF(ClassifierDF, RidgeClassifierCV):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeClassifierCV`; accepts and returns
    data frames.
    """

    pass


#
# semi-supervised
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class LabelPropagationDF(ClassifierDF, LabelPropagation):
    """
    Wraps :class:`sklearn.semi_supervised.label_propagation.LabelPropagation`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class LabelSpreadingDF(ClassifierDF, LabelSpreading):
    """
    Wraps :class:`sklearn.semi_supervised.label_propagation.LabelSpreading`; accepts and
    returns data frames.
    """

    pass


#
# multi-class
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_MetaClassifierWrapperDF)
class OneVsRestClassifierDF(ClassifierDF, OneVsRestClassifier):
    """
    Wraps :class:`sklearn.multiclass.OneVsRestClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_MetaClassifierWrapperDF)
class OneVsOneClassifierDF(ClassifierDF, OneVsOneClassifier):
    """
    Wraps :class:`sklearn.multiclass.OneVsOneClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_MetaClassifierWrapperDF)
class OutputCodeClassifierDF(ClassifierDF, OutputCodeClassifier):
    """
    Wraps :class:`sklearn.multiclass.OutputCodeClassifier`; accepts and returns data
    frames.
    """

    pass


#
# multi-output
#


class _MultiOutputClassifierWrapperDF(
    _MetaClassifierWrapperDF[sklearn.multioutput.MultiOutputClassifier],
    metaclass=ABCMeta,
):
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
        sup = super()

        # estimators attribute of abstract class MultiOutputEstimator
        # usually the delegate estimator will provide a list of estimators used
        # to predict each output. If present, use these estimators to get
        # individual class labels for each output; otherwise we cannot assign class
        # labels
        if hasattr(delegate_estimator, "estimators_"):
            return [
                sup._prediction_with_class_labels(
                    X=X, y=output, classes=getattr(estimator, "classes_", None)
                )
                for estimator, output in zip(
                    getattr(delegate_estimator, "estimators_"), y
                )
            ]
        else:
            return [
                sup._prediction_with_class_labels(X=X, y=output, classes=None)
                for output in y
            ]


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_MultiOutputClassifierWrapperDF)
class MultiOutputClassifierDF(ClassifierDF, MultiOutputClassifier):
    """
    Wraps :class:`sklearn.multioutput.MultiOutputClassifier`; accepts and returns data
    frames.
    """

    pass


#
# chaining
#


class _ClassifierChainWrapperDF(
    _MetaClassifierWrapperDF[sklearn.multioutput.ClassifierChain], metaclass=ABCMeta
):
    # noinspection PyPep8Naming
    def _prediction_with_class_labels(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame, list, np.ndarray],
        classes: Optional[Sequence[Any]] = None,
    ) -> Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]:
        return super()._prediction_with_class_labels(
            X=X, y=y, classes=range(self.n_outputs_)
        )


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierChainWrapperDF)
class ClassifierChainDF(ClassifierDF, ClassifierChain):
    """
    Wraps :class:`sklearn.multioutput.ClassifierChain`; accepts and returns data frames.
    """

    pass


#
# neural network
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class MLPClassifierDF(ClassifierDF, MLPClassifier):
    """
    Wraps :class:`sklearn.neural_network.multilayer_perceptron.MLPClassifier`; accepts
    and returns data frames.
    """

    pass


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
