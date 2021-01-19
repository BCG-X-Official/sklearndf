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

from .._wrapper import (
    _ClassifierWrapperDF,
    _MetaClassifierWrapperDF,
    make_df_classifier,
)
from ..transformation._wrapper import _NComponentsDimensionalityReductionWrapperDF

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

DummyClassifierDF = make_df_classifier(DummyClassifier)


#
# neighbors
#

NearestCentroidDF = make_df_classifier(NearestCentroid)
KNeighborsClassifierDF = make_df_classifier(KNeighborsClassifier)
RadiusNeighborsClassifierDF = make_df_classifier(RadiusNeighborsClassifier)


#
# voting
#

VotingClassifierDF = make_df_classifier(
    VotingClassifier, df_wrapper_type=_MetaClassifierWrapperDF
)


#
# ensemble
#


RandomForestClassifierDF = make_df_classifier(RandomForestClassifier)
ExtraTreesClassifierDF = make_df_classifier(ExtraTreesClassifier)
GradientBoostingClassifierDF = make_df_classifier(GradientBoostingClassifier)
AdaBoostClassifierDF = make_df_classifier(AdaBoostClassifier)
BaggingClassifierDF = make_df_classifier(BaggingClassifier)


#
# tree
#

DecisionTreeClassifierDF = make_df_classifier(DecisionTreeClassifier)
ExtraTreeClassifierDF = make_df_classifier(ExtraTreeClassifier)


#
# discriminant analysis
#


class _LinearDiscriminantAnalysisWrapperDF(
    _NComponentsDimensionalityReductionWrapperDF[LinearDiscriminantAnalysis],
    _ClassifierWrapperDF[LinearDiscriminantAnalysis],
    metaclass=ABCMeta,
):
    pass


LinearDiscriminantAnalysisDF = make_df_classifier(
    LinearDiscriminantAnalysis,
    df_wrapper_type=_LinearDiscriminantAnalysisWrapperDF,
)

QuadraticDiscriminantAnalysisDF = make_df_classifier(QuadraticDiscriminantAnalysis)


#
# naive bayes
#


GaussianNBDF = make_df_classifier(GaussianNB)
MultinomialNBDF = make_df_classifier(MultinomialNB)
ComplementNBDF = make_df_classifier(ComplementNB)
BernoulliNBDF = make_df_classifier(BernoulliNB)


#
# calibration
#

CalibratedClassifierCVDF = make_df_classifier(
    CalibratedClassifierCV, df_wrapper_type=_MetaClassifierWrapperDF
)


#
# SVM
#

SVCDF = make_df_classifier(SVC)
NuSVCDF = make_df_classifier(NuSVC)
LinearSVCDF = make_df_classifier(LinearSVC)


#
# gaussian process
#

GaussianProcessClassifierDF = make_df_classifier(GaussianProcessClassifier)


#
# linear model
#


LogisticRegressionDF = make_df_classifier(LogisticRegression)
LogisticRegressionCVDF = make_df_classifier(LogisticRegressionCV)
PassiveAggressiveClassifierDF = make_df_classifier(PassiveAggressiveClassifier)
PerceptronDF = make_df_classifier(Perceptron)
SGDClassifierDF = make_df_classifier(SGDClassifier)
RidgeClassifierDF = make_df_classifier(RidgeClassifier)
RidgeClassifierCVDF = make_df_classifier(RidgeClassifierCV)


#
# semi-supervised
#

LabelPropagationDF = make_df_classifier(LabelPropagation)
LabelSpreadingDF = make_df_classifier(LabelSpreading)


#
# multi-class
#

OneVsRestClassifierDF = make_df_classifier(
    OneVsRestClassifier, df_wrapper_type=_MetaClassifierWrapperDF
)

OneVsOneClassifierDF = make_df_classifier(
    OneVsOneClassifier, df_wrapper_type=_MetaClassifierWrapperDF
)

OutputCodeClassifierDF = make_df_classifier(
    OutputCodeClassifier, df_wrapper_type=_MetaClassifierWrapperDF
)


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


MultiOutputClassifierDF = make_df_classifier(
    MultiOutputClassifier, df_wrapper_type=_MultiOutputClassifierWrapperDF
)


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


ClassifierChainDF = make_df_classifier(
    ClassifierChain, df_wrapper_type=_ClassifierChainWrapperDF
)


#
# neural network
#

MLPClassifierDF = make_df_classifier(MLPClassifier)


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
