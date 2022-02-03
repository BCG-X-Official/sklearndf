"""
Core implementation of :mod:`sklearndf.classification`
"""
import logging
from typing import Type, TypeVar

from sklearn.base import ClassifierMixin
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

from ..wrapper import make_df_classifier
from .wrapper import (
    ClassifierChainWrapperDF,
    LinearDiscriminantAnalysisWrapperDF,
    MetaClassifierWrapperDF,
    MultiOutputClassifierWrapperDF,
)
from sklearndf.wrapper import ClassifierWrapperDF

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
# Type constructors
#

T_NativeClassifier = TypeVar("T_NativeClassifier", bound=ClassifierMixin)
T_Wrapper = Type[ClassifierWrapperDF[T_NativeClassifier]]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals(), allow_imported_definitions=True)


#
# Class definitions
#


#
# Dummy
#

DummyClassifierDF: T_Wrapper[DummyClassifier] = make_df_classifier(DummyClassifier)


#
# neighbors
#

NearestCentroidDF: T_Wrapper[NearestCentroid] = make_df_classifier(NearestCentroid)
KNeighborsClassifierDF: T_Wrapper[KNeighborsClassifier] = make_df_classifier(
    KNeighborsClassifier
)
RadiusNeighborsClassifierDF: T_Wrapper[RadiusNeighborsClassifier] = make_df_classifier(
    RadiusNeighborsClassifier
)


#
# voting
#

VotingClassifierDF: T_Wrapper[VotingClassifier] = make_df_classifier(
    VotingClassifier, base_wrapper=MetaClassifierWrapperDF
)


#
# ensemble
#


RandomForestClassifierDF: T_Wrapper[RandomForestClassifier] = make_df_classifier(
    RandomForestClassifier
)
ExtraTreesClassifierDF: T_Wrapper[ExtraTreesClassifier] = make_df_classifier(
    ExtraTreesClassifier
)
GradientBoostingClassifierDF: T_Wrapper[
    GradientBoostingClassifier
] = make_df_classifier(GradientBoostingClassifier)
AdaBoostClassifierDF: T_Wrapper[AdaBoostClassifier] = make_df_classifier(
    AdaBoostClassifier
)
BaggingClassifierDF: T_Wrapper[BaggingClassifier] = make_df_classifier(
    BaggingClassifier
)


#
# tree
#

DecisionTreeClassifierDF: T_Wrapper[DecisionTreeClassifier] = make_df_classifier(
    DecisionTreeClassifier
)
ExtraTreeClassifierDF: T_Wrapper[ExtraTreeClassifier] = make_df_classifier(
    ExtraTreeClassifier
)


#
# discriminant analysis
#


LinearDiscriminantAnalysisDF: T_Wrapper[
    LinearDiscriminantAnalysis
] = make_df_classifier(
    LinearDiscriminantAnalysis,
    base_wrapper=LinearDiscriminantAnalysisWrapperDF,
)

QuadraticDiscriminantAnalysisDF: T_Wrapper[
    QuadraticDiscriminantAnalysis
] = make_df_classifier(QuadraticDiscriminantAnalysis)


#
# naive bayes
#


GaussianNBDF: T_Wrapper[GaussianNB] = make_df_classifier(GaussianNB)
MultinomialNBDF: T_Wrapper[MultinomialNB] = make_df_classifier(MultinomialNB)
ComplementNBDF: T_Wrapper[ComplementNB] = make_df_classifier(ComplementNB)
BernoulliNBDF: T_Wrapper[BernoulliNB] = make_df_classifier(BernoulliNB)


#
# calibration
#

CalibratedClassifierCVDF: T_Wrapper[MetaClassifierWrapperDF] = make_df_classifier(
    CalibratedClassifierCV, base_wrapper=MetaClassifierWrapperDF
)


#
# SVM
#

SVCDF: T_Wrapper[SVC] = make_df_classifier(SVC)
NuSVCDF: T_Wrapper[NuSVC] = make_df_classifier(NuSVC)
LinearSVCDF: T_Wrapper[LinearSVC] = make_df_classifier(LinearSVC)


#
# gaussian process
#

GaussianProcessClassifierDF: T_Wrapper[GaussianProcessClassifier] = make_df_classifier(
    GaussianProcessClassifier
)


#
# linear model
#


LogisticRegressionDF: T_Wrapper[LogisticRegression] = make_df_classifier(
    LogisticRegression
)
LogisticRegressionCVDF: T_Wrapper[LogisticRegressionCV] = make_df_classifier(
    LogisticRegressionCV
)
PassiveAggressiveClassifierDF: T_Wrapper[
    PassiveAggressiveClassifier
] = make_df_classifier(PassiveAggressiveClassifier)
PerceptronDF: T_Wrapper[Perceptron] = make_df_classifier(Perceptron)
SGDClassifierDF: T_Wrapper[SGDClassifier] = make_df_classifier(SGDClassifier)
RidgeClassifierDF: T_Wrapper[RidgeClassifier] = make_df_classifier(RidgeClassifier)
RidgeClassifierCVDF: T_Wrapper[RidgeClassifierCV] = make_df_classifier(
    RidgeClassifierCV
)


#
# semi-supervised
#

LabelPropagationDF: T_Wrapper[LabelPropagation] = make_df_classifier(LabelPropagation)
LabelSpreadingDF: T_Wrapper[LabelSpreading] = make_df_classifier(LabelSpreading)


#
# multi-class
#

OneVsRestClassifierDF: T_Wrapper[OneVsRestClassifier] = make_df_classifier(
    OneVsRestClassifier, base_wrapper=MetaClassifierWrapperDF
)

OneVsOneClassifierDF: T_Wrapper[OneVsOneClassifier] = make_df_classifier(
    OneVsOneClassifier, base_wrapper=MetaClassifierWrapperDF
)

OutputCodeClassifierDF: T_Wrapper[OutputCodeClassifier] = make_df_classifier(
    OutputCodeClassifier, base_wrapper=MetaClassifierWrapperDF
)


#
# multi-output
#


MultiOutputClassifierDF: T_Wrapper[MultiOutputClassifier] = make_df_classifier(
    MultiOutputClassifier, base_wrapper=MultiOutputClassifierWrapperDF
)


#
# chaining
#


ClassifierChainDF: T_Wrapper[ClassifierChain] = make_df_classifier(
    ClassifierChain, base_wrapper=ClassifierChainWrapperDF
)


#
# neural network
#

MLPClassifierDF: T_Wrapper[MLPClassifier] = make_df_classifier(MLPClassifier)


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
