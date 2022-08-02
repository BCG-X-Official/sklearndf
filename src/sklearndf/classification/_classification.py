"""
Core implementation of :mod:`sklearndf.classification`
"""
import logging

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

from ..wrapper import ClassifierWrapperDF, MetaEstimatorWrapperDF
from .wrapper import (
    ClassifierChainWrapperDF,
    LinearDiscriminantAnalysisWrapperDF,
    MetaClassifierWrapperDF,
    MultiOutputClassifierWrapperDF,
    PartialFitClassifierWrapperDF,
)

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


class DummyClassifierDF(
    ClassifierWrapperDF[DummyClassifier],
    native=DummyClassifier,
):
    """Stub for DF wrapper of class ``DummyClassifier``"""


#
# neighbors
#


class NearestCentroidDF(
    ClassifierWrapperDF[NearestCentroid],
    native=NearestCentroid,
):
    """Stub for DF wrapper of class ``NearestCentroid``"""


class KNeighborsClassifierDF(
    ClassifierWrapperDF[KNeighborsClassifier],
    native=KNeighborsClassifier,
):
    """Stub for DF wrapper of class ``KNeighborsClassifier``"""


class RadiusNeighborsClassifierDF(
    ClassifierWrapperDF[RadiusNeighborsClassifier],
    native=RadiusNeighborsClassifier,
):
    """Stub for DF wrapper of class ``RadiusNeighborsClassifier``"""


#
# voting
#


class VotingClassifierDF(
    MetaClassifierWrapperDF[VotingClassifier],
    native=VotingClassifier,
):
    """Stub for DF wrapper of class ``VotingClassifier``"""


#
# ensemble
#


class RandomForestClassifierDF(
    ClassifierWrapperDF[RandomForestClassifier],
    native=RandomForestClassifier,
):
    """Stub for DF wrapper of class ``RandomForestClassifier``"""


class ExtraTreesClassifierDF(
    ClassifierWrapperDF[ExtraTreesClassifier],
    native=ExtraTreesClassifier,
):
    """Stub for DF wrapper of class ``ExtraTreesClassifier``"""


# noinspection PyAbstractClass
class GradientBoostingClassifierDF(
    ClassifierWrapperDF[GradientBoostingClassifier],
    native=GradientBoostingClassifier,
):
    """Stub for DF wrapper of class ``GradientBoostingClassifier``"""


class AdaBoostClassifierDF(
    ClassifierWrapperDF[AdaBoostClassifier],
    native=AdaBoostClassifier,
):
    """Stub for DF wrapper of class ``AdaBoostClassifier``"""


class BaggingClassifierDF(
    ClassifierWrapperDF[BaggingClassifier],
    native=BaggingClassifier,
):
    """Stub for DF wrapper of class ``BaggingClassifier``"""


#
# tree
#


class DecisionTreeClassifierDF(
    ClassifierWrapperDF[DecisionTreeClassifier],
    native=DecisionTreeClassifier,
):
    """Stub for DF wrapper of class ``DecisionTreeClassifier``"""


class ExtraTreeClassifierDF(
    ClassifierWrapperDF[ExtraTreeClassifier],
    native=ExtraTreeClassifier,
):
    """Stub for DF wrapper of class ``ExtraTreeClassifier``"""


#
# discriminant analysis
#


class LinearDiscriminantAnalysisDF(
    LinearDiscriminantAnalysisWrapperDF,
    native=LinearDiscriminantAnalysis,
):
    """Stub for DF wrapper of class ``LinearDiscriminantAnalysis``"""


class QuadraticDiscriminantAnalysisDF(
    ClassifierWrapperDF[QuadraticDiscriminantAnalysis],
    native=QuadraticDiscriminantAnalysis,
):
    """Stub for DF wrapper of class ``QuadraticDiscriminantAnalysis``"""


#
# naive bayes
#


class GaussianNBDF(
    PartialFitClassifierWrapperDF[GaussianNB],
    native=GaussianNB,
):
    """Stub for DF wrapper of class ``GaussianNB``"""


class MultinomialNBDF(
    PartialFitClassifierWrapperDF[MultinomialNB],
    native=MultinomialNB,
):
    """Stub for DF wrapper of class ``MultinomialNB``"""


class ComplementNBDF(
    PartialFitClassifierWrapperDF[ComplementNB],
    native=ComplementNB,
):
    """Stub for DF wrapper of class ``ComplementNB``"""


class BernoulliNBDF(
    PartialFitClassifierWrapperDF[BernoulliNB],
    native=BernoulliNB,
):
    """Stub for DF wrapper of class ``BernoulliNB``"""


#
# calibration
#


class CalibratedClassifierCVDF(
    MetaClassifierWrapperDF[CalibratedClassifierCV],
    native=CalibratedClassifierCV,
):
    """Stub for DF wrapper of class ``CalibratedClassifierCV``"""


#
# SVM
#


class SVCDF(ClassifierWrapperDF[SVC], native=SVC):
    """Stub for DF wrapper of class ``SVC``"""


class NuSVCDF(ClassifierWrapperDF[NuSVC], native=NuSVC):
    """Stub for DF wrapper of class ``NuSVC``"""


class LinearSVCDF(ClassifierWrapperDF[LinearSVC], native=LinearSVC):
    """Stub for DF wrapper of class ``LinearSVC``"""


#
# gaussian process
#


class GaussianProcessClassifierDF(
    ClassifierWrapperDF[GaussianProcessClassifier],
    native=GaussianProcessClassifier,
):
    """Stub for DF wrapper of class ``GaussianProcessClassifier``"""


#
# linear model
#


class LogisticRegressionDF(
    ClassifierWrapperDF[LogisticRegression],
    native=LogisticRegression,
):
    """Stub for DF wrapper of class ``LogisticRegression``"""


class LogisticRegressionCVDF(
    ClassifierWrapperDF[LogisticRegressionCV],
    native=LogisticRegressionCV,
):
    """Stub for DF wrapper of class ``LogisticRegressionCV``"""


class PassiveAggressiveClassifierDF(
    PartialFitClassifierWrapperDF[PassiveAggressiveClassifier],
    native=PassiveAggressiveClassifier,
):
    """Stub for DF wrapper of class ``PassiveAggressiveClassifier``"""


class PerceptronDF(
    PartialFitClassifierWrapperDF[Perceptron],
    native=Perceptron,
):
    """Stub for DF wrapper of class ``Perceptron``"""


class SGDClassifierDF(
    PartialFitClassifierWrapperDF[SGDClassifier],
    native=SGDClassifier,
):
    """Stub for DF wrapper of class ``SGDClassifier``"""


class RidgeClassifierDF(
    ClassifierWrapperDF[RidgeClassifier],
    native=RidgeClassifier,
):
    """Stub for DF wrapper of class ``RidgeClassifier``"""


class RidgeClassifierCVDF(
    ClassifierWrapperDF[RidgeClassifierCV],
    native=RidgeClassifierCV,
):
    """Stub for DF wrapper of class ``RidgeClassifierCV``"""


#
# semi-supervised
#


class LabelPropagationDF(
    ClassifierWrapperDF[LabelPropagation],
    native=LabelPropagation,
):
    """Stub for DF wrapper of class ``LabelPropagation``"""


class LabelSpreadingDF(
    ClassifierWrapperDF[LabelSpreading],
    native=LabelSpreading,
):
    """Stub for DF wrapper of class ``LabelSpreading``"""


#
# multi-class
#


class OneVsRestClassifierDF(
    MetaClassifierWrapperDF[OneVsRestClassifier],
    native=OneVsRestClassifier,
):
    """Stub for DF wrapper of class ``OneVsRestClassifier``"""


class OneVsOneClassifierDF(
    ClassifierWrapperDF[OneVsOneClassifier],
    MetaEstimatorWrapperDF[OneVsOneClassifier],
    native=OneVsOneClassifier,
):
    """Stub for DF wrapper of class ``OneVsOneClassifier``"""


class OutputCodeClassifierDF(
    ClassifierWrapperDF[OutputCodeClassifier],
    MetaEstimatorWrapperDF[OutputCodeClassifier],
    native=OutputCodeClassifier,
):
    """Stub for DF wrapper of class ``OutputCodeClassifier``"""


#
# multi-output
#


class MultiOutputClassifierDF(
    MultiOutputClassifierWrapperDF,
    native=MultiOutputClassifier,
):
    """Stub for DF wrapper of class ``MultiOutputClassifier``"""


#
# chaining
#


class ClassifierChainDF(
    ClassifierChainWrapperDF,
    native=ClassifierChain,
):
    """Stub for DF wrapper of class ``ClassifierChain``"""


#
# neural network
#


class MLPClassifierDF(
    PartialFitClassifierWrapperDF[MLPClassifier],
    native=MLPClassifier,
):
    """Stub for DF wrapper of class ``MLPClassifier``"""


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
