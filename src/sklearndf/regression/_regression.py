"""
Core implementation of :mod:`sklearndf.regression`
"""
import logging

from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge,
    RidgeCV,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from pytools.api import AllTracker

from ..wrapper import RegressorWrapperDF
from .wrapper import (
    IsotonicRegressionWrapperDF,
    MetaRegressorWrapperDF,
    MultiOutputRegressorWrapperDF,
    PartialFitRegressorWrapperDF,
    RegressorTransformerWrapperDF,
)

# noinspection PyProtectedMember

log = logging.getLogger(__name__)

__all__ = [
    "AdaBoostRegressorDF",
    "ARDRegressionDF",
    "BaggingRegressorDF",
    "BayesianRidgeDF",
    "CCADF",
    "DecisionTreeRegressorDF",
    "DummyRegressorDF",
    "ElasticNetCVDF",
    "ElasticNetDF",
    "ExtraTreeRegressorDF",
    "ExtraTreesRegressorDF",
    "GaussianProcessRegressorDF",
    "GradientBoostingRegressorDF",
    "HuberRegressorDF",
    "IsotonicRegressionDF",
    "KernelRidgeDF",
    "KNeighborsRegressorDF",
    "LarsCVDF",
    "LarsDF",
    "LassoCVDF",
    "LassoDF",
    "LassoLarsCVDF",
    "LassoLarsDF",
    "LassoLarsICDF",
    "LinearRegressionDF",
    "LinearSVRDF",
    "MLPRegressorDF",
    "MultiOutputRegressorDF",
    "MultiTaskElasticNetCVDF",
    "MultiTaskElasticNetDF",
    "MultiTaskLassoCVDF",
    "MultiTaskLassoDF",
    "NuSVRDF",
    "OrthogonalMatchingPursuitCVDF",
    "OrthogonalMatchingPursuitDF",
    "PassiveAggressiveRegressorDF",
    "PLSCanonicalDF",
    "PLSRegressionDF",
    "RadiusNeighborsRegressorDF",
    "RandomForestRegressorDF",
    "RANSACRegressorDF",
    "RegressorChainDF",
    "RidgeCVDF",
    "RidgeDF",
    "SGDRegressorDF",
    "SVRDF",
    "TheilSenRegressorDF",
    "TransformedTargetRegressorDF",
    "VotingRegressorDF",
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


class DummyRegressorDF(
    RegressorWrapperDF[DummyRegressor],
    native=DummyRegressor,
):
    """Stub for DF wrapper of class ``DummyRegressor``"""


#
# SVM
#


class LinearSVRDF(RegressorWrapperDF[LinearSVR], native=LinearSVR):
    """Stub for DF wrapper of class ``LinearSVR``"""


class SVRDF(RegressorWrapperDF[SVR], native=SVR):
    """Stub for DF wrapper of class ``SVR``"""


class NuSVRDF(RegressorWrapperDF[NuSVR], native=NuSVR):
    """Stub for DF wrapper of class ``NuSVR``"""


#
# multi-output
#


class MultiOutputRegressorDF(
    MultiOutputRegressorWrapperDF,
    native=MultiOutputRegressor,
):
    """Stub for DF wrapper of class ``MultiOutputRegressor``"""


class RegressorChainDF(
    MetaRegressorWrapperDF[RegressorChain],
    native=RegressorChain,
):
    """Stub for DF wrapper of class ``RegressorChain``"""


#
# neighbors
#


class KNeighborsRegressorDF(
    RegressorWrapperDF[KNeighborsRegressor],
    native=KNeighborsRegressor,
):
    """Stub for DF wrapper of class ``KNeighborsRegressor``"""


class RadiusNeighborsRegressorDF(
    RegressorWrapperDF[RadiusNeighborsRegressor],
    native=RadiusNeighborsRegressor,
):
    """Stub for DF wrapper of class ``RadiusNeighborsRegressor``"""


#
# neural_network
#


class MLPRegressorDF(
    PartialFitRegressorWrapperDF[MLPRegressor],
    native=MLPRegressor,
):
    """Stub for DF wrapper of class ``MLPRegressor``"""


#
# linear_model
#


class LinearRegressionDF(
    RegressorWrapperDF[LinearRegression],
    native=LinearRegression,
):
    """Stub for DF wrapper of class ``LinearRegression``"""


class RidgeDF(RegressorWrapperDF[Ridge], native=Ridge):
    """Stub for DF wrapper of class ``Ridge``"""


class RidgeCVDF(RegressorWrapperDF[RidgeCV], native=RidgeCV):
    """Stub for DF wrapper of class ``RidgeCV``"""


class SGDRegressorDF(
    PartialFitRegressorWrapperDF[SGDRegressor],
    native=SGDRegressor,
):
    """Stub for DF wrapper of class ``SGDRegressor``"""


class HuberRegressorDF(
    RegressorWrapperDF[HuberRegressor],
    native=HuberRegressor,
):
    """Stub for DF wrapper of class ``HuberRegressor``"""


class TheilSenRegressorDF(
    RegressorWrapperDF[TheilSenRegressor],
    native=TheilSenRegressor,
):
    """Stub for DF wrapper of class ``TheilSenRegressor``"""


class BayesianRidgeDF(
    RegressorWrapperDF[BayesianRidge],
    native=BayesianRidge,
):
    """Stub for DF wrapper of class ``BayesianRidge``"""


class ARDRegressionDF(
    RegressorWrapperDF[ARDRegression],
    native=ARDRegression,
):
    """Stub for DF wrapper of class ``ARDRegression``"""


class OrthogonalMatchingPursuitDF(
    RegressorWrapperDF[OrthogonalMatchingPursuit],
    native=OrthogonalMatchingPursuit,
):
    """Stub for DF wrapper of class ``OrthogonalMatchingPursuit``"""


class OrthogonalMatchingPursuitCVDF(
    RegressorWrapperDF[OrthogonalMatchingPursuitCV],
    native=OrthogonalMatchingPursuitCV,
):
    """Stub for DF wrapper of class ``OrthogonalMatchingPursuitCV``"""


class RANSACRegressorDF(
    RegressorWrapperDF[RANSACRegressor],
    native=RANSACRegressor,
):
    """Stub for DF wrapper of class ``RANSACRegressor``"""


class ElasticNetDF(RegressorWrapperDF[ElasticNet], native=ElasticNet):
    """Stub for DF wrapper of class ``ElasticNet``"""


class LassoCVDF(RegressorWrapperDF[LassoCV], native=LassoCV):
    """Stub for DF wrapper of class ``LassoCV``"""


class ElasticNetCVDF(RegressorWrapperDF[ElasticNetCV], native=ElasticNetCV):
    """Stub for DF wrapper of class ``ElasticNetCV``"""


class MultiTaskElasticNetCVDF(
    RegressorWrapperDF[MultiTaskElasticNetCV],
    native=MultiTaskElasticNetCV,
):
    """Stub for DF wrapper of class ``MultiTaskElasticNetCV``"""


class MultiTaskLassoCVDF(
    RegressorWrapperDF[MultiTaskLassoCV],
    native=MultiTaskLassoCV,
):
    """Stub for DF wrapper of class ``MultiTaskLassoCV``"""


class MultiTaskElasticNetDF(
    RegressorWrapperDF[MultiTaskElasticNet],
    native=MultiTaskElasticNet,
):
    """Stub for DF wrapper of class ``MultiTaskElasticNet``"""


class MultiTaskLassoDF(
    RegressorWrapperDF[MultiTaskLasso],
    native=MultiTaskLasso,
):
    """Stub for DF wrapper of class ``MultiTaskLasso``"""


class LassoDF(RegressorWrapperDF[Lasso], native=Lasso):
    """Stub for DF wrapper of class ``Lasso``"""


class PassiveAggressiveRegressorDF(
    PartialFitRegressorWrapperDF[PassiveAggressiveRegressor],
    native=PassiveAggressiveRegressor,
):
    """Stub for DF wrapper of class ``PassiveAggressiveRegressor``"""


class LarsDF(RegressorWrapperDF[Lars], native=Lars):
    """Stub for DF wrapper of class ``Lars``"""


class LassoLarsDF(RegressorWrapperDF[LassoLars], native=LassoLars):
    """Stub for DF wrapper of class ``LassoLars``"""


class LassoLarsICDF(RegressorWrapperDF[LassoLarsIC], native=LassoLarsIC):
    """Stub for DF wrapper of class ``LassoLarsIC``"""


class LarsCVDF(RegressorWrapperDF[LarsCV], native=LarsCV):
    """Stub for DF wrapper of class ``LarsCV``"""


class LassoLarsCVDF(RegressorWrapperDF[LassoLarsCV], native=LassoLarsCV):
    """Stub for DF wrapper of class ``LassoLarsCV``"""


#
# ensemble
#


class BaggingRegressorDF(
    RegressorWrapperDF[BaggingRegressor],
    native=BaggingRegressor,
):
    """Stub for DF wrapper of class ``BaggingRegressor``"""


# noinspection PyAbstractClass
class GradientBoostingRegressorDF(
    RegressorWrapperDF[GradientBoostingRegressor],
    native=GradientBoostingRegressor,
):
    """Stub for DF wrapper of class ``GradientBoostingRegressor``"""


class AdaBoostRegressorDF(
    RegressorWrapperDF[AdaBoostRegressor],
    native=AdaBoostRegressor,
):
    """Stub for DF wrapper of class ``AdaBoostRegressor``"""


class RandomForestRegressorDF(
    RegressorWrapperDF[RandomForestRegressor],
    native=RandomForestRegressor,
):
    """Stub for DF wrapper of class ``RandomForestRegressor``"""


class ExtraTreesRegressorDF(
    RegressorWrapperDF[ExtraTreesRegressor],
    native=ExtraTreesRegressor,
):
    """Stub for DF wrapper of class ``ExtraTreesRegressor``"""


class VotingRegressorDF(
    MetaRegressorWrapperDF[VotingRegressor],
    native=VotingRegressor,
):
    """Stub for DF wrapper of class ``VotingRegressor``"""


#
# gaussian_process
#


class GaussianProcessRegressorDF(
    RegressorWrapperDF[GaussianProcessRegressor],
    native=GaussianProcessRegressor,
):
    """Stub for DF wrapper of class ``GaussianProcessRegressor``"""


#
# isotonic
#


class IsotonicRegressionDF(
    IsotonicRegressionWrapperDF,
    native=IsotonicRegression,
):
    """Stub for DF wrapper of class ``IsotonicRegression``"""


#
# compose
#


class TransformedTargetRegressorDF(
    RegressorWrapperDF[TransformedTargetRegressor],
    native=TransformedTargetRegressor,
):
    """Stub for DF wrapper of class ``TransformedTargetRegressor``"""


#
# kernel_ridge
#


class KernelRidgeDF(RegressorWrapperDF[KernelRidge], native=KernelRidge):
    """Stub for DF wrapper of class ``KernelRidge``"""


#
# tree
#


class DecisionTreeRegressorDF(
    RegressorWrapperDF[DecisionTreeRegressor],
    native=DecisionTreeRegressor,
):
    """Stub for DF wrapper of class ``DecisionTreeRegressor``"""


class ExtraTreeRegressorDF(
    RegressorWrapperDF[ExtraTreeRegressor],
    native=ExtraTreeRegressor,
):
    """Stub for DF wrapper of class ``ExtraTreeRegressor``"""


#
# cross_decomposition
#


class CCADF(RegressorTransformerWrapperDF[CCA], native=CCA):
    """Stub for DF wrapper of class ``CCA``"""


class PLSRegressionDF(
    RegressorTransformerWrapperDF[PLSRegression],
    native=PLSRegression,
):
    """Stub for DF wrapper of class ``PLSRegression``"""


class PLSCanonicalDF(
    RegressorTransformerWrapperDF[PLSCanonical],
    native=PLSCanonical,
):
    """Stub for DF wrapper of class ``PLSCanonical``"""


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
