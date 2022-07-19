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

__tracker = AllTracker(globals(), allow_imported_definitions=True)


#
# Class definitions
#


#
# Dummy
#


class DummyRegressorDF(
    RegressorWrapperDF[DummyRegressor],
    DummyRegressor,  # type: ignore
    native=DummyRegressor,
):
    """Stub for DF wrapper of class ``DummyRegressor``"""


#
# SVM
#


class LinearSVRDF(
    RegressorWrapperDF[LinearSVR], LinearSVR, native=LinearSVR  # type: ignore
):
    """Stub for DF wrapper of class ``LinearSVR``"""


class SVRDF(RegressorWrapperDF[SVR], SVR, native=SVR):  # type: ignore
    """Stub for DF wrapper of class ``SVR``"""


class NuSVRDF(RegressorWrapperDF[NuSVR], NuSVR, native=NuSVR):  # type: ignore
    """Stub for DF wrapper of class ``NuSVR``"""


#
# multi-output
#


class MultiOutputRegressorDF(
    MultiOutputRegressorWrapperDF,
    MultiOutputRegressor,  # type: ignore
    native=MultiOutputRegressor,
):
    """Stub for DF wrapper of class ``MultiOutputRegressor``"""


class RegressorChainDF(
    MetaRegressorWrapperDF[RegressorChain],
    RegressorChain,  # type: ignore
    native=RegressorChain,
):
    """Stub for DF wrapper of class ``RegressorChain``"""


#
# neighbors
#


class KNeighborsRegressorDF(
    RegressorWrapperDF[KNeighborsRegressor],
    KNeighborsRegressor,  # type: ignore
    native=KNeighborsRegressor,
):
    """Stub for DF wrapper of class ``KNeighborsRegressor``"""


class RadiusNeighborsRegressorDF(
    RegressorWrapperDF[RadiusNeighborsRegressor],
    RadiusNeighborsRegressor,  # type: ignore
    native=RadiusNeighborsRegressor,
):
    """Stub for DF wrapper of class ``RadiusNeighborsRegressor``"""


#
# neural_network
#


class MLPRegressorDF(
    PartialFitRegressorWrapperDF[MLPRegressor],
    MLPRegressor,  # type: ignore
    native=MLPRegressor,
):
    """Stub for DF wrapper of class ``MLPRegressor``"""


#
# linear_model
#


class LinearRegressionDF(
    RegressorWrapperDF[LinearRegression],
    LinearRegression,  # type: ignore
    native=LinearRegression,
):
    """Stub for DF wrapper of class ``LinearRegression``"""


class RidgeDF(RegressorWrapperDF[Ridge], Ridge, native=Ridge):  # type: ignore
    """Stub for DF wrapper of class ``Ridge``"""


class RidgeCVDF(RegressorWrapperDF[RidgeCV], RidgeCV, native=RidgeCV):  # type: ignore
    """Stub for DF wrapper of class ``RidgeCV``"""


class SGDRegressorDF(
    PartialFitRegressorWrapperDF[SGDRegressor],
    SGDRegressor,  # type: ignore
    native=SGDRegressor,
):
    """Stub for DF wrapper of class ``SGDRegressor``"""


class HuberRegressorDF(
    RegressorWrapperDF[HuberRegressor],
    HuberRegressor,  # type: ignore
    native=HuberRegressor,
):
    """Stub for DF wrapper of class ``HuberRegressor``"""


class TheilSenRegressorDF(
    RegressorWrapperDF[TheilSenRegressor],
    TheilSenRegressor,  # type: ignore
    native=TheilSenRegressor,
):
    """Stub for DF wrapper of class ``TheilSenRegressor``"""


class BayesianRidgeDF(
    RegressorWrapperDF[BayesianRidge],
    BayesianRidge,  # type: ignore
    native=BayesianRidge,
):
    """Stub for DF wrapper of class ``BayesianRidge``"""


class ARDRegressionDF(
    RegressorWrapperDF[ARDRegression],
    ARDRegression,  # type: ignore
    native=ARDRegression,
):
    """Stub for DF wrapper of class ``ARDRegression``"""


class OrthogonalMatchingPursuitDF(
    RegressorWrapperDF[OrthogonalMatchingPursuit],
    OrthogonalMatchingPursuit,  # type: ignore
    native=OrthogonalMatchingPursuit,
):
    """Stub for DF wrapper of class ``OrthogonalMatchingPursuit``"""


class OrthogonalMatchingPursuitCVDF(
    RegressorWrapperDF[OrthogonalMatchingPursuitCV],
    OrthogonalMatchingPursuitCV,  # type: ignore
    native=OrthogonalMatchingPursuitCV,
):
    """Stub for DF wrapper of class ``OrthogonalMatchingPursuitCV``"""


class RANSACRegressorDF(
    RegressorWrapperDF[RANSACRegressor],
    RANSACRegressor,  # type: ignore
    native=RANSACRegressor,
):
    """Stub for DF wrapper of class ``RANSACRegressor``"""


class ElasticNetDF(
    RegressorWrapperDF[ElasticNet], ElasticNet, native=ElasticNet  # type: ignore
):
    """Stub for DF wrapper of class ``ElasticNet``"""


class LassoCVDF(RegressorWrapperDF[LassoCV], LassoCV, native=LassoCV):  # type: ignore
    """Stub for DF wrapper of class ``LassoCV``"""


class ElasticNetCVDF(
    RegressorWrapperDF[ElasticNetCV], ElasticNetCV, native=ElasticNetCV  # type: ignore
):
    """Stub for DF wrapper of class ``ElasticNetCV``"""


class MultiTaskElasticNetCVDF(
    RegressorWrapperDF[MultiTaskElasticNetCV],
    MultiTaskElasticNetCV,  # type: ignore
    native=MultiTaskElasticNetCV,
):
    """Stub for DF wrapper of class ``MultiTaskElasticNetCV``"""


class MultiTaskLassoCVDF(
    RegressorWrapperDF[MultiTaskLassoCV],
    MultiTaskLassoCV,  # type: ignore
    native=MultiTaskLassoCV,
):
    """Stub for DF wrapper of class ``MultiTaskLassoCV``"""


class MultiTaskElasticNetDF(
    RegressorWrapperDF[MultiTaskElasticNet],
    MultiTaskElasticNet,  # type: ignore
    native=MultiTaskElasticNet,
):
    """Stub for DF wrapper of class ``MultiTaskElasticNet``"""


class MultiTaskLassoDF(
    RegressorWrapperDF[MultiTaskLasso],
    MultiTaskLasso,  # type: ignore
    native=MultiTaskLasso,
):
    """Stub for DF wrapper of class ``MultiTaskLasso``"""


class LassoDF(RegressorWrapperDF[Lasso], Lasso, native=Lasso):  # type: ignore
    """Stub for DF wrapper of class ``Lasso``"""


class PassiveAggressiveRegressorDF(
    PartialFitRegressorWrapperDF[PassiveAggressiveRegressor],
    PassiveAggressiveRegressor,  # type: ignore
    native=PassiveAggressiveRegressor,
):
    """Stub for DF wrapper of class ``PassiveAggressiveRegressor``"""


class LarsDF(RegressorWrapperDF[Lars], Lars, native=Lars):  # type: ignore
    """Stub for DF wrapper of class ``Lars``"""


class LassoLarsDF(
    RegressorWrapperDF[LassoLars], LassoLars, native=LassoLars  # type: ignore
):
    """Stub for DF wrapper of class ``LassoLars``"""


class LassoLarsICDF(
    RegressorWrapperDF[LassoLarsIC], LassoLarsIC, native=LassoLarsIC  # type: ignore
):
    """Stub for DF wrapper of class ``LassoLarsIC``"""


class LarsCVDF(RegressorWrapperDF[LarsCV], LarsCV, native=LarsCV):  # type: ignore
    """Stub for DF wrapper of class ``LarsCV``"""


class LassoLarsCVDF(
    RegressorWrapperDF[LassoLarsCV], LassoLarsCV, native=LassoLarsCV  # type: ignore
):
    """Stub for DF wrapper of class ``LassoLarsCV``"""


#
# ensemble
#


class BaggingRegressorDF(
    RegressorWrapperDF[BaggingRegressor],
    BaggingRegressor,  # type: ignore
    native=BaggingRegressor,
):
    """Stub for DF wrapper of class ``BaggingRegressor``"""


# noinspection PyAbstractClass
class GradientBoostingRegressorDF(
    RegressorWrapperDF[GradientBoostingRegressor],
    GradientBoostingRegressor,  # type: ignore
    native=GradientBoostingRegressor,
):
    """Stub for DF wrapper of class ``GradientBoostingRegressor``"""


class AdaBoostRegressorDF(
    RegressorWrapperDF[AdaBoostRegressor],
    AdaBoostRegressor,  # type: ignore
    native=AdaBoostRegressor,
):
    """Stub for DF wrapper of class ``AdaBoostRegressor``"""


class RandomForestRegressorDF(
    RegressorWrapperDF[RandomForestRegressor],
    RandomForestRegressor,  # type: ignore
    native=RandomForestRegressor,
):
    """Stub for DF wrapper of class ``RandomForestRegressor``"""


class ExtraTreesRegressorDF(
    RegressorWrapperDF[ExtraTreesRegressor],
    ExtraTreesRegressor,  # type: ignore
    native=ExtraTreesRegressor,
):
    """Stub for DF wrapper of class ``ExtraTreesRegressor``"""


class VotingRegressorDF(
    MetaRegressorWrapperDF[VotingRegressor],
    VotingRegressor,  # type: ignore
    native=VotingRegressor,
):
    """Stub for DF wrapper of class ``VotingRegressor``"""


#
# gaussian_process
#


class GaussianProcessRegressorDF(
    RegressorWrapperDF[GaussianProcessRegressor],
    GaussianProcessRegressor,  # type: ignore
    native=GaussianProcessRegressor,
):
    """Stub for DF wrapper of class ``GaussianProcessRegressor``"""


#
# isotonic
#


class IsotonicRegressionDF(
    IsotonicRegressionWrapperDF,
    IsotonicRegression,  # type: ignore
    native=IsotonicRegression,
):
    """Stub for DF wrapper of class ``IsotonicRegression``"""


#
# compose
#


class TransformedTargetRegressorDF(
    RegressorWrapperDF[TransformedTargetRegressor],
    TransformedTargetRegressor,  # type: ignore
    native=TransformedTargetRegressor,
):
    """Stub for DF wrapper of class ``TransformedTargetRegressor``"""


#
# kernel_ridge
#


class KernelRidgeDF(
    RegressorWrapperDF[KernelRidge], KernelRidge, native=KernelRidge  # type: ignore
):
    """Stub for DF wrapper of class ``KernelRidge``"""


#
# tree
#


class DecisionTreeRegressorDF(
    RegressorWrapperDF[DecisionTreeRegressor],
    DecisionTreeRegressor,  # type: ignore
    native=DecisionTreeRegressor,
):
    """Stub for DF wrapper of class ``DecisionTreeRegressor``"""


class ExtraTreeRegressorDF(
    RegressorWrapperDF[ExtraTreeRegressor],
    ExtraTreeRegressor,  # type: ignore
    native=ExtraTreeRegressor,
):
    """Stub for DF wrapper of class ``ExtraTreeRegressor``"""


#
# cross_decomposition
#


class CCADF(RegressorTransformerWrapperDF[CCA], CCA, native=CCA):  # type: ignore
    """Stub for DF wrapper of class ``CCA``"""


class PLSRegressionDF(
    RegressorTransformerWrapperDF[PLSRegression],
    PLSRegression,  # type: ignore
    native=PLSRegression,
):
    """Stub for DF wrapper of class ``PLSRegression``"""


class PLSCanonicalDF(
    RegressorTransformerWrapperDF[PLSCanonical],
    PLSCanonical,  # type: ignore
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
