"""
Core implementation of :mod:`sklearndf.regression`
"""
import logging
from typing import Type, TypeVar

from sklearn.base import RegressorMixin
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

from ..wrapper import make_df_regressor
from .wrapper import (
    IsotonicRegressionWrapperDF,
    MetaRegressorWrapperDF,
    RegressorTransformerWrapperDF,
)
from sklearndf.wrapper import RegressorWrapperDF

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
# Type constructors
#

T_NativeRegressor = TypeVar("T_NativeRegressor", bound=RegressorMixin)
T_Wrapper = Type[RegressorWrapperDF[T_NativeRegressor]]


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

DummyRegressorDF: T_Wrapper[DummyRegressor] = make_df_regressor(DummyRegressor)


#
# SVM
#

LinearSVRDF: T_Wrapper[LinearSVR] = make_df_regressor(LinearSVR)
SVRDF: T_Wrapper[SVR] = make_df_regressor(SVR)
NuSVRDF: T_Wrapper[NuSVR] = make_df_regressor(NuSVR)


#
# multi-output
#

MultiOutputRegressorDF: T_Wrapper[MultiOutputRegressor] = make_df_regressor(
    MultiOutputRegressor, base_wrapper=MetaRegressorWrapperDF
)

RegressorChainDF: T_Wrapper[RegressorChain] = make_df_regressor(
    RegressorChain, base_wrapper=MetaRegressorWrapperDF
)


#
# neighbors
#

KNeighborsRegressorDF: T_Wrapper[KNeighborsRegressor] = make_df_regressor(
    KNeighborsRegressor
)
RadiusNeighborsRegressorDF: T_Wrapper[RadiusNeighborsRegressor] = make_df_regressor(
    RadiusNeighborsRegressor
)


#
# neural_network
#

MLPRegressorDF: T_Wrapper[MLPRegressor] = make_df_regressor(MLPRegressor)


#
# linear_model
#

LinearRegressionDF: T_Wrapper[LinearRegression] = make_df_regressor(LinearRegression)
RidgeDF: T_Wrapper[Ridge] = make_df_regressor(Ridge)
RidgeCVDF: T_Wrapper[RidgeCV] = make_df_regressor(RidgeCV)
SGDRegressorDF: T_Wrapper[SGDRegressor] = make_df_regressor(SGDRegressor)
HuberRegressorDF: T_Wrapper[HuberRegressor] = make_df_regressor(HuberRegressor)
TheilSenRegressorDF: T_Wrapper[TheilSenRegressor] = make_df_regressor(TheilSenRegressor)
BayesianRidgeDF: T_Wrapper[BayesianRidge] = make_df_regressor(BayesianRidge)
ARDRegressionDF: T_Wrapper[ARDRegression] = make_df_regressor(ARDRegression)
OrthogonalMatchingPursuitDF: T_Wrapper[OrthogonalMatchingPursuit] = make_df_regressor(
    OrthogonalMatchingPursuit
)
OrthogonalMatchingPursuitCVDF: T_Wrapper[
    OrthogonalMatchingPursuitCV
] = make_df_regressor(OrthogonalMatchingPursuitCV)
RANSACRegressorDF: T_Wrapper[RANSACRegressor] = make_df_regressor(RANSACRegressor)
ElasticNetDF: T_Wrapper[ElasticNet] = make_df_regressor(ElasticNet)
LassoCVDF: T_Wrapper[LassoCV] = make_df_regressor(LassoCV)
ElasticNetCVDF: T_Wrapper[ElasticNetCV] = make_df_regressor(ElasticNetCV)
MultiTaskElasticNetCVDF: T_Wrapper[MultiTaskElasticNetCV] = make_df_regressor(
    MultiTaskElasticNetCV
)
MultiTaskLassoCVDF: T_Wrapper[MultiTaskLassoCV] = make_df_regressor(MultiTaskLassoCV)
MultiTaskElasticNetDF: T_Wrapper[MultiTaskElasticNet] = make_df_regressor(
    MultiTaskElasticNet
)
MultiTaskLassoDF: T_Wrapper[MultiTaskLasso] = make_df_regressor(MultiTaskLasso)
LassoDF: T_Wrapper[Lasso] = make_df_regressor(Lasso)
PassiveAggressiveRegressorDF: T_Wrapper[PassiveAggressiveRegressor] = make_df_regressor(
    PassiveAggressiveRegressor
)
LarsDF: T_Wrapper[Lars] = make_df_regressor(Lars)
LassoLarsDF: T_Wrapper[LassoLars] = make_df_regressor(LassoLars)
LassoLarsICDF: T_Wrapper[LassoLarsIC] = make_df_regressor(LassoLarsIC)
LarsCVDF: T_Wrapper[LarsCV] = make_df_regressor(LarsCV)
LassoLarsCVDF: T_Wrapper[LassoLarsCV] = make_df_regressor(LassoLarsCV)


#
# ensemble
#

BaggingRegressorDF: T_Wrapper[BaggingRegressor] = make_df_regressor(BaggingRegressor)
GradientBoostingRegressorDF: T_Wrapper[GradientBoostingRegressor] = make_df_regressor(
    GradientBoostingRegressor
)
AdaBoostRegressorDF: T_Wrapper[AdaBoostRegressor] = make_df_regressor(AdaBoostRegressor)
RandomForestRegressorDF: T_Wrapper[RandomForestRegressor] = make_df_regressor(
    RandomForestRegressor
)
ExtraTreesRegressorDF: T_Wrapper[ExtraTreesRegressor] = make_df_regressor(
    ExtraTreesRegressor
)

VotingRegressorDF: T_Wrapper[VotingRegressor] = make_df_regressor(
    VotingRegressor, base_wrapper=MetaRegressorWrapperDF
)


#
# gaussian_process
#

GaussianProcessRegressorDF: T_Wrapper[GaussianProcessRegressor] = make_df_regressor(
    GaussianProcessRegressor
)


#
# isotonic
#

IsotonicRegressionDF: T_Wrapper[IsotonicRegression] = make_df_regressor(
    IsotonicRegression, base_wrapper=IsotonicRegressionWrapperDF
)


#
# compose
#

TransformedTargetRegressorDF: T_Wrapper[TransformedTargetRegressor] = make_df_regressor(
    TransformedTargetRegressor
)


#
# kernel_ridge
#

KernelRidgeDF: T_Wrapper[KernelRidge] = make_df_regressor(KernelRidge)


#
# tree
#

DecisionTreeRegressorDF: T_Wrapper[DecisionTreeRegressor] = make_df_regressor(
    DecisionTreeRegressor
)
ExtraTreeRegressorDF: T_Wrapper[ExtraTreeRegressor] = make_df_regressor(
    ExtraTreeRegressor
)


#
# cross_decomposition
#


CCADF: T_Wrapper[CCA] = make_df_regressor(
    CCA, base_wrapper=RegressorTransformerWrapperDF
)

PLSRegressionDF: T_Wrapper[PLSRegression] = make_df_regressor(
    PLSRegression, base_wrapper=RegressorTransformerWrapperDF
)

PLSCanonicalDF: T_Wrapper[PLSCanonical] = make_df_regressor(
    PLSCanonical, base_wrapper=RegressorTransformerWrapperDF
)


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
