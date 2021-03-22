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

from ..wrapper import make_df_regressor
from .wrapper import (
    IsotonicRegressionWrapperDF,
    MetaRegressorWrapperDF,
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

DummyRegressorDF = make_df_regressor(DummyRegressor)


#
# SVM
#

LinearSVRDF = make_df_regressor(LinearSVR)
SVRDF = make_df_regressor(SVR)
NuSVRDF = make_df_regressor(NuSVR)


#
# multi-output
#

MultiOutputRegressorDF = make_df_regressor(
    MultiOutputRegressor, base_wrapper=MetaRegressorWrapperDF
)

RegressorChainDF = make_df_regressor(
    RegressorChain, base_wrapper=MetaRegressorWrapperDF
)


#
# neighbors
#

KNeighborsRegressorDF = make_df_regressor(KNeighborsRegressor)
RadiusNeighborsRegressorDF = make_df_regressor(RadiusNeighborsRegressor)


#
# neural_network
#

MLPRegressorDF = make_df_regressor(MLPRegressor)


#
# linear_model
#

LinearRegressionDF = make_df_regressor(LinearRegression)
RidgeDF = make_df_regressor(Ridge)
RidgeCVDF = make_df_regressor(RidgeCV)
SGDRegressorDF = make_df_regressor(SGDRegressor)
HuberRegressorDF = make_df_regressor(HuberRegressor)
TheilSenRegressorDF = make_df_regressor(TheilSenRegressor)
BayesianRidgeDF = make_df_regressor(BayesianRidge)
ARDRegressionDF = make_df_regressor(ARDRegression)
OrthogonalMatchingPursuitDF = make_df_regressor(OrthogonalMatchingPursuit)
OrthogonalMatchingPursuitCVDF = make_df_regressor(OrthogonalMatchingPursuitCV)
RANSACRegressorDF = make_df_regressor(RANSACRegressor)
ElasticNetDF = make_df_regressor(ElasticNet)
LassoCVDF = make_df_regressor(LassoCV)
ElasticNetCVDF = make_df_regressor(ElasticNetCV)
MultiTaskElasticNetCVDF = make_df_regressor(MultiTaskElasticNetCV)
MultiTaskLassoCVDF = make_df_regressor(MultiTaskLassoCV)
MultiTaskElasticNetDF = make_df_regressor(MultiTaskElasticNet)
MultiTaskLassoDF = make_df_regressor(MultiTaskLasso)
LassoDF = make_df_regressor(Lasso)
PassiveAggressiveRegressorDF = make_df_regressor(PassiveAggressiveRegressor)
LarsDF = make_df_regressor(Lars)
LassoLarsDF = make_df_regressor(LassoLars)
LassoLarsICDF = make_df_regressor(LassoLarsIC)
LarsCVDF = make_df_regressor(LarsCV)
LassoLarsCVDF = make_df_regressor(LassoLarsCV)


#
# ensemble
#

BaggingRegressorDF = make_df_regressor(BaggingRegressor)
GradientBoostingRegressorDF = make_df_regressor(GradientBoostingRegressor)
AdaBoostRegressorDF = make_df_regressor(AdaBoostRegressor)
RandomForestRegressorDF = make_df_regressor(RandomForestRegressor)
ExtraTreesRegressorDF = make_df_regressor(ExtraTreesRegressor)

VotingRegressorDF = make_df_regressor(
    VotingRegressor, base_wrapper=MetaRegressorWrapperDF
)


#
# gaussian_process
#

GaussianProcessRegressorDF = make_df_regressor(GaussianProcessRegressor)


#
# isotonic
#

IsotonicRegressionDF = make_df_regressor(
    IsotonicRegression, base_wrapper=IsotonicRegressionWrapperDF
)


#
# compose
#

TransformedTargetRegressorDF = make_df_regressor(TransformedTargetRegressor)


#
# kernel_ridge
#

KernelRidgeDF = make_df_regressor(KernelRidge)


#
# tree
#

DecisionTreeRegressorDF = make_df_regressor(DecisionTreeRegressor)
ExtraTreeRegressorDF = make_df_regressor(ExtraTreeRegressor)


#
# cross_decomposition
#


CCADF = make_df_regressor(CCA, base_wrapper=RegressorTransformerWrapperDF)

PLSRegressionDF = make_df_regressor(
    PLSRegression, base_wrapper=RegressorTransformerWrapperDF
)

PLSCanonicalDF = make_df_regressor(
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
