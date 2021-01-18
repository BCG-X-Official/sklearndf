"""
Core implementation of :mod:`sklearndf.regression`
"""
import functools
import logging
from abc import ABCMeta
from typing import Any, Generic, Optional, TypeVar, Union

import pandas as pd
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

from .._wrapper import _MetaRegressorWrapperDF, _RegressorWrapperDF, make_df_regressor

# noinspection PyProtectedMember
from ..transformation._wrapper import _ColumnPreservingTransformerWrapperDF

log = logging.getLogger(__name__)

__all__ = [
    "ARDRegressionDF",
    "AdaBoostRegressorDF",
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
    "KNeighborsRegressorDF",
    "KernelRidgeDF",
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
    "PLSCanonicalDF",
    "PLSRegressionDF",
    "PassiveAggressiveRegressorDF",
    "RANSACRegressorDF",
    "RadiusNeighborsRegressorDF",
    "RandomForestRegressorDF",
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
# type variables
#

T_Regressor = TypeVar("T_Regressor", bound=RegressorMixin)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())

#
# Set the module for new DF classes
#

make_df_regressor = functools.partial(make_df_regressor, module="sklearndf.regression")


#
# Class definitions
#


#
# wrapper for hybrid regressor/transformer classes
#


class _RegressorTransformerWrapperDF(
    _RegressorWrapperDF[T_Regressor],
    _ColumnPreservingTransformerWrapperDF[T_Regressor],
    Generic[T_Regressor],
    metaclass=ABCMeta,
):
    """
    Wraps a combined regressor and constant column transformer
    """

    pass


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
    MultiOutputRegressor, df_wrapper_type=_MetaRegressorWrapperDF
)

RegressorChainDF = make_df_regressor(
    RegressorChain, df_wrapper_type=_MetaRegressorWrapperDF
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
VotingRegressorDF = make_df_regressor(
    VotingRegressor, df_wrapper_type=_MetaRegressorWrapperDF
)
GradientBoostingRegressorDF = make_df_regressor(GradientBoostingRegressor)
AdaBoostRegressorDF = make_df_regressor(AdaBoostRegressor)
RandomForestRegressorDF = make_df_regressor(RandomForestRegressor)
ExtraTreesRegressorDF = make_df_regressor(ExtraTreesRegressor)


#
# gaussian_process
#

GaussianProcessRegressorDF = make_df_regressor(GaussianProcessRegressor)


#
# isotonic
#


class _IsotonicRegressionWrapperDF(
    _RegressorTransformerWrapperDF[IsotonicRegression], metaclass=ABCMeta
):
    # noinspection PyPep8Naming
    def _check_parameter_types(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        super()._check_parameter_types(X=X, y=y)
        if X.shape[1] != 1:
            raise ValueError(
                f"arg X expected to have exactly 1 column but has {X.shape[1]} columns"
            )

    # noinspection PyPep8Naming
    def _convert_X_for_delegate(self, X: pd.DataFrame) -> Any:
        return super()._convert_X_for_delegate(X).iloc[:, 0].values

    def _convert_y_for_delegate(
        self, y: Optional[Union[pd.Series, pd.DataFrame]]
    ) -> Any:
        y = super()._convert_y_for_delegate(y)
        return None if y is None else y.values


IsotonicRegressionDF = make_df_regressor(
    IsotonicRegression, df_wrapper_type=_IsotonicRegressionWrapperDF
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


CCADF = make_df_regressor(CCA, df_wrapper_type=_RegressorTransformerWrapperDF)

PLSRegressionDF = make_df_regressor(
    PLSRegression, df_wrapper_type=_RegressorTransformerWrapperDF
)

PLSCanonicalDF = make_df_regressor(
    PLSCanonical, df_wrapper_type=_RegressorTransformerWrapperDF
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
