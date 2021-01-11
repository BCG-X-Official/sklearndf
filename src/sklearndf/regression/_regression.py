"""
Core implementation of :mod:`sklearndf.regression`
"""
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

from .. import RegressorDF, TransformerDF
from .._wrapper import _MetaRegressorWrapperDF, _RegressorWrapperDF, df_estimator

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

# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class DummyRegressorDF(RegressorDF, DummyRegressor):
    """
    Wraps :class:`sklearn.dummy.DummyRegressor`; accepts and
    returns data frames.
    """

    pass


#
# SVM
#


# noinspection PyAbstractClass


@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class LinearSVRDF(RegressorDF, LinearSVR):
    """
    Wraps :class:`sklearn.svm.classes.LinearSVR`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class SVRDF(RegressorDF, SVR):
    """
    Wraps :class:`sklearn.svm.classes.SVR`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class NuSVRDF(RegressorDF, NuSVR):
    """
    Wraps :class:`sklearn.svm.classes.NuSVR`; accepts and returns data frames.
    """

    pass


#
# multi-output
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_MetaRegressorWrapperDF)
class MultiOutputRegressorDF(RegressorDF, MultiOutputRegressor):
    """
    Wraps :class:`sklearn.multioutput.MultiOutputRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_MetaRegressorWrapperDF)
class RegressorChainDF(RegressorDF, RegressorChain):
    """
    Wraps :class:`sklearn.multioutput.RegressorChain`; accepts and returns data frames.
    """

    pass


#
# neighbors
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class KNeighborsRegressorDF(RegressorDF, KNeighborsRegressor):
    """
    Wraps :class:`sklearn.neighbors.regression.KNeighborsRegressor`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class RadiusNeighborsRegressorDF(RegressorDF, RadiusNeighborsRegressor):
    """
    Wraps :class:`sklearn.neighbors.regression.RadiusNeighborsRegressor`; accepts and
    returns data frames.
    """

    pass


#
# neural_network
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class MLPRegressorDF(RegressorDF, MLPRegressor):
    """
    Wraps :class:`sklearn.neural_network.multilayer_perceptron.MLPRegressor`; accepts
    and returns data frames.
    """

    pass


#
# linear_model
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class LinearRegressionDF(RegressorDF, LinearRegression):
    """
    Wraps :class:`sklearn.linear_model.base.LinearRegression`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class RidgeDF(RegressorDF, Ridge):
    """
    Wraps :class:`sklearn.linear_model.ridge.Ridge`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class RidgeCVDF(RegressorDF, RidgeCV):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeCV`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class SGDRegressorDF(RegressorDF, SGDRegressor):
    """
    Wraps :class:`sklearn.linear_model.stochastic_gradient.SGDRegressor`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class HuberRegressorDF(RegressorDF, HuberRegressor):
    """
    Wraps :class:`sklearn.linear_model.huber.HuberRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class TheilSenRegressorDF(RegressorDF, TheilSenRegressor):
    """
    Wraps :class:`sklearn.linear_model.theil_sen.TheilSenRegressor`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class BayesianRidgeDF(RegressorDF, BayesianRidge):
    """
    Wraps :class:`sklearn.linear_model.bayes.BayesianRidge`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class ARDRegressionDF(RegressorDF, ARDRegression):
    """
    Wraps :class:`sklearn.linear_model.bayes.ARDRegression`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class OrthogonalMatchingPursuitDF(RegressorDF, OrthogonalMatchingPursuit):
    """
    Wraps :class:`sklearn.linear_model.omp.OrthogonalMatchingPursuit`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class OrthogonalMatchingPursuitCVDF(RegressorDF, OrthogonalMatchingPursuitCV):
    """
    Wraps :class:`sklearn.linear_model.omp.OrthogonalMatchingPursuitCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class RANSACRegressorDF(RegressorDF, RANSACRegressor):
    """
    Wraps :class:`sklearn.linear_model.ransac.RANSACRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class ElasticNetDF(RegressorDF, ElasticNet):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.ElasticNet`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class LassoCVDF(RegressorDF, LassoCV):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.LassoCV`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class ElasticNetCVDF(RegressorDF, ElasticNetCV):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.ElasticNetCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class MultiTaskElasticNetCVDF(RegressorDF, MultiTaskElasticNetCV):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskElasticNetCV`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class MultiTaskLassoCVDF(RegressorDF, MultiTaskLassoCV):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskLassoCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class MultiTaskElasticNetDF(RegressorDF, MultiTaskElasticNet):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskElasticNet`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class MultiTaskLassoDF(RegressorDF, MultiTaskLasso):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskLasso`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class LassoDF(RegressorDF, Lasso):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.Lasso`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class PassiveAggressiveRegressorDF(RegressorDF, PassiveAggressiveRegressor):
    """
    Wraps :class:`sklearn.linear_model.passive_aggressive.PassiveAggressiveRegressor`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class LarsDF(RegressorDF, Lars):
    """
    Wraps :class:`sklearn.linear_model.least_angle.Lars`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class LassoLarsDF(RegressorDF, LassoLars):
    """
    Wraps :class:`sklearn.linear_model.least_angle.LassoLars`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class LassoLarsICDF(RegressorDF, LassoLarsIC):
    """
    Wraps :class:`sklearn.linear_model.least_angle.LassoLarsIC`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class LarsCVDF(RegressorDF, LarsCV):
    """
    Wraps :class:`sklearn.linear_model.least_angle.LarsCV`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class LassoLarsCVDF(RegressorDF, LassoLarsCV):
    """
    Wraps :class:`sklearn.linear_model.least_angle.LassoLarsCV`; accepts and returns
    data frames.
    """

    pass


#
# ensemble
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class BaggingRegressorDF(RegressorDF, BaggingRegressor):
    """
    Wraps :class:`sklearn.ensemble.bagging.BaggingRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_MetaRegressorWrapperDF)
class VotingRegressorDF(RegressorDF, VotingRegressor):
    """
    Wraps :class:`sklearn.ensemble.voting.VotingRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class GradientBoostingRegressorDF(RegressorDF, GradientBoostingRegressor):
    """
    Wraps :class:`sklearn.ensemble.gradient_boosting.GradientBoostingRegressor`; accepts
    and returns data frames.
    """


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class AdaBoostRegressorDF(RegressorDF, AdaBoostRegressor):
    """
    Wraps :class:`sklearn.ensemble.weight_boosting.AdaBoostRegressor`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class RandomForestRegressorDF(RegressorDF, RandomForestRegressor):
    """
    Wraps :class:`sklearn.ensemble.forest.RandomForestRegressor`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class ExtraTreesRegressorDF(RegressorDF, ExtraTreesRegressor):
    """
    Wraps :class:`sklearn.ensemble.forest.ExtraTreesRegressor`; accepts and returns data
    frames.
    """

    pass


#
# gaussian_process
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class GaussianProcessRegressorDF(RegressorDF, GaussianProcessRegressor):
    """
    Wraps :class:`sklearn.gaussian_process.gpr.GaussianProcessRegressor`; accepts and
    returns data frames.
    """

    pass


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


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_IsotonicRegressionWrapperDF)
class IsotonicRegressionDF(RegressorDF, TransformerDF, IsotonicRegression):
    """
    Wraps :class:`sklearn.isotonic.IsotonicRegression`; accepts and returns data frames.
    """

    pass


#
# compose
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class TransformedTargetRegressorDF(RegressorDF, TransformedTargetRegressor):
    """
    Wraps :class:`sklearn.compose._target.TransformedTargetRegressor`; accepts and
    returns data frames.
    """

    pass


#
# kernel_ridge
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class KernelRidgeDF(RegressorDF, KernelRidge):
    """
    Wraps :class:`sklearn.kernel_ridge.KernelRidge`; accepts and returns data frames.
    """

    pass


#
# tree
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class DecisionTreeRegressorDF(RegressorDF, DecisionTreeRegressor):
    """
    Wraps :class:`sklearn.tree.tree.DecisionTreeRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class ExtraTreeRegressorDF(RegressorDF, ExtraTreeRegressor):
    """
    Wraps :class:`sklearn.tree.tree.ExtraTreeRegressor`; accepts and returns data
    frames.
    """

    pass


#
# cross_decomposition
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorTransformerWrapperDF)
class CCADF(RegressorDF, TransformerDF, CCA):
    """
    Wraps :class:`sklearn.cross_decomposition.cca_.CCA`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorTransformerWrapperDF)
class PLSRegressionDF(RegressorDF, TransformerDF, PLSRegression):
    """
    Wraps :class:`sklearn.cross_decomposition.pls_.PLSRegression`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorTransformerWrapperDF)
class PLSCanonicalDF(RegressorDF, TransformerDF, PLSCanonical):
    """
    Wraps :class:`sklearn.cross_decomposition.pls_.PLSCanonical`; accepts and returns
    data frames.
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
