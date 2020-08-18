"""
Extended versions of all Scikit-Learn regressors with enhanced E2E support for data
frames.
"""
from packaging.version import parse as __parse_version
from sklearn import __version__ as __sklearn_version__

from ._regression import *

if __parse_version(__sklearn_version__) >= __parse_version("0.22"):
    from ._regression_v0_22 import *

if __parse_version(__sklearn_version__) >= __parse_version("0.23"):
    from ._regression_v0_23 import *

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
    "StackingRegressorDF",
    "GammaRegressorDF",
    "GeneralizedLinearRegressorDF",
    "PoissonRegressorDF",
    "TweedieRegressorDF",
]
