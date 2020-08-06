"""
Core implementation of :mod:`sklearndf.regression.extra`
"""
import logging
import warnings

from ... import ClassifierDF, RegressorDF
from ..._wrapper import (
    _ClassifierWrapperDF,
    _RegressorWrapperDF,
    df_estimator,
)

log = logging.getLogger(__name__)

__all__ = ["LGBMRegressorDF", "LGBMClassifierDF"]
__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

#
# lightgbm
#


# since we install LGBM via conda, the warning about the Clang compiler is irrelevant
warnings.filterwarnings("ignore", message=r"Starting from version 2\.2\.1")
# cross-validation will invariably generate sliced subsets, so the following warning
# is not helpful
warnings.filterwarnings(
    "ignore", message=r"Usage of np\.ndarray subset \(sliced data\) is not recommended"
)
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier

# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_RegressorWrapperDF)
class LGBMRegressorDF(RegressorDF, LGBMRegressor):
    """
    Wraps :class:`lightgbm.sklearn.LGBMRegressor`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class LGBMClassifierDF(ClassifierDF, LGBMClassifier):
    """
    Wraps :class:`lightgbm.sklearn.LGBMClassifier`; accepts and returns data frames.
    """

    pass


#
# validate that __all__ comprises all symbols ending in "DF", and no others
#

__estimators = [
    sym
    for sym in dir()
    if sym.endswith("DF")
    and sym not in __imported_estimators
    and not sym.startswith("_")
]
if set(__estimators) != set(__all__):
    raise RuntimeError(
        "__all__ does not contain exactly all DF estimators; expected value is:\n"
        f"{__estimators}"
    )
