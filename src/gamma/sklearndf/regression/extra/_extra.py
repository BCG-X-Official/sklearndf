"""
Core implementation of :mod:`gamma.ml.regression.extra`
"""
import logging
import warnings

from gamma.sklearndf import RegressorDF
from gamma.sklearndf._wrapper import RegressorWrapperDF, df_estimator

log = logging.getLogger(__name__)

__all__ = ["LGBMRegressorDF"]
__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

#
# lightgbm
#


warnings.filterwarnings(
    "ignore", message=r"Starting from version 2\.2\.1", category=UserWarning
)
from lightgbm.sklearn import LGBMRegressor

# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=RegressorWrapperDF)
class LGBMRegressorDF(RegressorDF, LGBMRegressor):
    """
    Wraps :class:`lightgbm.sklearn.LGBMRegressor`; accepts and returns data frames.
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
