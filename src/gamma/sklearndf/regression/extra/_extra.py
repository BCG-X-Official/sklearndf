"""
Core implementation of :mod:`gamma.ml.regression.extra`
"""
import logging

from lightgbm.sklearn import LGBMRegressor

from gamma.sklearndf import RegressorDF
from gamma.sklearndf._wrapper import df_estimator, RegressorWrapperDF

log = logging.getLogger(__name__)

#
# lightgbm
#

# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=RegressorWrapperDF)
class LGBMRegressorDF(RegressorDF, LGBMRegressor):
    """
    Wraps :class:`lightgbm.sklearn.LGBMRegressor`; accepts and returns data frames.
    """

    pass


#
# export all symbols ending in "DF"
#

__all__ = [sym for sym in dir() if sym.endswith("DF") and not sym.startswith("_")]
