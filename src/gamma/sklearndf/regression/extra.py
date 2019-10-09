#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
Additional sklearndf regressors from Gamma or 3rd parties.
"""

import logging

from lightgbm.sklearn import LGBMRegressor

from gamma.sklearndf import RegressorDF
from gamma.sklearndf.wrapper import df_estimator, RegressorWrapperDF

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
