"""
Core implementation of :mod:`sklearndf.transformation.extra`
"""
from __future__ import annotations

import logging

import pandas as pd
from boruta import BorutaPy

from pytools.api import AllTracker

from ...wrapper import MetaEstimatorWrapperDF, make_df_transformer
from ..wrapper import ColumnSubsetTransformerWrapperDF, NumpyTransformerWrapperDF

log = logging.getLogger(__name__)

__all__ = ["BorutaPyWrapperDF", "BorutaDF"]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals(), allow_imported_definitions=True)


#
# Class definitions
#


class BorutaPyWrapperDF(
    MetaEstimatorWrapperDF[BorutaPy],
    NumpyTransformerWrapperDF[BorutaPy],
    ColumnSubsetTransformerWrapperDF[BorutaPy],
):
    """
    DF wrapper for :class:`~boruta.BorutaPy`.
    """

    def _get_features_out(self) -> pd.Index:
        return self.feature_names_in_[self.native_estimator.support_]


BorutaDF = make_df_transformer(
    BorutaPy, name="BorutaDF", base_wrapper=BorutaPyWrapperDF
)

__tracker.validate()
