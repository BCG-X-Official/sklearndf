"""
Core implementation of :mod:`sklearndf.transformation.extra.wrapper`
"""

from __future__ import annotations

import logging
from typing import Generic, TypeVar

import pandas as pd
from sklearn.feature_selection import SelectorMixin

from pytools.api import AllTracker

from ....wrapper import MetaEstimatorWrapperDF
from ...wrapper import ColumnSubsetTransformerWrapperDF, NumpyTransformerWrapperDF

log = logging.getLogger(__name__)

__all__ = ["BorutaPyWrapperDF", "ARFSWrapperDF"]

try:
    # import boruta classes only if installed
    from boruta import BorutaPy
except ImportError:
    BorutaPy = None


#
# Type variables
#

T_FeatureSelector = TypeVar("T_FeatureSelector", bound=SelectorMixin)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


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

    def _get_sparse_threshold(self) -> float:
        # don't allow sparse input
        return 0.0


class ARFSWrapperDF(
    MetaEstimatorWrapperDF[T_FeatureSelector],
    ColumnSubsetTransformerWrapperDF[T_FeatureSelector],
    Generic[T_FeatureSelector],
):
    """
    DF wrapper for :class:`~boruta.BorutaPy`.
    """

    def _get_features_out(self) -> pd.Index:
        return self.feature_names_in_[self.native_estimator.support_]

    def _get_sparse_threshold(self) -> float:
        # don't allow sparse input
        return 0.0


__tracker.validate()
