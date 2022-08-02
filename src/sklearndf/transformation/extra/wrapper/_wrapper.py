"""
Core implementation of :mod:`sklearndf.transformation.extra.wrapper`
"""
from __future__ import annotations

import logging

import pandas as pd

from pytools.api import AllTracker

from ....wrapper import MetaEstimatorWrapperDF
from ...wrapper import ColumnSubsetTransformerWrapperDF, NumpyTransformerWrapperDF

log = logging.getLogger(__name__)

__all__ = ["BorutaPyWrapperDF"]

try:
    # import boruta classes only if installed
    from boruta import BorutaPy
except ImportError:
    BorutaPy = None

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())

#
# Class definitions
#

if BorutaPy:

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

else:
    __all__.remove("BorutaPyWrapperDF")

__tracker.validate()
