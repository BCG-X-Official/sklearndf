"""
Core implementation of :mod:`sklearndf.pipeline`
"""

import logging
from typing import Type

from sklearn.pipeline import FeatureUnion, Pipeline

from pytools.api import AllTracker

from ..wrapper import EstimatorWrapperDF, make_df_estimator, make_df_transformer
from .wrapper import FeatureUnionWrapperDF, PipelineWrapperDF

log = logging.getLogger(__name__)

__all__ = ["PipelineDF", "FeatureUnionDF"]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals(), allow_imported_definitions=True)


#
# Class definitions
#


PipelineDF: Type[EstimatorWrapperDF[Pipeline]] = make_df_estimator(
    Pipeline, base_wrapper=PipelineWrapperDF
)
FeatureUnionDF: Type[EstimatorWrapperDF[FeatureUnion]] = make_df_transformer(
    FeatureUnion, base_wrapper=FeatureUnionWrapperDF
)


#
# Validate __all__
#

__tracker.validate()
