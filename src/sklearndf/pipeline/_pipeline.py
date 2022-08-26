"""
Core implementation of :mod:`sklearndf.pipeline`
"""

import logging

from sklearn.pipeline import FeatureUnion, Pipeline

from pytools.api import AllTracker

from .wrapper import FeatureUnionWrapperDF, PipelineWrapperDF

log = logging.getLogger(__name__)

__all__ = ["PipelineDF", "FeatureUnionDF"]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class PipelineDF(PipelineWrapperDF, native=Pipeline):
    """Stub for DF wrapper of class ``Pipeline``"""


class FeatureUnionDF(
    FeatureUnionWrapperDF,
    native=FeatureUnion,
):
    """Stub for DF wrapper of class ``FeatureUnion``"""


#
# Validate __all__
#

__tracker.validate()
