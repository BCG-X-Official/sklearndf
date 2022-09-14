"""
Core implementation of :mod:`sklearndf.clustering`
"""
import logging

from sklearn.cluster import BisectingKMeans

from pytools.api import AllTracker

from .wrapper import KMeansBaseWrapperDF

log = logging.getLogger(__name__)

__all__ = [
    "BisectingKMeansDF",
]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class BisectingKMeansDF(KMeansBaseWrapperDF[BisectingKMeans], native=BisectingKMeans):
    """Stub for DF wrapper of class ``MiniBatchKMeans``"""


#
# Validate __all__
#

__tracker.validate()
