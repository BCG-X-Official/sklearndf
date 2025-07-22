"""
Core implementation of :mod:`sklearndf.clustering`
"""

import logging

from sklearn.cluster import HDBSCAN

from pytools.api import AllTracker

from ..wrapper import ClusterWrapperDF

log = logging.getLogger(__name__)

__all__ = [
    "HDBSCANDF",
]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class HDBSCANDF(ClusterWrapperDF[HDBSCAN], native=HDBSCAN):
    """Stub for DF wrapper of class ``DBSCAN``"""


#
# Validate __all__
#

__tracker.validate()
