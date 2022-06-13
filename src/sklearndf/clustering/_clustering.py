"""
Core implementation of :mod:`sklearndf.clustering`
"""
import logging

from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    FeatureAgglomeration,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
)

from pytools.api import AllTracker

from ..wrapper import ClustererWrapperDF
from .wrapper import FeatureAgglomerationWrapperDF, KMeansBaseWrapperDF

log = logging.getLogger(__name__)

__all__ = [
    "AffinityPropagationDF",
    "AgglomerativeClusteringDF",
    "BirchDF",
    "DBSCANDF",
    "FeatureAgglomerationDF",
    "KMeansDF",
    "MiniBatchKMeansDF",
    "MeanShiftDF",
    "OPTICSDF",
    "SpectralClusteringDF",
]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals(), allow_imported_definitions=True)


#
# Class definitions
#


class AffinityPropagationDF(
    ClustererWrapperDF, AffinityPropagation, native=AffinityPropagation
):
    """Stub for DF wrapper of class ``AffinityPropagation``"""


class AgglomerativeClusteringDF(
    ClustererWrapperDF, AgglomerativeClustering, native=AgglomerativeClustering
):
    """Stub for DF wrapper of class ``AgglomerativeClustering``"""


class BirchDF(ClustererWrapperDF, Birch, native=Birch):
    """Stub for DF wrapper of class ``Birch``"""


class DBSCANDF(ClustererWrapperDF, DBSCAN, native=DBSCAN):
    """Stub for DF wrapper of class ``DBSCAN``"""


class KMeansDF(KMeansBaseWrapperDF, KMeans, native=KMeans):
    """Stub for DF wrapper of class ``KMeans``"""


class MiniBatchKMeansDF(KMeansBaseWrapperDF, MiniBatchKMeans, native=MiniBatchKMeans):
    """Stub for DF wrapper of class ``MiniBatchKMeans``"""


class MeanShiftDF(ClustererWrapperDF, MeanShift, native=MeanShift):
    """Stub for DF wrapper of class ``MeanShift``"""


class OPTICSDF(ClustererWrapperDF, OPTICS, native=OPTICS):
    """Stub for DF wrapper of class ``OPTICS``"""


class SpectralClusteringDF(
    ClustererWrapperDF, SpectralClustering, native=SpectralClustering
):
    """Stub for DF wrapper of class ``SpectralClustering``"""


class FeatureAgglomerationDF(
    FeatureAgglomerationWrapperDF, FeatureAgglomeration, native=FeatureAgglomeration
):
    """Stub for DF wrapper of class ``FeatureAgglomeration``"""


#
# Validate __all__
#

__tracker.validate()
