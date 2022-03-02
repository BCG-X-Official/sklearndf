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

from ..wrapper import make_df_clusterer
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

AffinityPropagationDF = make_df_clusterer(AffinityPropagation)
AgglomerativeClusteringDF = make_df_clusterer(AgglomerativeClustering)
BirchDF = make_df_clusterer(Birch)
DBSCANDF = make_df_clusterer(DBSCAN)
KMeansDF = make_df_clusterer(KMeans, base_wrapper=KMeansBaseWrapperDF)
MiniBatchKMeansDF = make_df_clusterer(MiniBatchKMeans, base_wrapper=KMeansBaseWrapperDF)
MeanShiftDF = make_df_clusterer(MeanShift)
OPTICSDF = make_df_clusterer(OPTICS)
SpectralClusteringDF = make_df_clusterer(SpectralClustering)
FeatureAgglomerationDF = make_df_clusterer(
    FeatureAgglomeration, base_wrapper=FeatureAgglomerationWrapperDF
)

#
# Validate __all__
#

__tracker.validate()
