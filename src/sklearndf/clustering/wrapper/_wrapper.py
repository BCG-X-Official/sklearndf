"""
Core implementation of :mod:`sklearndf.clustering.wrapper`
"""

import logging
from abc import ABCMeta
from typing import Generic, TypeVar

import pandas as pd
from sklearn.cluster import FeatureAgglomeration, KMeans, MiniBatchKMeans

from pytools.api import AllTracker

from sklearndf.transformation.wrapper import ColumnPreservingTransformerWrapperDF
from sklearndf.wrapper import ClusterWrapperDF

log = logging.getLogger(__name__)

__all__ = [
    "KMeansBaseWrapperDF",
    "FeatureAgglomerationWrapperDF",
]

#
# Type variables
#

T_NativeKMeans = TypeVar("T_NativeKMeans", KMeans, MiniBatchKMeans)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Wrapper classes
#


# noinspection PyPep8Naming
class KMeansBaseWrapperDF(
    ClusterWrapperDF[T_NativeKMeans], Generic[T_NativeKMeans], metaclass=ABCMeta
):
    """
    DF wrapper for KMeans-like algorithms, e.g., :class:`sklearn.cluster.KMeans`.
    """

    #: the name of the index representing clusters
    IDX_CLUSTER = "cluster"

    @property
    def cluster_centers_(self) -> pd.DataFrame:
        """
        The cluster centers as a data frame, with clusters as rows and feature values
        as columns.
        """

        self.ensure_fitted()

        raw_cluster_centers = self._native_estimator.cluster_centers_
        return pd.DataFrame(
            raw_cluster_centers,
            columns=self.feature_names_in_,
            index=pd.RangeIndex(
                len(raw_cluster_centers), name=KMeansBaseWrapperDF.IDX_CLUSTER
            ),
        )


class FeatureAgglomerationWrapperDF(
    ClusterWrapperDF[FeatureAgglomeration],
    ColumnPreservingTransformerWrapperDF[FeatureAgglomeration],
    metaclass=ABCMeta,
):
    """
    DF wrapper for FeatureAgglomeration that combines clusterer and transformer.
    """

    pass


#
# Validate __all__
#

__tracker.validate()
