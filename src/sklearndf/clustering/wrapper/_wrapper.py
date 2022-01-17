"""
Core implementation of :mod:`sklearndf.clustering.wrapper`
"""

import logging
from abc import ABCMeta
from typing import Generic, TypeVar

import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from pytools.api import AllTracker

from sklearndf.wrapper import ClustererWrapperDF

log = logging.getLogger(__name__)

__all__ = [
    "KMeansBaseWrapperDF",
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
    ClustererWrapperDF[T_NativeKMeans], Generic[T_NativeKMeans], metaclass=ABCMeta
):
    """
    DF wrapper for KMeans-like algorithms, e.g., :class:`sklearn.cluster.KMeans`.
    """

    @property
    def cluster_centers_(self) -> pd.DataFrame:
        """
        The data frame containing cluster centers designated by a clustering algorithm.
        In the resulting data frame each row is a cluster center decided by KMeans and
        columns are the same as input data frame used for training.
        """
        COL_CLUSTER = "cluster"

        self._ensure_fitted()
        raw_cluster_centers = self._native_estimator.cluster_centers_

        return pd.DataFrame(
            data=raw_cluster_centers, columns=self.feature_names_in_
        ).rename_axis(index=COL_CLUSTER)


#
# Validate __all__
#

__tracker.validate()
