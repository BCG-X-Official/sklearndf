import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

from sklearndf.clustering import KMeansDF
from sklearndf.pipeline import ClusterPipelineDF
from test.sklearndf.pipeline import make_simple_transformer


def test_clustering_pipeline_df(
    iris_features: pd.DataFrame, iris_target_sr: pd.DataFrame
) -> None:

    cls_p_df = ClusterPipelineDF(
        clusterer=KMeansDF(n_clusters=4),
        preprocessing=make_simple_transformer(
            impute_median_columns=iris_features.select_dtypes(
                include=np.number
            ).columns,
            one_hot_encode_columns=iris_features.select_dtypes(include=object).columns,
        ),
    )

    cls_p_df.fit(X=iris_features, y=iris_target_sr)
    cls_p_df.predict(X=iris_features)

    # test-type check within constructor:
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        ClusterPipelineDF(clusterer=KMeans(n_clusters=4), preprocessing=OneHotEncoder())
