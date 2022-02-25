from typing import Type

import pandas as pd
import pytest

import sklearndf.clustering
from sklearndf import ClustererDF
from sklearndf.clustering import FeatureAgglomerationDF
from test.sklearndf import iterate_classes

CLUSTERERS_TO_TEST = iterate_classes(
    from_modules=sklearndf.clustering,
    matching=r".*DF",
    excluding=[ClustererDF.__name__, r".*WrapperDF", FeatureAgglomerationDF.__name__],
)
# FeatureAgglomeration doesn't support `fit_predict` method
CLUSTERERS_WITH_AGGLOMERATION = CLUSTERERS_TO_TEST + [FeatureAgglomerationDF]


@pytest.mark.parametrize(argnames="sklearn_clusterer_cls", argvalues=CLUSTERERS_TO_TEST)
def test_clusterer_fit_predict_call(
    iris_features: pd.DataFrame, sklearn_clusterer_cls: Type
) -> None:
    """ Check if each sklearndf clusterer supports fit_predict method"""

    clusterer_instance = sklearn_clusterer_cls()

    assert not clusterer_instance.is_fitted
    result_prediction = clusterer_instance.fit_predict(iris_features)
    assert type(result_prediction) == pd.Series
    assert clusterer_instance.is_fitted


@pytest.mark.parametrize(
    argnames="sklearn_clusterer_cls", argvalues=CLUSTERERS_WITH_AGGLOMERATION
)
def test_clusterer_fit_call(
    iris_features: pd.DataFrame, sklearn_clusterer_cls: Type
) -> None:
    """ Check if each sklearndf clusterer supports fit method"""

    clusterer_instance = sklearn_clusterer_cls()

    assert not clusterer_instance.is_fitted
    clusterer_instance.fit(iris_features)
    assert clusterer_instance.is_fitted
