from typing import Type

import pandas as pd
import pytest

import sklearndf.clustering
from sklearndf import ClusterDF, __sklearn_1_1__, __sklearn_version__
from sklearndf.clustering import FeatureAgglomerationDF
from test.sklearndf import iterate_classes

# noinspection PyTypeChecker
CLUSTERERS_TO_TEST = iterate_classes(
    from_modules=sklearndf.clustering,
    matching=r".*DF",
    excluding=[ClusterDF.__name__, r".*WrapperDF", FeatureAgglomerationDF.__name__],
)
# FeatureAgglomeration doesn't support `fit_predict` method
CLUSTERERS_WITH_AGGLOMERATION = CLUSTERERS_TO_TEST + [FeatureAgglomerationDF]


def test_clusterer_count() -> None:
    n = len(CLUSTERERS_TO_TEST)

    print(f"Testing {n} clusterers.")

    if __sklearn_version__ < __sklearn_1_1__:
        assert n == 9
    else:
        assert n == 10


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearn_clusterer_cls", argvalues=CLUSTERERS_TO_TEST
)
def test_clusterer_fit_predict_call(
    iris_features: pd.DataFrame, sklearn_clusterer_cls: Type[ClusterDF]
) -> None:
    """Check if each sklearndf clusterer supports fit_predict method"""

    clusterer_instance = sklearn_clusterer_cls()

    assert not clusterer_instance.is_fitted
    result_prediction = clusterer_instance.fit_predict(iris_features)
    assert type(result_prediction) == pd.Series
    assert clusterer_instance.is_fitted


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearn_clusterer_cls", argvalues=CLUSTERERS_WITH_AGGLOMERATION
)
def test_clusterer_fit_call(
    iris_features: pd.DataFrame, sklearn_clusterer_cls: Type[ClusterDF]
) -> None:
    """Check if each sklearndf clusterer supports fit method"""

    clusterer_instance = sklearn_clusterer_cls()

    assert not clusterer_instance.is_fitted
    clusterer_instance.fit(iris_features)
    assert clusterer_instance.is_fitted
