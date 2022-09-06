import logging

import numpy as np
import pandas as pd

# noinspection PyPackageRequirements
import pytest
import sklearn
from sklearn import datasets
from sklearn.utils import Bunch

from sklearndf.transformation import OneHotEncoderDF

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# todo: Keep this up to date, consider implementing manifold/neighbors
UNSUPPORTED_SKLEARN_PACKAGES = [sklearn.manifold, sklearn.neighbors]


@pytest.fixture  # type: ignore
def boston_target() -> str:
    return "price"


@pytest.fixture  # type: ignore
def iris_target_name() -> str:
    return "species"


@pytest.fixture  # type: ignore
def n_jobs() -> int:
    return -3


@pytest.fixture  # type: ignore
def boston_df(boston_target: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    boston: Bunch = datasets.load_boston()

    return pd.DataFrame(
        data=np.c_[boston.data, boston.target],
        columns=[*map(str, boston.feature_names), boston_target],
    )


@pytest.fixture  # type: ignore
def boston_features(boston_df: pd.DataFrame, boston_target: str) -> pd.DataFrame:
    return boston_df.drop(labels=[boston_target], axis=1)


@pytest.fixture  # type: ignore
def boston_target_sr(boston_df: pd.DataFrame, boston_target: str) -> pd.Series:
    return boston_df.loc[:, boston_target]


@pytest.fixture  # type: ignore
def boston_target_df(boston_df: pd.DataFrame, boston_target: str) -> pd.DataFrame:
    target = boston_df.loc[:, [boston_target]]
    target.loc[:, f"{boston_target}_2"] = target.loc[:, boston_target] * 2
    return target


@pytest.fixture  # type: ignore
def iris_dataset() -> Bunch:
    return datasets.load_iris()


@pytest.fixture  # type: ignore
def iris_df(iris_dataset: Bunch, iris_target_name: str) -> pd.DataFrame:
    #  convert sklearn iris data set to data frame
    return pd.DataFrame(
        data=np.c_[iris_dataset.data, iris_dataset.target],
        columns=[*map(str, iris_dataset.feature_names), iris_target_name],
    )


@pytest.fixture  # type: ignore
def iris_features(iris_df: pd.DataFrame, iris_target_name: str) -> pd.DataFrame:
    return iris_df.drop(labels=[iris_target_name], axis=1)


@pytest.fixture  # type: ignore
def iris_target_sr(
    iris_dataset: Bunch, iris_df: pd.DataFrame, iris_target_name: str
) -> pd.Series:
    # replace numerical targets with actual class labels
    return iris_df.loc[:, iris_target_name].apply(
        lambda x: iris_dataset.target_names[int(x)]
    )


@pytest.fixture  # type: ignore
def iris_targets_df(iris_df: pd.DataFrame, iris_target_name: str) -> pd.DataFrame:
    return iris_df.loc[:, [iris_target_name, iris_target_name]]


@pytest.fixture  # type: ignore
def iris_targets_binary_df(iris_target_sr: pd.Series) -> pd.DataFrame:
    return OneHotEncoderDF(sparse=False).fit_transform(X=iris_target_sr.to_frame())


@pytest.fixture  # type:ignore
def test_data_categorical() -> pd.DataFrame:
    return pd.DataFrame(
        data=[
            ["yes", "red", "child"],
            ["yes", "blue", "father"],
            ["no", "green", "mother"],
        ],
        columns=["a", "b", "c"],
    )
