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

UNSUPPORTED_SKLEARN_PACKAGES = [
    sklearn.manifold,
    sklearn.neighbors,
    sklearn.feature_extraction.image,
]


@pytest.fixture  # type: ignore
def diabetes_target() -> str:
    return "disease_progression_1yr"


@pytest.fixture  # type: ignore
def iris_target_name() -> str:
    return "species"


@pytest.fixture  # type: ignore
def n_jobs() -> int:
    return -3


@pytest.fixture  # type: ignore
def diabetes_df(diabetes_target: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    diabetes: Bunch
    diabetes = datasets.load_diabetes(scaled=False)

    return pd.DataFrame(
        data=np.c_[diabetes.data, diabetes.target],
        columns=[*map(str, diabetes.feature_names), diabetes_target],
    ).astype(dtype={"sex": "category"})


@pytest.fixture  # type: ignore
def diabetes_features(diabetes_df: pd.DataFrame, diabetes_target: str) -> pd.DataFrame:
    return diabetes_df.drop(labels=[diabetes_target], axis=1)


@pytest.fixture  # type: ignore
def diabetes_target_sr(diabetes_df: pd.DataFrame, diabetes_target: str) -> pd.Series:
    return diabetes_df.loc[:, diabetes_target]


@pytest.fixture  # type: ignore
def diabetes_target_df(diabetes_df: pd.DataFrame, diabetes_target: str) -> pd.DataFrame:
    target = diabetes_df.loc[:, [diabetes_target]]
    target.loc[:, f"{diabetes_target}_2"] = target.loc[:, diabetes_target] * 2
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
    return (
        OneHotEncoderDF(sparse_output=False)
        .fit_transform(X=iris_target_sr.to_frame())
        .astype(int)
    )


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
