import logging

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.utils import Bunch

from test import read_test_config
from test.paths import TEST_DATA_CSV

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


@pytest.fixture
def boston_target() -> str:
    return "target"


@pytest.fixture
def iris_target() -> str:
    return "target"


@pytest.fixture
def n_jobs() -> int:
    return -3


@pytest.fixture
def batch_table() -> pd.DataFrame:

    # Note: this file is not included within the git repository!
    inputfile_config = read_test_config(section="inputfile")
    return pd.read_csv(
        filepath_or_buffer=TEST_DATA_CSV,
        delimiter=inputfile_config["delimiter"],
        header=inputfile_config["header"],
        decimal=inputfile_config["decimal"],
    )


@pytest.fixture
def boston_df(boston_target: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    boston: Bunch = datasets.load_boston()

    return pd.DataFrame(
        data=np.c_[boston.data, boston.target],
        columns=[*boston.feature_names, boston_target],
    )


@pytest.fixture
def boston_features(boston_df: pd.DataFrame, boston_target: str) -> pd.DataFrame:
    return boston_df.drop(labels=[boston_target], axis=1)


@pytest.fixture
def boston_target_sr(boston_df: pd.DataFrame, boston_target: str) -> pd.Series:
    return boston_df.loc[:, boston_target]


@pytest.fixture
def iris_df(iris_target: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    iris: Bunch = datasets.load_iris()

    iris_df = pd.DataFrame(
        data=np.c_[iris.data, iris.target], columns=[*iris.feature_names, iris_target]
    )

    # replace target numericals with actual class labels
    iris_df.loc[:, iris_target] = iris_df.loc[:, iris_target].apply(
        lambda x: iris.target_names[int(x)]
    )

    return iris_df


@pytest.fixture
def iris_features(iris_df: pd.DataFrame, iris_target: str) -> pd.DataFrame:
    return iris_df.drop(labels=[iris_target], axis=1)


@pytest.fixture
def iris_target_sr(iris_df: pd.DataFrame, iris_target: str) -> pd.Series:
    return iris_df.loc[:, iris_target]
