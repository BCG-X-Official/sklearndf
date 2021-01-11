import itertools
import logging
from typing import Type

import numpy as np
import pandas as pd
import pytest

import sklearndf.transformation
from sklearndf import TransformerDF
from test.sklearndf import list_classes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

IMPUTERS_TO_TEST = list_classes(
    from_modules=sklearndf.transformation, matching=r".*Imputer*DF", excluding=[]
)


@pytest.fixture
def test_data_x() -> pd.DataFrame:
    return pd.DataFrame(
        data=[[7, 2, 3], [4, np.nan, 6], [10, 5, 9]], columns=["a", "b", "c"]
    )


@pytest.fixture
def test_data_x_with_all_nan() -> pd.DataFrame:
    return pd.DataFrame(
        data=[[7, np.nan, 3], [4, np.nan, 6], [np.nan, np.nan, np.nan]],
        columns=["a", "b", "c"],
    )


@pytest.fixture
def test_data_y() -> pd.DataFrame:
    return pd.DataFrame(
        data=[[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]], columns=["a", "b", "c"]
    )


@pytest.mark.parametrize(
    argnames=["imputer_cls", "add_indicator"],
    argvalues=itertools.product(IMPUTERS_TO_TEST, (True, False)),
)
def test_imputer(
    imputer_cls: Type[TransformerDF],
    add_indicator: bool,
    test_data_x: pd.DataFrame,
    test_data_y: pd.DataFrame,
    test_data_x_with_all_nan: pd.DataFrame,
) -> None:
    imputerdf = imputer_cls(add_indicator=add_indicator)
    imputer_cls_orig = type(imputerdf.native_estimator)

    # noinspection PyArgumentList
    imputer_orig = imputer_cls_orig(add_indicator=add_indicator)
    # noinspection PyUnresolvedReferences
    imputer_orig.fit(test_data_x.values)
    # noinspection PyUnresolvedReferences
    y_transformed = imputer_orig.transform(test_data_y)

    imputerdf.fit(test_data_x)
    y_transformed_df = imputerdf.transform(test_data_y)

    assert np.array_equal(
        np.round(y_transformed, 4), np.round(y_transformed_df.values, 4)
    ), (
        f"Different imputation results! "
        f"sklearn:{y_transformed} "
        f"sklearndf: {y_transformed_df.values}"
    )

    # test correct imputation (and returned column labels)
    # for the case when a full input series is NaN
    # noinspection PyUnresolvedReferences
    imputer_orig.fit(test_data_x_with_all_nan.values)
    # noinspection PyUnresolvedReferences
    y_transformed = imputer_orig.transform(test_data_y)

    imputerdf.fit(test_data_x_with_all_nan)
    y_transformed_df = imputerdf.transform(test_data_y)

    assert np.array_equal(
        np.round(y_transformed, 4), np.round(y_transformed_df.values, 4)
    ), (
        f"Different imputation results! "
        f"sklearn:{y_transformed} "
        f"sklearndf: {y_transformed_df.values}"
    )
