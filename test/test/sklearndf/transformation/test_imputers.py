import itertools
import logging
from typing import Type

import numpy as np
import pandas as pd
import pytest

import sklearndf.transformation
from sklearndf import TransformerDF
from test.sklearndf import iterate_classes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

IMPUTERS_TO_TEST = iterate_classes(
    from_modules=sklearndf.transformation, matching=r".*Imputer.*DF", excluding=[]
)


@pytest.mark.parametrize(  # type: ignore
    argnames=["imputer_cls", "add_indicator"],
    argvalues=itertools.product(IMPUTERS_TO_TEST, [True, False]),
)
def test_imputer(
    imputer_cls: Type[TransformerDF],
    add_indicator: bool,
) -> None:
    """
    Test imputer classes using the combinations of arguments from
    ``@pytest.mark.parametrize``

    :param imputer_cls: the imputer class to test
    :param add_indicator: whether to add an indicator column
    :return:
    """
    imputer_df = imputer_cls(add_indicator=add_indicator)
    imputer_cls_orig = type(imputer_df.native_estimator)

    test_data_x = pd.DataFrame(
        data=[[7, 2, 3], [4, np.nan, 6], [10, 5, 9]], columns=["a", "b", "c"]
    )
    test_data_x_with_all_nan = pd.DataFrame(
        data=[[7, np.nan, 3], [4, np.nan, 6], [np.nan, np.nan, np.nan]],
        columns=["a", "b", "c"],
    )
    test_data_y = pd.DataFrame(
        data=[[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]], columns=["a", "b", "c"]
    )

    # noinspection PyArgumentList
    imputer_orig = imputer_cls_orig(add_indicator=add_indicator)
    # noinspection PyUnresolvedReferences
    imputer_orig.fit(test_data_x.values)
    # noinspection PyUnresolvedReferences
    y_transformed = imputer_orig.transform(test_data_y)

    imputer_df.fit(test_data_x)
    y_transformed_df = imputer_df.transform(test_data_y)

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

    imputer_df.fit(test_data_x_with_all_nan)
    y_transformed_df = imputer_df.transform(test_data_y)

    assert np.array_equal(
        np.round(y_transformed, 4), np.round(y_transformed_df.values, 4)
    ), (
        f"Different imputation results! "
        f"sklearn:{y_transformed} "
        f"sklearndf: {y_transformed_df.values}"
    )
