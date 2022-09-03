from typing import Type

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.base import is_regressor
from sklearn.preprocessing import OneHotEncoder

from sklearndf import RegressorDF
from sklearndf.pipeline import RegressorPipelineDF
from sklearndf.regression import RandomForestRegressorDF
from sklearndf.regression.extra import LGBMRegressorDF
from test.sklearndf.pipeline import make_simple_transformer


@pytest.mark.parametrize(  # type: ignore
    argnames="regressor_df_cls", argvalues=[RandomForestRegressorDF, LGBMRegressorDF]
)
def test_regression_pipeline_df(
    boston_features: pd.DataFrame,
    boston_target_sr: pd.Series,
    regressor_df_cls: Type[RegressorDF],
) -> None:

    rpdf = RegressorPipelineDF(
        regressor=regressor_df_cls(),
        preprocessing=make_simple_transformer(
            impute_median_columns=boston_features.select_dtypes(
                include=np.number
            ).columns,
            one_hot_encode_columns=boston_features.select_dtypes(
                include=object
            ).columns,
        ),
    )

    assert is_regressor(rpdf)

    rpdf.fit(X=boston_features, y=boston_target_sr)
    rpdf.predict(X=boston_features)

    # test type check within constructor
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        RegressorPipelineDF(regressor=LGBMRegressor(), preprocessing=OneHotEncoder())
