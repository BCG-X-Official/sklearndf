import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_regressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from sklearndf.pipeline import RegressorPipelineDF
from sklearndf.regression import LinearRegressionDF
from test.sklearndf.pipeline import make_simple_transformer


def test_regression_pipeline_df(
    boston_features: pd.DataFrame, boston_target_sr: pd.Series
) -> None:

    rpdf = RegressorPipelineDF(
        regressor=LinearRegressionDF(),
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
        RegressorPipelineDF(regressor=LinearRegression(), preprocessing=OneHotEncoder())
