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
    diabetes_features: pd.DataFrame,
    diabetes_target_sr: pd.Series,
    regressor_df_cls: type[RegressorDF],
) -> None:
    rpdf = RegressorPipelineDF(
        regressor=regressor_df_cls(),
        preprocessing=make_simple_transformer(
            impute_median_columns=diabetes_features.select_dtypes(
                include=np.number
            ).columns,
            one_hot_encode_columns=diabetes_features.select_dtypes(
                include=object
            ).columns,
        ),
    )

    assert is_regressor(rpdf)

    rpdf.fit(X=diabetes_features, y=diabetes_target_sr)
    rpdf.predict(X=diabetes_features)

    # test type check within constructor
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        RegressorPipelineDF(regressor=LGBMRegressor(), preprocessing=OneHotEncoder())
