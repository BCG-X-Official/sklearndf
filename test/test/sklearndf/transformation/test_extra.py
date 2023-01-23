import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearndf.pipeline import PipelineDF
from sklearndf.regression import RandomForestRegressorDF
from sklearndf.transformation import (
    ColumnTransformerDF,
    OneHotEncoderDF,
    SimpleImputerDF,
)
from sklearndf.transformation.extra import BorutaDF


def test_boruta_df() -> None:
    """Test basic functionality of BorutaDF with both sklearn and sklearndf predictor"""
    df = pd.DataFrame(data=np.random.randn(100, 5), columns=list("abcde"))
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    boruta = BorutaDF(estimator=RandomForestRegressor(), max_iter=10)
    boruta.fit(x, y)

    boruta_df = BorutaDF(estimator=RandomForestRegressorDF(), max_iter=10)
    boruta_df.fit(x, y)


def test_boruta_pipeline(diabetes_df: pd.DataFrame, diabetes_target: str) -> None:
    """Test a pipeline with on the diabetes dataset"""

    boruta_selector = PipelineDF(
        steps=[
            (
                "preprocess",
                PipelineDF(
                    steps=[
                        ("imputer", SimpleImputerDF()),
                        (
                            "onehot",
                            ColumnTransformerDF(
                                [
                                    (
                                        "onehot",
                                        OneHotEncoderDF(
                                            drop="first", categories="auto"
                                        ),
                                        ["sex"],
                                    ),
                                ],
                                remainder="passthrough",
                            ),
                        ),
                    ]
                ),
            ),
            (
                "boruta",
                BorutaDF(
                    estimator=RandomForestRegressor(), n_estimators=10, max_iter=10
                ),
            ),
        ]
    )

    x = diabetes_df.drop(columns=diabetes_target)
    y = diabetes_df.loc[:, diabetes_target]

    boruta_selector.fit(x, y)
