from typing import Any, Callable, Dict, Type

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from sklearndf import TransformerDF
from sklearndf.pipeline import PipelineDF
from sklearndf.regression import RandomForestRegressorDF
from sklearndf.regression.extra import LGBMRegressorDF
from sklearndf.transformation import SimpleImputerDF
from sklearndf.transformation.extra import BoostAGrootaDF, BorutaDF, GrootCVDF, LeshyDF
from sklearndf.wrapper import MissingEstimator

regressor_params = dict(max_depth=5, n_jobs=-3, random_state=42, n_estimators=100)
lgbm_regressor = LGBMRegressor(**regressor_params)
lgbm_regressor_df = LGBMRegressorDF(**regressor_params)

parametrize_feature_selector_cls: Callable[
    [Callable[..., None]], Callable[..., None]
] = pytest.mark.parametrize(
    # the class/parameter combinations to test for feature selection
    argnames=["feature_selector_cls", "feature_selector_params"],
    argvalues=[
        (cls, params)
        for cls, params in [
            # Boruta selector
            (
                BorutaDF,
                dict(
                    estimator=RandomForestRegressorDF(
                        max_depth=5, n_jobs=-3, random_state=42, n_estimators=100
                    )
                ),
            ),
            # Various ARFS selectors
            (LeshyDF, dict(estimator=lgbm_regressor, random_state=42, perc=90)),
            (LeshyDF, dict(estimator=lgbm_regressor_df, random_state=42, perc=90)),
            (BoostAGrootaDF, dict(est=lgbm_regressor, cutoff=1.1)),
            (GrootCVDF, dict()),
        ]
        if not issubclass(cls.__wrapped__, MissingEstimator)
    ],
)


@parametrize_feature_selector_cls
def test_feature_selection_df(
    feature_selector_cls: Type[TransformerDF], feature_selector_params: Dict[str, Any]
) -> None:
    """
    Test feature selection using the Boruta or ARFS package using a simple synthetic
    dataset.

    :param feature_selector_cls: The feature selector class to test.
    :param feature_selector_params: The parameters to use for the feature selector.
    """

    df = pd.DataFrame(data=np.random.randn(100, 5), columns=list("abcde"))
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_selector = feature_selector_cls(**feature_selector_params)
    feature_selector.fit(x, y)
    assert set(feature_selector.feature_names_out_) <= {"a", "b", "c", "d", "e"}


@parametrize_feature_selector_cls
def test_feature_selection_pipeline_df(
    feature_selector_cls: Type[TransformerDF],
    feature_selector_params: Dict[str, Any],
    diabetes_df: pd.DataFrame,
    diabetes_target: str,
) -> None:
    """
    Test feature selection using the Boruta or ARFS package using the diabetes
    dataset.

    :param feature_selector_cls: The feature selector class to test.
    :param feature_selector_params: The parameters to use for the feature selector.
    :param diabetes_df: The diabetes dataset.
    :param diabetes_target: The diabetes target column.
    """

    feature_selector = feature_selector_cls(**feature_selector_params)

    diabetes_df = diabetes_df.sample(frac=0.5, random_state=42)

    feature_selection_pipeline = PipelineDF(
        steps=[
            (
                "preprocess",
                PipelineDF(
                    steps=[
                        ("imputer", SimpleImputerDF()),
                    ]
                ),
            ),
            ("selector", feature_selector),
        ]
    )

    x = diabetes_df.drop(columns=diabetes_target)
    y = diabetes_df.loc[:, diabetes_target]

    feature_selection_pipeline.fit(x, y)

    try:
        assert set(feature_selection_pipeline.feature_names_out_) == set(
            feature_selector.selected_features_
        )
    except AttributeError:
        pass
    assert set(feature_selection_pipeline.feature_names_out_) in [
        {"bmi", "bp", "s3", "s5"},
        *({"bmi", "bp", "s3", "s5", extra} for extra in ["sex", "s1", "s2", "s6"]),
        {"bmi", "bp", "s2", "s5", "s6"},
    ]
