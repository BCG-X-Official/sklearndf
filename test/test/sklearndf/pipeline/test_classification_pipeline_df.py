from typing import Type

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from sklearndf import ClassifierDF
from sklearndf.classification import RandomForestClassifierDF
from sklearndf.classification.extra import LGBMClassifierDF
from sklearndf.pipeline import ClassifierPipelineDF
from test.sklearndf.pipeline import make_simple_transformer


@pytest.mark.parametrize(  # type: ignore
    argnames="classifier_df_cls",
    argvalues=[RandomForestClassifierDF, LGBMClassifierDF],
)
def test_classification_pipeline_df(
    iris_features: pd.DataFrame,
    iris_target_sr: pd.DataFrame,
    classifier_df_cls: Type[ClassifierDF],
) -> None:

    cls_p_df = ClassifierPipelineDF(
        classifier=classifier_df_cls(),
        preprocessing=make_simple_transformer(
            impute_median_columns=iris_features.select_dtypes(
                include=np.number
            ).columns,
            one_hot_encode_columns=iris_features.select_dtypes(include=object).columns,
        ),
    )

    assert is_classifier(cls_p_df)

    cls_p_df.fit(X=iris_features, y=iris_target_sr)
    cls_p_df.predict(X=iris_features)

    # test-type check within constructor:
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        ClassifierPipelineDF(
            classifier=RandomForestClassifier(), preprocessing=OneHotEncoder()
        )
