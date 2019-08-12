from typing import *

import numpy as np
import pandas as pd
import pytest

import gamma.sklearndf.classification
from gamma.sklearndf import ClassifierDF
from gamma.sklearndf.classification import RandomForestClassifierDF
from test.gamma.sklearndf import get_classes


@pytest.mark.parametrize(
    argnames="sklearndf_cls",
    argvalues=get_classes(
        from_module=gamma.sklearndf.classification,
        regex=r".*DF",
        ignore=["ClassifierDF", "ClassifierWrapperDF"],
    ),
)
def test_wrapped_constructor(sklearndf_cls: Type) -> None:
    try:
        cls: ClassifierDF = sklearndf_cls()
    except TypeError as te:
        # some Classifiers need special kwargs in __init__ we can't easily infer:
        if "missing 1 required positional argument" in str(te):
            # parameter 'base_estimator' is expected:
            if "'base_estimator'" in str(te):
                cls = sklearndf_cls(base_estimator=RandomForestClassifierDF())
            # parameter 'estimator' is expected:
            elif "'estimator'" in str(te):
                cls = sklearndf_cls(estimator=RandomForestClassifierDF())
            # parameter 'estimators' is expected:
            elif "'estimators'" in str(te):
                cls = sklearndf_cls(estimators=[RandomForestClassifierDF()])
            # unknown Exception, raise it:
            else:
                raise te
        # unknown Exception, raise it:
        else:
            raise te


def test_classifier_df(iris_df: pd.DataFrame, iris_target: str) -> None:
    classifier_df = RandomForestClassifierDF()

    x = iris_df.drop(columns=iris_target)
    y = iris_df.loc[:, iris_target]

    classifier_df.fit(X=x, y=y)

    predictions = classifier_df.predict(X=x)

    # test predictions data-type, length and values
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(y)
    assert np.all(predictions.isin(y.unique()))

    # test predict_proba & predict_log_proba
    for func in (classifier_df.predict_proba, classifier_df.predict_log_proba):
        predicted_probas = func(X=x)

        # test data-type and shape
        assert isinstance(predicted_probas, pd.DataFrame)
        assert len(predicted_probas) == len(y)
        assert predicted_probas.shape == (len(y), len(y.unique()))

        # check correct labels are set as columns
        assert list(y.unique()) == list(predicted_probas.columns)
