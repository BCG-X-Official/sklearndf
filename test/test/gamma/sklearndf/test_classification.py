from typing import *

import numpy as np
import pandas as pd
import pytest

import gamma.sklearndf.classification
from gamma.sklearndf.classification import (
    ClassifierDF,
    ClassifierWrapperDF,
    RandomForestClassifierDF,
    SVCDF,
)
from test.gamma.sklearndf import get_classes, get_missing_init_parameter

CLASSIFIERS_TO_TEST = get_classes(
    from_module=gamma.sklearndf.classification,
    matching=r".*DF",
    excluding=[ClassifierDF.__name__, r".*WrapperDF"],
)

DEFAULT_INIT_PARAMETERS = {
    "estimator": {"estimator": RandomForestClassifierDF()},
    "base_estimator": {"base_estimator": RandomForestClassifierDF()},
    "estimators": {
        "estimators": [("rfc", RandomForestClassifierDF()), ("svmc", SVCDF())]
    },
}


@pytest.mark.parametrize(argnames="sklearndf_cls", argvalues=CLASSIFIERS_TO_TEST)
def test_wrapped_constructor(sklearndf_cls: Type) -> None:
    """ Test standard constructor of wrapped sklearn classifiers """
    try:
        cls: ClassifierDF = sklearndf_cls()
    except TypeError as te:
        # some classifiers need additional parameters, look up their key and add
        # default:
        missing_init_parameter = get_missing_init_parameter(te=te)
        if missing_init_parameter not in DEFAULT_INIT_PARAMETERS:
            raise te
        else:
            cls = sklearndf_cls(**DEFAULT_INIT_PARAMETERS[missing_init_parameter])


@pytest.mark.parametrize(argnames="sklearndf_cls", argvalues=CLASSIFIERS_TO_TEST)
def test_wrapped_fit_predict(
    sklearndf_cls: Type[ClassifierDF],
    iris_features: pd.DataFrame,
    iris_target_sr: pd.Series,
) -> None:
    """ Test fit & predict & predict[_log]_proba of wrapped sklearn classifiers """
    try:
        classifier = sklearndf_cls()
    except TypeError as te:
        # some classifiers need additional parameters, look up their key and add
        # default:
        missing_init_parameter = get_missing_init_parameter(te=te)
        if missing_init_parameter not in DEFAULT_INIT_PARAMETERS:
            raise te
        else:
            classifier = sklearndf_cls(
                **DEFAULT_INIT_PARAMETERS[missing_init_parameter]
            )

    classifier.fit(X=iris_features, y=iris_target_sr)
    predictions = classifier.predict(X=iris_features)

    # test predictions data-type, length and values
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(iris_target_sr)
    assert np.all(predictions.isin(iris_target_sr.unique()))

    # test predict_proba & predict_log_proba if delegate-classifier has them:
    test_funcs = [
        getattr(classifier, attr)
        for attr in ["predict_proba", "predict_log_proba"]
        if hasattr(classifier.delegate_estimator, attr)
    ]
    for func in test_funcs:
        predicted_probas = func(X=iris_features)

        # test data-type and shape
        assert isinstance(predicted_probas, pd.DataFrame)
        assert len(predicted_probas) == len(iris_target_sr)
        assert predicted_probas.shape == (
            len(iris_target_sr),
            len(iris_target_sr.unique()),
        )

        # check correct labels are set as columns
        assert list(iris_target_sr.unique()) == list(predicted_probas.columns)
