from typing import *

import numpy as np
import pandas as pd
import pytest
from sklearn.multioutput import MultiOutputEstimator

import gamma.sklearndf.classification
from gamma.sklearndf.classification import ClassifierDF, RandomForestClassifierDF, SVCDF
from test.gamma.sklearndf import get_classes

CLASSIFIERS_TO_TEST = get_classes(
    from_module=gamma.sklearndf.classification,
    matching=r".*DF",
    excluding=[ClassifierDF.__name__, r".*WrapperDF"],
)


CLASSIFIER_INIT_PARAMETERS = {
    "CalibratedClassifierCVDF": {"base_estimator": RandomForestClassifierDF()},
    "ClassifierChainDF": {"base_estimator": RandomForestClassifierDF()},
    "MultiOutputClassifierDF": {"estimator": RandomForestClassifierDF()},
    "OneVsOneClassifierDF": {"estimator": RandomForestClassifierDF()},
    "OneVsRestClassifierDF": {"estimator": RandomForestClassifierDF()},
    "OutputCodeClassifierDF": {"estimator": RandomForestClassifierDF()},
    "VotingClassifierDF": {
        "estimators": [
            ("rfc", RandomForestClassifierDF()),
            ("svmc", SVCDF(probability=True)),
        ],
        "voting": "soft",
    },
}


@pytest.mark.parametrize(argnames="sklearndf_cls", argvalues=CLASSIFIERS_TO_TEST)
def test_wrapped_constructor(sklearndf_cls: Type[ClassifierDF]) -> None:
    """ Test standard constructor of wrapped sklearn classifiers """
    # noinspection PyArgumentList
    _: ClassifierDF = sklearndf_cls(
        **CLASSIFIER_INIT_PARAMETERS.get(sklearndf_cls.__name__, {})
    )


@pytest.mark.parametrize(argnames="sklearndf_cls", argvalues=CLASSIFIERS_TO_TEST)
def test_wrapped_fit_predict(
    sklearndf_cls: Type[ClassifierDF],
    iris_features: pd.DataFrame,
    iris_target_sr: pd.Series,
    iris_target_df: pd.DataFrame,
) -> None:
    """ Test fit & predict & predict[_log]_proba of wrapped sklearn classifiers """
    # noinspection PyArgumentList
    classifier: ClassifierDF = sklearndf_cls(
        **CLASSIFIER_INIT_PARAMETERS.get(sklearndf_cls.__name__, {})
    )

    is_multi_output = isinstance(classifier.root_estimator, MultiOutputEstimator)

    if is_multi_output:
        classifier.fit(X=iris_features, y=iris_target_df)
    else:
        classifier.fit(X=iris_features, y=iris_target_sr)

    predictions = classifier.predict(X=iris_features)

    # test predictions data-type, length and values
    assert isinstance(predictions, pd.DataFrame if is_multi_output else pd.Series)
    assert len(predictions) == len(iris_target_sr)
    assert np.all(predictions.isin(iris_target_sr.unique()))

    # test predict_proba & predict_log_proba only if the root classifier has them:
    test_funcs = [
        getattr(classifier, attr)
        for attr in ["predict_proba", "predict_log_proba"]
        if hasattr(classifier.root_estimator, attr)
    ]
    for method_name in ["predict_proba", "predict_log_proba"]:
        method = getattr(classifier, method_name, None)

        if hasattr(classifier.root_estimator, method_name):
            predictions = method(X=iris_features)

            if is_multi_output:
                assert isinstance(predictions, list)
                assert classifier.n_outputs == len(predictions)
            else:
                assert classifier.n_outputs == 1
                predictions = [predictions]

            for prediction in predictions:
                # test type and shape of predictions
                assert isinstance(prediction, pd.DataFrame)
                assert len(prediction) == len(iris_target_sr)
                assert prediction.shape == (
                    len(iris_target_sr),
                    len(iris_target_sr.unique()),
                )
                # check correct labels are set as columns
                assert list(iris_target_sr.unique()) == list(prediction.columns)
        else:
            with pytest.raises(NotImplementedError):
                method(X=iris_features)
