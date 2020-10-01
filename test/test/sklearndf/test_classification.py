from itertools import chain
from typing import Type

import numpy as np
import pandas as pd
import pytest
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

import sklearndf.classification as classification
from sklearndf import ClassifierDF
from test.sklearndf import check_expected_not_fitted_error, list_classes

CLASSIFIERS_TO_TEST = list_classes(
    from_modules=classification,
    matching=r".*DF",
    excluding=[ClassifierDF.__name__, r".*WrapperDF"],
)


CLASSIFIER_INIT_PARAMETERS = {
    "CalibratedClassifierCVDF": {
        "base_estimator": classification.RandomForestClassifierDF()
    },
    "ClassifierChainDF": {"base_estimator": classification.RandomForestClassifierDF()},
    "MultiOutputClassifierDF": {"estimator": classification.RandomForestClassifierDF()},
    "OneVsOneClassifierDF": {"estimator": classification.RandomForestClassifierDF()},
    "OneVsRestClassifierDF": {"estimator": classification.RandomForestClassifierDF()},
    "OutputCodeClassifierDF": {"estimator": classification.RandomForestClassifierDF()},
    "VotingClassifierDF": {
        "estimators": [
            ("rfc", classification.RandomForestClassifierDF()),
            ("svmc", classification.SVCDF(probability=True)),
        ],
        "voting": "soft",
    },
    "StackingClassifierDF": {
        "estimators": (
            ("Forest", classification.RandomForestClassifierDF()),
            ("SVC", classification.SVCDF()),
            ("AdaBoost", classification.AdaBoostClassifierDF()),
        )
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
    iris_targets_df: pd.DataFrame,
    iris_targets_binary_df: pd.DataFrame,
) -> None:
    """ Test fit & predict & predict[_log]_proba of wrapped sklearn classifiers """
    # noinspection PyArgumentList
    classifier: ClassifierDF = sklearndf_cls(
        **CLASSIFIER_INIT_PARAMETERS.get(sklearndf_cls.__name__, {})
    )

    is_chain = isinstance(classifier.native_estimator, ClassifierChain)

    is_multi_output = isinstance(classifier.native_estimator, MultiOutputClassifier)
    check_expected_not_fitted_error(estimator=classifier)

    if is_chain:
        # for chain classifiers, classes must be numerical so the preceding
        # classification can act as input to the next classification
        classes = set(range(iris_targets_binary_df.shape[1]))
        classifier.fit(X=iris_features, y=iris_targets_binary_df)
    elif is_multi_output:
        classes = set(
            chain(
                *(
                    list(iris_targets_df.iloc[:, col].unique())
                    for col in range(iris_targets_df.shape[1])
                )
            )
        )
        classifier.fit(X=iris_features, y=iris_targets_df)
    else:
        classes = set(iris_target_sr.unique())
        classifier.fit(X=iris_features, y=iris_target_sr)

    predictions = classifier.predict(X=iris_features)

    # test predictions data-type, length and values
    assert isinstance(
        predictions, pd.DataFrame if is_multi_output or is_chain else pd.Series
    )
    assert len(predictions) == len(iris_target_sr)
    assert np.all(predictions.isin(classes))

    # test predict_proba & predict_log_proba:
    for method_name in ["predict_proba", "predict_log_proba"]:
        method = getattr(classifier, method_name, None)

        if hasattr(classifier.native_estimator, method_name):
            predictions = method(X=iris_features)

            if is_multi_output:
                assert isinstance(predictions, list)
                assert classifier.n_outputs_ == len(predictions)
            else:
                assert classifier.n_outputs_ == predictions.shape[1] if is_chain else 1
                predictions = [predictions]

            for prediction in predictions:
                # test type and shape of predictions
                assert isinstance(prediction, pd.DataFrame)
                assert len(prediction) == len(iris_target_sr)
                assert prediction.shape == (len(iris_target_sr), len(classes))
                # check correct labels are set as columns
                assert classes == set(prediction.columns)
        else:
            with pytest.raises(NotImplementedError):
                method(X=iris_features)
