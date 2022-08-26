from itertools import chain
from typing import Any, Dict, Type

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

import sklearndf.classification as classification
from sklearndf import (
    ClassifierDF,
    __sklearn_0_22__,
    __sklearn_1_0__,
    __sklearn_version__,
)
from test.sklearndf import check_expected_not_fitted_error, iterate_classes

CLASSIFIERS_TO_TEST = iterate_classes(
    from_modules=classification,
    matching=r".*DF",
    excluding=[ClassifierDF.__name__, r".*WrapperDF", r"^_"],
)


def test_classifier_count() -> None:
    n = len(CLASSIFIERS_TO_TEST)

    print(f"Testing {n} classifiers.")
    if __sklearn_version__ < __sklearn_0_22__:
        assert n == 38
    elif __sklearn_version__ < __sklearn_1_0__:
        assert n == 40
    else:
        assert n == 41


CLASSIFIER_INIT_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "CalibratedClassifierCVDF": {
        "base_estimator": classification.RandomForestClassifierDF()
    },
    "ClassifierChainDF": {"base_estimator": classification.RandomForestClassifierDF()},
    "MultiOutputClassifierDF": {"estimator": classification.RandomForestClassifierDF()},
    "MultiOutputClassifierDF_partial_fit": {"estimator": classification.PerceptronDF()},
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
            ("Forest", classification.RandomForestClassifierDF(max_depth=5)),
            ("Logit", classification.LogisticRegressionCVDF()),
            ("AdaBoost", classification.AdaBoostClassifierDF()),
        )
    },
}


CLASSIFIERS_PARTIAL_FIT = [
    classification.BernoulliNBDF,
    classification.MultinomialNBDF,
    classification.PerceptronDF,
    classification.SGDClassifierDF,
    classification.PassiveAggressiveClassifierDF,
    classification.GaussianNBDF,
    classification.ComplementNBDF,
    classification.MultiOutputClassifierDF,
]
if __sklearn_version__ >= __sklearn_0_22__:
    CLASSIFIERS_PARTIAL_FIT.append(classification.CategoricalNBDF)


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearndf_cls", argvalues=CLASSIFIERS_TO_TEST
)
def test_wrapped_fit_predict(
    sklearndf_cls: Type[ClassifierDF],
    iris_features: pd.DataFrame,
    iris_target_sr: pd.Series,
    iris_targets_df: pd.DataFrame,
    iris_targets_binary_df: pd.DataFrame,
) -> None:
    """Test fit & predict & predict[_log]_proba of wrapped sklearn classifiers"""
    # noinspection PyArgumentList
    parameters: Dict[str, Any] = CLASSIFIER_INIT_PARAMETERS.get(
        sklearndf_cls.__name__, {}
    )
    # noinspection PyArgumentList
    classifier: ClassifierDF = sklearndf_cls(**parameters)

    assert is_classifier(classifier)

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
        method = getattr(classifier, method_name)

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
                assert set(prediction.columns) == classes
        else:
            with pytest.raises(NotImplementedError):
                method(X=iris_features)


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearndf_cls", argvalues=CLASSIFIERS_PARTIAL_FIT
)
def test_wrapped_partial_fit(
    sklearndf_cls: Type[ClassifierDF],
    iris_features: pd.DataFrame,
    iris_target_sr: pd.Series,
    iris_targets_df: pd.DataFrame,
) -> None:
    # noinspection PyArgumentList
    classifier: ClassifierDF = sklearndf_cls(
        **CLASSIFIER_INIT_PARAMETERS.get(f"{sklearndf_cls.__name__}_partial_fit", {})
    )

    is_multi_output = isinstance(classifier.native_estimator, MultiOutputClassifier)
    if is_multi_output:
        classes = iris_targets_df.apply(lambda col: col.unique()).transpose().values
        iris_target = iris_targets_df
    else:
        classes = iris_target_sr.unique()
        iris_target = iris_target_sr

    with pytest.raises(
        ValueError,
        match="classes must be passed on the first call to partial_fit.",
    ):
        classifier.partial_fit(iris_features, iris_target)

    classifier.partial_fit(iris_features, iris_target, classes)
