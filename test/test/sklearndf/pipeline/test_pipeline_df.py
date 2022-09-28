"""
Test module for PipelineDF inspired by:
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tests/test_pipeline.py
"""
from __future__ import annotations

import shutil
import time
from tempfile import mkdtemp
from typing import Any, Dict, Mapping, cast

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_no_warnings
from pandas.testing import assert_frame_equal
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier, is_regressor
from sklearn.feature_selection import f_classif

from sklearndf import __sklearn_1_1__, __sklearn_version__
from sklearndf.classification import SVCDF, LogisticRegressionDF
from sklearndf.pipeline import FeatureUnionDF, PipelineDF
from sklearndf.regression import DummyRegressorDF, LassoDF, LinearRegressionDF
from sklearndf.transformation import OneHotEncoderDF, SelectKBestDF, SimpleImputerDF
from sklearndf.transformation.wrapper import ColumnPreservingTransformerWrapperDF


def test_set_params_nested_pipeline_df() -> None:
    """Test parameter setting for nested pipelines - adapted from
    sklearn.tests.test_pipeline"""

    PipelineDF([("b", SimpleImputerDF(strategy="median"))])

    estimator = PipelineDF([("a", PipelineDF([("b", DummyRegressorDF())]))])

    estimator.set_params(a__b__alpha=0.001, a__b=LassoDF())
    estimator.set_params(a__steps=[("b", LogisticRegressionDF())], a__b__C=5)


class NoFit(
    BaseEstimator,  # type: ignore
):
    """Small class to test parameter dispatching."""

    def __init__(self, a: Any = None, b: Any = None) -> None:
        self.a = a
        self.b = b


class NoTransformer(
    NoFit,
    TransformerMixin,  # type: ignore
):
    """
    Not a transformer
    """

    # noinspection PyPep8Naming
    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> NoTransformer:
        return self

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b}

    def set_params(self, a: Any = None, **params: Dict[str, Any]) -> NoTransformer:
        self.a = a
        return self


class NoInvTransformer(
    NoTransformer,
    TransformerMixin,  # type: ignore
):
    # noinspection PyPep8Naming
    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return X


class Transformer(NoInvTransformer):
    # noinspection PyPep8Naming
    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return X

    # noinspection PyPep8Naming,PyMethodMayBeStatic
    def inverse_transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return X


class DummyTransformer(Transformer):
    """Transformer which stores the column means"""

    def __init__(self, a: Any = None, b: Any = None) -> None:
        super().__init__(a, b)

    # noinspection PyPep8Naming,PyAttributeOutsideInit
    def fit(
        self, X: npt.NDArray[Any], y: Any = None, **fit_params: Any
    ) -> DummyTransformer:
        self.means_: npt.NDArray[Any] = np.mean(X, axis=0)
        # store timestamp to figure out whether the result of 'fit' has been
        # cached or not
        self.timestamp_: float = time.time()
        return self


class DummyTransformerDF(
    ColumnPreservingTransformerWrapperDF[DummyTransformer], native=DummyTransformer
):
    """dummy transformer"""


class NoTransformerDF(
    ColumnPreservingTransformerWrapperDF[NoTransformer], native=NoTransformer
):
    """not a transformer"""


def test_pipeline_df_memory(
    iris_features: pd.DataFrame, iris_target_sr: pd.Series
) -> None:
    """Test memory caching in PipelineDF - taken almost 1:1 from
    sklearn.tests.test_pipeline"""

    cache_dir = mkdtemp()

    try:
        memory = joblib.Memory(location=cache_dir, verbose=10)

        # Test with Transformer + SVC
        clf = SVCDF(probability=True, random_state=0)
        tx = DummyTransformerDF()
        pipe = PipelineDF([("tx", clone(tx)), ("svc", clf)])
        cached_pipe = PipelineDF([("tx", tx), ("svc", clf)], memory=memory)

        # Memoize the transformer at the first fit
        cached_pipe.fit(iris_features, iris_target_sr)
        pipe.fit(iris_features, iris_target_sr)
        # Get the time stamp of the transformer in the cached pipeline
        ts = cached_pipe.named_steps["tx"].timestamp_
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(
            pipe.predict(iris_features), cached_pipe.predict(iris_features)
        )
        assert_array_equal(
            pipe.predict_proba(iris_features), cached_pipe.predict_proba(iris_features)
        )
        assert_array_equal(
            pipe.predict_log_proba(iris_features),
            cached_pipe.predict_log_proba(iris_features),
        )
        assert_array_equal(
            pipe.score(iris_features, iris_target_sr),
            cached_pipe.score(iris_features, iris_target_sr),
        )
        assert_array_equal(
            pipe.named_steps["tx"].means_, cached_pipe.named_steps["tx"].means_
        )
        assert not hasattr(tx, "means_")
        # Check that we are reading the cache while fitting
        # a second time
        cached_pipe.fit(iris_features, iris_target_sr)
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(
            pipe.predict(iris_features), cached_pipe.predict(iris_features)
        )
        assert_array_equal(
            pipe.predict_proba(iris_features), cached_pipe.predict_proba(iris_features)
        )
        assert_array_equal(
            pipe.predict_log_proba(iris_features),
            cached_pipe.predict_log_proba(iris_features),
        )
        assert_array_equal(
            pipe.score(iris_features, iris_target_sr),
            cached_pipe.score(iris_features, iris_target_sr),
        )
        assert_array_equal(
            pipe.named_steps["tx"].means_, cached_pipe.named_steps["tx"].means_
        )
        assert ts == cached_pipe.named_steps["tx"].timestamp_
        # Create a new pipeline with cloned estimators
        # Check that even changing the name step does not affect the cache hit
        clf_2 = SVCDF(probability=True, random_state=0)
        tx_2 = DummyTransformerDF()
        cached_pipe_2 = PipelineDF([("tx_2", tx_2), ("svc", clf_2)], memory=memory)
        cached_pipe_2.fit(iris_features, iris_target_sr)

        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(
            pipe.predict(iris_features), cached_pipe_2.predict(iris_features)
        )
        assert_array_equal(
            pipe.predict_proba(iris_features),
            cached_pipe_2.predict_proba(iris_features),
        )
        assert_array_equal(
            pipe.predict_log_proba(iris_features),
            cached_pipe_2.predict_log_proba(iris_features),
        )
        assert_array_equal(
            pipe.score(iris_features, iris_target_sr),
            cached_pipe_2.score(iris_features, iris_target_sr),
        )
        assert_array_equal(
            pipe.named_steps["tx"].means_, cached_pipe_2.named_steps["tx_2"].means_
        )
        assert ts == cached_pipe_2.named_steps["tx_2"].timestamp_
    finally:
        shutil.rmtree(cache_dir)


def test_pipeline_df__init() -> None:
    """Test the various init parameters of the pipeline."""

    with pytest.raises(TypeError):
        PipelineDF()

    # Check that we can't instantiate pipelines with objects without fit
    # method
    if __sklearn_version__ < __sklearn_1_1__:
        with pytest.raises(
            TypeError,
            match=(
                "Last step of Pipeline should implement fit "
                "or be the string 'passthrough'"
                ".*NoFit.*"
            ),
        ):
            PipelineDF([("clf", NoFit())])
    else:
        with pytest.raises(
            ValueError,
            match=(
                "expected final step 'clf' to be an EstimatorDF or passthrough, "
                "but found an instance of NoFit"
            ),
        ):
            PipelineDF([("clf", NoFit())])

    # Smoke test with only an estimator
    clf = NoTransformerDF()

    # step names
    step_svc = "svc"
    step_anova = "anova"

    pipe = PipelineDF([(step_svc, clf)])
    assert pipe.get_params(deep=True) == dict(
        svc__a=None, svc__b=None, svc=clf, **pipe.get_params(deep=False)
    )

    assert not is_classifier(pipe)
    assert not is_regressor(pipe)
    assert pipe._estimator_type is None

    # Check that params are set
    pipe.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    # Smoke test the repr:
    _ = repr(pipe)

    # Test with two objects
    clf = SVCDF()
    filter1 = SelectKBestDF(f_classif)
    pipe = PipelineDF([(step_anova, filter1), (step_svc, clf)])

    # Check that the pipeline correctly detects that it is a classifier
    assert is_classifier(pipe)

    # Check that estimators are not cloned on pipeline construction
    assert pipe.named_steps[step_anova] is filter1
    assert pipe.named_steps[step_svc] is clf

    # Check that params are set
    pipe.set_params(svc__C=0.1)
    assert clf.C == 0.1
    # Smoke test the repr:
    _ = repr(pipe)

    # Check that params are not set when naming them wrong
    with pytest.raises(ValueError):
        pipe.set_params(anova__C=0.1)

    # Test clone
    pipe2 = cast(PipelineDF, assert_no_warnings(clone, pipe))
    assert pipe.named_steps[step_svc] is not pipe2.named_steps[step_svc]

    # Check that apart from estimators, the parameters are the same

    def _get_deep_params(_pipe: PipelineDF) -> Mapping[str, Any]:
        top_level_params = {*_pipe.get_params(deep=False), step_svc, step_anova}
        return {
            k: v
            for k, v in _pipe.get_params(deep=True).items()
            if k not in top_level_params
        }

    params = _get_deep_params(pipe)
    params2 = _get_deep_params(pipe2)

    assert params == params2


def test_pipeline_df_raise_set_params_error() -> None:
    """Test pipeline raises set params error message for nested models."""
    pipe = PipelineDF([("cls", LinearRegressionDF())])

    with pytest.raises(
        ValueError,
        match=r"Invalid parameter (fake|'fake') for estimator Pipeline",
    ):
        pipe.set_params(fake="nope")

    # nested model check
    with pytest.raises(
        ValueError,
        match=r"Invalid parameter (fake|'fake') for estimator Pipeline",
    ):
        pipe.set_params(fake__estimator="nope")


def test_feature_union(test_data_categorical: pd.DataFrame) -> None:
    feature_union = FeatureUnionDF(
        [
            ("oh", OneHotEncoderDF(drop="first", sparse=False)),
        ]
    )

    assert_frame_equal(
        feature_union.fit_transform(test_data_categorical),
        pd.DataFrame(
            dict(
                oh__a_yes=[1.0, 1.0, 0.0],
                oh__b_green=[0.0, 0.0, 1.0],
                oh__b_red=[1.0, 0.0, 0.0],
                oh__c_father=[0.0, 1.0, 0.0],
                oh__c_mother=[0.0, 0.0, 1.0],
            ),
        ).rename_axis(columns="feature_out"),
    )

    if __sklearn_version__ >= __sklearn_1_1__:
        feature_union = FeatureUnionDF(
            [
                ("oh", OneHotEncoderDF(drop="first", sparse=False)),
                ("pass", "passthrough"),
                ("pass_again", "passthrough"),
            ]
        )

        assert_frame_equal(
            feature_union.fit_transform(test_data_categorical),
            pd.concat(
                [
                    pd.DataFrame(
                        dict(
                            oh__a_yes=[1.0, 1.0, 0.0],
                            oh__b_green=[0.0, 0.0, 1.0],
                            oh__b_red=[1.0, 0.0, 0.0],
                            oh__c_father=[0.0, 1.0, 0.0],
                            oh__c_mother=[0.0, 0.0, 1.0],
                        ),
                        dtype=object,
                    ),
                    test_data_categorical.add_prefix("pass__"),
                    test_data_categorical.add_prefix("pass_again__"),
                ],
                axis=1,
            ).rename_axis(columns="feature_out"),
        )
