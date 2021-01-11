"""
Test module for PipelineDF inspired by:
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tests/test_pipeline.py
"""
import shutil
import time
from distutils.version import LooseVersion
from tempfile import mkdtemp
from typing import Any, Dict, Mapping

import joblib
import numpy as np
import pandas as pd
from numpy.testing import (
    assert_array_equal,
    assert_no_warnings,
    assert_raises,
    assert_raises_regex,
)
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif

from sklearndf import TransformerDF
from sklearndf._wrapper import df_estimator
from sklearndf.classification import SVCDF, LogisticRegressionDF
from sklearndf.pipeline import PipelineDF
from sklearndf.regression import DummyRegressorDF, LassoDF, LinearRegressionDF
from sklearndf.transformation import SelectKBestDF, SimpleImputerDF
from sklearndf.transformation._wrapper import _ColumnPreservingTransformerWrapperDF


def test_set_params_nested_pipeline_df() -> None:
    """Test parameter setting for nested pipelines - adapted from
    sklearn.tests.test_pipeline"""

    PipelineDF([("b", SimpleImputerDF(strategy="median"))])

    estimator = PipelineDF([("a", PipelineDF([("b", DummyRegressorDF())]))])

    estimator.set_params(a__b__alpha=0.001, a__b=LassoDF())
    estimator.set_params(a__steps=[("b", LogisticRegressionDF())], a__b__C=5)


class NoFit(BaseEstimator):
    """Small class to test parameter dispatching."""

    def __init__(self, a: str = None, b: str = None) -> None:
        self.a = a
        self.b = b


class NoTransformer(NoFit):
    """
    Not a transformer
    """

    # noinspection PyPep8Naming
    def fit(self, X, y=None, **fit_params) -> "NoTransformer":
        return self

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b}

    def set_params(self, **params: Dict[str, Any]) -> "NoTransformer":
        self.a = params["a"]
        return self


class NoInvTransformer(NoTransformer, TransformerMixin):
    # noinspection PyPep8Naming
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class Transformer(NoInvTransformer):
    # noinspection PyPep8Naming
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    # noinspection PyPep8Naming,PyMethodMayBeStatic
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X


class DummyTransformer(Transformer):
    """Transformer which stores the column means"""

    def __init__(self, a: str = None, b: str = None) -> None:
        super().__init__(a, b)

    # noinspection PyPep8Naming,PyAttributeOutsideInit
    def fit(self, X, y=None, **fit_params) -> "DummyTransformer":
        self.means_: np.ndarray = np.mean(X, axis=0)
        # store timestamp to figure out whether the result of 'fit' has been
        # cached or not
        self.timestamp_: float = time.time()
        return self


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class DummyTransformerDF(TransformerDF, DummyTransformer):
    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class NoTransformerDF(TransformerDF, NoTransformer):
    pass


def test_pipeline_df_memory(
    iris_features: pd.DataFrame, iris_target_sr: pd.Series
) -> None:
    """Test memory caching in PipelineDF - taken almost 1:1 from
    sklearn.tests.test_pipeline"""

    cache_dir = mkdtemp()

    try:
        if LooseVersion(joblib.__version__) < LooseVersion("0.12"):
            # Deal with change of API in joblib
            memory = joblib.Memory(cachedir=cache_dir, verbose=10)
        else:
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
    """ Test the various init parameters of the pipeline. """

    assert_raises(TypeError, PipelineDF)
    # Check that we can't instantiate pipelines with objects without fit
    # method
    assert_raises_regex(
        TypeError,
        "Last step of Pipeline should implement fit "
        "or be the string 'passthrough'"
        ".*NoFit.*",
        PipelineDF,
        [("clf", NoFit())],
    )

    # Smoke test with only an estimator
    clf = NoTransformerDF()

    # step names
    step_svc = "svc"
    step_anova = "anova"

    pipe = PipelineDF([(step_svc, clf)])
    assert pipe.get_params(deep=True) == dict(
        svc__a=None, svc__b=None, svc=clf, **pipe.get_params(deep=False)
    )

    # Check that params are set
    pipe.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    # Smoke test the repr:
    repr(pipe)

    # Test with two objects
    clf = SVCDF()
    filter1 = SelectKBestDF(f_classif)
    pipe = PipelineDF([(step_anova, filter1), (step_svc, clf)])

    # Check that estimators are not cloned on pipeline construction
    assert pipe.named_steps[step_anova] is filter1
    assert pipe.named_steps[step_svc] is clf

    # Check that params are set
    pipe.set_params(svc__C=0.1)
    assert clf.C == 0.1
    # Smoke test the repr:
    repr(pipe)

    # Check that params are not set when naming them wrong
    assert_raises(ValueError, pipe.set_params, anova__C=0.1)

    # Test clone
    pipe2 = assert_no_warnings(clone, pipe)
    assert not pipe.named_steps[step_svc] is pipe2.named_steps[step_svc]

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
    """ Test pipeline raises set params error message for nested models. """
    pipe = PipelineDF([("cls", LinearRegressionDF())])

    assert_raises_regex(
        ValueError,
        "Invalid parameter fake for estimator Pipeline",
        pipe.set_params,
        fake="nope",
    )

    # nested model check
    assert_raises_regex(
        ValueError,
        "Invalid parameter fake for estimator Pipeline",
        pipe.set_params,
        fake__estimator="nope",
    )
