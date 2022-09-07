# inspired by:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tests/test_base.py
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_equal
from sklearn import clone
from sklearn.base import BaseEstimator, is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pytools.expression import freeze, make_expression
from pytools.expression.atomic import Id

from sklearndf.classification import SVCDF, DecisionTreeClassifierDF
from sklearndf.clustering.wrapper import KMeansBaseWrapperDF
from sklearndf.pipeline import PipelineDF
from sklearndf.pipeline.wrapper import FeatureUnionWrapperDF
from sklearndf.transformation import OneHotEncoderDF
from sklearndf.wrapper import (
    ClassifierWrapperDF,
    EstimatorWrapperDF,
    RegressorWrapperDF,
)


class DummyEstimator(
    BaseEstimator,  # type: ignore
):
    def __init__(self, l1: int = 0, empty: Any = None) -> None:
        self.l1 = l1
        self.empty = empty


class DummyEstimator2(
    BaseEstimator,  # type: ignore
):
    def __init__(self, a: Any = None, b: Any = None) -> None:
        self.a = a
        self.b = b


class DummyEstimator3(
    BaseEstimator,  # type: ignore
):
    def __init__(self, c: int = 0, d: Any = None) -> None:
        self.c = c
        self.d = d


class DummyEstimatorDF(EstimatorWrapperDF[DummyEstimator], native=DummyEstimator):
    """A trivial estimator."""


class DummyEstimator2DF(EstimatorWrapperDF[DummyEstimator2], native=DummyEstimator2):
    """A trivial estimator."""


class DummyEstimator3DF(EstimatorWrapperDF[DummyEstimator3], native=DummyEstimator3):
    """A trivial estimator."""


def test_clone() -> None:
    # Tests that clone creates a correct deep copy.
    # We create an estimator, make a copy of its original state
    # (which, in this case, is the current state of the estimator),
    # and check that the obtained copy is a correct deep copy.

    encoder = OneHotEncoderDF(drop="first", sparse=False)
    new_encoder = encoder.clone()
    assert encoder is not new_encoder
    assert encoder.get_params() == new_encoder.get_params()

    encoder = OneHotEncoderDF(handle_unknown="ignore", sparse=False)
    new_encoder = clone(encoder)

    assert encoder is not new_encoder


def test_clone_2() -> None:
    # Tests that clone doesn't copy everything.
    # We first create an estimator, give it an own attribute, and
    # make a copy of its original state. Then we check that the copy doesn't
    # have the specific attribute we manually added to the initial estimator.

    encoder = OneHotEncoderDF(drop="first", sparse=False)

    encoder.own_attribute = "test"
    new_encoder = encoder.clone()

    assert not hasattr(new_encoder, "own_attribute")


def test_clone_empty_array() -> None:
    # Regression test for cloning estimators with empty arrays
    clf = DummyEstimatorDF(empty=np.array([]))
    clf2 = clf.clone()
    assert_array_equal(clf.empty, clf2.empty)

    clf = DummyEstimatorDF(empty=sp.csr_matrix(np.array([[0]])))
    clf2 = clf.clone()
    assert_array_equal(clf.empty.data, clf2.empty.data)


def test_clone_nan() -> None:
    # Regression test for cloning estimators with default parameter as np.nan
    clf = DummyEstimatorDF(empty=np.nan)
    clf2 = clf.clone()

    assert clf.empty is clf2.empty


def test_clone_sparse_matrices() -> None:
    sparse_matrix_classes = [
        getattr(sp, name) for name in dir(sp) if name.endswith("_matrix")
    ]

    for cls in sparse_matrix_classes:
        sparse_matrix = cls(np.eye(5))
        clf = DummyEstimatorDF(empty=sparse_matrix)
        clf_cloned = clf.clone()
        assert clf.empty.__class__ is clf_cloned.empty.__class__
        assert_array_equal(clf.empty.toarray(), clf_cloned.empty.toarray())


def test_clone_estimator_types() -> None:
    # Check that clone works for parameters that are types rather than
    # instances
    clf = DummyEstimatorDF(empty=DummyEstimator)
    clf2 = clf.clone()

    assert clf.empty is clf2.empty


def test_repr() -> None:
    # Smoke test the repr of the base estimator.
    repr(DummyEstimatorDF())

    estimator = DummyEstimator2DF(
        a=DummyEstimator3DF(c=None), b=DummyEstimator3DF(c=1, d=2)
    )
    assert freeze(make_expression(estimator)) == freeze(
        Id.DummyEstimator2DF(
            a=Id.DummyEstimator3DF(c=None), b=Id.DummyEstimator3DF(c=1, d=2)
        )
    )
    assert repr(estimator) == (
        "DummyEstimator2DF(a=DummyEstimator3DF(c=None), "
        "b=DummyEstimator3DF(c=1, d=2))"
    )

    assert len(repr(DummyEstimator2DF(a=["long_params"] * 1000))) == 15021


def test_str() -> None:
    # Smoke test the str of the base estimator
    my_estimator = DummyEstimatorDF()
    str(my_estimator)


def test_get_params() -> None:
    test = DummyEstimator2DF(DummyEstimator3DF(), DummyEstimator3DF())

    assert "a__d" in test.get_params(deep=True)
    assert "a__d" not in test.get_params(deep=False)

    # noinspection PyTypeChecker
    test.set_params(a__d=2)
    assert test.a.d == 2
    with pytest.raises(ValueError):
        test.set_params(a__a=2)


def test_is_classifier() -> None:
    svc = SVCDF()
    assert is_classifier(svc)
    assert is_classifier(GridSearchCV(svc, {"C": [0.1, 1]}))
    assert is_classifier(PipelineDF([("svc", svc)]))
    assert is_classifier(Pipeline([("svc_cv", GridSearchCV(svc, {"C": [0.1, 1]}))]))


def test_set_params() -> None:
    # test nested estimator parameter setting
    clf = Pipeline([("svc", SVCDF())])
    # non-existing parameter in svc
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        clf.set_params(svc__stupid_param=True)
    # non-existing parameter of pipeline
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        clf.set_params(svm__stupid_param=True)


def test_set_params_updates_valid_params() -> None:
    # Check that set_params tries to set SVC().C, not
    # DecisionTreeClassifier().C
    gs = GridSearchCV(DecisionTreeClassifierDF(), {})
    # noinspection PyTypeChecker
    gs.set_params(estimator=SVCDF(), estimator__C=42.0)
    assert gs.estimator.C == 42.0


# noinspection PyUnusedLocal
def test_native_class_validation() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "native class RandomForestClassifier cannot be used with wrapper class "
            "MismatchedNativeClass1 because it does not implement RegressorMixin"
        ),
    ):

        class MismatchedNativeClass1(
            RegressorWrapperDF[RandomForestClassifier], native=RandomForestClassifier
        ):
            pass

    with pytest.raises(
        TypeError,
        match=(
            "native class RandomForestRegressor cannot be used with wrapper class "
            "MismatchedNativeClass2 because it does not implement ClassifierMixin"
        ),
    ):

        class MismatchedNativeClass2(
            ClassifierWrapperDF[RandomForestRegressor], native=RandomForestRegressor
        ):
            pass

    with pytest.raises(
        TypeError,
        match=(
            "native class RandomForestRegressor cannot be used with wrapper class "
            "MismatchedNativeClass3 because it does not implement ClusterMixin"
        ),
    ):

        class MismatchedNativeClass3(
            KMeansBaseWrapperDF[RandomForestRegressor], native=RandomForestRegressor
        ):
            pass

    with pytest.raises(
        TypeError,
        match=(
            "native class RandomForestRegressor cannot be used with wrapper class "
            "MismatchedNativeClass4 because it does not implement TransformerMixin"
        ),
    ):

        class MismatchedNativeClass4(
            FeatureUnionWrapperDF, native=RandomForestRegressor
        ):
            pass

    with pytest.raises(
        TypeError,
        match=(
            "native class RandomForestRegressor cannot be used with wrapper class "
            "MismatchedNativeClass5 because it does not implement Pipeline"
        ),
    ):

        class MismatchedNativeClass5(PipelineDF, native=RandomForestRegressor):
            pass
