# inspired by:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tests/test_base.py
from abc import ABCMeta

import numpy as np

# noinspection PyPackageRequirements
import scipy.sparse as sp
from numpy.testing import assert_array_equal, assert_raises
from sklearn import clone
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearndf import EstimatorDF

# noinspection PyProtectedMember
from sklearndf._wrapper import _EstimatorWrapperDF, df_estimator
from sklearndf.classification import SVCDF, DecisionTreeClassifierDF
from sklearndf.pipeline import PipelineDF
from sklearndf.transformation import OneHotEncoderDF


class _DummyEstimator(BaseEstimator):
    def __init__(self, l1=0, empty=None) -> None:
        self.l1 = l1
        self.empty = empty


class _DummyEstimator2(BaseEstimator):
    def __init__(self, a=None, b=None) -> None:
        self.a = a
        self.b = b


class _DummyEstimator3(BaseEstimator):
    def __init__(self, c=None, d=None) -> None:
        self.c = c
        self.d = d


@df_estimator(df_wrapper_type=_EstimatorWrapperDF)
class _DummyEstimatorDF(EstimatorDF, _DummyEstimator, metaclass=ABCMeta):
    pass


@df_estimator(df_wrapper_type=_EstimatorWrapperDF)
class _DummyEstimator2DF(EstimatorDF, _DummyEstimator2, metaclass=ABCMeta):
    pass


@df_estimator(df_wrapper_type=_EstimatorWrapperDF)
class _DummyEstimator3DF(EstimatorDF, _DummyEstimator3, metaclass=ABCMeta):
    pass


def test_clone() -> None:
    # Tests that clone creates a correct deep copy.
    # We create an estimator, make a copy of its original state
    # (which, in this case, is the current state of the estimator),
    # and check that the obtained copy is a correct deep copy.

    encoder = OneHotEncoderDF(drop="first", sparse=False)
    new_encoder = clone(encoder)
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
    new_encoder = clone(encoder)

    assert not hasattr(new_encoder, "own_attribute")


def test_clone_empty_array() -> None:
    # Regression test for cloning estimators with empty arrays
    clf = _DummyEstimatorDF(empty=np.array([]))
    clf2 = clone(clf)
    assert_array_equal(clf.empty, clf2.empty)

    clf = _DummyEstimatorDF(empty=sp.csr_matrix(np.array([[0]])))
    clf2 = clone(clf)
    assert_array_equal(clf.empty.data, clf2.empty.data)


def test_clone_nan() -> None:
    # Regression test for cloning estimators with default parameter as np.nan
    clf = _DummyEstimatorDF(empty=np.nan)
    clf2 = clone(clf)

    assert clf.empty is clf2.empty


def test_clone_sparse_matrices() -> None:
    sparse_matrix_classes = [
        getattr(sp, name) for name in dir(sp) if name.endswith("_matrix")
    ]

    for cls in sparse_matrix_classes:
        sparse_matrix = cls(np.eye(5))
        clf = _DummyEstimatorDF(empty=sparse_matrix)
        clf_cloned = clone(clf)
        assert clf.empty.__class__ is clf_cloned.empty.__class__
        assert_array_equal(clf.empty.toarray(), clf_cloned.empty.toarray())


def test_clone_estimator_types() -> None:
    # Check that clone works for parameters that are types rather than
    # instances
    clf = _DummyEstimatorDF(empty=_DummyEstimator)
    clf2 = clone(clf)

    assert clf.empty is clf2.empty


def test_repr() -> None:
    # Smoke test the repr of the base estimator.
    my_estimator = _DummyEstimatorDF()
    repr(my_estimator)

    test = _DummyEstimator2DF(_DummyEstimator3DF(), _DummyEstimator3DF())

    assert repr(test) == (
        "_DummyEstimator2DF(a=_DummyEstimator3DF(c=None, d=None),\n"
        "                   b=_DummyEstimator3DF(c=None, d=None))"
    )

    some_est = _DummyEstimator2DF(a=["long_params"] * 1000)

    assert len(repr(some_est)) == 702


def test_str() -> None:
    # Smoke test the str of the base estimator
    my_estimator = _DummyEstimatorDF()
    str(my_estimator)


def test_get_params() -> None:
    test = _DummyEstimator2DF(_DummyEstimator3DF(), _DummyEstimator3DF())

    assert "a__d" in test.get_params(deep=True)
    assert "a__d" not in test.get_params(deep=False)

    test.set_params(a__d=2)
    assert test.a.d == 2
    assert_raises(ValueError, test.set_params, a__a=2)


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
    assert_raises(ValueError, clf.set_params, svc__stupid_param=True)
    # non-existing parameter of pipeline
    assert_raises(ValueError, clf.set_params, svm__stupid_param=True)


def test_set_params_updates_valid_params() -> None:
    # Check that set_params tries to set SVC().C, not
    # DecisionTreeClassifier().C
    gs = GridSearchCV(DecisionTreeClassifierDF(), {})
    # noinspection PyTypeChecker
    gs.set_params(estimator=SVCDF(), estimator__C=42.0)
    assert gs.estimator.C == 42.0
