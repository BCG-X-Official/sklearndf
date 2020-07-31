from typing import *

import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin

import gamma.sklearndf.classification
import gamma.sklearndf.regression
from test import check_sklearn_version

from test.gamma.sklearndf import find_all_submodules, list_classes, sklearndf_to_wrapped

Module: type = Any


CLASSIFIER_COVERAGE_EXCLUDES = {
    # todo: check to prune parts of this list automatically,
    #  for when one mod/pkg is private
    # Base classes and Mixins -->
    sklearn.naive_bayes._BaseNB.__name__,
    sklearn.linear_model._base.LinearClassifierMixin.__name__,
    sklearn.naive_bayes._BaseDiscreteNB.__name__,
    sklearn.base.ClassifierMixin.__name__,
    sklearn.svm._base.BaseSVC.__name__,
    sklearn.semi_supervised._label_propagation.BaseLabelPropagation.__name__,
    sklearn.ensemble._forest.ForestClassifier.__name__,
    sklearn.linear_model._stochastic_gradient.BaseSGDClassifier.__name__,
    # <--- Base classes and Mixins
}

if check_sklearn_version(maximum="0.23"):
    up_to_v0_24 = (
        # deprecated in version 0.22 and will be removed in version 0.24! -->
        sklearn.naive_bayes.BaseNB.__name__,
        sklearn.naive_bayes.BaseDiscreteNB.__name__,
        # <-- deprecated in version 0.22 and will be removed in version 0.24!
    )
    CLASSIFIER_COVERAGE_EXCLUDES.update(up_to_v0_24)

if check_sklearn_version(minimum="0.23"):
    added_in_v023 = (
        sklearn.naive_bayes.BaseNB.__name__,
        sklearn.naive_bayes.BaseDiscreteNB.__name__,
        sklearn.linear_model._ridge._IdentityClassifier.__name__,
    )
    CLASSIFIER_COVERAGE_EXCLUDES.update(added_in_v023)

REGRESSOR_COVERAGE_EXCLUDES = (
    # Base classes and Mixins -->
    sklearn.linear_model._stochastic_gradient.BaseSGDRegressor.__name__,
    sklearn.base.RegressorMixin.__name__,
    sklearn.ensemble._forest.ForestRegressor.__name__,
    # <--- Base classes and Mixins
    # Private classes -->
    sklearn.calibration._SigmoidCalibration.__name__,
    sklearn.cross_decomposition._pls._PLS.__name__
    # <-- Private classes
)


def test_classifier_coverage() -> None:
    """ Check if each sklearn classifier has a wrapped sklearndf counterpart. """
    sklearn_classifier_classes = [
        cls
        for cls in list_classes(
            from_modules=find_all_submodules(sklearn),
            matching=".*",
            excluding=CLASSIFIER_COVERAGE_EXCLUDES,
        )
        if issubclass(cls, ClassifierMixin)
    ]
    sklearndf_cls_to_sklearn_cls = sklearndf_to_wrapped(gamma.sklearndf.classification)

    missing = []

    for sklearn_cls in sklearn_classifier_classes:
        if sklearn_cls not in sklearndf_cls_to_sklearn_cls.values():
            missing.append(sklearn_cls)

    if missing:
        raise ValueError(
            f"Class(es): {','.join([m.__module__ +'.'+ m.__name__ for m in missing])} is/are not wrapped!"
        )


def test_regressor_coverage() -> None:
    """ Check if each sklearn regressor has a wrapped sklearndf counterpart. """
    sklearn_regressor_classes = [
        cls
        for cls in list_classes(
            from_modules=find_all_submodules(sklearn),
            matching=".*",
            excluding=REGRESSOR_COVERAGE_EXCLUDES,
        )
        if issubclass(cls, RegressorMixin)
    ]
    sklearndf_cls_to_sklearn_cls = sklearndf_to_wrapped(gamma.sklearndf.regression)

    missing = []

    for sklearn_cls in sklearn_regressor_classes:
        if sklearn_cls not in sklearndf_cls_to_sklearn_cls.values():
            missing.append(sklearn_cls)

    if missing:
        raise ValueError(
            f"Class(es): {','.join([m.__module__ +'.'+ m.__name__ for m in missing])} is/are not wrapped!"
        )
