from typing import *

import pytest
import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin

import gamma.sklearndf.classification
import gamma.sklearndf.regression
import gamma.sklearndf.transformation
from test import check_sklearn_version

from test.gamma.sklearndf import find_all_submodules, list_classes, sklearndf_to_wrapped

Module: type = Any


CLASSIFIER_COVERAGE_EXCLUDES = {
    # exclude all Base classes, named starting with "Base" or "_Base":
    r"^_?Base.*",
    # exclude all Mixin classes, named ending on Mixin:
    r".*Mixin$",
    # Base classes and Mixins, not following the convention
    sklearn.ensemble._forest.ForestClassifier.__name__,
}

if check_sklearn_version(minimum="0.23"):
    added_in_v023 = (sklearn.linear_model._ridge._IdentityClassifier.__name__,)
    CLASSIFIER_COVERAGE_EXCLUDES.update(added_in_v023)

REGRESSOR_COVERAGE_EXCLUDES = {
    # exclude all Base classes, named starting with "Base" or "_Base":
    r"^_?Base.*",
    # exclude all Mixin classes, named ending on Mixin:
    r".*Mixin$",
    # Base classes and Mixins, not following the convention -->
    sklearn.ensemble._forest.ForestRegressor.__name__,
    # <--- Base classes and Mixins, not following the convention
    # Private classes -->
    sklearn.calibration._SigmoidCalibration.__name__,
    sklearn.cross_decomposition._pls._PLS.__name__
    # <-- Private classes
}

TRANSFORMER_COVERAGE_EXCLUDES = CLASSIFIER_COVERAGE_EXCLUDES.union(
    REGRESSOR_COVERAGE_EXCLUDES
)


@pytest.fixture
def sklearn_classifier_classes() -> List[Type]:
    return [
        cls
        for cls in list_classes(
            from_modules=find_all_submodules(sklearn),
            matching=".*",
            excluding=CLASSIFIER_COVERAGE_EXCLUDES,
        )
        if issubclass(cls, ClassifierMixin)
    ]


@pytest.fixture
def sklearn_regressor_classes() -> List[Type]:
    return [
        cls
        for cls in list_classes(
            from_modules=find_all_submodules(sklearn),
            matching=".*",
            excluding=REGRESSOR_COVERAGE_EXCLUDES,
        )
        if issubclass(cls, RegressorMixin)
    ]


@pytest.fixture
def sklearn_transformer_classes(
    sklearn_classifier_classes: List[Type], sklearn_regressor_classes: List[Type]
) -> List[Type]:
    """ Return all classses that are 'just' transformers, not learners."""
    transfomer_mixin_classes = [
        cls
        for cls in list_classes(
            from_modules=find_all_submodules(sklearn),
            matching=".*",
            excluding=TRANSFORMER_COVERAGE_EXCLUDES,
        )
        if issubclass(cls, TransformerMixin)
    ]

    transformer_classes = list(
        set(transfomer_mixin_classes)
        .difference(sklearn_classifier_classes)
        .difference(sklearn_regressor_classes)
    )

    return transformer_classes


def test_classifier_coverage(sklearn_classifier_classes: List[Type]) -> None:
    """ Check if each sklearn classifier has a wrapped sklearndf counterpart. """
    sklearndf_cls_to_sklearn_cls = sklearndf_to_wrapped(gamma.sklearndf.classification)

    missing = []

    for sklearn_cls in sklearn_classifier_classes:
        if sklearn_cls not in sklearndf_cls_to_sklearn_cls.values():
            missing.append(sklearn_cls)

    if missing:
        raise ValueError(
            f"Class(es): {','.join([m.__module__ +'.'+ m.__name__ for m in missing])} is/are not wrapped!"
        )


def test_regressor_coverage(sklearn_regressor_classes: List[Type]) -> None:
    """ Check if each sklearn regressor has a wrapped sklearndf counterpart. """
    sklearndf_cls_to_sklearn_cls = sklearndf_to_wrapped(gamma.sklearndf.regression)

    missing = []

    for sklearn_cls in sklearn_regressor_classes:
        if sklearn_cls not in sklearndf_cls_to_sklearn_cls.values():
            missing.append(sklearn_cls)

    if missing:
        raise ValueError(
            f"Class(es): {','.join([m.__module__ +'.'+ m.__name__ for m in missing])} is/are not wrapped!"
        )


def test_transformer_coverage(sklearn_transformer_classes: List[Type]) -> None:
    """ Check if each sklearn transformer has a wrapped sklearndf counterpart. """

    sklearndf_cls_to_sklearn_cls = sklearndf_to_wrapped(gamma.sklearndf.transformation)

    missing = []

    for sklearn_cls in sklearn_transformer_classes:
        if sklearn_cls not in sklearndf_cls_to_sklearn_cls.values():
            missing.append(sklearn_cls)

    if missing:
        raise ValueError(
            f"Class(es): {','.join([m.__module__ +'.'+ m.__name__ for m in missing])} is/are not wrapped!"
        )
