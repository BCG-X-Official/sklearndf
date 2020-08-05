import itertools
from typing import *
from typing import Dict

import pytest
import sklearn
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.utils.metaestimators import _BaseComposition

import gamma.sklearndf.classification
import gamma.sklearndf.pipeline
import gamma.sklearndf.regression
import gamma.sklearndf.transformation
from gamma.sklearndf import BaseEstimatorDF
from test import check_sklearn_version
from test.conftest import UNSUPPORTED_SKLEARN_PACKAGES
from test.gamma.sklearndf import (
    find_all_submodules,
    list_classes,
    sklearn_delegate_classes,
)

Module = type(sklearn)


CLASSIFIER_COVERAGE_EXCLUDES = {
    # exclude all Base classes, named starting with "Base" or "_Base":
    r"^_?Base.*",
    # exclude all Mixin classes, named ending on Mixin:
    r".*Mixin$",
    # Base classes and Mixins, not following the convention
    "ForestClassifier",
}

if check_sklearn_version(minimum="0.23"):
    added_in_v023 = ("_IdentityClassifier",)
    CLASSIFIER_COVERAGE_EXCLUDES.update(added_in_v023)

REGRESSOR_COVERAGE_EXCLUDES = {
    # exclude all Base classes, named starting with "Base" or "_Base":
    r"^_?Base.*",
    # exclude all Mixin classes, named ending on Mixin:
    r".*Mixin$",
    # Base classes and Mixins, not following the convention -->
    "ForestRegressor",
    # <--- Base classes and Mixins, not following the convention
    # Private classes -->
    "_SigmoidCalibration",
    "_PLS"
    # <-- Private classes
}


TRANSFORMER_COVERAGE_EXCLUDES = (
    {
        # class "Imputer" was deprecated in 0.20 and removed in 0.22
        "Imputer"
    }
    | CLASSIFIER_COVERAGE_EXCLUDES
    | REGRESSOR_COVERAGE_EXCLUDES
)


PIPELINE_COVERAGE_EXCLUDES = {
    # exclude all Base classes, named starting with "Base" or "_Base":
    r"^_?Base.*",
    # exclude all Mixin classes, named ending on Mixin:
    r".*Mixin$",
}

UNSUPPORTED_SKLEARN_CLASSES = {
    sklearn_class.__name__
    for sklearn_class in list_classes(
        from_modules=itertools.chain.from_iterable(
            find_all_submodules(p) for p in UNSUPPORTED_SKLEARN_PACKAGES
        ),
        matching=".*",
    )
}


def _find_sklearn_classes_to_cover(
    from_modules: Union[Module, Iterable[Module]],
    subclass_of: Type,
    excluding: Optional[Union[str, Iterable[str]]] = None,
) -> List[Type]:
    return [
        cls
        for cls in list_classes(
            from_modules=from_modules, matching=".*", excluding=excluding
        )
        if issubclass(cls, subclass_of)
    ]


def sklearn_classifier_classes() -> List[Type]:
    return _find_sklearn_classes_to_cover(
        from_modules=find_all_submodules(sklearn),
        subclass_of=ClassifierMixin,
        excluding=CLASSIFIER_COVERAGE_EXCLUDES,
    )


def sklearn_regressor_classes() -> List[Type]:
    return _find_sklearn_classes_to_cover(
        from_modules=find_all_submodules(sklearn),
        subclass_of=RegressorMixin,
        excluding=REGRESSOR_COVERAGE_EXCLUDES,
    )


def sklearn_pipeline_classes() -> List[Type]:

    pipeline_modules = find_all_submodules(sklearn.pipeline)
    pipeline_modules.add(sklearn.pipeline)

    return _find_sklearn_classes_to_cover(
        from_modules=pipeline_modules,
        subclass_of=_BaseComposition,
        excluding=PIPELINE_COVERAGE_EXCLUDES,
    )


def sklearn_transformer_classes() -> List[Type]:
    """ Return all classses that are 'just' transformers, not learners or pipelines."""
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
        .difference(sklearn_classifier_classes())
        .difference(sklearn_regressor_classes())
        .difference(sklearn_pipeline_classes())
    )

    return transformer_classes


def _check_unexpected_sklearn_class(cls: Type) -> None:
    f_cls_name = f"{cls.__module__}.{cls.__name__}"
    if cls.__name__ in UNSUPPORTED_SKLEARN_CLASSES:
        pytest.skip(f"Class '{f_cls_name} is not wrapped but marked as unsupported' ")
    else:
        raise ValueError(f"Class '{f_cls_name}' is not wrapped")


@pytest.mark.parametrize(
    argnames="sklearn_classifier_cls", argvalues=sklearn_classifier_classes()
)
def test_classifier_coverage(sklearn_classifier_cls: Type[ClassifierMixin]) -> None:
    """ Check if each sklearn classifier has a wrapped sklearndf counterpart. """
    sklearn_classes: Dict[
        BaseEstimator, _BaseEstimatorWrapperDF
    ] = sklearn_delegate_classes(gamma.sklearndf.classification)

    if sklearn_classifier_cls not in sklearn_classes:
        _check_unexpected_sklearn_class(sklearn_classifier_cls)


@pytest.mark.parametrize(
    argnames="sklearn_regressor_cls", argvalues=sklearn_regressor_classes()
)
def test_regressor_coverage(sklearn_regressor_cls: Type[RegressorMixin]) -> None:
    """ Check if each sklearn regressor has a wrapped sklearndf counterpart. """
    sklearn_classes: Dict[
        BaseEstimator, _BaseEstimatorWrapperDF
    ] = sklearn_delegate_classes(gamma.sklearndf.regression)

    if sklearn_regressor_cls not in sklearn_classes:
        _check_unexpected_sklearn_class(sklearn_regressor_cls)


@pytest.mark.parametrize(
    argnames="sklearn_transformer_cls", argvalues=sklearn_transformer_classes()
)
def test_transformer_coverage(sklearn_transformer_cls: Type[TransformerMixin]) -> None:
    """ Check if each sklearn transformer has a wrapped sklearndf counterpart. """

    sklearn_classes: Dict[
        BaseEstimator, _BaseEstimatorWrapperDF
    ] = sklearn_delegate_classes(gamma.sklearndf.transformation)

    if sklearn_transformer_cls not in sklearn_classes:
        _check_unexpected_sklearn_class(sklearn_transformer_cls)


@pytest.mark.parametrize(
    argnames="sklearn_pipeline_cls", argvalues=sklearn_pipeline_classes()
)
def test_pipeline_coverage(sklearn_pipeline_cls: Type) -> None:
    """ Check if each sklearn pipeline estimator has
        a wrapped sklearndf counterpart. """

    sklearn_classes = sklearn_delegate_classes(gamma.sklearndf.pipeline)

    if sklearn_pipeline_cls not in sklearn_classes:
        _check_unexpected_sklearn_class(sklearn_pipeline_cls)
