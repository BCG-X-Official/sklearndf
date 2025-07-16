import itertools
from collections.abc import Iterable
from types import ModuleType
from typing import Optional, TypeVar, Union

import pytest
import sklearn
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import _BaseComposition

import sklearndf.classification
import sklearndf.clustering
import sklearndf.pipeline
import sklearndf.regression
import sklearndf.transformation
from ..conftest import UNSUPPORTED_SKLEARN_PACKAGES
from . import find_all_submodules, iterate_classes, sklearn_delegate_classes
from sklearndf import EstimatorDF

T = TypeVar("T")


GENERAL_COVERAGE_EXCLUSIONS = {
    # exclude all private classes:
    r"^_",
    # exclude all Base classes:
    r"^Base[A-Z]",
    # exclude all Mixin classes:
    r".*Mixin$",
}

CLASSIFIER_COVERAGE_EXCLUSIONS = {
    *GENERAL_COVERAGE_EXCLUSIONS,
    # Base classes and Mixins not following the convention
    "ForestClassifier",
    "_IdentityClassifier",
}


REGRESSOR_COVERAGE_EXCLUSIONS = {
    *GENERAL_COVERAGE_EXCLUSIONS,
    # Base classes and mix-ins
    "ForestRegressor",
    "GeneralizedLinearRegressor",
    # Private classes
    "_SigmoidCalibration",
    "_PLS",
}


TRANSFORMER_COVERAGE_EXCLUSIONS = {
    *GENERAL_COVERAGE_EXCLUSIONS,
    # class "Imputer" was deprecated in 0.20 and removed in 0.22
    "Imputer",
    # class "AgglomerationTransform" is just a mix-in class and
    # isn't meant to be used directly
    "AgglomerationTransform",
}


PIPELINE_COVERAGE_EXCLUSIONS = GENERAL_COVERAGE_EXCLUSIONS


CLUSTERER_COVERAGE_EXCLUSIONS = {
    *GENERAL_COVERAGE_EXCLUSIONS,
}


UNSUPPORTED_SKLEARN_CLASSES = {
    sklearn_class.__name__
    for sklearn_class in iterate_classes(
        from_modules=itertools.chain.from_iterable(
            (p, *find_all_submodules(p)) for p in UNSUPPORTED_SKLEARN_PACKAGES
        ),
        matching=".*",
    )
}


def _find_sklearn_classes_to_cover(
    from_modules: Union[ModuleType, Iterable[ModuleType]],
    subclass_of: type[T],
    excluding: Optional[Union[str, Iterable[str]]] = None,
) -> list[type[T]]:
    return [
        cls
        for cls in iterate_classes(
            from_modules=from_modules, matching=".*", excluding=excluding
        )
        if issubclass(cls, subclass_of)
    ]


def sklearn_classifier_classes() -> list[type]:
    return _find_sklearn_classes_to_cover(
        from_modules=find_all_submodules(sklearn),
        subclass_of=ClassifierMixin,
        excluding=CLASSIFIER_COVERAGE_EXCLUSIONS,
    )


def sklearn_regressor_classes() -> list[type]:
    return _find_sklearn_classes_to_cover(
        from_modules=find_all_submodules(sklearn),
        subclass_of=RegressorMixin,
        excluding=REGRESSOR_COVERAGE_EXCLUSIONS,
    )


def sklearn_pipeline_classes() -> list[type]:
    pipeline_modules = find_all_submodules(sklearn.pipeline)
    pipeline_modules.add(sklearn.pipeline)

    return _find_sklearn_classes_to_cover(
        from_modules=pipeline_modules,
        subclass_of=_BaseComposition,
        excluding=PIPELINE_COVERAGE_EXCLUSIONS,
    )


def sklearn_transformer_classes() -> list[type]:
    """Return all classes that are 'just' transformers, not learners or pipelines."""
    transformer_mixin_classes = [
        cls
        for cls in iterate_classes(
            from_modules=find_all_submodules(sklearn),
            matching=".*",
            excluding=TRANSFORMER_COVERAGE_EXCLUSIONS,
        )
        if issubclass(cls, TransformerMixin)
    ]

    transformer_classes = list(
        set(transformer_mixin_classes)
        .difference(sklearn_classifier_classes())
        .difference(sklearn_regressor_classes())
        .difference(sklearn_pipeline_classes())
        .difference(sklearn_clusterer_classes())
    )

    return transformer_classes


def sklearn_clusterer_classes() -> list[type]:
    return _find_sklearn_classes_to_cover(
        from_modules=find_all_submodules(sklearn),
        subclass_of=ClusterMixin,
        excluding=CLUSTERER_COVERAGE_EXCLUSIONS,
    )


def _check_unexpected_sklearn_class(cls: type) -> None:
    f_cls_name = f"{cls.__module__}.{cls.__name__}"
    if cls.__name__ in UNSUPPORTED_SKLEARN_CLASSES:
        pytest.skip(f"Class {f_cls_name} is not wrapped but marked as unsupported")
    else:
        raise ValueError(f"Class {f_cls_name} is not wrapped")


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearn_classifier_cls", argvalues=sklearn_classifier_classes()
)
def test_classifier_coverage(sklearn_classifier_cls: type[ClassifierMixin]) -> None:
    """Check if each sklearn classifier has a wrapped sklearndf counterpart."""
    sklearn_classes: dict[type[BaseEstimator], type[EstimatorDF]] = (
        sklearn_delegate_classes(sklearndf.classification)
    )

    if sklearn_classifier_cls not in sklearn_classes:
        _check_unexpected_sklearn_class(sklearn_classifier_cls)


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearn_regressor_cls", argvalues=sklearn_regressor_classes()
)
def test_regressor_coverage(sklearn_regressor_cls: type[RegressorMixin]) -> None:
    """Check if each sklearn regressor has a wrapped sklearndf counterpart."""
    sklearn_classes: dict[type[BaseEstimator], type[EstimatorDF]] = (
        sklearn_delegate_classes(sklearndf.regression)
    )

    if sklearn_regressor_cls not in sklearn_classes:
        _check_unexpected_sklearn_class(sklearn_regressor_cls)


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearn_transformer_cls", argvalues=sklearn_transformer_classes()
)
def test_transformer_coverage(sklearn_transformer_cls: type[TransformerMixin]) -> None:
    """Check if each sklearn transformer has a wrapped sklearndf counterpart."""

    sklearn_classes: dict[type[BaseEstimator], type[EstimatorDF]] = (
        sklearn_delegate_classes(sklearndf.transformation)
    )

    if sklearn_transformer_cls not in sklearn_classes:
        _check_unexpected_sklearn_class(sklearn_transformer_cls)


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearn_pipeline_cls", argvalues=sklearn_pipeline_classes()
)
def test_pipeline_coverage(sklearn_pipeline_cls: type[Pipeline]) -> None:
    """Check if each sklearn pipeline estimator has
    a wrapped sklearndf counterpart."""

    # noinspection PyTypeChecker
    sklearn_classes = sklearn_delegate_classes(sklearndf.pipeline)

    if sklearn_pipeline_cls not in sklearn_classes:
        _check_unexpected_sklearn_class(sklearn_pipeline_cls)


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearn_clusterer_cls", argvalues=sklearn_clusterer_classes()
)
def test_clusterer_coverage(sklearn_clusterer_cls: type[ClusterMixin]) -> None:
    """Check if each sklearn clusterer has a wrapped sklearndf counterpart."""
    sklearn_classes: dict[type[BaseEstimator], type[EstimatorDF]] = (
        sklearn_delegate_classes(sklearndf.clustering)
    )

    if sklearn_clusterer_cls not in sklearn_classes:
        _check_unexpected_sklearn_class(sklearn_clusterer_cls)
