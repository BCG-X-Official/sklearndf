import itertools
from typing import *

import pytest
import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition

import gamma.sklearndf.classification
import gamma.sklearndf.regression
import gamma.sklearndf.transformation
import gamma.sklearndf.pipeline
from test import check_sklearn_version
from test.conftest import UNSUPPORTED_SKLEARN_PACKAGES

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

PIPELINE_COVERAGE_EXCLUDES = {
    # exclude all Base classes, named starting with "Base" or "_Base":
    r"^_?Base.*",
    # exclude all Mixin classes, named ending on Mixin:
    r".*Mixin$",
}

SKIP_COVERAGE_FOR = {
    c.__name__
    for c in list_classes(
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
            from_modules=from_modules, matching=".*", excluding=excluding,
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


def _handle_not_covered_type(cls: Type) -> None:
    f_cls_name = f"{cls.__module__}.{cls.__name__}"
    if cls.__name__ in SKIP_COVERAGE_FOR:
        pytest.skip(f"Class '{f_cls_name} is not wrapped but marked as unsupported!' ")
    else:
        raise ValueError(f"Class '{f_cls_name}' is not wrapped!")


@pytest.mark.parametrize(
    argnames="sklearn_classifier_cls", argvalues=sklearn_classifier_classes()
)
def test_classifier_coverage(sklearn_classifier_cls: Type) -> None:
    """ Check if each sklearn classifier has a wrapped sklearndf counterpart. """
    sklearndf_cls_to_sklearn_cls = sklearndf_to_wrapped(gamma.sklearndf.classification)

    if sklearn_classifier_cls not in sklearndf_cls_to_sklearn_cls.values():
        _handle_not_covered_type(sklearn_classifier_cls)


@pytest.mark.parametrize(
    argnames="sklearn_regressor_cls", argvalues=sklearn_regressor_classes()
)
def test_regressor_coverage(sklearn_regressor_cls: List[Type]) -> None:
    """ Check if each sklearn regressor has a wrapped sklearndf counterpart. """
    sklearndf_cls_to_sklearn_cls = sklearndf_to_wrapped(gamma.sklearndf.regression)

    if sklearn_regressor_cls not in sklearndf_cls_to_sklearn_cls.values():
        _handle_not_covered_type(sklearn_regressor_cls)


@pytest.mark.parametrize(
    argnames="sklearn_transformer_cls", argvalues=sklearn_transformer_classes()
)
def test_transformer_coverage(sklearn_transformer_cls: Type) -> None:
    """ Check if each sklearn transformer has a wrapped sklearndf counterpart. """

    sklearndf_cls_to_sklearn_cls = sklearndf_to_wrapped(gamma.sklearndf.transformation)

    if sklearn_transformer_cls not in sklearndf_cls_to_sklearn_cls.values():
        _handle_not_covered_type(sklearn_transformer_cls)


@pytest.mark.parametrize(
    argnames="sklearn_pipeline_cls", argvalues=sklearn_pipeline_classes()
)
def test_pipeline_coverage(sklearn_pipeline_cls: Type) -> None:
    """ Check if each sklearn pipeline estimator has
        a wrapped sklearndf counterpart. """

    sklearndf_cls_to_sklearn_cls = sklearndf_to_wrapped(gamma.sklearndf.pipeline)

    if sklearn_pipeline_cls not in sklearndf_cls_to_sklearn_cls.values():
        _handle_not_covered_type(sklearn_pipeline_cls)
