import re
import sys
from collections.abc import Iterable
from types import ModuleType
from typing import Optional, Union

import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion

from sklearndf import EstimatorDF, LearnerDF, TransformerDF, __sklearn_version__
from sklearndf.pipeline.wrapper import FeatureUnionSparseFrames
from sklearndf.transformation.wrapper import ColumnTransformerSparseFrames
from sklearndf.wrapper import EstimatorWrapperDF

OVERRIDDEN_SKLEARN_CLASSES = {
    ColumnTransformerSparseFrames: ColumnTransformer,
    FeatureUnionSparseFrames: FeatureUnion,
}


def find_all_classes(
    *modules: ModuleType,
) -> set[type[EstimatorWrapperDF[BaseEstimator]]]:
    """Finds all Class members in given module/modules."""
    types: set[type[EstimatorWrapperDF[BaseEstimator]]] = set()

    def _add_classes_from_module(_m: ModuleType) -> None:
        member: type[EstimatorWrapperDF[BaseEstimator]]
        for member in vars(_m).values():
            if isinstance(member, type):
                types.add(member)

    for module in modules:
        _add_classes_from_module(module)

    return types


def find_all_submodules(parent_module: ModuleType) -> set[ModuleType]:
    """Finds all submodules for a parent module."""
    parent_name = f"{parent_module.__name__}."
    return {
        module
        for module_name, module in sys.modules.items()
        if module_name.startswith(parent_name)
    }


def sklearn_delegate_classes(
    module: ModuleType,
) -> dict[type[BaseEstimator], type[EstimatorWrapperDF[BaseEstimator]]]:
    """
    Create a dictionary mapping sklearn classes to their corresponding sklearndf
    classes.
    """
    return {
        OVERRIDDEN_SKLEARN_CLASSES.get(
            df_class.__wrapped__, df_class.__wrapped__
        ): df_class
        for df_class in find_all_classes(module)
        # we only consider non-abstract wrapper classes wrapping a specific native class
        if issubclass(df_class, EstimatorWrapperDF) and hasattr(df_class, "__wrapped__")
    }


def iterate_classes(
    from_modules: Union[ModuleType, Iterable[ModuleType]],
    matching: str,
    excluding: Optional[Union[str, Iterable[str]]] = None,
) -> list[type[EstimatorWrapperDF[BaseEstimator]]]:
    """Helper to return all classes with matching name from Python module(s)"""

    if not isinstance(from_modules, Iterable):
        from_modules = (from_modules,)

    if excluding is not None and not isinstance(excluding, str):
        excluding = "|".join(f"({exclude_pattern})" for exclude_pattern in excluding)

    return [
        m
        for m in find_all_classes(*from_modules)
        if re.match(matching, m.__name__)
        and ((excluding is None) or not re.match(excluding, m.__name__))
    ]


def get_sklearndf_wrapper_class(
    to_wrap: type[BaseEstimator], from_module: ModuleType
) -> type[EstimatorWrapperDF[BaseEstimator]]:
    """Helper to return the wrapped counterpart for a sklearn class"""
    try:
        return sklearn_delegate_classes(from_module)[to_wrap]

    except KeyError as cause:
        raise ValueError(
            f"There is no class that wraps '{to_wrap}' in {from_module}"
        ) from cause


def check_expected_not_fitted_error(estimator: EstimatorDF) -> None:
    """Check if transformers & learners raise NotFittedError"""

    test_x = pd.DataFrame(data=list(range(10)))

    def check_sklearndf_call(
        func_to_call: str, _estimator: Union[LearnerDF, TransformerDF]
    ) -> None:
        try:
            getattr(_estimator, func_to_call)(X=test_x)
        except sklearn.exceptions.NotFittedError:
            # This is the expected error, that sklearn[df] should raise
            return
        except Exception as sklearndf_exception:
            # Re-run the predict/transform ahead of fitting, and compare errors
            # across sklearn and sklearndf:
            try:
                if func_to_call == "transform":
                    x = test_x.values
                else:
                    x = test_x.values.reshape(-1)

                getattr(_estimator.native_estimator, func_to_call)(x)
            except sklearn.exceptions.NotFittedError:
                raise AssertionError(
                    "sklearndf did not return an expected NotFittedError"
                    f" for {_estimator.__class__.__name__}"
                )
            except Exception as sklearn_exception:
                assert repr(sklearndf_exception) == repr(sklearn_exception), (
                    "sklearndf raised a different error as sklearn"
                    f" for {_estimator.__class__.__name__}:\n"
                    f"sklearndf: {repr(sklearndf_exception)} \n"
                    f"sklearn:   {repr(sklearn_exception)}"
                )

    if isinstance(estimator, LearnerDF):
        check_sklearndf_call("predict", estimator)
    elif isinstance(estimator, TransformerDF):
        check_sklearndf_call("transform", estimator)
    else:
        raise TypeError(f"Estimator of unknown type:{estimator.__name__}")
