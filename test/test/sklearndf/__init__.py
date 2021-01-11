import re
import sys
from distutils import version
from typing import Any, Dict, Iterable, List, Optional, Set, Type, Union

import pandas as pd
import sklearn
from sklearn.base import BaseEstimator

from sklearndf import LearnerDF, TransformerDF
from sklearndf._wrapper import _EstimatorWrapperDF

Module: type = Any


def find_all_classes(*modules: Module) -> Set[Type[_EstimatorWrapperDF]]:
    """ Finds all Class members in given module/modules. """
    types: Set[Type[_EstimatorWrapperDF]] = set()

    def _add_classes_from_module(_m: Module) -> None:
        for member in vars(module).values():
            if isinstance(member, type):
                member: Type[_EstimatorWrapperDF]
                types.add(member)

    for module in modules:
        _add_classes_from_module(module)

    return types


def find_all_submodules(parent_module: Module) -> Set[Module]:
    """ Finds all submodules for a parent module. """
    parent_name = f"{parent_module.__name__}."
    return {
        module
        for module_name, module in sys.modules.items()
        if module_name.startswith(parent_name)
    }


def sklearn_delegate_classes(
    module: Module,
) -> Dict[Type[BaseEstimator], Type[_EstimatorWrapperDF]]:
    """ Creates a dictionary mapping from sklearndf -> sklearn classes. """
    return {
        df_class.__wrapped__: df_class
        for df_class in find_all_classes(module)
        if issubclass(df_class, _EstimatorWrapperDF)
    }


def list_classes(
    from_modules: Union[Module, Iterable[Module]],
    matching: str,
    excluding: Optional[Union[str, Iterable[str]]] = None,
) -> List[Type[_EstimatorWrapperDF]]:
    """ Helper to return all classes with matching name from Python module(s) """

    if not isinstance(from_modules, Iterable):
        from_modules = (from_modules,)

    if isinstance(excluding, Iterable):
        excluding = "|".join(f"({exclude_pattern})" for exclude_pattern in excluding)

    return [
        m
        for m in find_all_classes(*from_modules)
        if re.match(matching, m.__name__)
        and not (excluding and re.match(excluding, m.__name__))
    ]


def get_sklearndf_wrapper_class(
    to_wrap: Type[BaseEstimator], from_module=None
) -> Type[_EstimatorWrapperDF]:
    """ Helper to return the wrapped counterpart for a sklearn class """
    try:
        return sklearn_delegate_classes(from_module)[to_wrap]

    except KeyError as cause:
        raise ValueError(
            f"There is no class that wraps '{to_wrap}' in {from_module}"
        ) from cause


def check_expected_not_fitted_error(estimator: Union[LearnerDF, TransformerDF]):
    """ Check if transformers & learners raise NotFittedError (since sklearn 0.22)"""
    if version.LooseVersion(sklearn.__version__) <= "0.21":
        return

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
                    f" for {_estimator.__class__.__name__}:"
                    f"sklearndf: {repr(sklearndf_exception)} \n"
                    f"sklearn: {repr(sklearn_exception)}"
                )

    if isinstance(estimator, LearnerDF):
        check_sklearndf_call("predict", estimator)
    elif isinstance(estimator, TransformerDF):
        check_sklearndf_call("transform", estimator)
    else:
        raise TypeError(f"Estimator of unknown type:{estimator.__name__}")
