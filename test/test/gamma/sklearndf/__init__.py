import inspect
import re
from distutils import version
from typing import *
from typing import Type

import pandas as pd
import sklearn
import sys

from sklearn.base import BaseEstimator

from gamma.sklearndf import BaseEstimatorWrapperDF, BaseLearnerDF, TransformerDF

Module: type = Any


def find_all_classes(*modules: Module) -> Set[type]:
    """ Finds all Class members in given module/modules. """
    types: Set[type] = set()

    def _add_classes_from_module(_m: Module) -> None:
        for member in vars(module).values():
            if isinstance(member, type):
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


def sklearndf_to_wrapped(module: Module) -> Dict[BaseEstimatorWrapperDF, BaseEstimator]:
    """ Creates a dictionary mapping from sklearndf -> sklearn classes. """
    return {
        cdf: cdf.__wrapped__
        for cdf in find_all_classes((module))
        if issubclass(cdf, BaseEstimatorWrapperDF)
    }


def list_classes(
    from_modules: Union[Module, Iterable[Module]],
    matching: str,
    excluding: Optional[Union[str, List[str]]] = None,
) -> List[Type]:
    """ Helper to return all classes with matching name from Python module(s) """

    if not isinstance(from_modules, Iterable):
        from_modules = (from_modules,)

    if isinstance(excluding, list):
        excluding = "|".join([f"({exclude_pattern})" for exclude_pattern in excluding])

    return [
        m
        for m in find_all_classes(*from_modules)
        if re.match(matching, m.__name__)
        and not (excluding and re.match(excluding, m.__name__))
    ]


def get_wrapped_counterpart(to_wrap: Type, from_module=None) -> BaseEstimatorWrapperDF:
    """ Helper to return the wrapped counterpart for a sklearn class """
    for sklearndf_cls, sklearn_cls in sklearndf_to_wrapped(from_module).items():
        if sklearn_cls == to_wrap:
            return sklearndf_cls

    raise ValueError(f"There is no class that wraps '{to_wrap}' in {from_module}")


def check_expected_not_fitted_error(estimator: Union[BaseLearnerDF, TransformerDF]):
    """ Check if transformers & learners raise NotFittedError (since sklearn 0.22)"""
    if version.LooseVersion(sklearn.__version__) <= "0.21":
        return

    test_x = pd.DataFrame(data=list(range(10)))

    def check_sklearndf_call(
        func_to_call: str, estimator: Union[BaseLearnerDF, TransformerDF]
    ) -> None:
        try:
            getattr(estimator, func_to_call)(X=test_x)
        except sklearn.exceptions.NotFittedError:
            # This is the expected error, that sklearn[df] should raise
            return
        except Exception as sklearndf_exception:
            # Re-run the predict/transform ahead of fitting, and compare errors
            # across sklearn and sklearndf:
            try:
                getattr(estimator.root_estimator, func_to_call)(
                    test_x.values.reshape(-1)
                )
            except sklearn.exceptions.NotFittedError:
                raise AssertionError(
                    "sklearndf did not return an expected NotFittedError"
                    f" for {estimator.__class__.__name__}"
                )
            except Exception as sklearn_exception:
                assert repr(sklearndf_exception) == repr(sklearn_exception), (
                    "sklearndf raised a different error as sklearn"
                    f" for {estimator.__class__.__name__}"
                )

    if isinstance(estimator, BaseLearnerDF):
        check_sklearndf_call("predict", estimator)
    elif isinstance(estimator, TransformerDF):
        check_sklearndf_call("transform", estimator)
    else:
        raise TypeError(f"Estimator of unknown type:{estimator.__name__}")
