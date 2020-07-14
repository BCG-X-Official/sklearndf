import inspect
import re
from distutils import version
from typing import *
from typing import Type

import pandas as pd
import pytest
import sklearn

from gamma.sklearndf import BaseLearnerDF, TransformerDF


def get_classes(
    from_module, matching: str, excluding: Optional[Union[str, List[str]]] = None
) -> List[Type]:
    """ Helper to return all classes with matching name from a Python module """

    if isinstance(excluding, list):
        excluding = "|".join([f"({exclude_pattern})" for exclude_pattern in excluding])

    return [
        m[1]
        for m in inspect.getmembers(from_module)
        if isinstance(m[1], type)
        and re.match(matching, m[0])
        and not (excluding and re.match(excluding, m[0]))
    ]


def get_wrapped_counterpart(to_wrap: Type, from_package=None) -> Type:
    """ Helper to return the wrapped counterpart for a sklearn class """
    orig_name = to_wrap.__name__
    new_name = orig_name + "DF"

    if hasattr(from_package, new_name):
        return getattr(from_package, new_name)


def check_expected_not_fitted_error(estimator: Union[BaseLearnerDF, TransformerDF]):
    """ Check if transformers & learners raise NotFittedError (since sklearn 0.22)"""
    if version.LooseVersion(sklearn.__version__) <= "0.21":
        return

    if isinstance(estimator, BaseLearnerDF):
        with pytest.raises(expected_exception=sklearn.exceptions.NotFittedError):
            estimator.predict(X=pd.DataFrame(data=range(0, 10)))
    elif isinstance(estimator, TransformerDF):
        with pytest.raises(expected_exception=sklearn.exceptions.NotFittedError):
            estimator.transform(X=pd.DataFrame(data=range(0, 10)))
    else:
        raise TypeError(f"Estimator of unknown type:{estimator.__name__}")
