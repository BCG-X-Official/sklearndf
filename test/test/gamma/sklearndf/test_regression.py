from typing import *

import pandas as pd
import pytest

import gamma.sklearndf.regression
from gamma.sklearndf import RegressorDF
from gamma.sklearndf.regression import (
    _RegressorTransformerWrapperDF,
    ColumnPreservingTransformerWrapperDF,
    RandomForestRegressorDF,
    RegressorWrapperDF,
    SVRDF,
    TransformerDF,
)
from test.gamma.sklearndf import get_classes, get_missing_init_parameter

REGRESSORS_TO_TEST: List[Type] = get_classes(
    from_module=gamma.sklearndf.regression,
    matching=r".*DF",
    excluding=[RegressorDF.__name__, TransformerDF.__name__, r".*WrapperDF"],
)

DEFAULT_INIT_PARAMETERS = {
    "estimator": {"estimator": RandomForestRegressorDF()},
    "base_estimator": {"base_estimator": RandomForestRegressorDF()},
    "estimators": {
        "estimators": [("rfr", RandomForestRegressorDF()), ("svmr", SVRDF())]
    },
}


@pytest.mark.parametrize(argnames="sklearndf_cls", argvalues=REGRESSORS_TO_TEST)
def test_wrapped_constructor(sklearndf_cls: Type) -> None:
    """ Test standard constructor of wrapped sklearn regressors """
    try:
        rgr: RegressorDF = sklearndf_cls()
    except TypeError as te:
        # some regressors need additional parameters, look up their key and add default:
        missing_init_parameter = get_missing_init_parameter(te=te)
        if missing_init_parameter not in DEFAULT_INIT_PARAMETERS:
            raise te
        else:
            rgr = sklearndf_cls(**DEFAULT_INIT_PARAMETERS[missing_init_parameter])


@pytest.mark.parametrize(argnames="sklearndf_cls", argvalues=REGRESSORS_TO_TEST)
def test_wrapped_fit_predict(
    sklearndf_cls: Type, boston_features: pd.DataFrame, boston_target_sr: pd.DataFrame
) -> None:
    """ Test fit & predict of wrapped sklearn regressors """
    try:
        rgr: RegressorDF = sklearndf_cls()
    except TypeError as te:
        # some regressors need additional parameters, look up their key and add default:
        missing_init_parameter = get_missing_init_parameter(te=te)
        if missing_init_parameter not in DEFAULT_INIT_PARAMETERS:
            raise te
        else:
            rgr = sklearndf_cls(**DEFAULT_INIT_PARAMETERS[missing_init_parameter])

    rgr.fit(X=boston_features, y=boston_target_sr)
    predictions = rgr.predict(X=boston_features)

    # test predictions data-type, length and values
    assert isinstance(predictions, (pd.Series, pd.DataFrame))
    assert len(predictions) == len(boston_target_sr)
