from typing import *

import pandas as pd
import pytest

import gamma.sklearndf.regression
from gamma.sklearndf import RegressorDF
from gamma.sklearndf.regression import (
    RandomForestRegressorDF,
    SVRDF,
    TransformerDF,
)
from test.gamma.sklearndf import get_classes

REGRESSORS_TO_TEST: List[Type] = get_classes(
    from_module=gamma.sklearndf.regression,
    matching=r".*DF",
    excluding=[RegressorDF.__name__, TransformerDF.__name__, r".*WrapperDF"],
)

DEFAULT_REGRESSOR_PARAMETERS = {
    "MultiOutputRegressorDF": {"estimator": RandomForestRegressorDF()},
    "RegressorChainDF": {"base_estimator": RandomForestRegressorDF()},
    "VotingRegressorDF": {
        "estimators": [("rfr", RandomForestRegressorDF()), ("svmr", SVRDF())]
    },
}


@pytest.mark.parametrize(argnames="sklearndf_cls", argvalues=REGRESSORS_TO_TEST)
def test_wrapped_constructor(sklearndf_cls: Type) -> None:
    """ Test standard constructor of wrapped sklearn regressors """
    _: RegressorDF = sklearndf_cls(**DEFAULT_REGRESSOR_PARAMETERS.get(
        sklearndf_cls.__name__, {}))


@pytest.mark.parametrize(argnames="sklearndf_cls", argvalues=REGRESSORS_TO_TEST)
def test_wrapped_fit_predict(
    sklearndf_cls: Type, boston_features: pd.DataFrame, boston_target_sr: pd.Series
) -> None:
    """ Test fit & predict of wrapped sklearn regressors """
    regressor: RegressorDF = sklearndf_cls(**DEFAULT_REGRESSOR_PARAMETERS.get(
        sklearndf_cls.__name__, {}))

    regressor.fit(X=boston_features, y=boston_target_sr)
    predictions = regressor.predict(X=boston_features)

    # test predictions data-type, length and values
    assert isinstance(predictions, (pd.Series, pd.DataFrame))
    assert len(predictions) == len(boston_target_sr)
