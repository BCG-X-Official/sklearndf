from typing import *

import pandas as pd
import pytest
from sklearn.multioutput import MultiOutputEstimator, RegressorChain

import gamma.sklearndf.regression
from gamma.sklearndf import RegressorDF
from gamma.sklearndf.regression import (
    IsotonicRegressionDF,
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
    _: RegressorDF = sklearndf_cls(
        **DEFAULT_REGRESSOR_PARAMETERS.get(sklearndf_cls.__name__, {})
    )


@pytest.mark.parametrize(argnames="sklearndf_cls", argvalues=REGRESSORS_TO_TEST)
def test_wrapped_fit_predict(
    sklearndf_cls: Type,
    boston_features: pd.DataFrame,
    boston_target_sr: pd.Series,
    boston_target_df: pd.DataFrame,
) -> None:
    """ Test fit & predict of wrapped sklearn regressors """
    regressor: RegressorDF = sklearndf_cls(
        **DEFAULT_REGRESSOR_PARAMETERS.get(sklearndf_cls.__name__, {})
    )

    if (
        type(regressor).__name__.startswith("Multi")
        or isinstance(regressor.root_estimator, MultiOutputEstimator)
        or isinstance(regressor.root_estimator, RegressorChain)
    ):
        regressor.fit(X=boston_features, y=boston_target_df)

    else:
        if isinstance(regressor, IsotonicRegressionDF):
            # fit will fail when we have more than one feature
            with pytest.raises(ValueError):
                regressor.fit(X=boston_features, y=boston_target_sr)
            # eliminate all features except one then continue testing
            boston_features = boston_features.loc[:, ["LSTAT"]]

        regressor.fit(X=boston_features, y=boston_target_sr)

    predictions = regressor.predict(X=boston_features)

    # test predictions data-type, length and values
    assert isinstance(predictions, (pd.Series, pd.DataFrame))
    assert len(predictions) == len(boston_target_sr)
