from typing import List, Type

import pandas as pd
import pytest
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

import sklearndf.regression
from sklearndf import RegressorDF, TransformerDF
from sklearndf._wrapper import _EstimatorWrapperDF
from sklearndf.regression import (
    SVRDF,
    IsotonicRegressionDF,
    LinearRegressionDF,
    RandomForestRegressorDF,
)
from test.sklearndf import check_expected_not_fitted_error, list_classes

REGRESSORS_TO_TEST: List[Type[_EstimatorWrapperDF]] = list_classes(
    from_modules=sklearndf.regression,
    matching=r".*DF",
    excluding=[RegressorDF.__name__, TransformerDF.__name__, r".*WrapperDF"],
)

DEFAULT_REGRESSOR_PARAMETERS = {
    "MultiOutputRegressorDF": {"estimator": RandomForestRegressorDF()},
    "RegressorChainDF": {"base_estimator": RandomForestRegressorDF()},
    "VotingRegressorDF": {
        "estimators": [("rfr", RandomForestRegressorDF()), ("svr", SVRDF())]
    },
    "StackingRegressorDF": {
        "estimators": (
            ("Forest", RandomForestRegressorDF()),
            ("SVR", SVRDF()),
            ("Linear", LinearRegressionDF()),
        )
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

    check_expected_not_fitted_error(estimator=regressor)

    if (
        type(regressor).__name__.startswith("Multi")
        or isinstance(regressor.native_estimator, MultiOutputRegressor)
        or isinstance(regressor.native_estimator, RegressorChain)
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
