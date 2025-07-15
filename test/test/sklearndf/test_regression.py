from typing import Any

import pandas as pd
import pytest
from sklearn.base import BaseEstimator, is_regressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

import sklearndf.regression
from sklearndf import RegressorDF, TransformerDF, __sklearn_1_8__, __sklearn_version__
from sklearndf.regression import (
    SVRDF,
    IsotonicRegressionDF,
    LinearRegressionDF,
    MLPRegressorDF,
    MultiOutputRegressorDF,
    PassiveAggressiveRegressorDF,
    RandomForestRegressorDF,
    SGDRegressorDF,
)
from sklearndf.wrapper import EstimatorWrapperDF
from test.sklearndf import check_expected_not_fitted_error, iterate_classes

# noinspection PyTypeChecker
# ignore false alert about module type
REGRESSORS_TO_TEST: list[type[EstimatorWrapperDF[BaseEstimator]]] = iterate_classes(
    from_modules=sklearndf.regression,
    matching=r".*DF",
    excluding=[RegressorDF.__name__, TransformerDF.__name__, r".*WrapperDF"],
)


def test_regressor_count() -> None:
    n = len(REGRESSORS_TO_TEST)

    print(f"Testing {n} regressors.")
    if __sklearn_version__ < __sklearn_1_8__:
        assert n == 55
    else:
        pytest.fail(f"Unexpected scikit-learn version: {__sklearn_version__}")


DEFAULT_REGRESSOR_PARAMETERS: dict[str, dict[str, Any]] = {
    "MultiOutputRegressorDF": dict(estimator=RandomForestRegressorDF()),
    "MultiOutputRegressorDF_partial_fit": dict(estimator=SGDRegressorDF()),
    "RegressorChainDF": dict(base_estimator=RandomForestRegressorDF()),
    "VotingRegressorDF": dict(
        estimators=[("rfr", RandomForestRegressorDF()), ("svr", SVRDF())]
    ),
    "StackingRegressorDF": dict(
        estimators=(
            ("Forest", RandomForestRegressorDF()),
            ("SVR", SVRDF()),
            ("Linear", LinearRegressionDF()),
        )
    ),
    # the rank of Y is 1, so n_components needs to be 1
    "CCADF": dict(n_components=1),
    # the rank of Y is 1, so n_components needs to be 1
    "PLSCanonicalDF": dict(n_components=1),
    # use a solver that is still supported with scipy 1.11
    "QuantileRegressorDF": dict(solver="highs"),
}

REGRESSORS_PARTIAL_FIT = [
    SGDRegressorDF,
    PassiveAggressiveRegressorDF,
    MultiOutputRegressorDF,
    MLPRegressorDF,
]


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearndf_cls", argvalues=REGRESSORS_TO_TEST
)
def test_wrapped_fit_predict(
    sklearndf_cls: type[RegressorDF],
    diabetes_features: pd.DataFrame,
    diabetes_target_sr: pd.Series,
    diabetes_target_df: pd.DataFrame,
) -> None:
    """Test fit & predict of wrapped sklearn regressors"""
    parameters: dict[str, Any] = DEFAULT_REGRESSOR_PARAMETERS.get(
        sklearndf_cls.__name__, {}
    )

    # noinspection PyArgumentList
    regressor: RegressorDF = sklearndf_cls(**parameters)

    assert is_regressor(regressor)

    check_expected_not_fitted_error(estimator=regressor)

    if (
        type(regressor).__name__.startswith("Multi")
        or isinstance(regressor.native_estimator, MultiOutputRegressor)
        or isinstance(regressor.native_estimator, RegressorChain)
    ):
        regressor.fit(X=diabetes_features, y=diabetes_target_df)

    else:
        if isinstance(regressor, IsotonicRegressionDF):
            # fit will fail when we have more than one feature
            with pytest.raises(ValueError):
                regressor.fit(X=diabetes_features, y=diabetes_target_sr)
            # eliminate all features except one then continue testing
            diabetes_features = diabetes_features.loc[:, "bmi"]

        regressor.fit(X=diabetes_features, y=diabetes_target_sr)

    predictions = regressor.predict(X=diabetes_features)

    # test predictions data-type, length and values
    assert isinstance(predictions, (pd.Series, pd.DataFrame))
    assert len(predictions) == len(diabetes_target_sr)


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearndf_cls", argvalues=REGRESSORS_PARTIAL_FIT
)
def test_wrapped_partial_fit(
    sklearndf_cls: type[RegressorDF],
    diabetes_features: pd.DataFrame,
    diabetes_target_sr: pd.Series,
    diabetes_target_df: pd.DataFrame,
) -> None:
    # noinspection PyArgumentList
    regressor = sklearndf_cls(
        **DEFAULT_REGRESSOR_PARAMETERS.get(f"{sklearndf_cls.__name__}_partial_fit", {})
    )

    is_multi_output = isinstance(regressor.native_estimator, MultiOutputRegressor)
    diabetes_target = diabetes_target_df if is_multi_output else diabetes_target_sr

    # noinspection PyUnresolvedReferences
    regressor.partial_fit(diabetes_features, diabetes_target)
