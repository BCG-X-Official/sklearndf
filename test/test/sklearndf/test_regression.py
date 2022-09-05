from typing import Any, Dict, List, Type

import pandas as pd
import pytest
from sklearn.base import BaseEstimator, is_regressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

import sklearndf.regression
from sklearndf import (
    RegressorDF,
    TransformerDF,
    __sklearn_0_22__,
    __sklearn_0_23__,
    __sklearn_1_0__,
    __sklearn_version__,
)
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
REGRESSORS_TO_TEST: List[Type[EstimatorWrapperDF[BaseEstimator]]] = iterate_classes(
    from_modules=sklearndf.regression,
    matching=r".*DF",
    excluding=[RegressorDF.__name__, TransformerDF.__name__, r".*WrapperDF"],
)


def test_regressor_count() -> None:
    n = len(REGRESSORS_TO_TEST)

    print(f"Testing {n} regressors.")
    if __sklearn_version__ < __sklearn_0_22__:
        assert n == 49
    elif __sklearn_version__ < __sklearn_0_23__:
        assert n == 50
    elif __sklearn_version__ < __sklearn_1_0__:
        assert n == 53
    else:
        assert n == 55


DEFAULT_REGRESSOR_PARAMETERS: Dict[str, Dict[str, Any]] = {
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
    sklearndf_cls: Type[RegressorDF],
    boston_features: pd.DataFrame,
    boston_target_sr: pd.Series,
    boston_target_df: pd.DataFrame,
) -> None:
    """Test fit & predict of wrapped sklearn regressors"""
    parameters: Dict[str, Any] = DEFAULT_REGRESSOR_PARAMETERS.get(
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


@pytest.mark.parametrize(  # type: ignore
    argnames="sklearndf_cls", argvalues=REGRESSORS_PARTIAL_FIT
)
def test_wrapped_partial_fit(
    sklearndf_cls: Type[RegressorDF],
    boston_features: pd.DataFrame,
    boston_target_sr: pd.Series,
    boston_target_df: pd.DataFrame,
) -> None:

    # noinspection PyArgumentList
    regressor = sklearndf_cls(
        **DEFAULT_REGRESSOR_PARAMETERS.get(f"{sklearndf_cls.__name__}_partial_fit", {})
    )

    is_multi_output = isinstance(regressor.native_estimator, MultiOutputRegressor)
    boston_target = boston_target_df if is_multi_output else boston_target_sr

    regressor.partial_fit(boston_features, boston_target)
