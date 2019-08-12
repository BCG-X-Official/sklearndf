import pandas as pd

from gamma.sklearndf.regression import RandomForestRegressorDF


def test_regressor_df(boston_df: pd.DataFrame, boston_target: str) -> None:

    regressor_df = RandomForestRegressorDF()

    x = boston_df.drop(columns=boston_target)
    y = boston_df.loc[:, boston_target]

    regressor_df.fit(X=x, y=y)

    predictions = regressor_df.predict(X=x)

    # test predictions data-type, length and values
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(y)
