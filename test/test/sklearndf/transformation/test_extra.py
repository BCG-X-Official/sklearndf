import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from gamma.sklearndf.regression import RandomForestRegressorDF
from gamma.sklearndf.transformation.extra import BorutaDF


def test_boruta_df() -> None:
    """ Test basic functionality of BorutaDF with both sklearn & sklearndf predictor """
    df = pd.DataFrame(data=np.random.randn(100, 5), columns=list("abcde"))
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    boruta = BorutaDF(estimator=RandomForestRegressor(), max_iter=10)
    boruta.fit(x, y)

    boruta_df = BorutaDF(estimator=RandomForestRegressorDF(), max_iter=10)
    boruta_df.fit(x, y)
