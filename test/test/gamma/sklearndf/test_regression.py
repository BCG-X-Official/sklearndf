from typing import *

import pandas as pd
import pytest

import gamma.sklearndf.regression
from gamma.sklearndf import RegressorDF
from gamma.sklearndf.regression import RandomForestRegressorDF
from test.gamma.sklearndf import get_classes


@pytest.mark.parametrize(
    argnames="sklearndf_cls",
    argvalues=get_classes(
        from_module=gamma.sklearndf.regression,
        regex=r".*DF",
        ignore=[
            "RegressorDF",
            "RegressorWrapperDF",
            "ColumnPreservingTransformerWrapperDF",
            "_RegressorTransformerWrapperDF",
            "TransformerDF",
        ],
    ),
)
def test_wrapped_constructor(sklearndf_cls: Type) -> None:
    """ Test standard constructor of wrapped sklearn regressors """
    try:
        rgr: RegressorDF = sklearndf_cls()
    except TypeError as te:
        # some Regressors need special kwargs in __init__ we can't easily infer:
        if "missing 1 required positional argument" in str(te):
            # parameter 'base_estimator' is expected:
            if "'base_estimator'" in str(te):
                rgr = sklearndf_cls(base_estimator=RandomForestRegressorDF())
            # parameter 'estimator' is expected:
            elif "'estimator'" in str(te):
                rgr = sklearndf_cls(estimator=RandomForestRegressorDF())
            # parameter 'estimators' is expected:
            elif "'estimators'" in str(te):
                rgr = sklearndf_cls(estimators=[RandomForestRegressorDF()])
            # unknown Exception, raise it:
            else:
                raise te
        # unknown Exception, raise it:
        else:
            raise te


def test_regressor_df(boston_df: pd.DataFrame, boston_target: str) -> None:

    regressor_df = RandomForestRegressorDF()

    x = boston_df.drop(columns=boston_target)
    y = boston_df.loc[:, boston_target]

    regressor_df.fit(X=x, y=y)

    predictions = regressor_df.predict(X=x)

    # test predictions data-type, length and values
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(y)
