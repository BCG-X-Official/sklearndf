"""
Factory functions and wrapper class API for enhancing the functionality of native
estimators conforming with the scikit-learn API.

In many cases, creating a DF classifier or regressor is as simple as

.. code-block:: python

  RandomForestClassifierDF = make_df_classifier(RandomForestClassifier)
  RandomForestRegressorDF = make_df_regressor(RandomForestRegressor)

For more advanced examples, including the use of custom wrapper classes, see the many
examples in modules
:mod:`sklearndf.transformation`,
:mod:`sklearndf.classification`, and
:mod:`sklearndf.regression`.
"""

from ._wrapper import *
