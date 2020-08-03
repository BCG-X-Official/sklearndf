"""
The Gamma scikit-learn DF library.

Enhances scikit-learn estimators for E2E support of data frames.

The abstract class :class:`BaseEstimatorDF` and its subclasses wrap subclasses of
:class:`~sklearn.base.BaseEstimator` such that transform methods return data frames
with feature names in the column index.

The enhanced base estimators also offer attributes
:attr:`~BaseEstimatorDF.features_in`, :attr:`~TransformerDF.features_out`, and
:attr:`~TransformerDF.features_original`, which enable tracing features back to the
original inputs even across complex pipelines.
"""

import gamma.common.licensing as _licensing
from ._sklearndf import *

__version__ = "1.3.0rc0"
_licensing.check_license(__package__)
