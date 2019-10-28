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
from ._wrapper import *

_licensing.check_license(__package__)

# Export only the non-abstract sklearndf classes. Wrapper classes will be visible but
# will have to be imported explicitly
__all__ = [member for member in _sklearndf.__all__ if not member.startswith("Base")]
