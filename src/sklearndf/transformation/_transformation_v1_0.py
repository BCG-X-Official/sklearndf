"""
Core implementation of :mod:`sklearndf.transformation` loaded
from sklearn 1.0 onwards
"""


import logging

from sklearn.preprocessing import SplineTransformer

from .wrapper import PolynomialTransformerWrapperDF

log = logging.getLogger(__name__)

__all__ = ["SplineTransformerDF"]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

#
# preprocessing
#


class SplineTransformerDF(
    PolynomialTransformerWrapperDF[SplineTransformer],
    native=SplineTransformer,
):
    """Stub for DF wrapper of class ``SplineTransformer``"""


#
# validate that __all__ comprises all symbols ending in "DF", and no others
#

__estimators = [
    sym
    for sym in dir()
    if sym.endswith("DF")
    and sym not in __imported_estimators
    and not sym.startswith("_")
]
if set(__estimators) != set(__all__):
    raise RuntimeError(
        "__all__ does not contain exactly all DF estimators; expected value is:\n"
        f"{__estimators}"
    )
