"""
Core implementation of :mod:`sklearndf.transformation` loaded
from sklearn 0.24 onwards
"""


import logging

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.kernel_approximation import PolynomialCountSketch

from sklearndf.transformation.wrapper import (
    FeatureSelectionWrapperDF,
    NComponentsDimensionalityReductionWrapperDF,
)

log = logging.getLogger(__name__)

__all__ = ["PolynomialCountSketchDF", "SequentialFeatureSelectorDF"]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

#
# Transformers which have an n_components attribute
# Implemented through NComponentsDimensionalityReductionWrapperDF
#


class PolynomialCountSketchDF(
    NComponentsDimensionalityReductionWrapperDF[PolynomialCountSketch],
    native=PolynomialCountSketch,
):
    """Stub for DF wrapper of class ``PolynomialCountSketch``"""


#
# feature_selection
#
# Transformers with a get_support method, implemented via FeatureSelectionWrapperDF
#


class SequentialFeatureSelectorDF(
    FeatureSelectionWrapperDF[SequentialFeatureSelector],
    native=SequentialFeatureSelector,
):
    """Stub for DF wrapper of class ``SequentialFeatureSelector``"""


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
