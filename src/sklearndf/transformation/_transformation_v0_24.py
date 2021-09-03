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
from sklearndf.wrapper import make_df_transformer

log = logging.getLogger(__name__)

__all__ = ["PolynomialCountSketchDF", "SequentialFeatureSelectorDF"]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

#
# Transformers which have an n_components attribute
# Implemented through NComponentsDimensionalityReductionWrapperDF
#
PolynomialCountSketchDF = make_df_transformer(
    PolynomialCountSketch, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)


#
# feature_selection
#
# Transformers with a get_support method, implemented via FeatureSelectionWrapperDF
#

SequentialFeatureSelectorDF = make_df_transformer(
    SequentialFeatureSelector, base_wrapper=FeatureSelectionWrapperDF
)


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
