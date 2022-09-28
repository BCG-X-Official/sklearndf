"""
Core implementation of :mod:`sklearndf.transformation` loaded
from sklearn 1.1 onwards
"""


import logging

from sklearn.decomposition import MiniBatchNMF
from sklearn.ensemble import RandomTreesEmbedding

from .wrapper import ComponentsDimensionalityReductionWrapperDF, EmbeddingWrapperDF

log = logging.getLogger(__name__)

__all__ = ["MiniBatchNMFDF", "RandomTreesEmbeddingDF"]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

#
# preprocessing
#


class MiniBatchNMFDF(
    ComponentsDimensionalityReductionWrapperDF[MiniBatchNMF],
    native=MiniBatchNMF,
):
    """Stub for DF wrapper of class ``MiniBatchNMF``"""


class RandomTreesEmbeddingDF(
    EmbeddingWrapperDF[RandomTreesEmbedding],
    native=RandomTreesEmbedding,
):
    """Stub for DF wrapper of class ``RandomTreesEmbedding``"""


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
