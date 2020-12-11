"""
Core implementation of :mod:`sklearndf.transformation` loaded
from sklearn 0.22 onwards
"""

#
# To create the DF class stubs:
#
# - generate a list of all child classes of TransformerMixin in PyCharm using the
#   hierarchy view (^H)
# - remove all abstract base classes and non-sklearn classes from the list
# - unindent all lines
# - use replace with regular expressions
#   Find: (\w+)\([^\)]+\) \(([\w\.]+)\)
#   Replace: @_df_transformer\nclass $1DF(TransformerDF, $1):\n    """\n    Wraps
#            :class:`$2.$1`;\n    accepts and returns data frames.\n    """
#            \n    pass\n\n
# - clean up imports; import only the module names not the individual classes

import logging

from sklearn.impute import KNNImputer

from .. import TransformerDF
from .._wrapper import df_estimator
from ._transformation import _ImputerWrapperDF

log = logging.getLogger(__name__)

__all__ = ["KNNImputerDF"]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

#
# impute
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ImputerWrapperDF)
class KNNImputerDF(TransformerDF, KNNImputer):
    """
    Wraps :class:`sklearn.impute._knn.KNNImputer`;
    accepts and returns data frames.
    """

    pass


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
