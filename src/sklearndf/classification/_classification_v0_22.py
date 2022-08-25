"""
Additional implementation of :mod:`sklearndf.classification` loaded
from sklearn 0.22 onwards
"""
import logging

from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import CategoricalNB

from pytools.api import AllTracker

from ..wrapper.stacking import StackingClassifierWrapperDF
from .wrapper import PartialFitClassifierWrapperDF

log = logging.getLogger(__name__)

__all__ = ["CategoricalNBDF", "StackingClassifierDF"]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


#
# naive bayes
#


class CategoricalNBDF(
    PartialFitClassifierWrapperDF[CategoricalNB],
    native=CategoricalNB,
):
    """Stub for DF wrapper of class ``CategoricalNB``"""


class StackingClassifierDF(
    StackingClassifierWrapperDF[StackingClassifier],
    native=StackingClassifier,
):
    """Stub for DF wrapper of class ``StackingClassifier``"""


#
# validate __all__
#

__tracker.validate()


#
# validate that __all__ comprises all symbols ending in "DF", and no others
#

__estimators = {
    sym
    for sym in dir()
    if sym.endswith("DF")
    and sym not in __imported_estimators
    and not sym.startswith("_")
}
if __estimators != set(__all__):
    raise RuntimeError(
        "__all__ does not contain exactly all DF estimators; expected value is:\n"
        f"{__estimators}"
    )
