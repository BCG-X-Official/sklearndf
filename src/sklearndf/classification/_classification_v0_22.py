"""
Additional implementation of :mod:`sklearndf.classification` loaded
from sklearn 0.22 onwards
"""

import logging

from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import CategoricalNB

from pytools.api import AllTracker

from .. import ClassifierDF
from .._wrapper import _ClassifierWrapperDF, _StackingClassifierWrapperDF, df_estimator

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


# todo: add other classification implementations for sklearn 0.22

#
# naive bayes
#

# noinspection PyAbstractClass


@df_estimator(df_wrapper_type=_ClassifierWrapperDF)
class CategoricalNBDF(ClassifierDF, CategoricalNB):
    """
    Wraps :class:`sklearn.naive_bayes.CategoricalNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_StackingClassifierWrapperDF)
class StackingClassifierDF(ClassifierDF, StackingClassifier):
    """
    Wraps :class:`sklearn.ensemble._stacking.StackingClassifier`;
    accepts and returns data frames.
    """

    pass


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
