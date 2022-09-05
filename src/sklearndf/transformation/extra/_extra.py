"""
Core implementation of :mod:`sklearndf.transformation.extra`
"""
from __future__ import annotations

import logging

from sklearn.base import TransformerMixin

from pytools.api import AllTracker

from ...wrapper import MissingEstimator

log = logging.getLogger(__name__)

__all__ = ["BorutaDF"]

try:
    # import boruta classes only if installed
    from boruta import BorutaPy

except ImportError:

    class BorutaPy(  # type: ignore
        MissingEstimator,
        TransformerMixin,  # type: ignore
    ):
        """Mock-up for missing estimator."""


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


from .wrapper import BorutaPyWrapperDF as _BorutaPyWrapperDF


class BorutaDF(_BorutaPyWrapperDF, native=BorutaPy):
    """
    DF version of :class:`~boruta.BorutaPy`.
    """


#
# validate __all__
#

__tracker.validate()
