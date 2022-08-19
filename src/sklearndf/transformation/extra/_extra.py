"""
Core implementation of :mod:`sklearndf.transformation.extra`
"""
from __future__ import annotations

import logging

from pytools.api import AllTracker

log = logging.getLogger(__name__)

__all__ = ["BorutaDF"]


try:
    # import boruta classes only if installed
    from boruta import BorutaPy

except ImportError:
    BorutaPy = None

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#

if BorutaPy:

    from .wrapper import BorutaPyWrapperDF

    class BorutaDF(BorutaPyWrapperDF, native=BorutaPy):
        """
        DF version of :class:`~boruta.BorutaPy`.
        """

    # remove the wrapper class from the global namespace to pass AllTracker validation
    del BorutaPyWrapperDF

else:
    __all__.remove("BorutaDF")


__tracker.validate()
