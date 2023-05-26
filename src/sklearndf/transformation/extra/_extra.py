"""
Core implementation of :mod:`sklearndf.transformation.extra`
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from pytools.api import AllTracker

from ...wrapper import MissingEstimator

log = logging.getLogger(__name__)

__all__ = ["BoostAGrootaDF", "BorutaDF", "GrootCVDF", "LeshyDF"]

try:
    # import boruta classes only if installed
    from boruta import BorutaPy

    # Apply a hack to address boruta's incompatibility with numpy >= 1.24:
    # boruta uses np.float_ which is deprecated in numpy >= 1.20 and removed in 1.24.
    #
    # We check these types are already defined in numpy, and if not, we define them
    # as aliases to the corresponding new types with a trailing underscore.

    for __attr in ["bool", "int", "float"]:
        if not hasattr(np, __attr):
            setattr(np, __attr, getattr(np, f"{__attr}_"))
    del __attr

except ImportError:

    class BorutaPy(  # type: ignore
        MissingEstimator,
        TransformerMixin,  # type: ignore
    ):
        """Mock-up for missing estimator."""


try:
    # import boruta classes only if installed
    from arfs.feature_selection.allrelevant import BoostAGroota, GrootCV, Leshy

except ImportError:

    class BoostAGroota(  # type: ignore
        MissingEstimator,
        TransformerMixin,  # type: ignore
    ):
        """Mock-up for missing estimator."""

    class GrootCV(  # type: ignore
        MissingEstimator,
        TransformerMixin,  # type: ignore
    ):
        """Mock-up for missing estimator."""

    class Leshy(  # type: ignore
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


from .wrapper import ARFSWrapperDF as _ARFSWrapperDF
from .wrapper import BorutaPyWrapperDF as _BorutaPyWrapperDF


class BorutaDF(_BorutaPyWrapperDF, native=BorutaPy):
    """
    DF version of :class:`~boruta.BorutaPy`.
    """


class LeshyDF(_ARFSWrapperDF[Leshy], native=Leshy):
    """
    DF version of :class:`~arfs.feature_selection.allrelevant.Leshy`.
    """


class BoostAGrootaDF(_ARFSWrapperDF[BoostAGroota], native=BoostAGroota):
    """
    DF version of :class:`~arfs.feature_selection.allrelevant.BoostAGroota`.
    """

    @property
    def estimator(self) -> BaseEstimator:
        """
        Alias for the native estimator's :attr:`.est` attribute, to conform with
        the :class:`~sklearn.base.MetaEstimatorMixin` interface.

        :return: the value of the native estimator's :attr:`.est` attribute
        """
        return self.native_estimator.est

    @estimator.setter
    def estimator(self, est: BaseEstimator) -> None:
        """
        Alias for the native estimator's :attr:`.est` attribute, to conform with
        the :class:`~sklearn.base.MetaEstimatorMixin` interface.

        :param est: the new value for the native estimator's :attr:`.est` attribute
        """
        self.native_estimator.est = est

    @estimator.deleter
    def estimator(self) -> None:
        """
        Alias for the native estimator's :attr:`.est` attribute, to conform with
        the :class:`~sklearn.base.MetaEstimatorMixin` interface.
        """
        del self.native_estimator.est


class GrootCVDF(_ARFSWrapperDF[GrootCV], native=GrootCV):
    """
    DF version of :class:`~arfs.feature_selection.allrelevant.GrootCV`.
    """


#
# validate __all__
#

__tracker.validate()
