"""
Handling of mocked up native estimators.
"""
import logging
from typing import Any

from sklearn.base import BaseEstimator

from pytools.api import AllTracker

log = logging.getLogger(__name__)

__all__ = [
    "MissingEstimator",
]

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class declarations
#


class MissingEstimator(
    BaseEstimator,  # type: ignore
):
    """
    Base class of mocked up native estimators, for use in case an optional 3rd party
    estimator is not installed but is required to create the associated DF estimator.

    Raises a :class:`.RuntimeError` upon instantiation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        :param args: arbitrary positional arguments
        :param kwargs: arbitrary keyword arguments
        :raises RuntimeError: always raised upon instantiation
        """
        raise RuntimeError(
            f"Estimator {type(self).__name__} is not available. "
            f"Please install the package that implements it."
        )


#
# validate __all__
#

__tracker.validate()
