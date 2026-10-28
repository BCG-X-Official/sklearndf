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

    Subclasses can pass the exception that caused the failed import as class keyword
    argument ``import_error``, to include it in the error message. This helps to
    diagnose packages that are installed but cannot be imported.
    """

    _import_error: str | None = None

    def __init_subclass__(
        cls, import_error: BaseException | None = None, **kwargs: Any
    ) -> None:
        """
        :param import_error: the exception raised when trying to import the native
            estimator, if any
        :param kwargs: additional class keyword arguments
        """
        super().__init_subclass__(**kwargs)
        if import_error is not None:
            cls._import_error = f"{type(import_error).__name__}: {import_error}"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        :param args: arbitrary positional arguments
        :param kwargs: arbitrary keyword arguments
        :raises RuntimeError: always raised upon instantiation
        """
        message = (
            f"Estimator {type(self).__name__} is not available. "
            f"Please install the package that implements it."
        )
        if self._import_error is not None:
            message += f" (import failed: {self._import_error})"
        raise RuntimeError(message)


#
# validate __all__
#

__tracker.validate()
