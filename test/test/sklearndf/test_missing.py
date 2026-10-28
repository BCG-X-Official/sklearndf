import logging

import pytest

from sklearndf.wrapper import MissingEstimator

log = logging.getLogger(__name__)


def test_missing() -> None:
    class MyMissingEstimator(MissingEstimator):
        pass

    with pytest.raises(
        RuntimeError,
        match=(
            "Estimator MyMissingEstimator is not available. "
            "Please install the package that implements it."
        ),
    ):
        MyMissingEstimator(1, "2", a=2)


def test_missing_import_error() -> None:
    class MyMissingEstimator(
        MissingEstimator,
        import_error=ModuleNotFoundError("No module named 'xyz'"),
    ):
        pass

    with pytest.raises(
        RuntimeError,
        match=(
            r"^Estimator MyMissingEstimator is not available\. "
            r"Please install the package that implements it\. "
            r"\(import failed: ModuleNotFoundError: No module named 'xyz'\)$"
        ),
    ):
        MyMissingEstimator()
