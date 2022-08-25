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
