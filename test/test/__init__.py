# noinspection PyPackageRequirements
import sklearn
from packaging import version

from test.paths import TEST_CONFIG_YML


def check_sklearn_version(minimum: str = "0.21", maximum: str = "0.24"):
    """ Utility to check sklearn version against provided string. """
    v_sklearn = version.parse(sklearn.__version__)
    return version.parse(minimum) <= v_sklearn <= version.parse(maximum)
