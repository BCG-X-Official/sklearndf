# noinspection PyPackageRequirements
from typing import Optional

import sklearn
from packaging import version

from test.paths import TEST_CONFIG_YML


def check_sklearn_version(minimum: Optional[str], maximum: Optional[str] = None):
    """ Utility to check sklearn version against provided string. """
    v_sklearn = version.parse(sklearn.__version__)
    return not (
        (minimum and version.parse(minimum) > v_sklearn)
        or (maximum and v_sklearn > version.parse(maximum))
    )
