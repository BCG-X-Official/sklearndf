import logging
import re
from os import environ
from urllib import request
from xml.etree import ElementTree

from pytest import mark

import sklearndf

log = logging.getLogger(__name__)

ENV_RUN_PACKAGE_VERSION_TEST = "RUN_PACKAGE_VERSION_TEST"


@mark.skipif(
    condition=environ.get(ENV_RUN_PACKAGE_VERSION_TEST, "") != sklearndf.__name__,
    reason="Parent build is not primarily for sklearndf.",
)
def test_package_version() -> None:
    dev_version = sklearndf.__version__

    log.info(f"Test package version – version set to: {dev_version}")
    assert re.match(
        r"^(\d)+\.(\d)+\.(\d)+(rc\d+)?$", dev_version
    ), "sklearndf.__version__ is not in MAJOR.MINOR.PATCH[rcN] format."

    releases_uri = "https://pypi.org/rss/project/sklearndf/releases.xml"

    with request.urlopen(releases_uri) as response:
        assert response.getcode() == 200, "Error getting releases from PyPi"
        releases_xml = response.read()

    tree = ElementTree.fromstring(releases_xml)
    releases_nodes = tree.findall(path=".//channel//item//title")
    releases = [r.text for r in releases_nodes]

    log.info(f"Found these releases on PyPi:{releases}")

    assert (
        dev_version not in releases
    ), f"Current package version {dev_version} already on PyPi"
