# noinspection PyPackageRequirements
import sklearn
import yaml
from packaging import version

from test.paths import TEST_CONFIG_YML


def read_test_config(section: str = None):
    config_file_path = TEST_CONFIG_YML
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
        if section is None:
            return config
        else:
            for element in config:
                if type(element) == dict:
                    for key in element.keys():
                        if key == section:
                            return element[key]

            raise ValueError("Section %s not found in global config!" % section)


def check_sklearn_version(minimum: str = "0.21", maximum: str = "0.24"):
    """ Utility to check sklearn version against provided string. """
    v_sklearn = version.parse(sklearn.__version__)
    return version.parse(minimum) <= v_sklearn <= version.parse(maximum)
