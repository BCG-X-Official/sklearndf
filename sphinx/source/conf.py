"""
Configuration file for the Sphinx documentation builder.

Receives the majority of the configuration from pytools conf_base.py
"""

import os
import sys

_dir_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "base")
sys.path.insert(0, _dir_base)

from conf_base import set_config

# ----- set custom configuration -----

set_config(
    globals(),
    project="sklearndf",
    html_logo=os.path.join("_images", "gamma_sklearndf_logo.png"),
    intersphinx_mapping={
        "lightgbm": ("https://lightgbm.readthedocs.io/en/latest/", None),
        "pytools": ("https://bcg-gamma.github.io/pytools/", None),
        "sklearn": ("https://scikit-learn.org/stable", None),
        "xgboost": ("https://xgboost.readthedocs.io/en/latest/", None),
    },
)
