"""
Configuration file for the Sphinx documentation builder.

Receives majority of configuration from pytools conf_base.py
"""

import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            os.pardir,
            "pytools",
            "sphinx",
            "source",
        )
    )
)

from conf_base import *

# ----- custom configuration -----

set_config(project="sklearndf", modules=["pytools", "sklearndf"])
