"""
Additional 3rd party classifiers that implement the `scikit-learn` interface.

Note that 3rd party packages implementing the associated native estimators must be
installed explicitly: they are not included in `sklearndf`'s package requirements to
achieve a lean package footprint for default installs of `sklearndf`.
"""
from ._extra import *
