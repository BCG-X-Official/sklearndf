"""
Adapter classes that wrap DF estimators and accept numpy arrays for all DF estimator
methods that would usually only accept pandas data frames or series.

For use in meta-estimators that internally pass on numpy arrays to sub-estimators.
"""

from ._numpy import *
