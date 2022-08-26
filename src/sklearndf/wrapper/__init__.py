"""
Wrapper class API for enhancing the functionality of native estimators conforming with
the `scikit-learn` API.

In most cases, creating a DF classifier or regressor class is as simple as

.. code-block:: python

    class RandomForestClassifierDF(
        ClassifierWrapperDF[RandomForestClassifier],
        native=RandomForestClassifier,
    ):
        \"""Stub for DF wrapper of class ``RandomForestClassifier``\"""

    class RandomForestRegressorDF(
        RegressorWrapperDF[RandomForestRegressor],
        native=RandomForestRegressor,
    ):
        \"""Stub for DF wrapper of class ``RandomForestRegressor``\"""


Any class implementing the `scikit-learn` estimator protocol (and subclassing
:class:`.BaseEstimator`) can be used to create a DF wrapper by declaring a wrapper
class as follows:

.. code-block::

    class <name>DF(<DF wrapper class>, native=<native class>):
        \"""Stub for DF wrapper of class ``<native class>``\"""

The resulting wrapper class implements a *delegation* pattern, forwarding method calls
and attribute access to a native estimator instance while

- implementing enhanced functionality introduced by the :class:`.EstimatorDF` class
  hierarchy, managing feature names and translating between data frames and *numpy*
  arrays behind the scenes
- adopting all additional methods and attributes from the wrapped native estimator
- delegating relevant method calls and attribute access to the native estimator,
  thus replicating the original estimator's behaviour except for the enhanced
  functionality introduced by the :class:`.EstimatorDF` class hierarchy.

Most regressors, classifiers, and clusterers can be augmented using the
:class:`.RegressorWrapperDF`, :class:`.ClassifierWrapperDF`, and
:class:`.ClusterWrapperDF` wrappers, respectively.

More care must be taken to wrap transformer classes and some clusterer classes, which
may require a more dedicated wrapper class to support the specific behaviour of the
native transformer or clusterer.
See packages :mod:`sklearndf.transformation.wrapper` and
:mod:`sklearndf.clustering.wrapper` for more details on these.

For more advanced examples, including the use of custom wrapper classes, see the many
examples in modules
:mod:`sklearndf.transformation`, :mod:`sklearndf.classification`,
:mod:`sklearndf.regression`, and :mod:`sklearndf.clustering`.
"""

from ._missing import *
from ._wrapper import *
