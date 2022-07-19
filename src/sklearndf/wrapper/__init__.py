"""
Factory functions and wrapper class API for enhancing the functionality of native
estimators conforming with the `scikit-learn` API.

In most cases, creating a DF classifier or regressor class is as simple as

.. code-block:: python

    class RandomForestClassifierDF(
    ClassifierWrapperDF[RandomForestClassifier],
    RandomForestClassifier,  # type: ignore
    native=RandomForestClassifier):
        \"""Stub for DF wrapper of class ``RandomForestClassifier``\"""

    class RandomForestRegressorDF(
    RegressorWrapperDF[RandomForestRegressor],
    RandomForestRegressor,  # type: ignore
    native=RandomForestRegressor):
        \"""Stub for DF wrapper of class ``RandomForestRegressor``\"""


Any class implementing the `scikit-learn` estimator protocol (and subclassing
:class:`.BaseEstimator`) can be used to create a DF wrapper by declaring a wrapper
class as follows:

.. codeblock:: python

    class <name>DF(
        <DF wrapper class>, <native class>, native=<native class>
    ):
        \"""Stub for DF wrapper of class ``<native class>``\"""

The resulting wrapper class

- implements enhanced functionality introduced by the :class:`.EstimatorDF` class
  hierarchy
- adopts all additional methods and attributes from the wrapped native estimator
- delegates relevant method calls and attribute access to the native estimator,
  thus replicating the original estimator's behaviour except for the enhanced
  functionality introduced by the :class:`.EstimatorDF` class hierarchy

Note that the resulting class will be based only on the DF wrapper class, even though
the native class must be stated both as a base class and as class parameter ``native``.
This is a design choice to improve code completions in IDEs relying on static code
inspection; at runtime the native class is removed from the class hierarchy during class
construction.

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

from ._wrapper import *
