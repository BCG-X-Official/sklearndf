Release Notes
=============

*sklearndf* 1.2
---------------

This release adds support for `scikit-learn 0.24 <https://scikit-learn.org/0.24/>`_.


1.2.3
~~~~~

This is a maintenance release to catch up with *sklearndf* 1.1.3.


1.2.2
~~~~~

This release makes small API tweaks, and catches up with *sklearndf* 1.1.2.

- API: make type hints more specific in signatures for
  :func:`.make_df_transformer`, :func:`.make_df_classifier`, and
  :func:`.make_df_regressor`


1.2.1
~~~~~

This is a maintenance release to catch up with *sklearndf* 1.1.1.


1.2.0
~~~~~

- API: add `DF` adaptations for classes introduced by scikit-learn 0.24:
  :class:`.PolynomialCountSketchDF` and :class:`.SequentialFeatureSelectorDF`


*sklearndf* 1.1
---------------

1.1.3
~~~~~

This release relaxes package dependencies to support any `numpy` version `1.x` from
1.16.


1.1.2
~~~~~

This release improves compatibility with `scikit-learn` and fixes bugs.

- API: add full support for the
  `_estimator_type <https://scikit-learn.org/stable/glossary.html#term-_estimator_type>`__
  attribute
- FIX: do not reset transformers when calling :meth:`.TransformerDF.inverse_transform`
- FIX: accept `"passthrough"` as value for arg `remainder` of
  :class:`.ColumnTransformerDF`


1.1.1
~~~~~

This release addresses compatibility issues with meta-estimators.

- FIX: support complex DF estimators inside :class:`.StackingEstimatorDF`
- FIX: raise an exception if a base estimator is not supported by one of `sklearndf`'s
  implementations for DF meta-estimators


1.1.0
~~~~~

This release exposes the `wrapper` API used to generate augmented DF estimators from
native `scikit-learn` estimators.

- API: expose the :class:`.EstimatorWrapperDF` class hierarchy through the new
  :mod:`sklearndf.wrapper` package
- API: create new `scikit-learn` wrapper classes with the new functions
  :func:`.make_df_estimator`, :func:`.make_df_classifier`, :func:`.make_df_regressor`,
  and :func:`.make_df_transformer`


*sklearndf* 1.0
---------------

1.0.2
~~~~~

This is a maintenance release focusing on enhancements to the CI/CD pipeline and bug
fixes.

- FIX: correctly mirror ``__init__`` signatures of native estimators to their
  corresponding DF estimators
- FIX: do not mirror native estimator class attributes and protected members to
  DF estimators
- FIX: support ``"passthrough"`` transformer in :class:`.ColumnTransformerDF`
- FIX: support ``drop`` parameter in :class:`.OneHotEncoderDF`
- BUILD: add support for `numpy` 1.20
- BUILD: updates and changes to the CI/CD pipeline


1.0.1
~~~~~

Initial release.
