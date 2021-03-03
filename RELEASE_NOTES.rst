Release Notes
=============

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
- BUILD: add support for numpy 1.20
- BUILD: updates and changes to the CI/CD pipeline


1.0.1
~~~~~

Initial release.
