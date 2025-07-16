Release Notes
=============

.. |lightgbm| replace:: :external+lightgbm:doc:`lightgbm <index>`
.. |xgboost| replace:: :external+xgboost:doc:`xgboost <index>`
.. |mypy| replace:: :external+mypy:doc:`mypy <index>`
.. |nbsp| unicode:: 0xA0
   :trim:

*skelearndf* 2.4
----------------


2.4.0
~~~~~

*sklearndf* |nbsp| 2.4 adds support for
`scikit-learn 1.7 <https://scikit-learn.org/1.7>`_, and drops support for
*scikit-learn* |nbsp| 1.2 and earlier, and Python |nbsp| 3.8 and earlier.

- API: add DF wrapper classes :class:`.FixedThresholdClassifierDF`
  and :class:`.TunedThresholdClassifierCVDF` for native classifiers
  :class:`~sklearn.model_selection.FixedThresholdClassifier` and
  :class:`~sklearn.model_selection.TunedThresholdClassifierCV`, respectively
- API: add DF wrapper class :class:`.SelfTrainingClassifierDF` for native classifier
  :class:`~sklearn.semi_supervised.SelfTrainingClassifier`
- API: support tag propagation introduced in *scikit-learn* |nbsp| 1.6
- API: add utility function :func:`.is_scalar_nan` that had been removed from
  :mod:`sklearndf.utils`


*sklearndf* 2.3
---------------


2.3.0
~~~~~

*sklearndf* 2.3 adds support for
`scikit-learn 1.3 <https://scikit-learn.org/1.3>`_
and drops support for *scikit-learn* |nbsp| 0.24.

- API: add DF wrapper classe :class:`.HDBSCANDF` for native estimator
  :class:`~sklearn.cluster.HDBSCAN`
- API: add DF wrapper class :class:`.TargetEncoderDF` for native estimator
  :class:`~sklearn.preprocessing.TargetEncoder`


*sklearndf* 2.2
---------------

*sklearndf* 2.2 adds support for
`scikit-learn 1.2 <https://scikit-learn.org/1.2>`_, and enhances the EstimatorDF
API.


2.2.1
~~~~~

- VIZ: use *scikit-learn*'s native HTML representation of estimators, if available


2.2.0
~~~~~

*sklearndf* 2.2 adds support for
`scikit-learn 1.2 <https://scikit-learn.org/1.2>`_.
It drops support for *scikit-learn* |nbsp| 0.23 and earlier due to incomplete
support of sparse output (see below).

- API: DF estimators now support native estimators using sparse matrices as input or
  output, and automatically convert them to or from sparse :class:`~pandas.DataFrame`
  objects
- API: new property :attr:`.EstimatorDF.output_names_` to get the names of the output
  columns the estimator was fitted with
- API: new method :attr:`.LearnerPipelineDF.preprocess` to apply the preprocessing step
  to a data frame
- API: remove properties ``feature_names_out_`` and ``feature_names_original_`` from
  class :class:`.LearnerPipelineDF`
- API: :class:`~pandas.Index` instances obtained from
  :attr:`.EstimatorDF.feature_names_in_` and :attr:`.TransformerDF.feature_names_out_`
  are now named ``"feature"`` instead of ``"feature_in"`` and ``"feature_out"``,
  respectively, and :class:`~pandas.Series` instances obtained from
  :attr:`.TransformerDF.feature_names_original_` are now named ``"feature_original"``
  instead of ``"feature_in"``, and their indices are now named ``"feature"`` instead
  of ``"feature_out"``; this is to separate the semantics of the originating property
  from the column index, which may be used in other contexts



*sklearndf* 2.1
---------------

*sklearndf* 2.1 adds support for
`scikit-learn 1.1 <https://scikit-learn.org/1.1>`_.


2.1.1
~~~~~

This is a maintenance release to catch up with *sklearndf* |nbsp| 2.0.2.


2.1.0
~~~~~

- API: new clusterer :class:`.BisectingKMeansDF`
- API: new transformer :class:`.MiniBatchNMFDF`
- API: new transformer :class:`.RandomTreesEmbeddingDF`; note that class
  :class:`~sklearn.ensemble.RandomTreesEmbedding` existed previously in *scikit-learn*,
  but is based on :class:`~sklearn.base.TransformerMixin` only as of
  *scikit-learn* |nbsp| 1.1
- API: support parameters ``max_categories`` and ``min_frequency`` of
  :class:`.OneHotEncoderDF`, introduced in *scikit-learn* |nbsp| 1.1
- API: support array-like values for parameter ``drop`` of :class:`.OneHotEncoderDF`
- API: support ``"passthrough"`` as a transformer in :class:`.FeatureUnionDF`
- API: remove ``GeneralizedLinearRegressorDF`` since the underlying native estimator is
  a base class and not intended to be used as a regressor of its own


*sklearndf* 2.0
---------------

*sklearndf* 2.0 adds support for
`scikit-learn 1.0 <https://scikit-learn.org/1.0>`_,
adds data frame support for clusterers along with additional API enhancements and
improvements, and is now subject to static type checking with |mypy|.


2.0.2
~~~~~

- BUILD: add support for :mod:`pandas` 2.0 and above
- FIX: property :attr:`.PCADF.n_components_` now returns the value of
  :attr:`~sklearndf.decomposition.PCA.n_components_`, not
  :attr:`~sklearndf.decomposition.PCA.n_components`
- FIX: detect missing and extra columns when validating data frames resulting from
  transforms, even when the total column count is correct


2.0.1
~~~~~

- API: upon declaration of new wrapper classes, automatically validate that their
  associated native estimators are compatible with the wrapper class
- API: new public constants ``DROP`` and ``PASSTHROUGH`` in
  :class:`.ColumnTransformerDF`
- FIX: base :class:`.LGBMClassifierDF` and :class:`.XGBClassifierDF` on the
  the correct wrapper class :class:`.ClassifierWrapperDF`
- FIX: support array-like values for parameter ``drop`` of :class:`.OneHotEncoderDF`
- FIX: various minor tweaks and stability improvements


2.0.0
~~~~~

- API: :class:`.ClassifierDF` and :class:`.RegressorDF` get a new base class
  :class:`.SupervisedLearnerDF`, which in turn is based on :class:`.LearnerDF`;
  :class:`.SupervisedLearnerDF` implements method :meth:`~.SupervisedLearnerDF.score`,
  which is no longer implemented by :class:`.LearnerDF`
- API: new class :class:`.ClusterDF`, based on :class:`.LearnerDF`
- API: class :class:`.EstimatorDF` now implements the
  :class:`~pytools.expression.HasExpressionRepr` mix-in, rendering estimator
  representations as :class:`~pytools.expression.Expression` objects to enable better
  formatting
- API: added data frame support for method
  :meth:`~.PartialFitRegressorWrapperDF.partial_fit`
- API: removed ``OutlierRemoverDF``
- API: removed dependency on package |lightgbm|: :class:`.LGBMClassifierDF` and
  :class:`.LGBMRegressorDF` are still available if |lightgbm| is installed
- API: added support for |xgboost|: :class:`.XGBClassifierDF` and
  :class:`.XGBClassifierDF` are available if |xgboost| is installed
- API: DF wrapper classes are now created using proper class declarations to better
  conform with Python type conventions checked by |mypy|;
  see :mod:`sklearndf.wrapper` for details
- API: remove functions ``make_df_estimator``, ``make_df_classifier``,
  ``make_df_regressor``, and ``make_df_transformer`` which are now obsolete
- API: move some classes in :mod:`sklearndf.wrapper` to sub-packages
  :mod:`sklearndf.wrapper.stacking` and :mod:`sklearndf.wrapper.numpy` to improve
  package navigability and to achieve better de-coupling of the underlying code;
  this change also moves :class:`~.StackingClassifierWrapperDF` and
  :class:`~.StackingRegressorWrapperDF` to package :mod:`sklearndf.wrapper.stacking`


*sklearndf* 1.2
---------------

This release adds support for `scikit-learn 0.24 <https://scikit-learn.org/0.24/>`_.


1.2.3
~~~~~

This is a maintenance release to catch up with *sklearndf* |nbsp| 1.1.3.


1.2.2
~~~~~

This release makes small API tweaks, and catches up with *sklearndf* |nbsp| 1.1.2.

- API: make type hints more specific in signatures for
  :func:`.make_df_transformer`, :func:`.make_df_classifier`, and
  :func:`.make_df_regressor`


1.2.1
~~~~~

This is a maintenance release to catch up with *sklearndf* |nbsp| 1.1.1.


1.2.0
~~~~~

- API: add `DF` adaptations for classes introduced by *scikit-learn* |nbsp| 0.24:
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
- BUILD: add support for `numpy` |nbsp| 1.20
- BUILD: updates and changes to the CI/CD pipeline


1.0.1
~~~~~

Initial release.
