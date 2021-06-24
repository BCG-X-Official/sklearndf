Augmented scikit-learn classes are named after their native scikit-learn counterparts,
with `DF` added as a suffix:
:class:`.SimpleImputerDF` takes the place of :class:`~sklearn.impute.SimpleImputer`,
:class:`.RandomForestRegressorDF` takes the place of
:class:`~sklearn.ensemble.RandomForestRegressor`, and so on.

For all methods expecting an `X` argument for a
feature matrix and potentially a
`y` argument for one or more targets, `sklearndf`
estimators expect a pandas
:class:`~pandas.DataFrame` for `X` and a
pandas :class:`~pandas.Series` for a
1-dimensional `y`, or a pandas :class:`~pandas.DataFrame`
when fitting to multiple
targets or outputs.
This includes methods such as :meth:`~EstimatorDF.fit`,
:meth:`~TransformerDF.transform`, :meth:`~LearnerDF.predict`,
and so on.

All estimators enhanced by `sklearndf` also implement an
additional attribute
:attr:`~EstimatorDF.feature_names_in_`, keeping track of the
column names of the data
frame used to fit the estimator.

`sklearndf` transformers also implement
:attr:`~TransformerDF.feature_names_out_` and
:attr:`~TransformerDF.feature_names_original_`, keeping track
of the feature names of the
transformed outputs as well as mapping output features
back to the input features.
This  enables tracing features back to the original
inputs even across complex
pipelines (see also :class:`.PipelineDF`)

`sklearndf` classes implement a class hierarchy that
follows the taxonomy of
scikit-learn classes (but is only partially reflected
via class inheritance in the
original implementation):

- all `sklearndf` transformers are subclasses of :class:`.TransformerDF`, which in turn
  provides the API for all common transformer methods, e.g.,
  :meth:`~TransformerDF.transform`

- all `sklearndf` regressors are subclasses of :class:`.RegressorDF`, which
  in turn provides the API for all common regressor methods, e.g.,
  :meth:`~LearnerDF.predict`

- all `sklearndf` classifiers are subclasses of :class:`.ClassifierDF`, which
  in turn provides the API for all common classifier methods, e.g.,
  :meth:`~ClassifierDF.predict_proba`

- all `sklearndf` regressors and classifiers are subclasses of :class:`.LearnerDF`

- all `sklearndf` estimators are subclasses of :class:`.EstimatorDF`

`sklearndf` introduces two additional pipeline classes,
:class:`.RegressorPipelineDF` and
:class:`.ClassifierPipelineDF`, with an abstract base
class :class:`.LearnerPipelineDF`,
to allow for easier handling of common types of ML
pipelines.
These classes implement pipelines with two steps --
ne preprocessing step, followed by
a regressor or a classifier.

`sklearndf` also provides data frame support for a
selection of custom or 3rd-party
estimators, most notably :class:`.BorutaDF`,
:class:`.LGBMRegressorDF`, and
:class:`.LGBMClassifierDF`.

All `sklearndf` estimators are fully type hinted.

Please see the :ref:`release notes<release-notes>` for recent API updates and bug fixes.
