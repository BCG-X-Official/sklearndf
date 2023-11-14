.. image:: sphinx/source/_images/sklearndf_logo.png

----

.. Begin-Badges

|pypi| |conda| |azure_build| |azure_code_cov|
|python_versions| |code_style| |made_with_sphinx_doc| |License_badge|

.. End-Badges

*sklearndf* is an open source library designed to address a common need with
`scikit-learn <https://github.com/scikit-learn/scikit-learn>`__: the outputs of
transformers are numpy arrays, even when the input is a
data frame. However, to inspect a model it is essential to keep track of the
feature names.

To this end, *sklearndf* enhances scikit-learn's estimators as follows:

- **Preserve data frame structure**:
  Return data frames as results of transformations, preserving feature names as the
  column index.
- **Feature name tracing**:
  Add additional estimator properties to enable tracing a feature name back to its
  original input feature; this is especially useful for transformers that create new
  features (e.g., one-hot encode), and for pipelines that include such transformers.
- **Easy use**:
  Simply append DF at the end of your usual scikit-learn class names to get enhanced
  data frame support!

The following quickstart guide provides a minimal example workflow to get up and running
with *sklearndf*.
For additional tutorials and the API reference,
see the `sklearndf documentation <https://bcg-x-official.github.io/sklearndf/>`__.
Changes and additions to new versions are summarized in the
`release notes <https://bcg-x-official.github.io/sklearndf/release_notes.html>`__.


Installation
------------

*sklearndf* supports both PyPI and Anaconda.
We recommend to install *sklearndf* into a dedicated environment.


Anaconda
~~~~~~~~

.. code-block:: sh

    conda create -n sklearndf
    conda activate sklearndf
    conda install -c bcg_gamma -c conda-forge sklearndf


Pip
~~~

macOS and Linux:
^^^^^^^^^^^^^^^^

.. code-block:: sh

    python -m venv sklearndf
    source sklearndf/bin/activate
    pip install sklearndf

Windows:
^^^^^^^^

.. code-block:: dosbatch

    python -m venv sklearndf
    sklearndf\Scripts\activate.bat
    pip install sklearndf


Quickstart
----------

Creating a DataFrame-friendly scikit-learn preprocessing pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The titanic data set includes categorical features such as class and sex, and also has
missing values for numeric features (i.e., age) and categorical features (i.e., embarked).
The aim is to predict whether or not a passenger survived.
A standard sklearn example for this dataset can be found
`here <https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py>`_.


We will build a preprocessing pipeline which:

- for categorical variables fills missing values with the string 'Unknown' and then one-hot encodes
- for numerical values fills missing values using median values

The strength of *sklearndf* is to maintain the scikit-learn conventions and
expressiveness, while also preserving data frames, and hence feature names. We can see
this after using ``fit_transform`` on our preprocessing pipeline.

.. code-block:: Python

    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    # relevant sklearndf imports
    from sklearndf.transformation import (
        ColumnTransformerDF,
        OneHotEncoderDF,
        SimpleImputerDF,
    )
    from sklearndf.pipeline import (
        PipelineDF,
        ClassifierPipelineDF,
    )
    from sklearndf.classification import RandomForestClassifierDF

    # load titanic data
    titanic_X, titanic_y = fetch_openml(
        "titanic", version=1, as_frame=True, return_X_y=True
    )

    # select features
    numerical_features = ['age', 'fare']
    categorical_features = ['embarked', 'sex', 'pclass']

    # create a preprocessing pipeline
    preprocessing_numeric_df = SimpleImputerDF(strategy="median")

    preprocessing_categorical_df = PipelineDF(
        steps=[
            ('imputer', SimpleImputerDF(strategy='constant', fill_value='Unknown')),
            ('one-hot', OneHotEncoderDF(sparse=False, handle_unknown="ignore")),
        ]
    )

    preprocessing_df = ColumnTransformerDF(
        transformers=[
            ('categorical', preprocessing_categorical_df, categorical_features),
            ('numeric', preprocessing_numeric_df, numerical_features),
        ]
    )

    # run preprocessing
    transformed_df = preprocessing_df.fit_transform(X=titanic_X, y=titanic_y)
    transformed_df.head()


+-------------+------------+------------+---+------------+--------+--------+
| feature_out | embarked_C | embarked_Q | … | pclass_3.0 | age    | fare   |
+=============+============+============+===+============+========+========+
| **0**       | 0          | 0          | … | 0          | 29     | 211.34 |
+-------------+------------+------------+---+------------+--------+--------+
| **1**       | 0          | 0          | … | 0          | 0.9167 | 151.55 |
+-------------+------------+------------+---+------------+--------+--------+
| **2**       | 0          | 0          | … | 0          | 2      | 151.55 |
+-------------+------------+------------+---+------------+--------+--------+
| **3**       | 0          | 0          | … | 0          | 30     | 151.55 |
+-------------+------------+------------+---+------------+--------+--------+
| **4**       | 0          | 0          | … | 0          | 25     | 151.55 |
+-------------+------------+------------+---+------------+--------+--------+


Tracing features from post-transform to original 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *sklearndf* pipeline has a ``feature_names_original_`` attribute
which returns a *pandas* ``Series``, mapping the output column names (the series' index)
to the input column names (the series' values).
We can therefore easily select all output features generated from a given input feature,
such as in this case for embarked.

.. code-block:: Python

    embarked_type_derivatives = preprocessing_df.feature_names_original_ == "embarked"
    transformed_df.loc[:, embarked_type_derivatives].head()


+-------------+------------+------------+------------+------------------+
| feature_out | embarked_C | embarked_Q | embarked_S | embarked_Unknown |
+=============+============+============+============+==================+
| **0**       | 0.0        | 0.0        | 1.0        | 0.0              |
+-------------+------------+------------+------------+------------------+
| **1**       | 0.0        | 0.0        | 1.0        | 0.0              |
+-------------+------------+------------+------------+------------------+
| **2**       | 0.0        | 0.0        | 1.0        | 0.0              |
+-------------+------------+------------+------------+------------------+
| **3**       | 0.0        | 0.0        | 1.0        | 0.0              |
+-------------+------------+------------+------------+------------------+
| **4**       | 0.0        | 0.0        | 1.0        | 0.0              |
+-------------+------------+------------+------------+------------------+


Completing the pipeline with a classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scikit-learn regressors and classifiers have a *sklearndf* sibling obtained by appending
``DF`` to the class name; the API of the native estimators is preserved.
The result of any predict and decision function will be returned as a *pandas*
``Series`` (single output) or ``DataFrame`` (class probabilities or multi-output).

We can combine the preprocessing pipeline above with a classifier to create a full
predictive pipeline. *sklearndf* provides two useful, specialised pipeline objects for
this, ``RegressorPipelineDF`` and ``ClassifierPipelineDF``.
Both implement a special two-step pipeline with one preprocessing step and one
prediction step, while staying compatible with the general sklearn pipeline idiom.

Using ``ClassifierPipelineDF`` we can combine the preprocessing pipeline with
``RandomForestClassifierDF`` to fit a model to a selected training set and then score
on a test set.

.. code-block:: Python

    # create full pipeline
    pipeline_df = ClassifierPipelineDF(
        preprocessing=preprocessing_df,
        classifier=RandomForestClassifierDF(
            n_estimators=1000,
            max_features=2/3,
            max_depth=7,
            random_state=42,
            n_jobs=-3,
        )
    )

    # split data and then fit and score random forest classifier
    df_train, df_test, y_train, y_test = train_test_split(
        titanic_X, titanic_y, random_state=42
    )
    pipeline_df.fit(df_train, y_train)
    print(f"model score: {pipeline_df.score(df_test, y_test).round(2)}")


|

    model score: 0.79


Contributing
------------

*sklearndf* is stable and is being supported long-term.

Contributions to *sklearndf* are welcome and appreciated.
For any bug reports or feature requests/enhancements please use the appropriate
`GitHub form <https://github.com/BCG-X-Official/sklearndf/issues>`_, and if you wish to do
so, please open a PR addressing the issue.

We do ask that for any major changes please discuss these with us first via an issue.

For further information on contributing please see our
`contribution guide <https://bcg-x-official.github.io/sklearndf/contribution_guide.html>`__.


License
-------

*sklearndf* is licensed under Apache 2.0 as described in the
`LICENSE <https://github.com/BCG-X-Official/sklearndf/blob/develop/LICENSE>`_ file.


Acknowledgements
----------------

Learners and pipelining from the popular Machine Learning package
`scikit-learn <https://github.com/scikit-learn/scikit-learn>`__  support
the corresponding *sklearndf* implementations.


BCG GAMMA
---------

We are always on the lookout for passionate and talented data scientists to join the
BCG GAMMA team. If you would like to know more you can find out about
`BCG GAMMA <https://www.bcg.com/en-gb/beyond-consulting/bcg-gamma/default>`_,
or have a look at
`career opportunities <https://www.bcg.com/en-gb/beyond-consulting/bcg-gamma/careers>`_.

.. Begin-Badges

.. |conda| image:: https://anaconda.org/bcg_gamma/sklearndf/badges/version.svg
   :target: https://anaconda.org/BCG_Gamma/sklearndf

.. |pypi| image:: https://badge.fury.io/py/sklearndf.svg
   :target: https://pypi.org/project/sklearndf/

.. |azure_build| image:: https://dev.azure.com/gamma-facet/facet/_apis/build/status/BCG-X-Official.sklearndf?repoName=BCG-X-Official%2Fsklearndf&branchName=develop
   :target: https://dev.azure.com/gamma-facet/facet/_build?definitionId=8&_a=summary

.. |azure_code_cov| image:: https://img.shields.io/azure-devops/coverage/gamma-facet/facet/8/2.1.x
   :target: https://dev.azure.com/gamma-facet/facet/_build?definitionId=8&_a=summary

.. |python_versions| image:: https://img.shields.io/badge/python-3.7|3.8|3.9-blue.svg
   :target: https://www.python.org/downloads/release/python-380/

.. |code_style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. |made_with_sphinx_doc| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
   :target: https://bcg-x-official.github.io/sklearndf/index.html

.. |license_badge| image:: https://img.shields.io/badge/License-Apache%202.0-olivegreen.svg
   :target: https://opensource.org/licenses/Apache-2.0

.. End-Badges
