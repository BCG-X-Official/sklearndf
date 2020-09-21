sklearndf
=========

by BCG Gamma

.. image:: _static/gamma_logo.jpg

sklearndf is an open source library designed to address a common issue with scikit-learn: the outputs of transformers are numpy arrays, even when the input is a data frame. However, to inspect a model it is essential to keep track of the feature names.

TODO - add git badges as substitutions

Installation
---------------------

sklearndf supports both PyPI and Anaconda

Anaconda
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: RST

    conda install gamma-sklearndf

Pip
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: RST

    pip install gamma-sklearndf


Quickstart
----------------------

sklearndf ehances scikit-learn's estimators to achieve the following:

- **Preserve dataframe structure**:
    Return data frames as results of transformations, preserving feature names as the column index.
- **Feature name tracing**:
    Add additional estimator properties to enable tracing a feature name back to its original input feature; this is especially useful for transformers that create new features (e.g., one-hot encode), and for pipelines that include such transformers.
- **Easy use**:
    Simply append DF at the end of your usual scikit-learn class names to get enhanced data frame support!


Creating a DataFrame friendly scikit-learn pre-processing pipeline 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The titanic data set includes categorical features such as class and sex, and also has
missing values for numeric features (i.e., age) and categorical features (i.e., embarked).
The aim is to predict whether or not a passenger survived.
A standard sklearn example for this dataset can be found here:
https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py

We will build a preprocessing pipeline which:

- for categorical variables fills missing values with the string 'Unknown' and then one-hot encodes
- for numerical values fills missing values using median values

The strength of sklearndf is to maintain the scikit-learn conventions and expressivity,
while also preserving dataframes, and hence feature names. We can see this after using
fit_transform on our preprocessing pipeline.

.. code-block:: Python

    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    # Relevant sklearndf imports
    from sklearndf.transformation import (
        ColumnTransformerDF,
        OneHotEncoderDF,
        SimpleImputerDF,
    )
    from sklearndf.pipeline import (
        PipelineDF,
        ClassifierPipelineDF
    )
    from sklearndf.classification import RandomForestClassifierDF

    # Load titanic data
    titanic_X, titanic_y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    # Select features
    numerical_features = ['age', 'fare']
    categorical_features = ['embarked', 'sex', 'pclass']

    # Create a pre-processing pipeline
    preprocessing_numeric_df = SimpleImputerDF(strategy="median")

    preprocessing_categorical_df = PipelineDF(
        steps=[
            ('imputer', SimpleImputerDF(strategy='constant', fill_value='Unknown')),
            ('one-hot', OneHotEncoderDF(sparse=False, handle_unknown="ignore"))
        ]
    )

    preprocessing_df = ColumnTransformerDF(
        transformers=[
            ('categorical', preprocessing_categorical_df, categorical_features),
            ('numeric', preprocessing_numeric_df, numerical_features),
        ]
    )

    # Run pre-processing
    transformed_df = preprocessing_df.fit_transform(X=titanic_X, y=titanic_y)
    transformed_df.head()


+-------------+------------+------------+------------+------------------+------------+----------+------------+------------+------------+--------+----------+
| feature_out | embarked_C | embarked_Q | embarked_S | embarked_Unknown | sex_female | sex_male | pclass_1.0 | pclass_2.0 | pclass_3.0 | age    | fare     |
+=============+============+============+============+==================+============+==========+============+============+============+========+==========+
|0            |0           |0           |1           |0                 |1           |0         |1           |0           |0           |29      |211.3375  |
+-------------+------------+------------+------------+------------------+------------+----------+------------+------------+------------+--------+----------+
|1            |0           |0           |1           |0                 |0           |1         |1           |0           |0           |0.9167  |151.55    |
+-------------+------------+------------+------------+------------------+------------+----------+------------+------------+------------+--------+----------+
|2            |0           |0           |1           |0                 |1           |0         |1           |0           |0           |2       |151.55    |
+-------------+------------+------------+------------+------------------+------------+----------+------------+------------+------------+--------+----------+
|3            |0           |0           |1           |0                 |0           |1         |1           |0           |0           |30      |151.55    |
+-------------+------------+------------+------------+------------------+------------+----------+------------+------------+------------+--------+----------+
|4            |0           |0           |1           |0                 |1           |0         |1           |0           |0           |25      |151.55    |
+-------------+------------+------------+------------+------------------+------------+----------+------------+------------+------------+--------+----------+


Tracing features from post-transform to original 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sklearndf pipeline has a `features_original_` attribute which returns a series mapping
the output columns (the series' index) to the input columns (the series' values).
We can therefore easily select all output features generated from a given input feature,
such as in this case for embarked.

.. code-block:: Python

    embarked_type_derivatives = preprocessing_df.features_original_ == "embarked"
    transformed_df.loc[:, embarked_type_derivatives].head()


+-------------+------------+------------+------------+------------------+
| feature_out | embarked_C | embarked_Q | embarked_S | embarked_Unknown |
+=============+============+============+============+==================+
|0            |0.0         |0.0         |1.0         |0.0               |
+-------------+------------+------------+------------+------------------+
|1            |0.0         |0.0         |1.0         |0.0               |
+-------------+------------+------------+------------+------------------+
|2            |0.0         |0.0         |1.0         |0.0               |
+-------------+------------+------------+------------+------------------+
|3            |0.0         |0.0         |1.0         |0.0               |
+-------------+------------+------------+------------+------------------+
|4            |0.0         |0.0         |1.0         |0.0               |
+-------------+------------+------------+------------+------------------+


Completing the pipeline with a classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scikit-learn regressors and classifiers have a sklearndf sibling obtained by appending
DF to the class name; the API remains the same.
The result of any predict and decision function will be returned as a pandas series
(single output) or data frame (class probabilities or multi-output).

We can combine the preprocessing pipeline above with a classifier to create a full
predictive pipeline. sklearndf provides two useful, specialised pipeline objects for
this, RegressorPipelineDF and ClassifierPipelineDF. Both implement a special two-step
pipeline with one pre-processing step and one prediction step, while staying compatible
with the general sklearn pipeline idiom.

Using ClassifierPipelineDF we can combine the preprocessing pipeline with
RandomForestClassifierDF() to fit a model to a selected training set and then score
and a test set.

.. code-block:: Python

    # create full pipeline
    pipeline_df = ClassifierPipelineDF(
        preprocessing=preprocessing_df,
        classifier=RandomForestClassifierDF(
            n_estimators=1000,
            max_features=2/3,
            max_depth=7,
            random_state=42,
            n_jobs=-3
        )
    )

    # split data and then fit and score random forest classifier
    df_train, df_test, y_train, y_test = train_test_split(titanic_X, titanic_y, random_state=42)
    pipeline_df.fit(df_train, y_train)
    print(f"model score: {pipeline_df.score(df_test, y_test).round(2)}")


model score: 0.79

Development Guidelines
---------------------------

TBD - link to long section in documentation

Acknowledgements
---------------------------

This package provides a layer on top of some popular building blocks for Machine
Learning:

The `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ learners and
pipelining support the corresponding sklearndf implementations.