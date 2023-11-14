*sklearndf* is an open source library designed to address a common need with
`scikit-learn <https://github.com/scikit-learn/scikit-learn>`__: the outputs of
transformers are numpy arrays, even when the input is a
data frame. However, to inspect a model it is essential to keep track of the
feature names.

To this end, *sklearndf* enhances scikit-learn's estimators as follows:

- **Preserve data frame structure**:
    Return data frames as results of transformations, preserving feature names as the column index.
- **Feature name tracing**:
    Add additional estimator properties to enable tracing a feature name back to its original input feature; this is especially useful for transformers that create new features (e.g., one-hot encode), and for pipelines that include such transformers.
- **Easy use**:
    Simply append DF at the end of your usual scikit-learn class names to get enhanced data frame support!

.. Begin-Badges

|pypi| |conda| |python_versions| |code_style| |made_with_sphinx_doc| |License_badge|

.. End-Badges

License
---------------------------

*sklearndf* is licensed under Apache 2.0 as described in the
`LICENSE <https://github.com/BCG-X-Official/sklearndf/blob/develop/LICENSE>`_ file.

.. Begin-Badges

.. |conda| image:: https://anaconda.org/bcg_gamma/sklearndf/badges/version.svg
    :target: https://anaconda.org/BCG_Gamma/sklearndf

.. |pypi| image:: https://badge.fury.io/py/sklearndf.svg
    :target: https://pypi.org/project/sklearndf/

.. |python_versions| image:: https://img.shields.io/badge/python-3.7|3.8|3.9-blue.svg
    :target: https://www.python.org/downloads/release/python-380/

.. |code_style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |made_with_sphinx_doc| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
    :target: https://bcg-x-official.github.io/sklearndf/index.html

.. |license_badge| image:: https://img.shields.io/badge/License-Apache%202.0-olivegreen.svg
    :target: https://opensource.org/licenses/Apache-2.0

.. End-Badges