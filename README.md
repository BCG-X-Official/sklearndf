# gamma-sklearndf

Enable scikit-learn transfomers, classifiers/regressors and pipelines/feature unions
to accept and return dataframes.

Currently this project contains:

- `gamma.sklearndf.regression` decorated scikit-learn regressors, incl. LGBM  
- `gamma.sklearndf.classification` decorated scikit-learn classifiers
- `gamma.sklearndf.transformation` decorated scikit-learn transformers
- `gamma.sklearndf.pipeline` decorated scikit-learn pipeline & feature union


# Installation
Latest stable conda package `gamma-sklearndf` can be installed using:
`conda install -c https://machine-1511619-alpha:bcggamma2019@artifactory.gamma.bcg.com/artifactory/api/conda/local-conda-1511619-alpha-01 gamma-sklearndf`

Or add the alpha channel and this package to your `environment.yml`:
```
channels:
  - conda-forge
  - https://machine-1511619-alpha:bcggamma2019@artifactory.gamma.bcg.com/artifactory/api/conda/local-conda-1511619-alpha-01
dependencies:
    - gamma-sklearndf
```
# Documentation
Documentation for all of alpha's Python projects is available at: 
https://git.sourceai.io/pages/alpha/alpha/

# API-Reference
See: https://git.sourceai.io/pages/alpha/alpha/gamma.sklearndf.html

# Contribute & Develop
Check out https://git.sourceai.io/alpha/alpha for developer instructions and guidelines.