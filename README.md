# gamma-sklearndf

Enable scikit-learn transfomers, classifiers/regressors and pipelines/feature unions
to accept and return dataframes.

Currently this project contains:

- `gamma.sklearndf.regression` decorated scikit-learn regressors, incl. LGBM  
- `gamma.sklearndf.classification` decorated scikit-learn classifiers
- `gamma.sklearndf.transformation` decorated scikit-learn transformers
- `gamma.sklearndf.pipeline` decorated scikit-learn pipeline & feature union


# Installation
The pip-project `gamma-sklearndf` can be installed using:
- `pip install git+ssh://git@git.sourceai.io/alpha/gamma-sklearndf.git#egg=gamma.sklearndf`
 (*latest version*)
 - Check [this page](./../../releases) for available releases and use 
 `pip install git+ssh://git@git.sourceai.io/alpha/gamma-sklearndf.git@[VERSION-TAG]#egg=gamma.sklearndf`
 to install a specific version. E.g. to install `v1.0.0` use:
 `pip install git+ssh://git@git.sourceai.io/alpha/gamma-sklearndf.git@v1.0.0#egg=gamma.sklearndf`

Ensure that you have set up a working SSH key on git.sourceai.io!

# Documentation
Documentation for all of alpha's Python projects is available at: 
https://git.sourceai.io/pages/alpha/alpha/

# API-Reference
See: https://git.sourceai.io/pages/alpha/alpha/gamma.sklearndf.html

# Contribute & Develop
Check out https://git.sourceai.io/alpha/alpha for developer instructions and guidelines.