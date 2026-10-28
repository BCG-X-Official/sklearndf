# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

*sklearndf* wraps scikit-learn estimators so they accept and return `pandas` DataFrames/Series, preserving feature names and exposing feature-name tracing (`feature_names_in_`, `feature_names_out_`, `feature_names_original_`). Every scikit-learn class `Foo` gets a DF counterpart `FooDF` (e.g. `RandomForestRegressorDF`). Source lives in `src/sklearndf`, tests in `test/test`. Packaged with flit (`pyproject.toml`); supports Python >=3.9 (`requires-python`), tested up to 3.14; depends on `gamma-pytools` (provides `AllTracker`, `inheritdoc`, expression repr, `fitted_only`).

## Commands

```sh
# environment (conda env "sklearndf-develop" from environment.yml) + git hooks
./dev-setup.sh          # conda env create -f environment.yml; pre-commit install

# tests (pytest config is in tox.ini; testpaths = test/test/)
# the contribution guide says `pytest tests/` and to `export PYTHONPATH=./src/`;
# the test dir is actually test/test/, and PYTHONPATH is only needed if sklearndf isn't installed
# coverage: pytest --cov=src/sklearndf test/test/
pytest test/test/
pytest test/test/sklearndf/test_regression.py            # one file
pytest test/test/sklearndf/test_regression.py::test_name  # one test
pytest -k "<expr>"

# lint / format / types (all run via pre-commit: isort, black, flake8, mypy)
pre-commit run --all-files
pre-commit run mypy --all-files
```

Flake8 config is in `tox.ini` (max line length 88, black-compatible). Docs are built with Sphinx from `sphinx/` (`sphinx/make.py`); `make.py` at the root delegates to a sibling `../pytools` checkout for conda builds.

## Architecture

### Wrapper mechanism (`src/sklearndf/wrapper/_wrapper.py`)
The core of the library. DF classes are **stubs that subclass a generic wrapper and name the native class via a class keyword**:

```python
class TargetEncoderDF(
    ColumnPreservingTransformerWrapperDF[TargetEncoder], native=TargetEncoder
):
    """Stub for DF wrapper of class ``TargetEncoder``"""
```

`EstimatorWrapperDFMeta` (metaclass) uses `native=` to generate `__init__`, mirror the native class's public attributes/methods/docstrings onto the wrapper, and create the delegate instance at construction. Wrapper hierarchy: `EstimatorWrapperDF` → `TransformerWrapperDF`, `LearnerWrapperDF` → `SupervisedLearnerWrapperDF` → `RegressorWrapperDF` / `ClassifierWrapperDF`, plus `ClusterWrapperDF` and `MetaEstimatorWrapperDF` (for estimators that wrap other estimators, e.g. pipelines/ensembles). A meta-estimator wrapper requires exactly one delegate attribute (`estimator`, `base_estimator` or `estimators`) on the native class; it keeps DF delegates (returned by the properties of the same name), passes the native counterparts to the native meta-estimator, and accepts lists that mix DF and native estimators. Abstract interfaces (`EstimatorDF`, `TransformerDF`, `RegressorDF`, …) live in `src/sklearndf/_sklearndf.py`.

Transformers need a strategy for output column names, supplied by mixin wrappers in `transformation/wrapper/_wrapper.py` (e.g. `ColumnPreservingTransformerWrapperDF`, `NComponentsDimensionalityReductionWrapperDF`, `ColumnSubsetTransformerWrapperDF`). Picking the right one for a new transformer is the main per-class decision. Special wrappers: `wrapper/numpy` (estimators needing ndarray input), `wrapper/stacking`, `wrapper/_missing.py`, and per-domain `*/wrapper` packages. `wrapper/_missing.py` provides `MissingEstimator`, the base of the mock classes used when an optional third-party package cannot be imported; each `except ImportError as import_error:` passes the exception as class keyword `import_error` so that it appears in the `RuntimeError` message (a package can be installed but still fail to import).

### Per-version module layout
Each domain package (`classification`, `regression`, `clustering`, `transformation`, `pipeline`) is split into a base module plus version-gated modules named after the **scikit-learn version that introduced the classes** (`_transformation_v1_3.py`, `_regression_v0_23.py`, …). `__init__.py` star-imports them all. `extra/` subpackages hold wrappers for optional third-party estimators (lightgbm, xgboost, boruta, arfs).

Each of these modules follows a strict convention, enforced at import time:
- an `__all__` listing exactly the `*DF` classes defined in that module;
- a `__imported_estimators` snapshot taken before class definitions and a trailing check that raises `RuntimeError` if `__all__` doesn't match the newly defined `*DF` symbols. When adding a class, add it to `__all__`.

Version-dependent behaviour elsewhere is gated through `src/sklearndf/_sklearn_version.py` (`__sklearn_1_6__`, etc.). Supported sklearn range is declared in `pyproject.toml` (currently `>=1.3,<1.8a`); update `_sklearn_version.py` constants, pre-commit deps, `environment.yml` and `condabuild/` together when bumping.

### Tests
`test/test/sklearndf/test_sklearn_coverage.py` introspects installed scikit-learn and fails if any estimator lacks a `*DF` counterpart (exclusion lists at top of that file for deliberately unsupported classes; `UNSUPPORTED_SKLEARN_PACKAGES` in `conftest.py`). New scikit-learn releases typically surface as failures here, which is the cue to add stubs in the matching `_vX_Y.py` module. Optional-dependency tests (lightgbm, xgboost, boruta, arfs) are skipped when not installed; this includes arfs 2.4.0 on Python 3.14 with a recent setuptools, because arfs imports `pkg_resources`, which setuptools no longer ships, so the arfs wrappers are not exercised there unless setuptools <81 is installed. The arfs wrappers split into `ARFSWrapperDF` (plain selector, e.g. `GrootCVDF`) and `ARFSMetaEstimatorWrapperDF` (selectors with a delegate `estimator`, e.g. `LeshyDF`, `BoostAGrootaDF`); arfs 3 requires Python <3.13.

## Conventions
(Full details in `sphinx/source/contribution_guide.rst`.)

- Keep the package importable on Python 3.9: import `TypeAlias` from `typing_extensions` (a declared dependency), and use `Union`/`Optional` instead of `X | Y` wherever it is evaluated at runtime (type aliases, `cast(...)` arguments, class-level annotations that `AllTracker` resolves). Mixing `|` with string forward references also fails on Python 3.10 to 3.13. Python 3.14 needs `gamma-pytools >=3.0.3`, which is enforced in the max build matrix (see below); `requires` keeps `>=2.1,<4a`, so older Pythons also allow pytools 2.x.

- Black formatting, isort `profile=black`, 88-column lines; type-checked with mypy.
- Commit messages complete the sentence "This commit will …": imperative mood, e.g. `add method foo to class Bar`, not `added …`. The history also uses prefixes such as `FIX:`, `RF:`, `BUILD:`.
- Docstrings: mandatory for all public modules, classes, functions, properties and attributes; reStructuredText; no parameter types in docstrings (use type hints).
  - Functions/methods: imperative (`"""Fit the model."""`).
  - Modules/classes/attributes/properties: start with a descriptive phrase, as in a dictionary entry; properties are documented like attributes, not `:return:`.
  - Multi-line docstrings put a line break after the opening and before the closing `"""`.
  - Sentences are capitalised and end with punctuation.
- DF stub docstrings are auto-derived from the native class; keep stubs to a one-line docstring.
- Release notes go in `RELEASE_NOTES.rst`.

## Docs and releases
- Build docs: `cd sphinx && python make.py html` (output in `sphinx/build/html`). Base config is shared via `pytools/sphinx/base`; needs a local LaTeX install for `imgmath`.
- `README.rst` is reused as the Sphinx quickstart; the Titanic example in it must be kept in sync with `sphinx/auxiliary/Titanic_getting_started_example.ipynb` by hand. Tutorial notebooks under `sphinx/source/tutorials` are committed with outputs.
- The package version is `__version__` in `src/sklearndf/__init__.py` (semantic versioning).
- Conda packages are built with rattler-build from `condabuild/recipe.yaml` (`condabuild/meta.yaml` is deprecated); `make.py` delegates to the sibling `../pytools/make.py`.
- **Dependency versions and `FACET_V_*` variables.** `make.py` reads the run dependencies from `[tool.flit.metadata]` `requires` (the `default` build) and from `[build.matrix.min]` / `[build.matrix.max]`, and exports each package as environment variable `FACET_V_<PACKAGE>`: the package name is upper-cased and every non-word character becomes `_` (`gamma-pytools` -> `FACET_V_GAMMA_PYTOOLS`, `scikit-learn` -> `FACET_V_SCIKIT_LEARN`, `typing_extensions` -> `FACET_V_TYPING_EXTENSIONS`; `python` comes from `requires-python`). `condabuild/recipe.yaml` and `tox.ini` take every runtime and test-extra dependency version from these variables (`${{ env.get('FACET_V_...') }}` in the recipe) and never hard-code a version; only build tools (flit, pip, rattler-build) and test tools (pytest) are literals, and optional packages are guarded with `env.exists(...)`. Rules enforced by `make.py`: `requires` lists each package once as `name specifier` with no environment markers (an entry such as `pkg >=1; python_version >= '3.14'` fails every build with "invalid version spec"), must not contain `python`, and every `requires` entry needs a value in both the min and the max matrix. To add or change a runtime dependency, update `requires`, both matrices, `environment.yml` and the `run` section of `condabuild/recipe.yaml` together. Check the result with the `make.py` parser (`Builder._get_requirements_to_expose`) before pushing.
- CI (`azure-pipelines.yml`) runs on pushes to `2.4.x` and `release/*`, on PRs targeting either, and nightly on `2.4.x`; a push to a `dev/*` branch alone does not start it. The full min/default/max matrix runs for PRs from `dev/*` or `release/*` branches and nightly; other PRs run a small subset, and only if build config files changed. Whenever `flit` is installed with pip, also pin `flit_core>=2,<4`, since flit_core 4 rejects the `[tool.flit.metadata]` table.
- Releases run through Azure Pipelines (`azure-pipelines.yml`): PR `dev/<version>` onto `release/<version>`, merge with a merge commit (no squash), then merge back to the baseline branch and bump the version. Publishing (anaconda.org, PyPI, GitHub release, docs) happens only for builds of a `release/*` branch, never during the PR build, and is irreversible; always confirm before merging.
