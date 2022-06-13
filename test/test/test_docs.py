"""
Test docstrings.
"""

from pytools.api import DocValidator


def test_doc() -> None:
    assert DocValidator(
        root_dir="src",
        exclude_from_parameter_validation=(
            r"sklearndf\.(?:"
            + "|".join(
                f"(?:{pattern})"
                for pattern in (
                    # generated classes, except in the '.extra' subpackages
                    r"(?:classification|clustering|regression|transformation)"
                    r"\.(?!extra\.).*",
                    # LGBM estimators in the '.extra' packages
                    r"(?:classification|regression)\.extra\.LGBM.*",
                    # XGBoost estimators in the '.extra' packages
                    r"(?:classification|regression)\.extra\.XGB.*",
                    # BorutaDF
                    r"transformation\.extra\.BorutaDF.*",
                    # scikit-learn pipeline classes
                    r"pipeline\.(PipelineDF|FeatureUnionDF).*",
                )
            )
            + ")"
        ),
    ).validate_doc(), "docstrings and type hints are valid"
