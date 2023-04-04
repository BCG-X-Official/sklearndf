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
                    # BorutaPy package
                    r"transformation\.extra\.BorutaDF",
                    # ARFS package
                    r"transformation\.extra\.BoostAGrootaDF",
                    r"transformation\.extra\.GrootCVDF",
                    r"transformation\.extra\.LeshyDF",
                    # scikit-learn pipeline classes
                    r"pipeline\.(PipelineDF|FeatureUnionDF).*",
                    # sparse frames version of FeatureUnion
                    r"pipeline\.wrapper\.FeatureUnion\.",
                )
            )
            + ")"
        ),
    ).validate_doc(), "docstrings and type hints are valid"
