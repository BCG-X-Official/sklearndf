"""
Test docstrings.
"""

from pytools.api.doc import DocValidator


def test_docstrings() -> None:
    assert DocValidator(
        root_dir="src",
        exclude_from_parameter_validation=(
            r"sklearndf\.(?:"
            + "|".join(
                f"(?:{pattern})"
                for pattern in (
                    # generated classes, except in the '.extra' subpackages
                    r"(?:classification|regression|transformation)\.(?!extra\.).*",
                    # LGBM estimators in the '.extra' packages
                    r"(?:classification|regression)\.extra\.LGBM.*",
                    # BorutaDF
                    r"transformation\.extra\.BorutaDF.*",
                    # scikit-learn pipeline classes
                    r"pipeline\.(PipelineDF|FeatureUnionDF).*",
                )
            )
            + ")"
        ),
    ).validate_docstrings(), "docstrings are valid"
