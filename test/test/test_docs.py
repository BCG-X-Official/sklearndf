"""
Test docstrings.
"""

from pytools.api import DocValidator


def test_docstrings() -> None:
    assert DocValidator(
        root_dir="src",
        exclude_from_parameter_validation=(
            "|".join(
                f"(?:{pattern})"
                for pattern in (
                    (
                        r"sklearndf\.(?:classification|regression|transformation)\."
                        r"(?!extra\.).*"
                    ),
                    r"sklearndf\.(?:classification|regression)\.extra\.LGBM.*",
                    r"sklearndf\.transformation\.extra\.Boruta.*",
                    r"sklearndf\.pipeline\.(PipelineDF|FeatureUnionDF)\..*",
                )
            )
        ),
    ).validate_docstrings(), "docstrings are valid"
