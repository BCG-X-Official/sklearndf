from typing import List, Type, cast

import numpy as np
import pandas as pd
import pytest
import sklearn
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier, is_regressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, StandardScaler

import sklearndf.transformation
from ... import check_sklearn_version
from ...sklearndf import (
    check_expected_not_fitted_error,
    get_sklearndf_wrapper_class,
    iterate_classes,
)
from sklearndf import (
    ClassifierDF,
    RegressorDF,
    TransformerDF,
    __sklearn_0_24__,
    __sklearn_version__,
)
from sklearndf.classification import RandomForestClassifierDF
from sklearndf.transformation import (
    RFECVDF,
    RFEDF,
    ColumnTransformerDF,
    FeatureAgglomerationDF,
    KBinsDiscretizerDF,
    NormalizerDF,
    OneHotEncoderDF,
    SelectFromModelDF,
    SimpleImputerDF,
    SparseCoderDF,
    StandardScalerDF,
)
from sklearndf.transformation.extra import OutlierRemoverDF
from sklearndf.wrapper import TransformerWrapperDF

TRANSFORMER_EXCLUSIONS = [
    TransformerDF.__name__,
    OneHotEncoderDF.__name__,
    SelectFromModelDF.__name__,
    SparseCoderDF.__name__,
    ColumnTransformerDF.__name__,
    KBinsDiscretizerDF.__name__,
    RFECVDF.__name__,
    RFEDF.__name__,
    r".*WrapperDF",
]

if check_sklearn_version(minimum="0.24"):
    from sklearndf.transformation import SequentialFeatureSelectorDF

    TRANSFORMER_EXCLUSIONS.append(SequentialFeatureSelectorDF.__name__)

TRANSFORMERS_TO_TEST = iterate_classes(
    from_modules=sklearndf.transformation,
    matching=r".*DF",
    excluding=TRANSFORMER_EXCLUSIONS,
)


@pytest.fixture
def test_data() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "c0": [0.0, 1.0, 2.5, 3.0, 4.0, 5.2, 6.0, 7.0, 8.0, 9.0],
            "c1": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "c2": [2.5, 3.0, 4.0, 5.2, 6.0, 7.0, 8.0, 9.0, 0.0, 4.0],
            "c3": ["f", "g", "h", "i", "j", "a", "b", "c", "d", "e"],
        }
    )


@pytest.mark.parametrize(argnames="sklearndf_cls", argvalues=TRANSFORMERS_TO_TEST)
def test_wrapped_constructor(sklearndf_cls: Type[TransformerDF]) -> None:
    transformer_df: TransformerDF = sklearndf_cls()

    if isinstance(transformer_df, RegressorDF):
        assert is_regressor(transformer_df)
        assert not is_classifier(transformer_df)
    elif isinstance(transformer_df, ClassifierDF):
        assert is_classifier(transformer_df)
        assert not is_regressor(transformer_df)
    elif isinstance(transformer_df, TransformerWrapperDF):
        if isinstance(transformer_df, FeatureAgglomerationDF):
            assert transformer_df._estimator_type == "clusterer"
        else:
            # noinspection PyUnresolvedReferences
            assert transformer_df._estimator_type is None
    else:
        assert getattr(transformer_df, "_estimator_type", None) is None


def test_special_wrapped_constructors() -> None:
    rf = RandomForestClassifierDF()

    with pytest.raises(NotImplementedError):
        OneHotEncoderDF()
    OneHotEncoderDF(sparse=False)

    SelectFromModelDF(estimator=rf)

    SparseCoderDF(dictionary=np.array([]))

    ColumnTransformerDF(transformers=[])

    with pytest.raises(NotImplementedError):
        KBinsDiscretizerDF()
    KBinsDiscretizerDF(encode="onehot-dense")

    RFECVDF(estimator=rf)

    RFEDF(estimator=rf)

    if check_sklearn_version(minimum="0.24"):
        from sklearndf.transformation import SequentialFeatureSelectorDF

        SequentialFeatureSelectorDF(estimator=rf)


@pytest.mark.parametrize(
    argnames="sklearn_cls",
    argvalues=iterate_classes(
        from_modules=sklearn.preprocessing,
        matching=r".*PowerTransformer|QuantileTransformer|.*Scaler",
    ),
)
def test_fit_transform(
    sklearn_cls: Type[BaseEstimator], test_data: pd.DataFrame
) -> None:
    # we only need the numerical column of the test data
    test_data = test_data.select_dtypes(include=float)

    # get the wrapped counterpart for sklearn:
    wrapper_class = get_sklearndf_wrapper_class(
        to_wrap=sklearn_cls, from_module=sklearndf.transformation
    )

    assert issubclass(wrapper_class, TransformerDF)

    # initialize both kind of transformers
    transformer_native = cast(TransformerMixin, sklearn_cls())
    transformer_df = cast(TransformerDF, wrapper_class())

    # for sklearn >=0.22 - check if not_fitted error is raised properly:
    check_expected_not_fitted_error(estimator=transformer_df)

    # test fit followed by transform

    # noinspection PyUnresolvedReferences
    transformed_native = transformer_native.fit(X=test_data).transform(X=test_data)
    transformed_df = transformer_df.fit(X=test_data).transform(X=test_data)

    assert transformed_df.columns.equals(test_data.columns)
    assert_array_equal(transformed_df.values, transformed_native)

    # test fit transform

    transformed_native = transformer_native.fit_transform(X=test_data)
    transformed_df = transformer_df.fit_transform(X=test_data)

    assert transformed_df.columns.equals(test_data.columns)
    assert_array_equal(transformed_df.values, transformed_native)

    # test inverse transform

    inverse_transformed_df = transformer_df.inverse_transform(X=transformed_df)
    assert_frame_equal(
        inverse_transformed_df, test_data.rename_axis(columns="feature_in")
    )


def test_column_transformer(test_data: pd.DataFrame) -> None:
    numeric_columns: List[str] = test_data.select_dtypes(include=float).columns.tolist()
    assert numeric_columns == ["c0", "c2"]

    feature_names_in_expected = test_data.columns.rename("feature_in")

    for remainder, output_names in [
        ("drop", ["c0", "c2", "c3"]),
        ("passthrough", ["c0", "c2", "c3", "c1"]),
    ]:
        feature_names_out_expected = pd.Index(output_names, name="feature_out")

        # test fit-transform in connection with ColumnTransformer(DF)
        tx_df = StandardScalerDF()
        col_tx_df = ColumnTransformerDF(
            transformers=[
                ("tx", tx_df, numeric_columns),
                ("keep", "passthrough", ["c3"]),
            ],
            remainder=remainder,
        )
        transformed_df = col_tx_df.fit_transform(X=test_data)

        tx_native = StandardScaler()
        col_tx_native = ColumnTransformer(
            transformers=[
                ("tx", tx_native, numeric_columns),
                ("keep", "passthrough", ["c3"]),
            ],
            remainder=remainder,
        )
        transformed_native = col_tx_native.fit_transform(X=test_data)

        assert_frame_equal(
            transformed_df,
            pd.DataFrame(transformed_native, columns=feature_names_out_expected),
        )

        assert col_tx_df.feature_names_in_.equals(feature_names_in_expected)
        assert col_tx_df.feature_names_out_.equals(feature_names_out_expected)
        assert col_tx_df.feature_names_original_.equals(
            pd.Series(feature_names_out_expected, index=feature_names_out_expected)
        )


def test_normalizer_df() -> None:
    x = [[4.0, 1.0, 2.0, 2.0], [1.0, 3.0, 9.0, 3.0], [5.0, 7.0, 5.0, 1.0]]
    test_df = pd.DataFrame(x, columns=pd.Index(["a", "b", "c", "d"], name="feature_in"))

    non_df_normalizer = Normalizer(norm="l2")
    df_normalizer = NormalizerDF(norm="l2")

    transformed_non_df = pd.DataFrame(
        non_df_normalizer.fit_transform(X=x),
        columns=pd.Index(["a", "b", "c", "d"], name="feature_out"),
    )

    # test fit_trannsform

    transformed_df = df_normalizer.fit_transform(X=test_df)
    assert_frame_equal(transformed_df, transformed_non_df)

    # test transform

    transformed_df = df_normalizer.transform(X=test_df)
    assert_frame_equal(transformed_df, transformed_non_df)

    # test inverse transform

    with pytest.raises(
        NotImplementedError,
        match=r"^NormalizerDF does not implement method inverse_transform\(\)$",
    ):
        df_normalizer.inverse_transform(X=transformed_df)


def test_simple_imputer_df() -> None:
    x = np.array(
        [[4.0, 1.0, 2.0, np.nan], [1.0, np.nan, 9.0, 3.0], [np.nan, np.nan, 5.0, 1.0]]
    )
    x_df = pd.DataFrame(x, columns=pd.Index(["a", "b", "c", "d"], name="feature_in"))

    imputer_native = SimpleImputer(add_indicator=True)
    imputer_df = SimpleImputerDF(add_indicator=True)

    transformed_native = imputer_native.fit_transform(X=x)
    transformed_df_expected = pd.DataFrame(
        transformed_native,
        columns=pd.Index(
            ["a", "b", "c", "d", "a__missing", "b__missing", "d__missing"],
            name="feature_out",
        ),
    )

    # test fit and transform

    assert_frame_equal(
        imputer_df.fit(X=x_df).transform(X=x_df), transformed_df_expected
    )

    # test fit_trannsform

    transformed_df = imputer_df.fit_transform(X=x_df)
    assert_frame_equal(transformed_df, transformed_df_expected)

    # test inverse transform

    if __sklearn_version__ >= __sklearn_0_24__:
        inverse_transformed_df = imputer_df.inverse_transform(X=transformed_df)
        assert_frame_equal(inverse_transformed_df, x_df)


@pytest.fixture
def df_outlier() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "c0": [0, 1, 2, 3, 4],
            "c1": [-1, 0, 0, 0, 1],
            "c2": [-10, 0, 1, 2, 3],
            "c3": [0, 1, 2, 3, 10],
        }
    )


def test_outlier_remover(df_outlier: pd.DataFrame) -> None:
    outlier_remover = OutlierRemoverDF(iqr_multiple=2)
    df_transformed = outlier_remover.fit_transform(df_outlier)
    df_transformed_expected = pd.DataFrame(
        data={
            "c0": [0, 1, 2, 3, 4],
            "c1": [np.nan, 0, 0, 0, np.nan],
            "c2": [np.nan, 0, 1, 2, 3],
            "c3": [0, 1, 2, 3, np.nan],
        }
    )
    assert_frame_equal(df_transformed, df_transformed_expected)


def test_one_hot_encoding() -> None:
    test_data_categorical = pd.DataFrame(
        data=[
            ["yes", "red", "child"],
            ["yes", "blue", "father"],
            ["no", "green", "mother"],
        ],
        columns=["a", "b", "c"],
    )

    assert_frame_equal(
        OneHotEncoderDF(drop=None, sparse=False).fit_transform(test_data_categorical),
        pd.DataFrame(
            {
                "a_no": [0.0, 0.0, 1.0],
                "a_yes": [1.0, 1.0, 0.0],
                "b_blue": [0.0, 1.0, 0.0],
                "b_green": [0.0, 0.0, 1.0],
                "b_red": [1.0, 0.0, 0.0],
                "c_child": [1.0, 0.0, 0.0],
                "c_father": [0.0, 1.0, 0.0],
                "c_mother": [0.0, 0.0, 1.0],
            }
        ).rename_axis(columns="feature_out"),
    )

    if check_sklearn_version(minimum="0.23"):
        assert_frame_equal(
            OneHotEncoderDF(drop="if_binary", sparse=False).fit_transform(
                test_data_categorical
            ),
            pd.DataFrame(
                {
                    "a_yes": [1.0, 1.0, 0.0],
                    "b_blue": [0.0, 1.0, 0.0],
                    "b_green": [0.0, 0.0, 1.0],
                    "b_red": [1.0, 0.0, 0.0],
                    "c_child": [1.0, 0.0, 0.0],
                    "c_father": [0.0, 1.0, 0.0],
                    "c_mother": [0.0, 0.0, 1.0],
                }
            ).rename_axis(columns="feature_out"),
        )

    assert_frame_equal(
        OneHotEncoderDF(drop="first", sparse=False).fit_transform(
            test_data_categorical
        ),
        pd.DataFrame(
            {
                "a_yes": [1.0, 1.0, 0.0],
                "b_green": [0.0, 0.0, 1.0],
                "b_red": [1.0, 0.0, 0.0],
                "c_father": [0.0, 1.0, 0.0],
                "c_mother": [0.0, 0.0, 1.0],
            }
        ).rename_axis(columns="feature_out"),
    )
