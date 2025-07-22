"""
Core implementation of :mod:`sklearndf.transformation.wrapper`
"""

import itertools
import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Generic, Optional, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.manifold import Isomap
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder

from pytools.api import AllTracker

from ... import TransformerDF
from ..._util import hstack_frames, is_sparse_frame, sparse_frame_density
from ...wrapper import TransformerWrapperDF

log = logging.getLogger(__name__)

__all__ = [
    "AdditiveChi2SamplerWrapperDF",
    "BaseMultiOutputWrapperDF",
    "BaseMultipleInputsPerOutputTransformerWrapperDF",
    "ColumnPreservingTransformerWrapperDF",
    "ColumnSubsetTransformerWrapperDF",
    "ColumnTransformerSparseFrames",
    "ColumnTransformerWrapperDF",
    "ComponentsDimensionalityReductionWrapperDF",
    "EmbeddingWrapperDF",
    "FeatureSelectionWrapperDF",
    "ImputerWrapperDF",
    "IsomapWrapperDF",
    "KBinsDiscretizerWrapperDF",
    "MissingIndicatorWrapperDF",
    "NComponentsDimensionalityReductionWrapperDF",
    "NumpyTransformerWrapperDF",
    "OneHotEncoderWrapperDF",
    "PCAWrapperDF",
    "PolynomialTransformerWrapperDF",
    "SingleColumnTransformerWrapperDF",
    "VectorizerWrapperDF",
]


#
# Type variables
#


# noinspection PyProtectedMember
from sklearn.impute._iterative import IterativeImputer

# Once we drop support for sklearn 0.21, T_Imputer can be bound to _BaseImputer
T_Imputer = TypeVar("T_Imputer", SimpleImputer, IterativeImputer)
T_Polynomial = TypeVar("T_Polynomial", bound=TransformerMixin)
T_Target = TypeVar("T_Target", bound=Union[pd.Series, pd.DataFrame, None])
T_Transformer = TypeVar("T_Transformer", bound=TransformerMixin)
T_Vectorizer = TypeVar("T_Vectorizer", bound=TransformerMixin)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# wrapper classes for transformers
#


class NumpyTransformerWrapperDF(
    TransformerWrapperDF[T_Transformer], Generic[T_Transformer], metaclass=ABCMeta
):
    """
    Abstract base class of DF wrappers for transformers that only accept numpy arrays.

    Converts data frames to numpy arrays before handing off to the native transformer.

    Implementations must define :meth:`_get_features_original`.
    """

    # noinspection PyPep8Naming
    def _adjust_X_type_for_delegate(
        self,
        X: pd.DataFrame,
    ) -> Union[npt.NDArray[Any], sparse.csr_matrix]:
        return self._frame_or_series_to_array(X)

    def _adjust_y_type_for_delegate(
        self, y: Union[pd.Series, pd.DataFrame, None]
    ) -> Union[npt.NDArray[Any], pd.arrays.SparseArray, sparse.csr_matrix, None]:
        return None if y is None else self._frame_or_series_to_array(y)

    def _frame_or_series_to_array(
        self, df: Union[pd.Series, pd.DataFrame]
    ) -> Union[npt.NDArray[Any], pd.arrays.SparseArray, sparse.csr_matrix]:
        if df.ndim == 1:
            return df.values

        sparse_threshold = self._get_sparse_threshold()

        if sparse_threshold > 0.0 and sparse_frame_density(df) < sparse_threshold:
            return sparse.hstack(
                [
                    (
                        sr.to_frame().sparse.to_coo()
                        if isinstance(sr.dtype, pd.SparseDtype)
                        else sr.values.reshape(-1, 1)
                    )
                    for _, sr in df.items()
                ]
            ).tocsr()

        else:
            return df.values

    def _get_sparse_threshold(self) -> float:
        # return the density below which _frame_or_series_to_array will
        # convert series and data frames to sparse matrices instead of
        # dense arrays
        return 0.3


class SingleColumnTransformerWrapperDF(
    TransformerWrapperDF[T_Transformer], Generic[T_Transformer], metaclass=ABCMeta
):
    """
    Abstract base class of DF wrappers for transformers that only accept single-column
    inputs.
    """

    # noinspection PyPep8Naming
    def _validate_parameter_types(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: T_Target,
        *,
        expected_columns: pd.Index = None,
    ) -> tuple[pd.DataFrame, T_Target]:
        X, y = super()._validate_parameter_types(
            X, y, expected_columns=expected_columns
        )
        if X.shape[1] != 1:
            raise ValueError(
                f"arg X expected to have exactly 1 column but has {X.shape[1]} columns"
            )
        return X, y


class ColumnSubsetTransformerWrapperDF(
    TransformerWrapperDF[T_Transformer], Generic[T_Transformer], metaclass=ABCMeta
):
    """
    Abstract base class of DF wrappers for transformers that do not change column names,
    but that may remove one or more columns.

    Implementations must define :meth:`_get_features_out`.
    """

    @abstractmethod
    def _get_features_out(self) -> pd.Index:
        # return column labels for arrays returned by the fitted transformer.
        pass

    def _get_features_original(self) -> pd.Series:
        # return the series with output columns in index and output columns as values
        features_out = self._get_features_out()
        return pd.Series(index=features_out, data=features_out.values)


class ColumnPreservingTransformerWrapperDF(
    ColumnSubsetTransformerWrapperDF[T_Transformer],
    Generic[T_Transformer],
):
    """
    DF wrapper for transformers whose output columns match the input columns.

    The native transformer must not add, remove, reorder, or rename any of the input
    columns.
    """

    def _get_features_out(self) -> pd.Index:
        return self.feature_names_in_


class BaseMultipleInputsPerOutputTransformerWrapperDF(
    TransformerWrapperDF[T_Transformer], Generic[T_Transformer]
):
    """
    DF wrapper for transformers mapping multiple input columns to individual output
    columns.
    """

    @abstractmethod
    def _get_features_out(self) -> pd.Index:
        # make this method abstract to ensure subclasses override the default
        # behaviour, which usually relies on method ``_get_features_original``
        pass

    def _get_features_original(self) -> pd.Series:
        raise TypeError(
            f"{type(self.native_estimator).__name__} transformers map multiple "
            "inputs to individual output columns; current sklearndf implementation "
            "only supports many-to-1 mappings from output columns to input columns"
        )


class BaseMultiOutputWrapperDF(
    BaseMultipleInputsPerOutputTransformerWrapperDF[T_Transformer],
    Generic[T_Transformer],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for dimensionality-reducing transformers.

    The native transformer is considered to map all input columns to each output column.
    """

    @property
    @abstractmethod
    def _n_components_(self) -> int:
        pass

    def _get_features_out(self) -> pd.Index:
        return pd.Index([f"x_{i}" for i in range(self._n_components_)])


class NComponentsDimensionalityReductionWrapperDF(
    BaseMultiOutputWrapperDF[T_Transformer],
    Generic[T_Transformer],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for dimensionality-reducing transformers supporting the
    :attr:`n_components` attribute.

    Subclasses must implement :meth:`_get_features_original`.
    """

    _ATTR_N_COMPONENTS = "n_components"

    def _validate_delegate_estimator(self) -> None:
        self._validate_delegate_attribute(
            attribute_name=(
                NComponentsDimensionalityReductionWrapperDF._ATTR_N_COMPONENTS
            )
        )

    @property
    def _n_components_(self) -> int:
        return cast(
            int,
            getattr(
                self.native_estimator,
                NComponentsDimensionalityReductionWrapperDF._ATTR_N_COMPONENTS,
            ),
        )


class PCAWrapperDF(NComponentsDimensionalityReductionWrapperDF[PCA]):
    """
    DF wrapper for :class:`sklearn.decomposition.PCA`.

    Uses attribute :attr:`~sklearn.decomposition.PCA.n_components_` to get the number
    of components, which may be determined by the estimator during fitting.
    """

    @property
    def _n_components_(self) -> int:
        return cast(int, self._native_estimator.n_components_)


class ComponentsDimensionalityReductionWrapperDF(
    BaseMultiOutputWrapperDF[T_Transformer],
    Generic[T_Transformer],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for dimensionality-reducing transformers supporting the
    ``components_`` attribute.

    The native transformer must provide a ``components_`` attribute once fitted,
    as an array of shape (n_components, n_features).
    """

    _ATTR_COMPONENTS = "components_"

    # noinspection PyPep8Naming
    def _post_fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params: Any
    ) -> None:
        # noinspection PyProtectedMember
        super()._post_fit(X, y, **fit_params)
        self._validate_delegate_attribute(attribute_name=self._ATTR_COMPONENTS)

    @property
    def _n_components_(self) -> int:
        return len(getattr(self.native_estimator, self._ATTR_COMPONENTS))


class FeatureSelectionWrapperDF(
    ColumnSubsetTransformerWrapperDF[T_Transformer],
    Generic[T_Transformer],
    metaclass=ABCMeta,
):
    """
    DF wrapper for feature selection transformers.

    The native transformer must implement a ``get_support`` method, providing the
    indices of the selected input columns
    """

    _ATTR_GET_SUPPORT = "get_support"

    def _validate_delegate_estimator(self) -> None:
        self._validate_delegate_attribute(attribute_name=self._ATTR_GET_SUPPORT)

    def _get_features_out(self) -> pd.Index:
        get_support = getattr(self.native_estimator, self._ATTR_GET_SUPPORT)
        return self.feature_names_in_[get_support()]

    # noinspection PyPep8Naming


class ColumnTransformerSparseFrames(
    ColumnTransformer,  # type:ignore
):
    """
    ColumnTransformer that returns sparse data frames instead of arrays if one or more
    of its transformers return a sparse data frame.
    """

    # noinspection PyPep8Naming
    def _hstack(
        self,
        Xs: list[Union[npt.NDArray[Any], sparse.spmatrix, pd.DataFrame]],
        *,
        n_samples: Optional[int] = None,
    ) -> Union[npt.NDArray[Any], sparse.spmatrix, pd.DataFrame]:
        if self.verbose_feature_names_out:
            # Get the prefixes for the columns of each transformer, unless the
            # transformer does not process any columns
            prefixes = [name for name, _, cols in self.transformers if len(cols)]
            if self._remainder[2] and self.remainder != "drop":
                # remainder columns exist and are not being dropped
                prefixes.append("remainder")
            stacked = hstack_frames(Xs, prefixes=prefixes)
        else:
            stacked = hstack_frames(Xs)

        if stacked is None:
            if n_samples is None:
                # Do not pass n_samples if we did not receive it, for backwards
                # compatibility
                return super()._hstack(Xs)
            else:
                # Only pass n_samples if we receive it
                return super()._hstack(Xs, n_samples=n_samples)
        else:
            self.sparse_output_ = is_sparse_frame(stacked)
            return stacked


class ColumnTransformerWrapperDF(
    TransformerWrapperDF[ColumnTransformer], metaclass=ABCMeta
):
    """
    DF wrapper for :class:`sklearn.compose.ColumnTransformer`.

    Requires all transformers passed as the ``transformers`` parameter to implement
    :class:`.TransformerDF`.
    """

    #: Special transformer argument for use with parameters ``transformers`` and
    #: ``remainder``.
    DROP = "drop"

    #: Special transformer argument for use with parameters ``transformers`` and
    #: ``remainder``.
    PASSTHROUGH = "passthrough"

    __SPECIAL_TRANSFORMERS = (DROP, PASSTHROUGH)

    def _validate_delegate_estimator(self) -> None:
        column_transformer: ColumnTransformer = self.native_estimator

        if (
            column_transformer.remainder
            not in ColumnTransformerWrapperDF.__SPECIAL_TRANSFORMERS
        ):
            raise ValueError(
                f"unsupported value for arg remainder: {column_transformer.remainder!r}"
            )

        non_compliant_transformers: list[str] = [
            type(transformer).__name__
            for _, transformer, _ in column_transformer.transformers
            if not (
                isinstance(transformer, TransformerDF)
                or transformer in ColumnTransformerWrapperDF.__SPECIAL_TRANSFORMERS
            )
        ]
        if non_compliant_transformers:
            from .. import ColumnTransformerDF

            raise ValueError(
                f"{ColumnTransformerDF.__name__} only accepts instances of "
                f"{TransformerDF.__name__} or special values "
                f'"{" and ".join(ColumnTransformerWrapperDF.__SPECIAL_TRANSFORMERS)}" '
                "as valid transformers, but "
                f'also got: {", ".join(non_compliant_transformers)}'
            )

    def _get_features_original(self) -> pd.Series:
        """
        Return the series mapping output column names to original columns names.

        :return: the series with index the column names of the output dataframe and
        values the corresponding input column names.
        """

        # If True, get_feature_names_out will prefix all feature names with the name of
        # the transformer that generated that feature
        verbose_feature_names_out: bool = getattr(
            self.native_estimator, "verbose_feature_names_out", False
        )

        # noinspection PyShadowingNames
        def _features_original(
            transformer_name: str,
            df_transformer: Union[TransformerDF, str],
            columns: Iterable[Any],
        ) -> pd.Series:
            input_column_names: npt.NDArray[Any]
            output_column_names: npt.NDArray[Any]

            if df_transformer == ColumnTransformerWrapperDF.PASSTHROUGH or (
                # scikit-learn 1.4+ replaces 'passthrough' with a FunctionTransformer
                # using the identity function (represented by `None`), so we need to
                # check for that.
                isinstance(df_transformer, FunctionTransformer)
                and df_transformer.func is None
            ):
                # we may get positional indices for columns selected by the
                # 'passthrough' transformer, and in that case need to look up the
                # associated column names
                if all(isinstance(column, int) for column in columns):
                    output_column_names = self._get_features_in().to_numpy()[columns]
                else:
                    output_column_names = np.array(columns)
                input_column_names = output_column_names

            else:
                assert isinstance(df_transformer, TransformerDF), (
                    "expected TransformerDF but got a "
                    f"{type(df_transformer).__name__}: {df_transformer!r}"
                )
                feature_names_original_: pd.Series = (
                    df_transformer._get_features_original()
                )
                if verbose_feature_names_out:
                    input_column_names = feature_names_original_.to_numpy()
                    output_column_names = feature_names_original_.index.to_numpy()
                else:
                    return feature_names_original_

            if verbose_feature_names_out:
                output_column_names = np.array(
                    [f"{transformer_name}__{column}" for column in output_column_names]
                )

            return pd.Series(index=output_column_names, data=input_column_names)

        transformer_name: str
        df_transformer: Union[TransformerDF, str]
        columns: Union[Sequence[Any], npt.NDArray[Any]]
        return pd.concat(
            [
                _features_original(transformer_name, df_transformer, columns)
                for transformer_name, df_transformer, columns in (
                    self.native_estimator.transformers_
                )
                if (
                    len(columns) > 0
                    and df_transformer != ColumnTransformerWrapperDF.DROP
                )
            ]
        )


class ImputerWrapperDF(
    TransformerWrapperDF[T_Imputer], Generic[T_Imputer], metaclass=ABCMeta
):
    """
    DF wrapper for imputation transformers, e.g., :class:`sklearn.impute.SimpleImputer`.
    """

    def _get_features_original(self) -> pd.Series:
        # get the columns that were dropped during imputation
        delegate_estimator = self.native_estimator

        nan_mask: Union[list[bool], npt.NDArray[Any]] = []

        def _nan_mask_from_statistics(
            stats: npt.NDArray[Any],
        ) -> Union[list[bool], npt.NDArray[np.bool_]]:
            if issubclass(stats.dtype.type, float):
                return cast(npt.NDArray[np.bool_], np.isnan(stats))
            else:
                return [
                    x is None or (isinstance(x, float) and np.isnan(x)) for x in stats
                ]

        # implementation for SimpleImputer
        if hasattr(delegate_estimator, "statistics_"):
            nan_mask = _nan_mask_from_statistics(stats=delegate_estimator.statistics_)

        # implementation for IterativeImputer
        elif hasattr(delegate_estimator, "initial_imputer_"):
            initial_imputer: SimpleImputer = delegate_estimator.initial_imputer_
            nan_mask = _nan_mask_from_statistics(stats=initial_imputer.statistics_)

        # implementation for KNNImputer
        elif hasattr(delegate_estimator, "_mask_fit_X"):
            # noinspection PyProtectedMember
            nan_mask = np.all(delegate_estimator._mask_fit_X, axis=0)

        # the imputed columns are all ingoing columns, except the ones that were dropped
        imputed_columns = self.feature_names_in_.delete(np.argwhere(nan_mask).tolist())
        features_original = pd.Series(
            index=imputed_columns, data=imputed_columns.values
        )

        # if the add_indicator flag is set, we will get additional "missing" columns
        if delegate_estimator.add_indicator:
            from .. import MissingIndicatorDF

            missing_indicator = MissingIndicatorDF.from_fitted(
                estimator=delegate_estimator.indicator_,
                features_in=self.feature_names_in_,
                n_outputs=self.n_outputs_,
            )
            return pd.concat(
                [features_original, missing_indicator.feature_names_original_]
            )
        else:
            return features_original


class MissingIndicatorWrapperDF(
    TransformerWrapperDF[MissingIndicator], metaclass=ABCMeta
):
    """
    DF wrapper for :class:`sklearn.impute.MissingIndicator`.
    """

    def _get_features_original(self) -> pd.Series:
        features_original: npt.NDArray[Any] = self.feature_names_in_[
            self.native_estimator.features_
        ].values
        # noinspection SpellCheckingInspection
        features_out = pd.Index(
            [f"missingindicator_{name}" for name in features_original]
        )
        return pd.Series(index=features_out, data=features_original)


class IsomapWrapperDF(BaseMultiOutputWrapperDF[Isomap], metaclass=ABCMeta):
    """
    DF wrapper for :class:`sklearn.manifold.Isomap`.
    """

    @property
    def _n_components_(self) -> int:
        return cast(int, self.native_estimator.embedding_.shape[1])


class AdditiveChi2SamplerWrapperDF(
    BaseMultiOutputWrapperDF[AdditiveChi2Sampler], metaclass=ABCMeta
):
    """
    DF wrapper for :class:`sklearn.kernel_approximation.AdditiveChi2Sampler`.
    """

    @property
    def _n_components_(self) -> int:
        assert self._features_in is not None, "estimator is fitted"
        return len(self._features_in) * (
            2 * cast(int, self.native_estimator.sample_steps) + 1
        )


class PolynomialTransformerWrapperDF(
    BaseMultipleInputsPerOutputTransformerWrapperDF[T_Polynomial],
    Generic[T_Polynomial],
    metaclass=ABCMeta,
):
    """
    DF wrapper for :class:`sklearn.preprocessing.PolynomialFeatures`
    and :class:`sklearn.preprocessing.SplineTransformer`.
    """

    def _get_features_out(self) -> pd.Index:
        return pd.Index(
            self.native_estimator.get_feature_names_out(
                self.feature_names_in_.to_numpy().astype(str)
            )
        )


class OneHotEncoderWrapperDF(TransformerWrapperDF[OneHotEncoder], metaclass=ABCMeta):
    """
    DF wrapper for :class:`sklearn.preprocessing.OneHotEncoder`.
    """

    def _get_features_original(self) -> pd.Series:
        # Return the series mapping output column names to original column names.

        native_estimator: OneHotEncoder = self.native_estimator
        feature_names_in: pd.Index = self.feature_names_in_
        feature_names_out: pd.Index = pd.Index(
            native_estimator.get_feature_names_out(
                feature_names_in.to_numpy().astype(str)
            )
        )

        def _adjust_n_features_in(n: npt.NDArray[np.int_]) -> None:
            drop = self.drop
            if drop is None:
                return
            elif isinstance(drop, str):
                if drop == "first":
                    # drop one category for all feature
                    n -= 1
                    return
                elif drop == "if_binary":
                    # drop one category only for binary features
                    n[n == 2] = 1
                    return
            elif isinstance(drop, (Sequence, np.ndarray)):
                # drop is an array-like
                n -= 1
                return

            raise ValueError(f"unexpected value for arg drop: {drop!r}")

        n_features_in: npt.NDArray[np.int_] = np.array(
            [len(categories) for categories in native_estimator.categories_],
            dtype=np.int_,
        )

        if self.max_categories is not None or self.min_frequency is not None:
            # count number of infrequent categories per column
            n_infrequent = np.array(
                [
                    (
                        1
                        if infrequent_categories_for_column is None
                        else len(infrequent_categories_for_column)
                    )
                    for infrequent_categories_for_column in (
                        self.native_estimator.infrequent_categories_
                    )
                ],
                dtype=np.int_,
            )
            # all infrequent categories will be aggregated, so deduct number of
            # infrequent categories, except one for the aggregated category
            n_features_in -= n_infrequent - 1

        _adjust_n_features_in(n_features_in)

        feature_names_in_mapped = itertools.chain(
            *([feature] * n for feature, n in zip(feature_names_in, n_features_in))
        )

        return pd.Series(index=feature_names_out, data=feature_names_in_mapped)


class KBinsDiscretizerWrapperDF(
    TransformerWrapperDF[KBinsDiscretizer], metaclass=ABCMeta
):
    """
    DF wrapper for :class:`sklearn.preprocessing.KBinsDiscretizer`.
    """

    def _validate_delegate_estimator(self) -> None:
        if self.native_estimator.encode == "onehot":
            raise NotImplementedError(
                'property encode="onehot" is not supported due to sparse matrices;'
                'consider using "onehot-dense" instead'
            )

    def _get_features_original(self) -> pd.Series:
        """
        Return the series mapping output column names to original columns names.

        :return: the series with index the column names of the output dataframe and
        values the corresponding input column names.
        """
        if self.native_estimator.encode == "onehot-dense":
            n_bins_per_feature = self.native_estimator.n_bins_
            features_in, features_out = zip(
                *(
                    (feature_name, f"{feature_name}_bin_{bin_index}")
                    for feature_name, n_bins in zip(
                        self.feature_names_in_, n_bins_per_feature
                    )
                    for bin_index in range(n_bins)
                )
            )

            return pd.Series(index=features_out, data=features_in)

        elif self.native_estimator.encode == "ordinal":
            return pd.Series(
                index=self.feature_names_in_.astype(str) + "_bin",
                data=self.feature_names_in_,
            )
        else:
            raise ValueError(
                f"unexpected value for property encode={self.native_estimator.encode}"
            )


class EmbeddingWrapperDF(
    BaseMultiOutputWrapperDF[T_Transformer],
    Generic[T_Transformer],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for dimensionality-reducing transformers.

    The native transformer is considered to map all input columns to each output column.
    """

    def _get_n_outputs(self) -> int:
        return cast(int, self.native_estimator.n_outputs_)

    @property
    def _n_components_(self) -> int:
        return self.n_outputs_


#
# Helper functions
#


class VectorizerWrapperDF(
    SingleColumnTransformerWrapperDF[T_Transformer], Generic[T_Transformer]
):
    """
    DF wrapper for vectorizers, specifically
    :class:`~sklearn.feature_extraction.text.CountVectorizer` and
    :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    """

    def _get_features_original(self) -> pd.Series:
        return pd.Series(
            index=self._get_features_out(), data=self._get_features_in()[0]
        )

    def _get_features_out(self) -> pd.Index:
        try:
            feature_names = self.native_estimator.get_feature_names_out()
        except AttributeError:
            try:
                n_features = self.native_estimator.n_features
            except AttributeError:
                raise TypeError(
                    f"native vectorizer {type(self.native_estimator).__name__} "
                    "has no method get_feature_names_out() or attribute n_features"
                )
            else:
                return pd.RangeIndex(n_features)
        else:
            return pd.Index(feature_names)

    # noinspection PyPep8Naming
    def _adjust_X_type_for_delegate(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        assert len(X.columns) == 1
        return cast(npt.NDArray[Any], X.iloc[:, 0].values)


#
# validate __all__
#

__tracker.validate()
