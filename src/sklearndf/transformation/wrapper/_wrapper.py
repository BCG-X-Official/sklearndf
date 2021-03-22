"""
Core implementation of :mod:`sklearndf.transformation.wrapper`
"""

import logging
from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import Any, Generic, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.manifold import Isomap
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, PolynomialFeatures

from pytools.api import AllTracker

from ... import TransformerDF
from ...wrapper import TransformerWrapperDF

log = logging.getLogger(__name__)

__all__ = [
    "BaseDimensionalityReductionWrapperDF",
    "BaseMultipleInputsPerOutputTransformerWrapperDF",
    "ColumnPreservingTransformerWrapperDF",
    "ColumnSubsetTransformerWrapperDF",
    "ComponentsDimensionalityReductionWrapperDF",
    "FeatureSelectionWrapperDF",
    "NComponentsDimensionalityReductionWrapperDF",
    "NumpyTransformerWrapperDF",
    "ColumnTransformerWrapperDF",
    "IsomapWrapperDF",
    "ImputerWrapperDF",
    "MissingIndicatorWrapperDF",
    "AdditiveChi2SamplerWrapperDF",
    "KBinsDiscretizerWrapperDF",
    "PolynomialFeaturesWrapperDF",
    "OneHotEncoderWrapperDF",
]


#
# type variables
#

T_Transformer = TypeVar("T_Transformer", bound=TransformerMixin)

# T_Imputer is needed because sklearn's _BaseImputer only exists from v0.22 onwards.
# Once we drop support for sklearn 0.21, _BaseImputer can be used instead.
# The following TypeVar helps to annotate availability of "add_indicator" and
# "missing_values" attributes on an imputer instance for ImputerWrapperDF below
T_Imputer = TypeVar("T_Imputer", SimpleImputer, IterativeImputer)


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
    def _convert_X_for_delegate(self, X: pd.DataFrame) -> Any:
        return super()._convert_X_for_delegate(X).values

    def _convert_y_for_delegate(
        self, y: Optional[Union[pd.Series, pd.DataFrame]]
    ) -> Any:
        y = super()._convert_y_for_delegate(y)
        return None if y is None else y.values


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
        raise NotImplementedError(
            f"{type(self.native_estimator).__name__} transformers map multiple "
            "inputs to individual output columns; current sklearndf implementation "
            "only supports many-to-1 mappings from output columns to input columns"
        )


class BaseDimensionalityReductionWrapperDF(
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
    BaseDimensionalityReductionWrapperDF[T_Transformer],
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
        self._validate_delegate_attribute(attribute_name=self._ATTR_N_COMPONENTS)

    @property
    def _n_components_(self) -> int:
        return getattr(self.native_estimator, self._ATTR_N_COMPONENTS)


class ComponentsDimensionalityReductionWrapperDF(
    BaseDimensionalityReductionWrapperDF[T_Transformer],
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
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
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


class ColumnTransformerWrapperDF(
    TransformerWrapperDF[ColumnTransformer], metaclass=ABCMeta
):
    """
    DF wrapper for :class:`sklearn.compose.ColumnTransformer`.

    Requires all transformers passed as the ``transformers`` parameter to implement
    :class:`.TransformerDF`.
    """

    __DROP = "drop"
    __PASSTHROUGH = "passthrough"

    __SPECIAL_TRANSFORMERS = (__DROP, __PASSTHROUGH)

    def _validate_delegate_estimator(self) -> None:
        column_transformer: ColumnTransformer = self.native_estimator

        if column_transformer.remainder != ColumnTransformerWrapperDF.__DROP:
            raise ValueError(
                f"unsupported value for arg remainder: ({column_transformer.remainder})"
            )

        non_compliant_transformers: List[str] = [
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

        return reduce(
            lambda x, y: x.append(y),
            (
                (
                    pd.Series(index=columns, data=columns)
                    if df_transformer == ColumnTransformerWrapperDF.__PASSTHROUGH
                    else df_transformer.feature_names_original_
                )
                for _, df_transformer, columns in self.native_estimator.transformers_
                if (
                    len(columns) > 0
                    and df_transformer != ColumnTransformerWrapperDF.__DROP
                )
            ),
        )


class ImputerWrapperDF(TransformerWrapperDF[T_Imputer], metaclass=ABCMeta):
    """
    DF wrapper for imputation transformers, e.g., :class:`sklearn.impute.SimpleImputer`.
    """

    def _get_features_original(self) -> pd.Series:
        # get the columns that were dropped during imputation
        delegate_estimator = self.native_estimator

        nan_mask = []

        def _nan_mask_from_statistics(stats: np.array):
            if issubclass(stats.dtype.type, float):
                na_mask = np.isnan(stats)
            else:
                na_mask = [
                    x is None or (isinstance(x, float) and np.isnan(x)) for x in stats
                ]
            return na_mask

        # implementation for i.e. SimpleImputer
        if hasattr(delegate_estimator, "statistics_"):
            nan_mask = _nan_mask_from_statistics(stats=delegate_estimator.statistics_)

        # implementation for IterativeImputer
        elif hasattr(delegate_estimator, "initial_imputer_"):
            initial_imputer: SimpleImputer = delegate_estimator.initial_imputer_
            nan_mask = _nan_mask_from_statistics(stats=initial_imputer.statistics_)

        # implementation for i.e. KNNImputer
        elif hasattr(delegate_estimator, "_mask_fit_X"):
            # noinspection PyProtectedMember
            nan_mask = np.all(delegate_estimator._mask_fit_X, axis=0)

        # the imputed columns are all ingoing columns, except the ones that were dropped
        imputed_columns = self.feature_names_in_.delete(np.argwhere(nan_mask))
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
            return features_original.append(missing_indicator.feature_names_original_)
        else:
            return features_original


class MissingIndicatorWrapperDF(
    TransformerWrapperDF[MissingIndicator], metaclass=ABCMeta
):
    """
    DF wrapper for :class:`sklearn.impute.MissingIndicator`.
    """

    def _get_features_original(self) -> pd.Series:
        features_original: np.ndarray = self.feature_names_in_[
            self.native_estimator.features_
        ].values
        features_out = pd.Index([f"{name}__missing" for name in features_original])
        return pd.Series(index=features_out, data=features_original)


class IsomapWrapperDF(BaseDimensionalityReductionWrapperDF[Isomap], metaclass=ABCMeta):
    """
    DF wrapper for :class:`sklearn.manifold.Isomap`.
    """

    @property
    def _n_components_(self) -> int:
        return self.native_estimator.embedding_.shape[1]


class AdditiveChi2SamplerWrapperDF(
    BaseDimensionalityReductionWrapperDF[AdditiveChi2Sampler], metaclass=ABCMeta
):
    """
    DF wrapper for :class:`sklearn.kernel_approximation.AdditiveChi2Sampler`.
    """

    @property
    def _n_components_(self) -> int:
        return len(self._features_in) * (2 * self.native_estimator.sample_steps + 1)


class PolynomialFeaturesWrapperDF(
    BaseMultipleInputsPerOutputTransformerWrapperDF[PolynomialFeatures],
    metaclass=ABCMeta,
):
    """
    DF wrapper for :class:`sklearn.preprocessing.PolynomialFeatures`.
    """

    def _get_features_out(self) -> pd.Index:
        return pd.Index(
            data=self.native_estimator.get_feature_names(
                input_features=self.feature_names_in_.astype(str)
            )
        )


class OneHotEncoderWrapperDF(TransformerWrapperDF[OneHotEncoder], metaclass=ABCMeta):
    """
    DF wrapper for :class:`sklearn.preprocessing.OneHotEncoder`.
    """

    def _validate_delegate_estimator(self) -> None:
        if self.native_estimator.sparse:
            raise NotImplementedError("sparse matrices not supported; use sparse=False")

    def _get_features_original(self) -> pd.Series:
        # Return the series mapping output column names to original column names.
        #
        # Remove 1st category column if argument drop == 'first'
        # Remove 1st category column only of binary features if arg drop == 'if_binary'

        feature_names_out = pd.Index(
            self.native_estimator.get_feature_names(self.feature_names_in_)
        )

        if self.drop == "first":
            feature_names_in = [
                column_original
                for column_original, category in zip(
                    self.feature_names_in_, self.native_estimator.categories_
                )
                for _ in range(len(category) - 1)
            ]
        elif self.drop == "if_binary":
            feature_names_in = [
                column_original
                for column_original, category in zip(
                    self.feature_names_in_, self.native_estimator.categories_
                )
                for _ in (range(1) if len(category) == 2 else category)
            ]
        else:
            feature_names_in = [
                column_original
                for column_original, category in zip(
                    self.feature_names_in_, self.native_estimator.categories_
                )
                for _ in category
            ]

        return pd.Series(index=feature_names_out, data=feature_names_in)


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


#
# validate __all__
#

__tracker.validate()
