"""
Specialised transformer wrappers.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, Optional, TypeVar, Union

import pandas as pd
from sklearn.base import TransformerMixin

from .._wrapper import _TransformerWrapperDF

log = logging.getLogger(__name__)


#
# type variables
#

T_Transformer = TypeVar("T_Transformer", bound=TransformerMixin)


#
# wrapper classes for transformers
#


class _NDArrayTransformerWrapperDF(
    _TransformerWrapperDF[T_Transformer], Generic[T_Transformer], metaclass=ABCMeta
):
    """
    ``TransformerDF`` whose delegate transformer only accepts numpy ndarrays.

    Wraps around the delegate transformer and converts the data frame to an array when
    needed.
    """

    # noinspection PyPep8Naming
    def _convert_X_for_delegate(self, X: pd.DataFrame) -> Any:
        return super()._convert_X_for_delegate(X).values

    def _convert_y_for_delegate(
        self, y: Optional[Union[pd.Series, pd.DataFrame]]
    ) -> Any:
        y = super()._convert_y_for_delegate(y)
        return None if y is None else y.values


class _ColumnSubsetTransformerWrapperDF(
    _TransformerWrapperDF[T_Transformer], Generic[T_Transformer], metaclass=ABCMeta
):
    """
    Transforms a data frame without changing column names, but possibly removing
    columns.

    All output columns of a :class:`ColumnSubsetTransformerWrapperDF` have the same
    names as their associated input columns. Some columns can be removed.
    Implementations must define ``_make_delegate_estimator`` and ``_get_features_out``.
    """

    @abstractmethod
    def _get_features_out(self) -> pd.Index:
        # return column labels for arrays returned by the fitted transformer.
        pass

    def _get_features_original(self) -> pd.Series:
        # return the series with output columns in index and output columns as values
        features_out = self._get_features_out()
        return pd.Series(index=features_out, data=features_out.values)


class _ColumnPreservingTransformerWrapperDF(
    _ColumnSubsetTransformerWrapperDF[T_Transformer],
    Generic[T_Transformer],
    metaclass=ABCMeta,
):
    """
    Transform a data frame keeping exactly the same columns.

    A ``ColumnPreservingTransformerWrapperDF`` does not add, remove, or rename any of
    the input columns.
    """

    def _get_features_out(self) -> pd.Index:
        return self.feature_names_in_


class _BaseMultipleInputsPerOutputTransformerWrapperDF(
    _TransformerWrapperDF[T_Transformer], Generic[T_Transformer], metaclass=ABCMeta
):
    """
    Transform data whom output columns have multiple input columns.
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


class _BaseDimensionalityReductionWrapperDF(
    _BaseMultipleInputsPerOutputTransformerWrapperDF[T_Transformer],
    Generic[T_Transformer],
    metaclass=ABCMeta,
):
    """
    Transform data making dimensionality reduction style transform.
    """

    @property
    @abstractmethod
    def _n_components(self) -> int:
        pass

    def _get_features_out(self) -> pd.Index:
        return pd.Index([f"x_{i}" for i in range(self._n_components)])


class _NComponentsDimensionalityReductionWrapperDF(
    _BaseDimensionalityReductionWrapperDF[T_Transformer],
    Generic[T_Transformer],
    metaclass=ABCMeta,
):
    """
    Transform features doing dimensionality reductions.

    The delegate transformer has a ``n_components`` attribute.
    """

    _ATTR_N_COMPONENTS = "n_components"

    def _validate_delegate_estimator(self) -> None:
        self._validate_delegate_attribute(attribute_name=self._ATTR_N_COMPONENTS)

    @property
    def _n_components(self) -> int:
        return getattr(self.native_estimator, self._ATTR_N_COMPONENTS)


class _ComponentsDimensionalityReductionWrapperDF(
    _BaseDimensionalityReductionWrapperDF[T_Transformer],
    Generic[T_Transformer],
    metaclass=ABCMeta,
):
    """
    Apply dimensionality reduction on a data frame.

    The delegate transformer has a ``components_`` attribute which is an array of
    shape (n_components, n_features) and we use n_components to determine the number
    of output columns.
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
    def _n_components(self) -> int:
        return len(getattr(self.native_estimator, self._ATTR_COMPONENTS))


class _FeatureSelectionWrapperDF(
    _ColumnSubsetTransformerWrapperDF[T_Transformer],
    Generic[T_Transformer],
    metaclass=ABCMeta,
):
    """
    Wrapper for feature selection transformers.

    The delegate transformer has a ``get_support`` method providing the indices of the
    selected input columns
    """

    _ATTR_GET_SUPPORT = "get_support"

    def _validate_delegate_estimator(self) -> None:
        self._validate_delegate_attribute(attribute_name=self._ATTR_GET_SUPPORT)

    def _get_features_out(self) -> pd.Index:
        get_support = getattr(self.native_estimator, self._ATTR_GET_SUPPORT)
        return self.feature_names_in_[get_support()]
