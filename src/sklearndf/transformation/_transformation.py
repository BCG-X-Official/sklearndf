"""
Core implementation of :mod:`sklearndf.transformation`
"""

#
# To create the DF class stubs:
#
# - generate a list of all child classes of TransformerMixin in PyCharm using the
#   hierarchy view (^H)
# - remove all abstract base classes and non-sklearn classes from the list
# - unindent all lines
# - use replace with regular expressions
#   Find: (\w+)\([^\)]+\) \(([\w\.]+)\)
#   Replace: @_df_transformer\nclass $1DF(TransformerDF, $1):\n    """\n    Wraps
#            :class:`$2.$1`;\n    accepts and returns data frames.\n    """
#            \n    pass\n\n
# - clean up imports; import only the module names not the individual classes

import logging
from abc import ABCMeta
from functools import reduce
from typing import Iterable, List, TypeVar

import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSSVD
from sklearn.decomposition import (
    NMF,
    PCA,
    DictionaryLearning,
    FactorAnalysis,
    FastICA,
    IncrementalPCA,
    KernelPCA,
    LatentDirichletAllocation,
    MiniBatchDictionaryLearning,
    MiniBatchSparsePCA,
    SparseCoder,
    SparsePCA,
    TruncatedSVD,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.feature_selection import (
    RFE,
    RFECV,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.kernel_approximation import (
    AdditiveChi2Sampler,
    Nystroem,
    RBFSampler,
    SkewedChi2Sampler,
)
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    KernelCenterer,
    LabelBinarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    MultiLabelBinarizer,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from pytools.api import AllTracker

from .. import TransformerDF
from .._wrapper import _TransformerWrapperDF, df_estimator
from ._wrapper import (
    _BaseDimensionalityReductionWrapperDF,
    _BaseMultipleInputsPerOutputTransformerWrapperDF,
    _ColumnPreservingTransformerWrapperDF,
    _ComponentsDimensionalityReductionWrapperDF,
    _FeatureSelectionWrapperDF,
    _NComponentsDimensionalityReductionWrapperDF,
)

log = logging.getLogger(__name__)

__all__ = [
    "AdditiveChi2SamplerDF",
    "BernoulliRBMDF",
    "BinarizerDF",
    "ColumnTransformerDF",
    "DictVectorizerDF",
    "DictionaryLearningDF",
    "FactorAnalysisDF",
    "FastICADF",
    "FeatureAgglomerationDF",
    "FeatureHasherDF",
    "FunctionTransformerDF",
    "GaussianRandomProjectionDF",
    "GenericUnivariateSelectDF",
    "HashingVectorizerDF",
    "IncrementalPCADF",
    "IsomapDF",
    "IterativeImputerDF",
    "KBinsDiscretizerDF",
    "KernelCentererDF",
    "KernelPCADF",
    "LabelBinarizerDF",
    "LabelEncoderDF",
    "LatentDirichletAllocationDF",
    "LinearDiscriminantAnalysisDF",
    "LocallyLinearEmbeddingDF",
    "MaxAbsScalerDF",
    "MinMaxScalerDF",
    "MiniBatchDictionaryLearningDF",
    "MiniBatchSparsePCADF",
    "MissingIndicatorDF",
    "MultiLabelBinarizerDF",
    "NMFDF",
    "NeighborhoodComponentsAnalysisDF",
    "NormalizerDF",
    "NystroemDF",
    "OneHotEncoderDF",
    "OrdinalEncoderDF",
    "PCADF",
    "PLSSVDDF",
    "PolynomialFeaturesDF",
    "PowerTransformerDF",
    "QuantileTransformerDF",
    "RBFSamplerDF",
    "RFECVDF",
    "RFEDF",
    "RobustScalerDF",
    "SelectFdrDF",
    "SelectFprDF",
    "SelectFromModelDF",
    "SelectFweDF",
    "SelectKBestDF",
    "SelectPercentileDF",
    "SimpleImputerDF",
    "SkewedChi2SamplerDF",
    "SparseCoderDF",
    "SparsePCADF",
    "SparseRandomProjectionDF",
    "StandardScalerDF",
    "TfidfTransformerDF",
    "TruncatedSVDDF",
    "VarianceThresholdDF",
]

__imported_estimators = {name for name in globals().keys() if name.endswith("DF")}


# T_Imputer is needed, as sklearn's _BaseImputer only exists from their v0.22 onwards:
# once support for sklearn 0.21 is dropped, _BaseImputer could be used.
# The following TypeVar helps to annotate availability of "add_indicator" and
# "missing_values" attributes on an imputer instance for _ImputerWrapperDF below
T_Imputer = TypeVar("T_Imputer", SimpleImputer, IterativeImputer)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())

#
# cluster
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class FeatureAgglomerationDF(TransformerDF, FeatureAgglomeration):
    """
    Wraps :class:`sklearn.cluster.FeatureAgglomeration`;
    accepts and returns data frames.
    """

    pass


#
# compose
#


class _ColumnTransformerWrapperDF(
    _TransformerWrapperDF[ColumnTransformer], metaclass=ABCMeta
):
    """
    Wrap :class:`sklearn.compose.ColumnTransformer` and return a DataFrame.

    Like :class:`~sklearn.compose.ColumnTransformer`, it has a ``transformers``
    parameter
    (``None`` by default) which is a list of tuple of the form (name, transformer,
    column(s)),
    but here all the transformers must be of type
    :class:`~yieldengine.df.transform.TransformerDF`.
    """

    def _validate_delegate_estimator(self) -> None:
        column_transformer: ColumnTransformer = self.native_estimator

        if column_transformer.remainder != "drop":
            raise ValueError(
                f"unsupported value for arg remainder: ({column_transformer.remainder})"
            )

        non_compliant_transformers: List[str] = [
            type(transformer).__name__
            for _, transformer, _ in column_transformer.transformers
            if not (
                isinstance(transformer, str) or isinstance(transformer, TransformerDF)
            )
        ]
        if non_compliant_transformers:
            raise ValueError(
                f"{ColumnTransformerDF.__name__} only accepts strings or "
                f"instances of "
                f"{TransformerDF.__name__} as valid transformers, but "
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
                df_transformer.feature_names_original_
                for df_transformer in self._inner_transformers()
            ),
        )

    def _inner_transformers(self) -> Iterable[_TransformerWrapperDF]:
        return (
            df_transformer
            for _, df_transformer, columns in self.native_estimator.transformers_
            if len(columns) > 0
            if df_transformer != "drop"
        )


# noinspection PyAbstractClass,DuplicatedCode
@df_estimator(df_wrapper_type=_ColumnTransformerWrapperDF)
class ColumnTransformerDF(TransformerDF, ColumnTransformer):
    """
    Wraps :class:`sklearn.compose.ColumnTransformer`;
    accepts and returns data frames.
    """

    pass


#
# cross_decomposition
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class PLSSVDDF(TransformerDF, PLSSVD):
    """
    Wraps :class:`sklearn.cross_decomposition.pls_.PLSSVD`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class FeatureHasherDF(TransformerDF, FeatureHasher):
    """
    Wraps :class:`sklearn.feature_extraction.FeatureHasher`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class DictVectorizerDF(TransformerDF, DictVectorizer):
    """
    Wraps :class:`sklearn.feature_extraction.dict_vectorizer.DictVectorizer`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class HashingVectorizerDF(TransformerDF, HashingVectorizer):
    """
    Wraps :class:`sklearn.feature_extraction.text.HashingVectorizer`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class TfidfTransformerDF(TransformerDF, TfidfTransformer):
    """
    Wraps :class:`sklearn.feature_extraction.text.TfidfTransformer`;
    accepts and returns data frames.
    """

    pass


#
# impute
#

# we cannot move this to package _wrapper as it references MissingIndicatorDF


class _ImputerWrapperDF(_TransformerWrapperDF[T_Imputer], metaclass=ABCMeta):
    """
    Impute missing values with data frames as input and output.

    Wrap around :class:`impute.SimpleImputer`. The ``fit``,
    ``transform`` and ``fit_transform`` methods accept and return dataframes.
    The parameters are the same as the one passed to
    :class:`impute.SimpleImputer`.
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
            missing_indicator = MissingIndicatorDF.from_fitted(
                estimator=delegate_estimator.indicator_,
                features_in=self.feature_names_in_,
                n_outputs=self.n_outputs_,
            )
            return features_original.append(missing_indicator.feature_names_original_)
        else:
            return features_original


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ImputerWrapperDF)
class SimpleImputerDF(TransformerDF, SimpleImputer):
    """
    Wraps :class:`sklearn.impute.MissingIndicator`;
    accepts and returns data frames.
    """

    pass


class _MissingIndicatorWrapperDF(
    _TransformerWrapperDF[MissingIndicator], metaclass=ABCMeta
):
    def _get_features_original(self) -> pd.Series:
        features_original: np.ndarray = self.feature_names_in_[
            self.native_estimator.features_
        ].values
        features_out = pd.Index([f"{name}__missing" for name in features_original])
        return pd.Series(index=features_out, data=features_original)


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_MissingIndicatorWrapperDF)
class MissingIndicatorDF(TransformerDF, MissingIndicator):
    """
    Wraps :class:`sklearn.impute.MissingIndicator`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ImputerWrapperDF)
class IterativeImputerDF(TransformerDF, IterativeImputer):
    """
    Wraps :class:`sklearn.impute.IterativeImputer`;
    accepts and returns data frames.
    """

    pass


class _IsomapWrapperDF(
    _BaseDimensionalityReductionWrapperDF[Isomap], metaclass=ABCMeta
):
    @property
    def _n_components(self) -> int:
        return self.native_estimator.embedding_.shape[1]


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_IsomapWrapperDF)
class IsomapDF(TransformerDF, IterativeImputer):
    """
    Wraps :class:`sklearn.manifold.Isomap`;
    accepts and returns data frames.
    """

    pass


class _AdditiveChi2SamplerWrapperDF(
    _BaseDimensionalityReductionWrapperDF[AdditiveChi2Sampler], metaclass=ABCMeta
):
    @property
    def _n_components(self) -> int:
        return len(self._features_in) * (2 * self.native_estimator.sample_steps + 1)


# noinspection PyAbstractClass,DuplicatedCode
@df_estimator(df_wrapper_type=_AdditiveChi2SamplerWrapperDF)
class AdditiveChi2SamplerDF(TransformerDF, AdditiveChi2Sampler):
    """
    Wraps :class:`sklearn.kernel_approximation.AdditiveChi2Sampler`;
    accepts and returns data frames.
    """

    pass


#
# neighbors
#

# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class NeighborhoodComponentsAnalysisDF(TransformerDF, NeighborhoodComponentsAnalysis):
    """
    Wraps :class:`sklearn.neighbors.NeighborhoodComponentsAnalysis`;
    accepts and returns data frames.
    """

    pass


#
# preprocessing
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class MinMaxScalerDF(TransformerDF, MinMaxScaler):
    """
    Wraps :class:`sklearn.preprocessing.MinMaxScaler`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class StandardScalerDF(TransformerDF, StandardScaler):
    """
    Wraps :class:`sklearn.preprocessing.StandardScaler`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class MaxAbsScalerDF(TransformerDF, MaxAbsScaler):
    """
    Wraps :class:`sklearn.preprocessing.MaxAbsScaler`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class RobustScalerDF(TransformerDF, RobustScaler):
    """
    Wraps :class:`sklearn.preprocessing.RobustScaler`;
    accepts and returns data frames.
    """

    pass


class _PolynomialFeaturesWrapperDF(
    _BaseMultipleInputsPerOutputTransformerWrapperDF[PolynomialFeatures],
    metaclass=ABCMeta,
):
    def _get_features_out(self) -> pd.Index:
        return pd.Index(
            data=self.native_estimator.get_feature_names(
                input_features=self.feature_names_in_.astype(str)
            )
        )


# noinspection PyAbstractClass,DuplicatedCode
@df_estimator(df_wrapper_type=_PolynomialFeaturesWrapperDF)
class PolynomialFeaturesDF(TransformerDF, PolynomialFeatures):
    """
    Wraps :class:`sklearn.preprocessing.PolynomialFeatures`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class NormalizerDF(TransformerDF, Normalizer):
    """
    Wraps :class:`sklearn.preprocessing.Normalizer`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class BinarizerDF(TransformerDF, Binarizer):
    """
    Wraps :class:`sklearn.preprocessing.Binarizer`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class KernelCentererDF(TransformerDF, KernelCenterer):
    """
    Wraps :class:`sklearn.preprocessing.KernelCenterer`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class QuantileTransformerDF(TransformerDF, QuantileTransformer):
    """
    Wraps :class:`sklearn.preprocessing.QuantileTransformer`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class PowerTransformerDF(TransformerDF, PowerTransformer):
    """
    Wraps :class:`sklearn.preprocessing.PowerTransformer`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class FunctionTransformerDF(TransformerDF, FunctionTransformer):
    """
    Wraps :class:`sklearn.preprocessing.FunctionTransformer`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class LabelEncoderDF(TransformerDF, LabelEncoder):
    """
    Wraps :class:`sklearn.preprocessing.LabelEncoder`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class LabelBinarizerDF(TransformerDF, LabelBinarizer):
    """
    Wraps :class:`sklearn.preprocessing.LabelBinarizer`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class MultiLabelBinarizerDF(TransformerDF, MultiLabelBinarizer):
    """
    Wraps :class:`sklearn.preprocessing.MultiLabelBinarizer`;
    accepts and returns data frames.
    """

    pass


class _OneHotEncoderWrapperDF(_TransformerWrapperDF[OneHotEncoder], metaclass=ABCMeta):
    """
    One-hot encoder with dataframes as input and output.

    Wrap around :class:`preprocessing.OneHotEncoder`. The ``fit``,
    ``transform`` and ``fit_transform`` methods accept and return dataframes.
    The parameters are the same as the one passed to
    :class:`preprocessing.OneHotEncoder`.
    """

    def _validate_delegate_estimator(self) -> None:
        if self.native_estimator.sparse:
            raise NotImplementedError("sparse matrices not supported; use sparse=False")

    def _get_features_original(self) -> pd.Series:
        """
        Return the series mapping output column names to original columns names.

        :return: the series with index the column names of the output dataframe and
        values the corresponding input column names.
        """
        return pd.Series(
            index=pd.Index(
                self.native_estimator.get_feature_names(self.feature_names_in_)
            ),
            data=[
                column_original
                for column_original, category in zip(
                    self.feature_names_in_, self.native_estimator.categories_
                )
                for _ in category
            ],
        )


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_OneHotEncoderWrapperDF)
class OneHotEncoderDF(TransformerDF, OneHotEncoder):
    """
    Wraps :class:`sklearn.preprocessing.OneHotEncoder`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ColumnPreservingTransformerWrapperDF)
class OrdinalEncoderDF(TransformerDF, OrdinalEncoder):
    """
    Wraps :class:`sklearn.preprocessing.OrdinalEncoder`;
    accepts and returns data frames.
    """

    pass


class _KBinsDiscretizerWrapperDF(
    _TransformerWrapperDF[KBinsDiscretizer], metaclass=ABCMeta
):
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


# noinspection PyAbstractClass,DuplicatedCode
@df_estimator(df_wrapper_type=_KBinsDiscretizerWrapperDF)
class KBinsDiscretizerDF(TransformerDF, KBinsDiscretizer):
    """
    Wrap :class:`sklearn.preprocessing.KBinsDiscretizer`;
    accepts and returns dataframes.
    """

    pass


#
# Transformers which have a components_ attribute
# Implemented through _ComponentsDimensionalityReductionWrapperDF
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class BernoulliRBMDF(TransformerDF, BernoulliRBM):
    """
    Wraps :class:`sklearn.neural_network.BernoulliRBM`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class DictionaryLearningDF(TransformerDF, DictionaryLearning):
    """
    Wraps :class:`decomposition.dict_learning.DictionaryLearning`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class FactorAnalysisDF(TransformerDF, FactorAnalysis):
    """
    Wraps :class:`decomposition.factor_analysis.FactorAnalysis`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class FastICADF(TransformerDF, FastICA):
    """
    Wraps :class:`decomposition.fastica_.FastICA`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class GaussianRandomProjectionDF(TransformerDF, GaussianRandomProjection):
    """
    Wraps :class:`sklearn.random_projection.GaussianRandomProjection`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class IncrementalPCADF(TransformerDF, IncrementalPCA):
    """
    Wraps :class:`decomposition.incremental_pca.IncrementalPCA`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class LatentDirichletAllocationDF(TransformerDF, LatentDirichletAllocation):
    """
    Wraps :class:`decomposition.online_lda.LatentDirichletAllocation`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class MiniBatchDictionaryLearningDF(TransformerDF, MiniBatchDictionaryLearning):
    """
    Wraps :class:`decomposition.dict_learning.MiniBatchDictionaryLearning`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class MiniBatchSparsePCADF(TransformerDF, MiniBatchSparsePCA):
    """
    Wraps :class:`decomposition.sparse_pca.MiniBatchSparsePCA`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class NMFDF(NMF, TransformerDF):
    """
    Wraps :class:`decomposition.NMF`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF)
class PCADF(TransformerDF, PCA):
    """
    Wraps :class:`decomposition.pca.PCA`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class SparseCoderDF(TransformerDF, SparseCoder):
    """
    Wraps :class:`decomposition.dict_learning.SparseCoder`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class SparsePCADF(TransformerDF, SparsePCA):
    """
    Wraps :class:`decomposition.sparse_pca.SparsePCA`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class SparseRandomProjectionDF(TransformerDF, SparseRandomProjection):
    """
    Wraps :class:`sklearn.random_projection.SparseRandomProjection`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF)
class TruncatedSVDDF(TransformerDF, TruncatedSVD):
    """
    Wraps :class:`decomposition.truncated_svd.TruncatedSVD`;
    accepts and returns data frames.
    """

    pass


#
# Transformer which have an n_components attribute
# Implemented through NComponentsDimensionalityReductionWrapperDF
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF)
class KernelPCADF(TransformerDF, KernelPCA):
    """
    Wraps :class:`decomposition.kernel_pca.KernelPCA`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF)
class LinearDiscriminantAnalysisDF(TransformerDF, LinearDiscriminantAnalysis):
    """
    Wraps :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF)
class LocallyLinearEmbeddingDF(TransformerDF, LocallyLinearEmbedding):
    """
    Wraps :class:`sklearn.manifold.LocallyLinearEmbedding`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF)
class NystroemDF(TransformerDF, Nystroem):
    """
    Wraps :class:`sklearn.kernel_approximation.Nystroem`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF)
class RBFSamplerDF(TransformerDF, RBFSampler):
    """
    Wraps :class:`sklearn.kernel_approximation.RBFSampler`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF)
class SkewedChi2SamplerDF(TransformerDF, SkewedChi2Sampler):
    """
    Wraps :class:`sklearn.kernel_approximation.SkewedChi2Sampler`;
    accepts and returns data frames.
    """

    pass


#
# feature_selection
#
#
# Transformer which have an get_support method
# Implemented through _FeatureSelectionWrapperDF
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_FeatureSelectionWrapperDF)
class VarianceThresholdDF(TransformerDF, VarianceThreshold):
    """
    Wraps :class:`sklearn.feature_selection.VarianceThreshold`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_FeatureSelectionWrapperDF)
class RFEDF(TransformerDF, RFE):
    """
    Wraps :class:`sklearn.feature_selection.RFE`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_FeatureSelectionWrapperDF)
class RFECVDF(TransformerDF, RFECV):
    """
    Wraps :class:`sklearn.feature_selection.RFECV`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_FeatureSelectionWrapperDF)
class SelectFromModelDF(TransformerDF, SelectFromModel):
    """
    Wraps :class:`sklearn.feature_selection.SelectFromModel`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_FeatureSelectionWrapperDF)
class SelectPercentileDF(TransformerDF, SelectPercentile):
    """
    Wraps :class:`sklearn.feature_selection.SelectPercentile`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_FeatureSelectionWrapperDF)
class SelectKBestDF(TransformerDF, SelectKBest):
    """
    Wraps :class:`sklearn.feature_selection.SelectKBest`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_FeatureSelectionWrapperDF)
class SelectFprDF(TransformerDF, SelectFpr):
    """
    Wraps :class:`sklearn.feature_selection.SelectFpr`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_FeatureSelectionWrapperDF)
class SelectFdrDF(TransformerDF, SelectFdr):
    """
    Wraps :class:`sklearn.feature_selection.SelectFdr`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_FeatureSelectionWrapperDF)
class SelectFweDF(TransformerDF, SelectFwe):
    """
    Wraps :class:`sklearn.feature_selection.SelectFwe`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_FeatureSelectionWrapperDF)
class GenericUnivariateSelectDF(TransformerDF, GenericUnivariateSelect):
    """
    Wraps :class:`sklearn.feature_selection.GenericUnivariateSelect`;
    accepts and returns data frames.
    """

    pass


__tracker.validate()


#
# validate that __all__ comprises all symbols ending in "DF", and no others
#

__estimators = {
    sym
    for sym in dir()
    if sym.endswith("DF")
    and sym not in __imported_estimators
    and not sym.startswith("_")
}

if __estimators != set(__all__):
    raise RuntimeError(
        "__all__ does not contain exactly all DF estimators; expected value is:\n"
        f"{__estimators}"
    )
