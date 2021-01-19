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
from .._wrapper import _TransformerWrapperDF, make_df_transformer
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
# Class definitions
#

#
# cluster
#


FeatureAgglomerationDF = make_df_transformer(
    FeatureAgglomeration, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)


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


ColumnTransformerDF = make_df_transformer(
    ColumnTransformer, df_wrapper_type=_ColumnTransformerWrapperDF
)


#
# cross_decomposition
#


PLSSVDDF = make_df_transformer(
    PLSSVD, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

FeatureHasherDF = make_df_transformer(
    FeatureHasher, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

DictVectorizerDF = make_df_transformer(
    DictVectorizer, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

HashingVectorizerDF = make_df_transformer(
    HashingVectorizer, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

TfidfTransformerDF = make_df_transformer(
    TfidfTransformer, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)


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


SimpleImputerDF = make_df_transformer(SimpleImputer, df_wrapper_type=_ImputerWrapperDF)


class _MissingIndicatorWrapperDF(
    _TransformerWrapperDF[MissingIndicator], metaclass=ABCMeta
):
    def _get_features_original(self) -> pd.Series:
        features_original: np.ndarray = self.feature_names_in_[
            self.native_estimator.features_
        ].values
        features_out = pd.Index([f"{name}__missing" for name in features_original])
        return pd.Series(index=features_out, data=features_original)


MissingIndicatorDF = make_df_transformer(
    MissingIndicator, df_wrapper_type=_MissingIndicatorWrapperDF
)

IterativeImputerDF = make_df_transformer(
    IterativeImputer, df_wrapper_type=_ImputerWrapperDF
)


class _IsomapWrapperDF(
    _BaseDimensionalityReductionWrapperDF[Isomap], metaclass=ABCMeta
):
    @property
    def _n_components(self) -> int:
        return self.native_estimator.embedding_.shape[1]


IsomapDF = make_df_transformer(Isomap, df_wrapper_type=_IsomapWrapperDF)


class _AdditiveChi2SamplerWrapperDF(
    _BaseDimensionalityReductionWrapperDF[AdditiveChi2Sampler], metaclass=ABCMeta
):
    @property
    def _n_components(self) -> int:
        return len(self._features_in) * (2 * self.native_estimator.sample_steps + 1)


AdditiveChi2SamplerDF = make_df_transformer(
    AdditiveChi2Sampler, df_wrapper_type=_AdditiveChi2SamplerWrapperDF
)


#
# neighbors
#

NeighborhoodComponentsAnalysisDF = make_df_transformer(
    NeighborhoodComponentsAnalysis,
    df_wrapper_type=_ColumnPreservingTransformerWrapperDF,
)


#
# preprocessing
#


MinMaxScalerDF = make_df_transformer(
    MinMaxScaler, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

StandardScalerDF = make_df_transformer(
    StandardScaler, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

MaxAbsScalerDF = make_df_transformer(
    MaxAbsScaler, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

RobustScalerDF = make_df_transformer(
    RobustScaler, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)


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


PolynomialFeaturesDF = make_df_transformer(
    PolynomialFeatures, df_wrapper_type=_PolynomialFeaturesWrapperDF
)

NormalizerDF = make_df_transformer(
    Normalizer, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

BinarizerDF = make_df_transformer(
    Binarizer, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

KernelCentererDF = make_df_transformer(
    KernelCenterer, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

QuantileTransformerDF = make_df_transformer(
    QuantileTransformer, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

PowerTransformerDF = make_df_transformer(
    PowerTransformer, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

FunctionTransformerDF = make_df_transformer(
    FunctionTransformer, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

LabelEncoderDF = make_df_transformer(
    LabelEncoder, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

LabelBinarizerDF = make_df_transformer(
    LabelBinarizer, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)

MultiLabelBinarizerDF = make_df_transformer(
    MultiLabelBinarizer, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)


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


OneHotEncoderDF = make_df_transformer(
    OneHotEncoder, df_wrapper_type=_OneHotEncoderWrapperDF
)

OrdinalEncoderDF = make_df_transformer(
    OrdinalEncoder, df_wrapper_type=_ColumnPreservingTransformerWrapperDF
)


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


KBinsDiscretizerDF = make_df_transformer(
    KBinsDiscretizer, df_wrapper_type=_KBinsDiscretizerWrapperDF
)


#
# Transformers which have a components_ attribute
# Implemented through _ComponentsDimensionalityReductionWrapperDF
#

BernoulliRBMDF = make_df_transformer(
    BernoulliRBM, df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF
)

DictionaryLearningDF = make_df_transformer(
    DictionaryLearning, df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF
)

FactorAnalysisDF = make_df_transformer(
    FactorAnalysis, df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF
)

FastICADF = make_df_transformer(
    FastICA, df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF
)

GaussianRandomProjectionDF = make_df_transformer(
    GaussianRandomProjection,
    df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF,
)

IncrementalPCADF = make_df_transformer(
    IncrementalPCA, df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF
)

LatentDirichletAllocationDF = make_df_transformer(
    LatentDirichletAllocation,
    df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF,
)

MiniBatchDictionaryLearningDF = make_df_transformer(
    MiniBatchDictionaryLearning,
    df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF,
)

MiniBatchSparsePCADF = make_df_transformer(
    MiniBatchSparsePCA, df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF
)

NMFDF = make_df_transformer(
    NMF, df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF
)

PCADF = make_df_transformer(
    PCA, df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF
)

SparseCoderDF = make_df_transformer(
    SparseCoder, df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF
)

SparsePCADF = make_df_transformer(
    SparsePCA, df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF
)

SparseRandomProjectionDF = make_df_transformer(
    SparseRandomProjection, df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF
)

TruncatedSVDDF = make_df_transformer(
    TruncatedSVD, df_wrapper_type=_ComponentsDimensionalityReductionWrapperDF
)


#
# Transformers which have an n_components attribute
# Implemented through NComponentsDimensionalityReductionWrapperDF
#

KernelPCADF = make_df_transformer(
    KernelPCA, df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF
)

LocallyLinearEmbeddingDF = make_df_transformer(
    LocallyLinearEmbedding, df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF
)

NystroemDF = make_df_transformer(
    Nystroem, df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF
)

RBFSamplerDF = make_df_transformer(
    RBFSampler, df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF
)

SkewedChi2SamplerDF = make_df_transformer(
    SkewedChi2Sampler, df_wrapper_type=_NComponentsDimensionalityReductionWrapperDF
)


#
# feature_selection
#
# Transformers with a get_support method, implemented via _FeatureSelectionWrapperDF
#

VarianceThresholdDF = make_df_transformer(
    VarianceThreshold, df_wrapper_type=_FeatureSelectionWrapperDF
)

RFEDF = make_df_transformer(RFE, df_wrapper_type=_FeatureSelectionWrapperDF)

RFECVDF = make_df_transformer(RFECV, df_wrapper_type=_FeatureSelectionWrapperDF)

SelectFromModelDF = make_df_transformer(
    SelectFromModel, df_wrapper_type=_FeatureSelectionWrapperDF
)

SelectPercentileDF = make_df_transformer(
    SelectPercentile, df_wrapper_type=_FeatureSelectionWrapperDF
)

SelectKBestDF = make_df_transformer(
    SelectKBest, df_wrapper_type=_FeatureSelectionWrapperDF
)

SelectFprDF = make_df_transformer(SelectFpr, df_wrapper_type=_FeatureSelectionWrapperDF)

SelectFdrDF = make_df_transformer(SelectFdr, df_wrapper_type=_FeatureSelectionWrapperDF)

SelectFweDF = make_df_transformer(SelectFwe, df_wrapper_type=_FeatureSelectionWrapperDF)

GenericUnivariateSelectDF = make_df_transformer(
    GenericUnivariateSelect, df_wrapper_type=_FeatureSelectionWrapperDF
)


#
# validate __all__
#

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
