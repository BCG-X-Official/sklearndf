"""
Core implementation of :mod:`sklearndf.transformation`
"""

import logging
from typing import Type, TypeVar

from sklearn.base import TransformerMixin
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

from ..wrapper import make_df_transformer
from .wrapper import (
    AdditiveChi2SamplerWrapperDF,
    ColumnPreservingTransformerWrapperDF,
    ColumnTransformerWrapperDF,
    ComponentsDimensionalityReductionWrapperDF,
    FeatureSelectionWrapperDF,
    ImputerWrapperDF,
    IsomapWrapperDF,
    KBinsDiscretizerWrapperDF,
    MissingIndicatorWrapperDF,
    NComponentsDimensionalityReductionWrapperDF,
    OneHotEncoderWrapperDF,
    PolynomialFeaturesWrapperDF,
)
from sklearndf.wrapper import TransformerWrapperDF

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


#
# Type constructors
#

T_NativeTransformer = TypeVar("T_NativeTransformer", bound=TransformerMixin)
T_Wrapper = Type[TransformerWrapperDF[T_NativeTransformer]]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals(), allow_imported_definitions=True)


#
# Class definitions
#

#
# cluster
#


FeatureAgglomerationDF: T_Wrapper[FeatureAgglomeration] = make_df_transformer(
    FeatureAgglomeration, base_wrapper=ColumnPreservingTransformerWrapperDF
)


#
# compose
#


ColumnTransformerDF: T_Wrapper[ColumnTransformer] = make_df_transformer(
    ColumnTransformer, base_wrapper=ColumnTransformerWrapperDF
)


#
# cross_decomposition
#


PLSSVDDF: T_Wrapper[PLSSVD] = make_df_transformer(
    PLSSVD, base_wrapper=ColumnPreservingTransformerWrapperDF
)

FeatureHasherDF: T_Wrapper[FeatureHasher] = make_df_transformer(
    FeatureHasher, base_wrapper=ColumnPreservingTransformerWrapperDF
)

DictVectorizerDF: T_Wrapper[DictVectorizer] = make_df_transformer(
    DictVectorizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

HashingVectorizerDF: T_Wrapper[HashingVectorizer] = make_df_transformer(
    HashingVectorizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

TfidfTransformerDF: T_Wrapper[TfidfTransformer] = make_df_transformer(
    TfidfTransformer, base_wrapper=ColumnPreservingTransformerWrapperDF
)


#
# impute
#

# we cannot move this to package _wrapper as it references MissingIndicatorDF


SimpleImputerDF: T_Wrapper[SimpleImputer] = make_df_transformer(
    SimpleImputer, base_wrapper=ImputerWrapperDF
)

MissingIndicatorDF: T_Wrapper[MissingIndicator] = make_df_transformer(
    MissingIndicator, base_wrapper=MissingIndicatorWrapperDF
)

IterativeImputerDF: T_Wrapper[IterativeImputer] = make_df_transformer(
    IterativeImputer, base_wrapper=ImputerWrapperDF
)

IsomapDF: T_Wrapper[Isomap] = make_df_transformer(Isomap, base_wrapper=IsomapWrapperDF)

AdditiveChi2SamplerDF: T_Wrapper[AdditiveChi2Sampler] = make_df_transformer(
    AdditiveChi2Sampler, base_wrapper=AdditiveChi2SamplerWrapperDF
)


#
# neighbors
#

NeighborhoodComponentsAnalysisDF: T_Wrapper[
    NeighborhoodComponentsAnalysis
] = make_df_transformer(
    NeighborhoodComponentsAnalysis,
    base_wrapper=ColumnPreservingTransformerWrapperDF,
)


#
# preprocessing
#


MinMaxScalerDF: T_Wrapper[MinMaxScaler] = make_df_transformer(
    MinMaxScaler, base_wrapper=ColumnPreservingTransformerWrapperDF
)

StandardScalerDF: T_Wrapper[StandardScaler] = make_df_transformer(
    StandardScaler, base_wrapper=ColumnPreservingTransformerWrapperDF
)

MaxAbsScalerDF: T_Wrapper[MaxAbsScaler] = make_df_transformer(
    MaxAbsScaler, base_wrapper=ColumnPreservingTransformerWrapperDF
)

RobustScalerDF: T_Wrapper[RobustScaler] = make_df_transformer(
    RobustScaler, base_wrapper=ColumnPreservingTransformerWrapperDF
)

PolynomialFeaturesDF: T_Wrapper[PolynomialFeatures] = make_df_transformer(
    PolynomialFeatures, base_wrapper=PolynomialFeaturesWrapperDF
)

NormalizerDF: T_Wrapper[Normalizer] = make_df_transformer(
    Normalizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

BinarizerDF: T_Wrapper[Binarizer] = make_df_transformer(
    Binarizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

KernelCentererDF: T_Wrapper[KernelCenterer] = make_df_transformer(
    KernelCenterer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

QuantileTransformerDF: T_Wrapper[QuantileTransformer] = make_df_transformer(
    QuantileTransformer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

PowerTransformerDF: T_Wrapper[PowerTransformer] = make_df_transformer(
    PowerTransformer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

FunctionTransformerDF: T_Wrapper[FunctionTransformer] = make_df_transformer(
    FunctionTransformer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

LabelEncoderDF: T_Wrapper[LabelEncoder] = make_df_transformer(
    LabelEncoder, base_wrapper=ColumnPreservingTransformerWrapperDF
)

LabelBinarizerDF: T_Wrapper[LabelBinarizer] = make_df_transformer(
    LabelBinarizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

MultiLabelBinarizerDF: T_Wrapper[MultiLabelBinarizer] = make_df_transformer(
    MultiLabelBinarizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

OneHotEncoderDF: T_Wrapper[OneHotEncoder] = make_df_transformer(
    OneHotEncoder, base_wrapper=OneHotEncoderWrapperDF
)

OrdinalEncoderDF: T_Wrapper[OrdinalEncoder] = make_df_transformer(
    OrdinalEncoder, base_wrapper=ColumnPreservingTransformerWrapperDF
)

KBinsDiscretizerDF: T_Wrapper[KBinsDiscretizer] = make_df_transformer(
    KBinsDiscretizer, base_wrapper=KBinsDiscretizerWrapperDF
)


#
# Transformers which have a components_ attribute
# Implemented through ComponentsDimensionalityReductionWrapperDF
#

BernoulliRBMDF: T_Wrapper[BernoulliRBM] = make_df_transformer(
    BernoulliRBM, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

DictionaryLearningDF: T_Wrapper[DictionaryLearning] = make_df_transformer(
    DictionaryLearning, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

FactorAnalysisDF: T_Wrapper[FactorAnalysis] = make_df_transformer(
    FactorAnalysis, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

FastICADF: T_Wrapper[FastICA] = make_df_transformer(
    FastICA, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

GaussianRandomProjectionDF: T_Wrapper[GaussianRandomProjection] = make_df_transformer(
    GaussianRandomProjection,
    base_wrapper=ComponentsDimensionalityReductionWrapperDF,
)

IncrementalPCADF: T_Wrapper[IncrementalPCA] = make_df_transformer(
    IncrementalPCA, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

LatentDirichletAllocationDF: T_Wrapper[LatentDirichletAllocation] = make_df_transformer(
    LatentDirichletAllocation,
    base_wrapper=ComponentsDimensionalityReductionWrapperDF,
)

MiniBatchDictionaryLearningDF: T_Wrapper[
    MiniBatchDictionaryLearning
] = make_df_transformer(
    MiniBatchDictionaryLearning,
    base_wrapper=ComponentsDimensionalityReductionWrapperDF,
)

MiniBatchSparsePCADF: T_Wrapper[MiniBatchSparsePCA] = make_df_transformer(
    MiniBatchSparsePCA, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

NMFDF: T_Wrapper[NMF] = make_df_transformer(
    NMF, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

PCADF: T_Wrapper[PCA] = make_df_transformer(
    PCA, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)

SparseCoderDF: T_Wrapper[SparseCoder] = make_df_transformer(
    SparseCoder, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

SparsePCADF: T_Wrapper[SparsePCA] = make_df_transformer(
    SparsePCA, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

SparseRandomProjectionDF: T_Wrapper[SparseRandomProjection] = make_df_transformer(
    SparseRandomProjection, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

TruncatedSVDDF: T_Wrapper[TruncatedSVD] = make_df_transformer(
    TruncatedSVD, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)


#
# Transformers which have an n_components attribute
# Implemented through NComponentsDimensionalityReductionWrapperDF
#

KernelPCADF: T_Wrapper[KernelPCA] = make_df_transformer(
    KernelPCA, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)

LocallyLinearEmbeddingDF: T_Wrapper[LocallyLinearEmbedding] = make_df_transformer(
    LocallyLinearEmbedding, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)

NystroemDF: T_Wrapper[Nystroem] = make_df_transformer(
    Nystroem, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)

RBFSamplerDF: T_Wrapper[RBFSampler] = make_df_transformer(
    RBFSampler, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)

SkewedChi2SamplerDF: T_Wrapper[SkewedChi2Sampler] = make_df_transformer(
    SkewedChi2Sampler, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)


#
# feature_selection
#
# Transformers with a get_support method, implemented via FeatureSelectionWrapperDF
#

VarianceThresholdDF: T_Wrapper[VarianceThreshold] = make_df_transformer(
    VarianceThreshold, base_wrapper=FeatureSelectionWrapperDF
)

RFEDF: T_Wrapper[RFE] = make_df_transformer(RFE, base_wrapper=FeatureSelectionWrapperDF)

RFECVDF: T_Wrapper[RFECV] = make_df_transformer(
    RFECV, base_wrapper=FeatureSelectionWrapperDF
)

SelectFromModelDF: T_Wrapper[SelectFromModel] = make_df_transformer(
    SelectFromModel, base_wrapper=FeatureSelectionWrapperDF
)

SelectPercentileDF: T_Wrapper[SelectPercentile] = make_df_transformer(
    SelectPercentile, base_wrapper=FeatureSelectionWrapperDF
)

SelectKBestDF: T_Wrapper[SelectKBest] = make_df_transformer(
    SelectKBest, base_wrapper=FeatureSelectionWrapperDF
)

SelectFprDF: T_Wrapper[SelectFpr] = make_df_transformer(
    SelectFpr, base_wrapper=FeatureSelectionWrapperDF
)

SelectFdrDF: T_Wrapper[SelectFdr] = make_df_transformer(
    SelectFdr, base_wrapper=FeatureSelectionWrapperDF
)

SelectFweDF: T_Wrapper[SelectFwe] = make_df_transformer(
    SelectFwe, base_wrapper=FeatureSelectionWrapperDF
)

GenericUnivariateSelectDF: T_Wrapper[GenericUnivariateSelect] = make_df_transformer(
    GenericUnivariateSelect,
    base_wrapper=FeatureSelectionWrapperDF,
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
