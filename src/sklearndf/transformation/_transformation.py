"""
Core implementation of :mod:`sklearndf.transformation`
"""

import logging

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

log = logging.getLogger(__name__)

__all__ = [
    "AdditiveChi2SamplerDF",
    "BernoulliRBMDF",
    "BinarizerDF",
    "ColumnTransformerDF",
    "DictionaryLearningDF",
    "DictVectorizerDF",
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
    "MiniBatchDictionaryLearningDF",
    "MiniBatchSparsePCADF",
    "MinMaxScalerDF",
    "MissingIndicatorDF",
    "MultiLabelBinarizerDF",
    "NeighborhoodComponentsAnalysisDF",
    "NMFDF",
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
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals(), allow_imported_definitions=True)


#
# Class definitions
#

#
# cluster
#


FeatureAgglomerationDF = make_df_transformer(
    FeatureAgglomeration, base_wrapper=ColumnPreservingTransformerWrapperDF
)


#
# compose
#


ColumnTransformerDF = make_df_transformer(
    ColumnTransformer, base_wrapper=ColumnTransformerWrapperDF
)


#
# cross_decomposition
#


PLSSVDDF = make_df_transformer(
    PLSSVD, base_wrapper=ColumnPreservingTransformerWrapperDF
)

FeatureHasherDF = make_df_transformer(
    FeatureHasher, base_wrapper=ColumnPreservingTransformerWrapperDF
)

DictVectorizerDF = make_df_transformer(
    DictVectorizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

HashingVectorizerDF = make_df_transformer(
    HashingVectorizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

TfidfTransformerDF = make_df_transformer(
    TfidfTransformer, base_wrapper=ColumnPreservingTransformerWrapperDF
)


#
# impute
#

# we cannot move this to package _wrapper as it references MissingIndicatorDF


SimpleImputerDF = make_df_transformer(SimpleImputer, base_wrapper=ImputerWrapperDF)

MissingIndicatorDF = make_df_transformer(
    MissingIndicator, base_wrapper=MissingIndicatorWrapperDF
)

IterativeImputerDF = make_df_transformer(
    IterativeImputer, base_wrapper=ImputerWrapperDF
)

IsomapDF = make_df_transformer(Isomap, base_wrapper=IsomapWrapperDF)

AdditiveChi2SamplerDF = make_df_transformer(
    AdditiveChi2Sampler, base_wrapper=AdditiveChi2SamplerWrapperDF
)


#
# neighbors
#

NeighborhoodComponentsAnalysisDF = make_df_transformer(
    NeighborhoodComponentsAnalysis,
    base_wrapper=ColumnPreservingTransformerWrapperDF,
)


#
# preprocessing
#


MinMaxScalerDF = make_df_transformer(
    MinMaxScaler, base_wrapper=ColumnPreservingTransformerWrapperDF
)

StandardScalerDF = make_df_transformer(
    StandardScaler, base_wrapper=ColumnPreservingTransformerWrapperDF
)

MaxAbsScalerDF = make_df_transformer(
    MaxAbsScaler, base_wrapper=ColumnPreservingTransformerWrapperDF
)

RobustScalerDF = make_df_transformer(
    RobustScaler, base_wrapper=ColumnPreservingTransformerWrapperDF
)

PolynomialFeaturesDF = make_df_transformer(
    PolynomialFeatures, base_wrapper=PolynomialFeaturesWrapperDF
)

NormalizerDF = make_df_transformer(
    Normalizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

BinarizerDF = make_df_transformer(
    Binarizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

KernelCentererDF = make_df_transformer(
    KernelCenterer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

QuantileTransformerDF = make_df_transformer(
    QuantileTransformer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

PowerTransformerDF = make_df_transformer(
    PowerTransformer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

FunctionTransformerDF = make_df_transformer(
    FunctionTransformer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

LabelEncoderDF = make_df_transformer(
    LabelEncoder, base_wrapper=ColumnPreservingTransformerWrapperDF
)

LabelBinarizerDF = make_df_transformer(
    LabelBinarizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

MultiLabelBinarizerDF = make_df_transformer(
    MultiLabelBinarizer, base_wrapper=ColumnPreservingTransformerWrapperDF
)

OneHotEncoderDF = make_df_transformer(
    OneHotEncoder, base_wrapper=OneHotEncoderWrapperDF
)

OrdinalEncoderDF = make_df_transformer(
    OrdinalEncoder, base_wrapper=ColumnPreservingTransformerWrapperDF
)

KBinsDiscretizerDF = make_df_transformer(
    KBinsDiscretizer, base_wrapper=KBinsDiscretizerWrapperDF
)


#
# Transformers which have a components_ attribute
# Implemented through ComponentsDimensionalityReductionWrapperDF
#

BernoulliRBMDF = make_df_transformer(
    BernoulliRBM, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

DictionaryLearningDF = make_df_transformer(
    DictionaryLearning, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

FactorAnalysisDF = make_df_transformer(
    FactorAnalysis, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

FastICADF = make_df_transformer(
    FastICA, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

GaussianRandomProjectionDF = make_df_transformer(
    GaussianRandomProjection,
    base_wrapper=ComponentsDimensionalityReductionWrapperDF,
)

IncrementalPCADF = make_df_transformer(
    IncrementalPCA, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

LatentDirichletAllocationDF = make_df_transformer(
    LatentDirichletAllocation,
    base_wrapper=ComponentsDimensionalityReductionWrapperDF,
)

MiniBatchDictionaryLearningDF = make_df_transformer(
    MiniBatchDictionaryLearning,
    base_wrapper=ComponentsDimensionalityReductionWrapperDF,
)

MiniBatchSparsePCADF = make_df_transformer(
    MiniBatchSparsePCA, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

NMFDF = make_df_transformer(
    NMF, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

PCADF = make_df_transformer(
    PCA, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)

SparseCoderDF = make_df_transformer(
    SparseCoder, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

SparsePCADF = make_df_transformer(
    SparsePCA, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

SparseRandomProjectionDF = make_df_transformer(
    SparseRandomProjection, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)

TruncatedSVDDF = make_df_transformer(
    TruncatedSVD, base_wrapper=ComponentsDimensionalityReductionWrapperDF
)


#
# Transformers which have an n_components attribute
# Implemented through NComponentsDimensionalityReductionWrapperDF
#

KernelPCADF = make_df_transformer(
    KernelPCA, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)

LocallyLinearEmbeddingDF = make_df_transformer(
    LocallyLinearEmbedding, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)

NystroemDF = make_df_transformer(
    Nystroem, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)

RBFSamplerDF = make_df_transformer(
    RBFSampler, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)

SkewedChi2SamplerDF = make_df_transformer(
    SkewedChi2Sampler, base_wrapper=NComponentsDimensionalityReductionWrapperDF
)


#
# feature_selection
#
# Transformers with a get_support method, implemented via FeatureSelectionWrapperDF
#

VarianceThresholdDF = make_df_transformer(
    VarianceThreshold, base_wrapper=FeatureSelectionWrapperDF
)

RFEDF = make_df_transformer(RFE, base_wrapper=FeatureSelectionWrapperDF)

RFECVDF = make_df_transformer(RFECV, base_wrapper=FeatureSelectionWrapperDF)

SelectFromModelDF = make_df_transformer(
    SelectFromModel, base_wrapper=FeatureSelectionWrapperDF
)

SelectPercentileDF = make_df_transformer(
    SelectPercentile, base_wrapper=FeatureSelectionWrapperDF
)

SelectKBestDF = make_df_transformer(SelectKBest, base_wrapper=FeatureSelectionWrapperDF)

SelectFprDF = make_df_transformer(SelectFpr, base_wrapper=FeatureSelectionWrapperDF)

SelectFdrDF = make_df_transformer(SelectFdr, base_wrapper=FeatureSelectionWrapperDF)

SelectFweDF = make_df_transformer(SelectFwe, base_wrapper=FeatureSelectionWrapperDF)

GenericUnivariateSelectDF = make_df_transformer(
    GenericUnivariateSelect, base_wrapper=FeatureSelectionWrapperDF
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
