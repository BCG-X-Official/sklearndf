"""
Core implementation of :mod:`sklearndf.transformation`
"""

import logging

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

# noinspection PyProtectedMember
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
    PolynomialTransformerWrapperDF,
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
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals(), allow_imported_definitions=True)


#
# Class definitions
#

#
# compose
#


class ColumnTransformerDF(
    ColumnTransformerWrapperDF,
    ColumnTransformer,  # type: ignore
    native=ColumnTransformer,
):
    """Stub for DF wrapper of class ``ColumnTransformer``"""


#
# cross_decomposition
#


class PLSSVDDF(
    ColumnPreservingTransformerWrapperDF[PLSSVD], PLSSVD, native=PLSSVD  # type: ignore
):
    """Stub for DF wrapper of class ``PLSSVD``"""


class FeatureHasherDF(
    ColumnPreservingTransformerWrapperDF[FeatureHasher],
    FeatureHasher,  # type: ignore
    native=FeatureHasher,
):
    """Stub for DF wrapper of class ``FeatureHasher``"""


class DictVectorizerDF(
    ColumnPreservingTransformerWrapperDF[DictVectorizer],
    DictVectorizer,  # type: ignore
    native=DictVectorizer,
):
    """Stub for DF wrapper of class ``DictVectorizer``"""


class HashingVectorizerDF(
    ColumnPreservingTransformerWrapperDF[HashingVectorizer],
    HashingVectorizer,  # type: ignore
    native=HashingVectorizer,
):
    """Stub for DF wrapper of class ``HashingVectorizer``"""


class TfidfTransformerDF(
    ColumnPreservingTransformerWrapperDF[TfidfTransformer],
    TfidfTransformer,  # type: ignore
    native=TfidfTransformer,
):
    """Stub for DF wrapper of class ``TfidfTransformer``"""


#
# impute
#

# we cannot move this to package _wrapper as it references MissingIndicatorDF


class SimpleImputerDF(
    ImputerWrapperDF[SimpleImputer], SimpleImputer, native=SimpleImputer  # type: ignore
):
    """Stub for DF wrapper of class ``SimpleImputer``"""


class MissingIndicatorDF(
    MissingIndicatorWrapperDF,
    MissingIndicator,  # type: ignore
    native=MissingIndicator,
):
    """Stub for DF wrapper of class ``MissingIndicator``"""


class IterativeImputerDF(
    ImputerWrapperDF[IterativeImputer],
    IterativeImputer,  # type: ignore
    native=IterativeImputer,
):
    """Stub for DF wrapper of class ``IterativeImputer``"""


class IsomapDF(
    IsomapWrapperDF,
    Isomap,  # type: ignore
    native=Isomap,
):
    """Stub for DF wrapper of class ``Isomap``"""


class AdditiveChi2SamplerDF(
    AdditiveChi2SamplerWrapperDF,
    AdditiveChi2Sampler,  # type: ignore
    native=AdditiveChi2Sampler,
):
    """Stub for DF wrapper of class ``AdditiveChi2Sampler``"""


#
# neighbors
#


class NeighborhoodComponentsAnalysisDF(
    ColumnPreservingTransformerWrapperDF[NeighborhoodComponentsAnalysis],
    NeighborhoodComponentsAnalysis,  # type: ignore
    native=NeighborhoodComponentsAnalysis,
):
    """Stub for DF wrapper of class ``NeighborhoodComponentsAnalysis``"""


#
# preprocessing
#


class MinMaxScalerDF(
    ColumnPreservingTransformerWrapperDF[MinMaxScaler],
    MinMaxScaler,  # type: ignore
    native=MinMaxScaler,
):
    """Stub for DF wrapper of class ``MinMaxScaler``"""


class StandardScalerDF(
    ColumnPreservingTransformerWrapperDF[StandardScaler],
    StandardScaler,  # type: ignore
    native=StandardScaler,
):
    """Stub for DF wrapper of class ``StandardScaler``"""


class MaxAbsScalerDF(
    ColumnPreservingTransformerWrapperDF[MaxAbsScaler],
    MaxAbsScaler,  # type: ignore
    native=MaxAbsScaler,
):
    """Stub for DF wrapper of class ``MaxAbsScaler``"""


class RobustScalerDF(
    ColumnPreservingTransformerWrapperDF[RobustScaler],
    RobustScaler,  # type: ignore
    native=RobustScaler,
):
    """Stub for DF wrapper of class ``RobustScaler``"""


class PolynomialFeaturesDF(
    PolynomialTransformerWrapperDF[PolynomialFeatures],
    PolynomialFeatures,  # type: ignore
    native=PolynomialFeatures,
):
    """Stub for DF wrapper of class ``PolynomialFeatures``"""


class NormalizerDF(
    ColumnPreservingTransformerWrapperDF[Normalizer],
    Normalizer,  # type: ignore
    native=Normalizer,
):
    """Stub for DF wrapper of class ``Normalizer``"""


class BinarizerDF(
    ColumnPreservingTransformerWrapperDF[Binarizer],
    Binarizer,  # type: ignore
    native=Binarizer,
):
    """Stub for DF wrapper of class ``Binarizer``"""


class KernelCentererDF(
    ColumnPreservingTransformerWrapperDF[KernelCenterer],
    KernelCenterer,  # type: ignore
    native=KernelCenterer,
):
    """Stub for DF wrapper of class ``KernelCenterer``"""


class QuantileTransformerDF(
    ColumnPreservingTransformerWrapperDF[QuantileTransformer],
    QuantileTransformer,  # type: ignore
    native=QuantileTransformer,
):
    """Stub for DF wrapper of class ``QuantileTransformer``"""


class PowerTransformerDF(
    ColumnPreservingTransformerWrapperDF[PowerTransformer],
    PowerTransformer,  # type: ignore
    native=PowerTransformer,
):
    """Stub for DF wrapper of class ``PowerTransformer``"""


class FunctionTransformerDF(
    ColumnPreservingTransformerWrapperDF[FunctionTransformer],
    FunctionTransformer,  # type: ignore
    native=FunctionTransformer,
):
    """Stub for DF wrapper of class ``FunctionTransformer``"""


class LabelEncoderDF(
    ColumnPreservingTransformerWrapperDF[LabelEncoder],
    LabelEncoder,  # type: ignore
    native=LabelEncoder,
):
    """Stub for DF wrapper of class ``LabelEncoder``"""


class LabelBinarizerDF(
    ColumnPreservingTransformerWrapperDF[LabelBinarizer],
    LabelBinarizer,  # type: ignore
    native=LabelBinarizer,
):
    """Stub for DF wrapper of class ``LabelBinarizer``"""


class MultiLabelBinarizerDF(
    ColumnPreservingTransformerWrapperDF[MultiLabelBinarizer],
    MultiLabelBinarizer,  # type: ignore
    native=MultiLabelBinarizer,
):
    """Stub for DF wrapper of class ``MultiLabelBinarizer``"""


class OneHotEncoderDF(
    OneHotEncoderWrapperDF,
    OneHotEncoder,  # type: ignore
    native=OneHotEncoder,
):
    """Stub for DF wrapper of class ``OneHotEncoder``"""


class OrdinalEncoderDF(
    ColumnPreservingTransformerWrapperDF[OrdinalEncoder],
    OrdinalEncoder,  # type: ignore
    native=OrdinalEncoder,
):
    """Stub for DF wrapper of class ``OrdinalEncoder``"""


class KBinsDiscretizerDF(
    KBinsDiscretizerWrapperDF,
    KBinsDiscretizer,  # type: ignore
    native=KBinsDiscretizer,
):
    """Stub for DF wrapper of class ``KBinsDiscretizer``"""


#
# Transformers which have a components_ attribute
# Implemented through ComponentsDimensionalityReductionWrapperDF
#


class BernoulliRBMDF(
    ComponentsDimensionalityReductionWrapperDF[BernoulliRBM],
    BernoulliRBM,  # type: ignore
    native=BernoulliRBM,
):
    """Stub for DF wrapper of class ``BernoulliRBM``"""


class DictionaryLearningDF(
    ComponentsDimensionalityReductionWrapperDF[DictionaryLearning],
    DictionaryLearning,  # type: ignore
    native=DictionaryLearning,
):
    """Stub for DF wrapper of class ``DictionaryLearning``"""


class FactorAnalysisDF(
    ComponentsDimensionalityReductionWrapperDF[FactorAnalysis],
    FactorAnalysis,  # type: ignore
    native=FactorAnalysis,
):
    """Stub for DF wrapper of class ``FactorAnalysis``"""


class FastICADF(
    ComponentsDimensionalityReductionWrapperDF[FastICA],
    FastICA,  # type: ignore
    native=FastICA,
):
    """Stub for DF wrapper of class ``FastICA``"""


class GaussianRandomProjectionDF(
    ComponentsDimensionalityReductionWrapperDF[GaussianRandomProjection],
    GaussianRandomProjection,  # type: ignore
    native=GaussianRandomProjection,
):
    """Stub for DF wrapper of class ``GaussianRandomProjection``"""


class IncrementalPCADF(
    ComponentsDimensionalityReductionWrapperDF[IncrementalPCA],
    IncrementalPCA,  # type: ignore
    native=IncrementalPCA,
):
    """Stub for DF wrapper of class ``IncrementalPCA``"""


class LatentDirichletAllocationDF(
    ComponentsDimensionalityReductionWrapperDF[LatentDirichletAllocation],
    LatentDirichletAllocation,  # type: ignore
    native=LatentDirichletAllocation,
):
    """Stub for DF wrapper of class ``LatentDirichletAllocation``"""


class MiniBatchDictionaryLearningDF(
    ComponentsDimensionalityReductionWrapperDF[MiniBatchDictionaryLearning],
    MiniBatchDictionaryLearning,  # type: ignore
    native=MiniBatchDictionaryLearning,
):
    """Stub for DF wrapper of class ``MiniBatchDictionaryLearning``"""


class MiniBatchSparsePCADF(
    ComponentsDimensionalityReductionWrapperDF[MiniBatchSparsePCA],
    MiniBatchSparsePCA,  # type: ignore
    native=MiniBatchSparsePCA,
):
    """Stub for DF wrapper of class ``MiniBatchSparsePCA``"""


class NMFDF(
    ComponentsDimensionalityReductionWrapperDF[NMF], NMF, native=NMF  # type: ignore
):
    """Stub for DF wrapper of class ``NMF``"""


class PCADF(
    NComponentsDimensionalityReductionWrapperDF[PCA], PCA, native=PCA  # type: ignore
):
    """Stub for DF wrapper of class ``PCA``"""


class SparseCoderDF(
    ComponentsDimensionalityReductionWrapperDF[SparseCoder],
    SparseCoder,  # type: ignore
    native=SparseCoder,
):
    """Stub for DF wrapper of class ``SparseCoder``"""


class SparsePCADF(
    ComponentsDimensionalityReductionWrapperDF[SparsePCA],
    SparsePCA,  # type: ignore
    native=SparsePCA,
):
    """Stub for DF wrapper of class ``SparsePCA``"""


class SparseRandomProjectionDF(
    ComponentsDimensionalityReductionWrapperDF[SparseRandomProjection],
    SparseRandomProjection,  # type: ignore
    native=SparseRandomProjection,
):
    """Stub for DF wrapper of class ``SparseRandomProjection``"""


class TruncatedSVDDF(
    ComponentsDimensionalityReductionWrapperDF[TruncatedSVD],
    TruncatedSVD,  # type: ignore
    native=TruncatedSVD,
):
    """Stub for DF wrapper of class ``TruncatedSVD``"""


#
# Transformers which have an n_components attribute
# Implemented through NComponentsDimensionalityReductionWrapperDF
#


class KernelPCADF(
    NComponentsDimensionalityReductionWrapperDF[KernelPCA],
    KernelPCA,  # type: ignore
    native=KernelPCA,
):
    """Stub for DF wrapper of class ``KernelPCA``"""


class LocallyLinearEmbeddingDF(
    NComponentsDimensionalityReductionWrapperDF[LocallyLinearEmbedding],
    LocallyLinearEmbedding,  # type: ignore
    native=LocallyLinearEmbedding,
):
    """Stub for DF wrapper of class ``LocallyLinearEmbedding``"""


class NystroemDF(
    NComponentsDimensionalityReductionWrapperDF[Nystroem],
    Nystroem,  # type: ignore
    native=Nystroem,
):
    """Stub for DF wrapper of class ``Nystroem``"""


class RBFSamplerDF(
    NComponentsDimensionalityReductionWrapperDF[RBFSampler],
    RBFSampler,  # type: ignore
    native=RBFSampler,
):
    """Stub for DF wrapper of class ``RBFSampler``"""


class SkewedChi2SamplerDF(
    NComponentsDimensionalityReductionWrapperDF[SkewedChi2Sampler],
    SkewedChi2Sampler,  # type: ignore
    native=SkewedChi2Sampler,
):
    """Stub for DF wrapper of class ``SkewedChi2Sampler``"""


#
# feature_selection
#
# Transformers with a get_support method, implemented via FeatureSelectionWrapperDF
#


class VarianceThresholdDF(
    FeatureSelectionWrapperDF[VarianceThreshold],
    VarianceThreshold,  # type: ignore
    native=VarianceThreshold,
):
    """Stub for DF wrapper of class ``VarianceThreshold``"""


class RFEDF(FeatureSelectionWrapperDF[RFE], RFE, native=RFE):  # type: ignore
    """Stub for DF wrapper of class ``RFE``"""


class RFECVDF(FeatureSelectionWrapperDF[RFECV], RFECV, native=RFECV):  # type: ignore
    """Stub for DF wrapper of class ``RFECV``"""


class SelectFromModelDF(
    FeatureSelectionWrapperDF[SelectFromModel],
    SelectFromModel,  # type: ignore
    native=SelectFromModel,
):
    """Stub for DF wrapper of class ``SelectFromModel``"""


class SelectPercentileDF(
    FeatureSelectionWrapperDF[SelectPercentile],
    SelectPercentile,  # type: ignore
    native=SelectPercentile,
):
    """Stub for DF wrapper of class ``SelectPercentile``"""


class SelectKBestDF(
    FeatureSelectionWrapperDF[SelectKBest],
    SelectKBest,  # type: ignore
    native=SelectKBest,
):
    """Stub for DF wrapper of class ``SelectKBest``"""


class SelectFprDF(
    FeatureSelectionWrapperDF[SelectFpr], SelectFpr, native=SelectFpr  # type: ignore
):
    """Stub for DF wrapper of class ``SelectFpr``"""


class SelectFdrDF(
    FeatureSelectionWrapperDF[SelectFdr], SelectFdr, native=SelectFdr  # type: ignore
):
    """Stub for DF wrapper of class ``SelectFdr``"""


class SelectFweDF(
    FeatureSelectionWrapperDF[SelectFwe], SelectFwe, native=SelectFwe  # type: ignore
):
    """Stub for DF wrapper of class ``SelectFwe``"""


class GenericUnivariateSelectDF(
    FeatureSelectionWrapperDF[GenericUnivariateSelect],
    GenericUnivariateSelect,  # type: ignore
    native=GenericUnivariateSelect,
):
    """Stub for DF wrapper of class ``GenericUnivariateSelect``"""


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
