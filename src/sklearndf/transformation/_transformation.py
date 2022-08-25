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

__tracker = AllTracker(globals())


#
# Class definitions
#

#
# compose
#


class ColumnTransformerDF(
    ColumnTransformerWrapperDF,
    native=ColumnTransformer,
):
    """Stub for DF wrapper of class ``ColumnTransformer``"""


#
# cross_decomposition
#


class PLSSVDDF(ColumnPreservingTransformerWrapperDF[PLSSVD], native=PLSSVD):
    """Stub for DF wrapper of class ``PLSSVD``"""


class FeatureHasherDF(
    ColumnPreservingTransformerWrapperDF[FeatureHasher],
    native=FeatureHasher,
):
    """Stub for DF wrapper of class ``FeatureHasher``"""


class DictVectorizerDF(
    ColumnPreservingTransformerWrapperDF[DictVectorizer],
    native=DictVectorizer,
):
    """Stub for DF wrapper of class ``DictVectorizer``"""


class HashingVectorizerDF(
    ColumnPreservingTransformerWrapperDF[HashingVectorizer],
    native=HashingVectorizer,
):
    """Stub for DF wrapper of class ``HashingVectorizer``"""


class TfidfTransformerDF(
    ColumnPreservingTransformerWrapperDF[TfidfTransformer],
    native=TfidfTransformer,
):
    """Stub for DF wrapper of class ``TfidfTransformer``"""


#
# impute
#

# we cannot move this to package _wrapper as it references MissingIndicatorDF


class SimpleImputerDF(ImputerWrapperDF[SimpleImputer], native=SimpleImputer):
    """Stub for DF wrapper of class ``SimpleImputer``"""


class MissingIndicatorDF(
    MissingIndicatorWrapperDF,
    native=MissingIndicator,
):
    """Stub for DF wrapper of class ``MissingIndicator``"""


class IterativeImputerDF(
    ImputerWrapperDF[IterativeImputer],
    native=IterativeImputer,
):
    """Stub for DF wrapper of class ``IterativeImputer``"""


class IsomapDF(
    IsomapWrapperDF,
    native=Isomap,
):
    """Stub for DF wrapper of class ``Isomap``"""


class AdditiveChi2SamplerDF(
    AdditiveChi2SamplerWrapperDF,
    native=AdditiveChi2Sampler,
):
    """Stub for DF wrapper of class ``AdditiveChi2Sampler``"""


#
# neighbors
#


class NeighborhoodComponentsAnalysisDF(
    ColumnPreservingTransformerWrapperDF[NeighborhoodComponentsAnalysis],
    native=NeighborhoodComponentsAnalysis,
):
    """Stub for DF wrapper of class ``NeighborhoodComponentsAnalysis``"""


#
# preprocessing
#


class MinMaxScalerDF(
    ColumnPreservingTransformerWrapperDF[MinMaxScaler],
    native=MinMaxScaler,
):
    """Stub for DF wrapper of class ``MinMaxScaler``"""


class StandardScalerDF(
    ColumnPreservingTransformerWrapperDF[StandardScaler],
    native=StandardScaler,
):
    """Stub for DF wrapper of class ``StandardScaler``"""


class MaxAbsScalerDF(
    ColumnPreservingTransformerWrapperDF[MaxAbsScaler],
    native=MaxAbsScaler,
):
    """Stub for DF wrapper of class ``MaxAbsScaler``"""


class RobustScalerDF(
    ColumnPreservingTransformerWrapperDF[RobustScaler],
    native=RobustScaler,
):
    """Stub for DF wrapper of class ``RobustScaler``"""


class PolynomialFeaturesDF(
    PolynomialTransformerWrapperDF[PolynomialFeatures],
    native=PolynomialFeatures,
):
    """Stub for DF wrapper of class ``PolynomialFeatures``"""


class NormalizerDF(
    ColumnPreservingTransformerWrapperDF[Normalizer],
    native=Normalizer,
):
    """Stub for DF wrapper of class ``Normalizer``"""


class BinarizerDF(
    ColumnPreservingTransformerWrapperDF[Binarizer],
    native=Binarizer,
):
    """Stub for DF wrapper of class ``Binarizer``"""


class KernelCentererDF(
    ColumnPreservingTransformerWrapperDF[KernelCenterer],
    native=KernelCenterer,
):
    """Stub for DF wrapper of class ``KernelCenterer``"""


class QuantileTransformerDF(
    ColumnPreservingTransformerWrapperDF[QuantileTransformer],
    native=QuantileTransformer,
):
    """Stub for DF wrapper of class ``QuantileTransformer``"""


class PowerTransformerDF(
    ColumnPreservingTransformerWrapperDF[PowerTransformer],
    native=PowerTransformer,
):
    """Stub for DF wrapper of class ``PowerTransformer``"""


class FunctionTransformerDF(
    ColumnPreservingTransformerWrapperDF[FunctionTransformer],
    native=FunctionTransformer,
):
    """Stub for DF wrapper of class ``FunctionTransformer``"""


class LabelEncoderDF(
    ColumnPreservingTransformerWrapperDF[LabelEncoder],
    native=LabelEncoder,
):
    """Stub for DF wrapper of class ``LabelEncoder``"""


class LabelBinarizerDF(
    ColumnPreservingTransformerWrapperDF[LabelBinarizer],
    native=LabelBinarizer,
):
    """Stub for DF wrapper of class ``LabelBinarizer``"""


class MultiLabelBinarizerDF(
    ColumnPreservingTransformerWrapperDF[MultiLabelBinarizer],
    native=MultiLabelBinarizer,
):
    """Stub for DF wrapper of class ``MultiLabelBinarizer``"""


class OneHotEncoderDF(
    OneHotEncoderWrapperDF,
    native=OneHotEncoder,
):
    """Stub for DF wrapper of class ``OneHotEncoder``"""


class OrdinalEncoderDF(
    ColumnPreservingTransformerWrapperDF[OrdinalEncoder],
    native=OrdinalEncoder,
):
    """Stub for DF wrapper of class ``OrdinalEncoder``"""


class KBinsDiscretizerDF(
    KBinsDiscretizerWrapperDF,
    native=KBinsDiscretizer,
):
    """Stub for DF wrapper of class ``KBinsDiscretizer``"""


#
# Transformers which have a components_ attribute
# Implemented through ComponentsDimensionalityReductionWrapperDF
#


class BernoulliRBMDF(
    ComponentsDimensionalityReductionWrapperDF[BernoulliRBM],
    native=BernoulliRBM,
):
    """Stub for DF wrapper of class ``BernoulliRBM``"""


class DictionaryLearningDF(
    ComponentsDimensionalityReductionWrapperDF[DictionaryLearning],
    native=DictionaryLearning,
):
    """Stub for DF wrapper of class ``DictionaryLearning``"""


class FactorAnalysisDF(
    ComponentsDimensionalityReductionWrapperDF[FactorAnalysis],
    native=FactorAnalysis,
):
    """Stub for DF wrapper of class ``FactorAnalysis``"""


class FastICADF(
    ComponentsDimensionalityReductionWrapperDF[FastICA],
    native=FastICA,
):
    """Stub for DF wrapper of class ``FastICA``"""


class GaussianRandomProjectionDF(
    ComponentsDimensionalityReductionWrapperDF[GaussianRandomProjection],
    native=GaussianRandomProjection,
):
    """Stub for DF wrapper of class ``GaussianRandomProjection``"""


class IncrementalPCADF(
    ComponentsDimensionalityReductionWrapperDF[IncrementalPCA],
    native=IncrementalPCA,
):
    """Stub for DF wrapper of class ``IncrementalPCA``"""


class LatentDirichletAllocationDF(
    ComponentsDimensionalityReductionWrapperDF[LatentDirichletAllocation],
    native=LatentDirichletAllocation,
):
    """Stub for DF wrapper of class ``LatentDirichletAllocation``"""


class MiniBatchDictionaryLearningDF(
    ComponentsDimensionalityReductionWrapperDF[MiniBatchDictionaryLearning],
    native=MiniBatchDictionaryLearning,
):
    """Stub for DF wrapper of class ``MiniBatchDictionaryLearning``"""


class MiniBatchSparsePCADF(
    ComponentsDimensionalityReductionWrapperDF[MiniBatchSparsePCA],
    native=MiniBatchSparsePCA,
):
    """Stub for DF wrapper of class ``MiniBatchSparsePCA``"""


class NMFDF(ComponentsDimensionalityReductionWrapperDF[NMF], native=NMF):
    """Stub for DF wrapper of class ``NMF``"""


class PCADF(NComponentsDimensionalityReductionWrapperDF[PCA], native=PCA):
    """Stub for DF wrapper of class ``PCA``"""


class SparseCoderDF(
    ComponentsDimensionalityReductionWrapperDF[SparseCoder],
    native=SparseCoder,
):
    """Stub for DF wrapper of class ``SparseCoder``"""


class SparsePCADF(
    ComponentsDimensionalityReductionWrapperDF[SparsePCA],
    native=SparsePCA,
):
    """Stub for DF wrapper of class ``SparsePCA``"""


class SparseRandomProjectionDF(
    ComponentsDimensionalityReductionWrapperDF[SparseRandomProjection],
    native=SparseRandomProjection,
):
    """Stub for DF wrapper of class ``SparseRandomProjection``"""


class TruncatedSVDDF(
    ComponentsDimensionalityReductionWrapperDF[TruncatedSVD],
    native=TruncatedSVD,
):
    """Stub for DF wrapper of class ``TruncatedSVD``"""


#
# Transformers which have an n_components attribute
# Implemented through NComponentsDimensionalityReductionWrapperDF
#


class KernelPCADF(
    NComponentsDimensionalityReductionWrapperDF[KernelPCA],
    native=KernelPCA,
):
    """Stub for DF wrapper of class ``KernelPCA``"""


class LocallyLinearEmbeddingDF(
    NComponentsDimensionalityReductionWrapperDF[LocallyLinearEmbedding],
    native=LocallyLinearEmbedding,
):
    """Stub for DF wrapper of class ``LocallyLinearEmbedding``"""


class NystroemDF(
    NComponentsDimensionalityReductionWrapperDF[Nystroem],
    native=Nystroem,
):
    """Stub for DF wrapper of class ``Nystroem``"""


class RBFSamplerDF(
    NComponentsDimensionalityReductionWrapperDF[RBFSampler],
    native=RBFSampler,
):
    """Stub for DF wrapper of class ``RBFSampler``"""


class SkewedChi2SamplerDF(
    NComponentsDimensionalityReductionWrapperDF[SkewedChi2Sampler],
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
    native=VarianceThreshold,
):
    """Stub for DF wrapper of class ``VarianceThreshold``"""


class RFEDF(FeatureSelectionWrapperDF[RFE], native=RFE):
    """Stub for DF wrapper of class ``RFE``"""


class RFECVDF(FeatureSelectionWrapperDF[RFECV], native=RFECV):
    """Stub for DF wrapper of class ``RFECV``"""


class SelectFromModelDF(
    FeatureSelectionWrapperDF[SelectFromModel],
    native=SelectFromModel,
):
    """Stub for DF wrapper of class ``SelectFromModel``"""


class SelectPercentileDF(
    FeatureSelectionWrapperDF[SelectPercentile],
    native=SelectPercentile,
):
    """Stub for DF wrapper of class ``SelectPercentile``"""


class SelectKBestDF(
    FeatureSelectionWrapperDF[SelectKBest],
    native=SelectKBest,
):
    """Stub for DF wrapper of class ``SelectKBest``"""


class SelectFprDF(FeatureSelectionWrapperDF[SelectFpr], native=SelectFpr):
    """Stub for DF wrapper of class ``SelectFpr``"""


class SelectFdrDF(FeatureSelectionWrapperDF[SelectFdr], native=SelectFdr):
    """Stub for DF wrapper of class ``SelectFdr``"""


class SelectFweDF(FeatureSelectionWrapperDF[SelectFwe], native=SelectFwe):
    """Stub for DF wrapper of class ``SelectFwe``"""


class GenericUnivariateSelectDF(
    FeatureSelectionWrapperDF[GenericUnivariateSelect],
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
