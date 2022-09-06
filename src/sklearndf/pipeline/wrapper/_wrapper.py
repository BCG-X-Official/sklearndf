"""
Core implementation of :mod:`sklearndf.pipeline.wrapper`
"""

import logging
from abc import ABCMeta
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union, cast

import numpy.typing as npt
import pandas as pd
from pandas.core.arrays import ExtensionArray
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from pytools.api import AllTracker

from sklearndf import EstimatorDF, TransformerDF
from sklearndf.wrapper import (
    ClassifierWrapperDF,
    RegressorWrapperDF,
    TransformerWrapperDF,
)

log = logging.getLogger(__name__)

__all__ = ["PipelineWrapperDF", "FeatureUnionWrapperDF"]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class PipelineWrapperDF(
    ClassifierWrapperDF[Pipeline],
    RegressorWrapperDF[Pipeline],
    TransformerWrapperDF[Pipeline],
    metaclass=ABCMeta,
):
    """
    DF wrapper for `scikit-learn` class :class:`~sklearn.pipeline.Pipeline`.
    """

    __native_base_class__ = Pipeline

    #: Placeholder that can be used in place of an estimator to designate a pipeline
    #: step that preserves the original ingoing data.
    PASSTHROUGH = "passthrough"

    def _validate_delegate_estimator(self) -> None:
        # ensure that all steps support data frames, and that all except the last
        # step are data frame transformers

        steps = self.steps

        if len(steps) == 0:
            return

        for name, transformer in steps[:-1]:
            if not (
                self._is_passthrough(transformer)
                or isinstance(transformer, TransformerDF)
            ):
                raise ValueError(
                    f"expected step {name!r} to be a {TransformerDF.__name__}, "
                    f"or {PipelineWrapperDF.PASSTHROUGH}, but found an instance of "
                    f"{type(transformer).__name__}"
                )

        final_step = steps[-1]
        final_estimator = final_step[1]
        if not (
            self._is_passthrough(final_estimator)
            or isinstance(final_estimator, EstimatorDF)
        ):
            raise ValueError(
                f"expected final step {final_step[0]!r} to be an "
                f"{EstimatorDF.__name__} or {PipelineWrapperDF.PASSTHROUGH}, "
                f"but found an instance of {type(final_estimator).__name__}"
            )

    @property
    def steps(self) -> List[Tuple[str, EstimatorDF]]:
        """
        The ``steps`` attribute of the underlying :class:`~sklearn.pipeline.Pipeline`.

        List of (name, transformer) tuples (transformers implement fit/transform).
        """
        return cast(List[Tuple[str, EstimatorDF]], self.native_estimator.steps)

    def __len__(self) -> int:
        """The number of steps of the pipeline."""
        return len(self.native_estimator.steps)

    def __getitem__(self, ind: Union[slice, int, str]) -> EstimatorDF:
        """
        Return a sub-pipeline or a single estimator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in ``steps`` will not change a copy.
        """

        if isinstance(ind, slice):
            base_pipeline = self.native_estimator
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")

            return cast(
                EstimatorDF,
                self.__class__(
                    steps=base_pipeline.steps[ind],
                    memory=base_pipeline.memory,
                    verbose=base_pipeline.verbose,
                ),
            )
        else:
            return cast(EstimatorDF, self.native_estimator[ind])

    @staticmethod
    def _is_passthrough(estimator: Union[EstimatorDF, str, None]) -> bool:
        # return True if the estimator is a "passthrough" (i.e. identity) transformer
        # in the pipeline
        return estimator is None or estimator == PipelineWrapperDF.PASSTHROUGH

    def _transformer_steps(self) -> Iterator[Tuple[str, TransformerDF]]:
        # make an iterator of all transform steps, i.e., excluding the final step
        # in case it is not a transformer
        # excludes steps whose transformer is ``None`` or ``"passthrough"``

        def _iter_not_none(
            transformer_steps: Sequence[Tuple[str, EstimatorDF]]
        ) -> Iterator[Tuple[str, TransformerDF]]:
            return (
                (name, cast(TransformerDF, transformer))
                for name, transformer in transformer_steps
                if not self._is_passthrough(transformer)
            )

        steps = self.steps

        if len(steps) == 0:
            return iter([])

        final_estimator = steps[-1][1]

        if isinstance(final_estimator, TransformerDF):
            return _iter_not_none(steps)
        else:
            return _iter_not_none(steps[:-1])

    def _get_features_original(self) -> pd.Series:
        col_mappings = [
            df_transformer.feature_names_original_
            for _, df_transformer in self._transformer_steps()
        ]

        _features_out: pd.Index
        _features_original: Union[npt.NDArray[Any], ExtensionArray]

        if len(col_mappings) == 0:
            _features_out = self.feature_names_in_
            _features_original = _features_out.values
        else:
            _features_out = col_mappings[-1].index
            _features_original = col_mappings[-1].values

            # iterate backwards starting from the penultimate item
            for preceding_out_to_original_mapping in col_mappings[-2::-1]:
                # join the original columns of my current transformer on the out columns
                # in the preceding transformer, then repeat
                if not all(
                    feature in preceding_out_to_original_mapping
                    for feature in _features_original
                ):
                    unknown_features = set(_features_original) - set(
                        preceding_out_to_original_mapping
                    )
                    raise KeyError(
                        f"unknown features encountered while tracing original "
                        f"features along pipeline: {unknown_features}"
                    )
                _features_original = preceding_out_to_original_mapping.loc[
                    _features_original
                ].values

        return pd.Series(index=_features_out, data=_features_original)

    def _get_features_out(self) -> pd.Index:
        for _, transformer in reversed(self.steps):
            if isinstance(transformer, TransformerDF):
                return transformer.feature_names_out_

        return self.feature_names_in_

    @property
    def _estimator_type(self) -> str:
        # noinspection PyProtectedMember
        return cast(str, self.native_estimator._estimator_type)

    def _more_tags(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any], getattr(self.native_estimator, "_more_tags", lambda: {})()
        )


class FeatureUnionWrapperDF(TransformerWrapperDF[FeatureUnion], metaclass=ABCMeta):
    """
    DF wrapper for `scikit-learn` class :class:`~sklearn.pipeline.FeatureUnion`.
    """

    DROP = "drop"
    PASSTHROUGH = "passthrough"

    @staticmethod
    def _prepend_features_out(features_out: pd.Index, name_prefix: str) -> pd.Index:
        return pd.Index(data=f"{name_prefix}__" + features_out.astype(str))

    def _get_features_original(self) -> pd.Series:
        # concatenate output-to-input mappings from all included transformers other than
        # ones stated as ``None`` or ``"drop"`` or any other string

        # prepend the name of the transformer so the resulting feature name is
        # `<name>__<output column of sub-transformer>

        def _prepend_features_original(
            features_original: pd.Series, name_prefix: str
        ) -> pd.Series:
            return pd.Series(
                data=features_original.values,
                index=self._prepend_features_out(
                    features_out=features_original.index, name_prefix=name_prefix
                ),
            )

        # noinspection PyProtectedMember
        return pd.concat(
            objs=(
                _prepend_features_original(
                    features_original=transformer.feature_names_original_,
                    name_prefix=name,
                )
                for name, transformer, _ in self.native_estimator._iter()
            )
        )

    def _get_features_out(self) -> pd.Index:
        # concatenate output columns from all included transformers other than
        # ones stated as ``None`` or ``"drop"`` or any other string

        # prepend the name of the transformer so the resulting feature name is
        # `<name>__<output column of sub-transformer>

        name: str
        transformer: Union[TransformerDF, str, FunctionTransformer]

        indices = [
            self._prepend_features_out(
                features_out=(
                    self._get_features_in()
                    if (
                        isinstance(transformer, FunctionTransformer)
                        and transformer.func is None
                    )
                    else cast(TransformerDF, transformer).feature_names_out_
                ),
                name_prefix=name,
            )
            for name, transformer in self.native_estimator.transformer_list
            if transformer != FeatureUnionWrapperDF.DROP
        ]

        if len(indices) == 0:
            return pd.Index()
        else:
            return indices[0].append(indices[1:])


#
# Validate __all__
#

__tracker.validate()
