#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
Extended versions of scikit-learn :class:`~sklearn.pipeline.Pipeline` and
:class:`~sklearn.pipeline.FeatureUnion`, providing enhanced support for data frames
"""

import abc as _abc
import logging as _logging
import typing as _t

import pandas as pd
import pandas.core.arrays as _pda
import sklearn.pipeline as _ppl

import gamma.sklearndf as _sdf
import gamma.sklearndf._wrapper as _wr
from gamma.sklearndf.pipeline import _model
from ._model import *

log = _logging.getLogger(__name__)

__all__ = ["PipelineDF", "FeatureUnionDF", *_model.__all__]


class _PipelineWrapperDF(
    _wr.ClassifierWrapperDF[_ppl.Pipeline],
    _wr.RegressorWrapperDF[_ppl.Pipeline],
    _wr.TransformerWrapperDF[_ppl.Pipeline],
    _abc.ABC,
):
    PASSTHROUGH = "passthrough"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # ensure that all steps support data frames, and that all except the last
        # step are data frame transformers

        steps = self.steps

        if len(steps) == 0:
            return

        for name, transformer in steps[:-1]:
            if not (
                self._is_passthrough(transformer)
                or isinstance(transformer, _sdf.TransformerDF)
            ):
                raise ValueError(
                    f'expected step "{name}" to contain a '
                    f"{_sdf.TransformerDF.__name__}, but found an instance of "
                    f"{type(transformer).__name__}"
                )

        final_step = steps[-1]
        final_estimator = final_step[1]
        if not (
            self._is_passthrough(final_estimator)
            or isinstance(final_estimator, _sdf.BaseEstimatorDF)
        ):
            raise ValueError(
                f'expected final step "{final_step[0]}" to contain a '
                f"{_sdf.BaseEstimatorDF.__name__}, but found an instance of "
                f"{type(final_estimator).__name__}"
            )

    @property
    def steps(self) -> _t.List[_t.Tuple[str, _sdf.BaseEstimatorDF]]:
        """
        The ``steps`` attribute of the underlying :class:`~sklearn.pipeline.Pipeline`.

        List of (name, transformer) tuples (transformers implement fit/transform).
        """
        return self.delegate_estimator.steps

    def __len__(self) -> int:
        """The number of steps of the pipeline."""
        return len(self.delegate_estimator.steps)

    def __getitem__(self, ind: _t.Union[slice, int, str]) -> _sdf.BaseEstimatorDF:
        """
        Return a sub-pipeline or a single estimator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `steps` will not change a copy.
        """

        if isinstance(ind, slice):
            base_pipeline = self.delegate_estimator
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")

            return _t.cast(
                _sdf.BaseEstimatorDF,
                self.__class__(
                    steps=base_pipeline.steps[ind],
                    memory=base_pipeline.memory,
                    verbose=base_pipeline.verbose,
                ),
            )
        else:
            return self.delegate_estimator[ind]

    @staticmethod
    def _is_passthrough(estimator: _t.Union[_sdf.BaseEstimatorDF, str, None]) -> bool:
        # return True if the estimator is a "passthrough" (i.e. identity) transformer
        # in the pipeline
        return estimator is None or estimator == _PipelineWrapperDF.PASSTHROUGH

    def _transformer_steps(self) -> _t.Iterator[_t.Tuple[str, _sdf.TransformerDF]]:
        # make an iterator of all transform steps, i.e. excluding the final step
        # in case it is not a transformer
        # excludes steps whose transformer is `None` or `"passthrough"`

        def _iter_not_none(
            transformer_steps: _t.Sequence[_t.Tuple[str, _sdf.BaseEstimatorDF]]
        ) -> _t.Iterator[_t.Tuple[str, _sdf.TransformerDF]]:
            return (
                (name, _t.cast(_sdf.TransformerDF, transformer))
                for name, transformer in transformer_steps
                if not self._is_passthrough(transformer)
            )

        steps = self.steps

        if len(steps) == 0:
            return iter([])

        final_estimator = steps[-1][1]

        if isinstance(final_estimator, _sdf.TransformerDF):
            return _iter_not_none(steps)
        else:
            return _iter_not_none(steps[:-1])

    def _get_features_original(self) -> pd.Series:
        col_mappings = [
            df_transformer.features_original
            for _, df_transformer in self._transformer_steps()
        ]

        if len(col_mappings) == 0:
            _features_out: pd.Index = self.features_in
            _features_original: _t.Union[
                pd.np.ndarray, _pda.ExtensionArray
            ] = _features_out.values
        else:
            _features_out: pd.Index = col_mappings[-1].index
            _features_original: _t.Union[
                pd.np.ndarray, _pda.ExtensionArray
            ] = col_mappings[-1].values

            # iterate backwards starting from the penultimate item
            for preceding_out_to_original_mapping in col_mappings[-2::-1]:
                # join the original columns of my current transformer on the out columns
                # in the preceding transformer, then repeat
                _features_original = preceding_out_to_original_mapping.loc[
                    _features_original
                ].values

        return pd.Series(index=_features_out, data=_features_original)

    def _get_features_out(self) -> pd.Index:
        for _, transformer in reversed(self.steps):
            if isinstance(transformer, _sdf.TransformerDF):
                return transformer.features_out

        return self.features_in


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_PipelineWrapperDF)
class PipelineDF(
    _sdf.ClassifierDF, _sdf.RegressorDF, _sdf.TransformerDF, _ppl.Pipeline
):
    """
    Wraps :class:`sklearn.pipeline.Pipeline`; accepts and returns data
    frames.
    """

    pass


class _FeatureUnionWrapperDF(_wr.TransformerWrapperDF[_ppl.FeatureUnion], _abc.ABC):
    @staticmethod
    def _prepend_features_out(features_out: pd.Index, name_prefix: str) -> pd.Index:
        return pd.Index(data=f"{name_prefix}__" + features_out.astype(str))

    def _get_features_original(self) -> pd.Series:
        # concatenate output->input mappings from all included transformers other than
        # ones stated as `None` or `"drop"` or any other string

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
                    features_original=transformer.features_original, name_prefix=name
                )
                for name, transformer, _ in self.delegate_estimator._iter()
            )
        )

    def _get_features_out(self) -> pd.Index:
        # concatenate output columns from all included transformers other than
        # ones stated as `None` or `"drop"` or any other string

        # prepend the name of the transformer so the resulting feature name is
        # `<name>__<output column of sub-transformer>

        # noinspection PyProtectedMember
        indices = [
            self._prepend_features_out(
                features_out=transformer.features_out, name_prefix=name
            )
            for name, transformer, _ in self.delegate_estimator._iter()
        ]

        if len(indices) == 0:
            return pd.Index()
        else:
            return indices[0].append(other=indices[1:])


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_FeatureUnionWrapperDF)
class FeatureUnionDF(_sdf.TransformerDF, _ppl.FeatureUnion):
    """
    Wraps :class:`sklearn.pipeline.FeatureUnion` for enhanced support of pandas data
    frames.
    """

    pass
