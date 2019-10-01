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
GAMMA custom pipelines
"""

import abc as _abc
import logging as _logging
import typing as _t

import pandas as _pd
import sklearn.base as _sb

import gamma.sklearndf as _sdf

log = _logging.getLogger(__name__)

__all__ = [
    "EstimatorPipelineDF",
    "LearnerPipelineDF",
    "RegressorPipelineDF",
    "ClassifierPipelineDF",
]

_T_FinalEstimatorDF = _t.TypeVar("_T_FinalEstimatorDF", bound=_sdf.BaseEstimatorDF)
_T_FinalLearnerDF = _t.TypeVar("_T_FinalLearnerDF", bound=_sdf.BaseLearnerDF)
_T_FinalRegressorDF = _t.TypeVar("_T_FinalRegressorDF", bound=_sdf.RegressorDF)
_T_FinalClassifierDF = _t.TypeVar("_T_FinalClassifierDF", bound=_sdf.ClassifierDF)


class EstimatorPipelineDF(
    _sdf.BaseEstimatorDF, _sb.BaseEstimator, _t.Generic[_T_FinalEstimatorDF], _abc.ABC
):
    """
    A data frame enabled pipeline with an optional preprocessing step and a
    mandatory estimator step.

    :param preprocessing: the preprocessing step in the pipeline (defaults to ``None``)
    """

    def __init__(self, preprocessing: _t.Optional[_sdf.TransformerDF] = None) -> None:
        super().__init__()

        if preprocessing is not None and not isinstance(
            preprocessing, _sdf.TransformerDF
        ):
            raise TypeError(
                "arg preprocessing expected to be a TransformerDF but is a "
                f"{type(preprocessing).__name__}"
            )

        self.preprocessing = preprocessing

    @property
    @_abc.abstractmethod
    def final_estimator_(self) -> _T_FinalEstimatorDF:
        """
        The final estimator following the preprocessing step.
        """
        pass

    @property
    def preprocessing_param_(self) -> str:
        """
        The name of the preprocessing step parameter.
        """
        return "preprocessing"

    @property
    @_abc.abstractmethod
    def final_estimator_param_(self) -> str:
        """
        The name of the estimator step parameter.
        """
        pass

    # noinspection PyPep8Naming
    def fit(
        self,
        X: _pd.DataFrame,
        y: _t.Optional[_t.Union[_pd.Series, _pd.DataFrame]] = None,
        **fit_params,
    ) -> "EstimatorPipelineDF[_T_FinalEstimatorDF]":
        self.final_estimator_.fit(
            self._pre_fit_transform(X, y, **fit_params), y, **fit_params
        )
        return self

    @property
    def is_fitted(self) -> bool:
        return self.preprocessing.is_fitted and self.final_estimator_.is_fitted

    def _get_features_in(self) -> _pd.Index:
        if self.preprocessing is not None:
            return self.preprocessing.features_in
        else:
            return self.final_estimator_.features_in

    # noinspection PyPep8Naming
    def _pre_transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        if self.preprocessing is not None:
            return self.preprocessing.transform(X)
        else:
            return X

    # noinspection PyPep8Naming
    def _pre_fit_transform(
        self, X: _pd.DataFrame, y: _pd.Series, **fit_params
    ) -> _pd.DataFrame:
        if self.preprocessing is not None:
            return self.preprocessing.fit_transform(X, y, **fit_params)
        else:
            return X


class LearnerPipelineDF(
    EstimatorPipelineDF[_T_FinalLearnerDF], _t.Generic[_T_FinalLearnerDF], _abc.ABC
):

    # noinspection PyPep8Naming
    def predict(
        self, X: _pd.DataFrame, **predict_params
    ) -> _t.Union[_pd.Series, _pd.DataFrame]:
        return self.final_estimator_.predict(self._pre_transform(X), **predict_params)

    # noinspection PyPep8Naming
    def fit_predict(self, X: _pd.DataFrame, y: _pd.Series, **fit_params) -> _pd.Series:
        return self.final_estimator_.fit_predict(
            self._pre_fit_transform(X, y, **fit_params), y, **fit_params
        )

    # noinspection PyPep8Naming
    def score(
        self,
        X: _pd.DataFrame,
        y: _t.Optional[_pd.Series] = None,
        sample_weight: _t.Optional[_t.Any] = None,
    ) -> float:
        if sample_weight is None:
            return self.final_estimator_.score(self._pre_transform(X), y)
        else:
            return self.final_estimator_.score(
                self._pre_transform(X), y, sample_weight=sample_weight
            )

    @property
    def n_outputs(self) -> int:
        return self.final_estimator_.n_outputs


class RegressorPipelineDF(
    LearnerPipelineDF[_T_FinalRegressorDF],
    _sdf.RegressorDF,
    _t.Generic[_T_FinalRegressorDF],
):
    """
    A data frame enabled pipeline with an optional preprocessing step and a
    mandatory regression step.

    :param preprocessing: the preprocessing step in the pipeline (defaults to ``None``)
    :param regressor: the classifier used in the pipeline
    :type regressor: :class:`.RegressorDF`
    """

    def __init__(
        self,
        regressor: _T_FinalRegressorDF,
        preprocessing: _t.Optional[_sdf.TransformerDF] = None,
    ) -> None:
        super().__init__(preprocessing=preprocessing)

        if not isinstance(regressor, _sdf.RegressorDF):
            raise TypeError(
                f"arg regressor expected to be a {_sdf.RegressorDF.__name__} but is a "
                f"{type(regressor).__name__}"
            )

        self.regressor = regressor

    @property
    def final_estimator_(self) -> _T_FinalRegressorDF:
        return self.regressor

    @property
    def final_estimator_param_(self) -> str:
        return "regressor"


class ClassifierPipelineDF(
    LearnerPipelineDF[_T_FinalClassifierDF],
    _sdf.ClassifierDF,
    _t.Generic[_T_FinalClassifierDF],
):
    """
    A data frame enabled pipeline with an optional preprocessing step and a
    mandatory classification step.

    :param preprocessing: the preprocessing step in the pipeline (defaults to ``None``)
    :param classifier: the classifier used in the pipeline
    :type classifier: :class:`.ClassifierDF`
    """

    def __init__(
        self,
        classifier: _T_FinalClassifierDF,
        preprocessing: _t.Optional[_sdf.TransformerDF] = None,
    ) -> None:
        super().__init__(preprocessing=preprocessing)

        if not isinstance(classifier, _sdf.ClassifierDF):
            raise TypeError(
                f"arg predictor expected to be a {_sdf.ClassifierDF.__name__} but is a "
                f"{type(classifier).__name__}"
            )
        self.classifier = classifier

    @property
    def final_estimator_(self) -> _T_FinalClassifierDF:
        return self.classifier

    @property
    def final_estimator_param_(self) -> str:
        return "classifier"

    # noinspection PyPep8Naming
    def predict_proba(
        self, X: _pd.DataFrame
    ) -> _t.Union[_pd.DataFrame, _t.List[_pd.DataFrame]]:
        return self.classifier.predict_proba(self._pre_transform(X))

    # noinspection PyPep8Naming
    def predict_log_proba(
        self, X: _pd.DataFrame
    ) -> _t.Union[_pd.DataFrame, _t.List[_pd.DataFrame]]:
        return self.classifier.predict_log_proba(self._pre_transform(X))

    # noinspection PyPep8Naming
    def decision_function(
        self, X: _pd.DataFrame
    ) -> _t.Union[_pd.Series, _pd.DataFrame]:
        return self.classifier.decision_function(self._pre_transform(X))
