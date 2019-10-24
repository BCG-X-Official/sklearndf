"""
GAMMA custom two-step pipelines
"""

import abc as _abc
import logging as _logging
import typing as _t

import pandas as _pd
import sklearn.base as _sb

import gamma.sklearndf as _sdf

log = _logging.getLogger(__name__)

__all__ = ["BaseLearnerPipelineDF", "RegressorPipelineDF", "ClassifierPipelineDF"]

T_FinalEstimatorDF = _t.TypeVar("T_FinalEstimatorDF", bound=_sdf.BaseEstimatorDF)
T_FinalLearnerDF = _t.TypeVar("T_FinalLearnerDF", bound=_sdf.BaseLearnerDF)
T_FinalRegressorDF = _t.TypeVar("T_FinalRegressorDF", bound=_sdf.RegressorDF)
T_FinalClassifierDF = _t.TypeVar("T_FinalClassifierDF", bound=_sdf.ClassifierDF)


class BaseEstimatorPipelineDF(
    _sb.BaseEstimator, _sdf.BaseEstimatorDF, _t.Generic[T_FinalEstimatorDF], _abc.ABC
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
    def final_estimator(self) -> T_FinalEstimatorDF:
        """
        The final estimator following the preprocessing step.
        """
        pass

    @property
    def preprocessing_name(self) -> str:
        """
        The name of the preprocessing step parameter.
        """
        return "preprocessing"

    @property
    @_abc.abstractmethod
    def final_estimator_name(self) -> str:
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
    ) -> "BaseEstimatorPipelineDF[T_FinalEstimatorDF]":
        self.final_estimator.fit(
            self._pre_fit_transform(X, y, **fit_params), y, **fit_params
        )
        return self

    @property
    def is_fitted(self) -> bool:
        return (
            self.preprocessing is None or self.preprocessing.is_fitted
        ) and self.final_estimator.is_fitted

    def _get_features_in(self) -> _pd.Index:
        if self.preprocessing is not None:
            return self.preprocessing.features_in
        else:
            return self.final_estimator.features_in

    def _get_n_outputs(self) -> int:
        if self.preprocessing is not None:
            return self.preprocessing.n_outputs
        else:
            return self.final_estimator.n_outputs

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


class BaseLearnerPipelineDF(
    BaseEstimatorPipelineDF[T_FinalLearnerDF], _t.Generic[T_FinalLearnerDF], _abc.ABC
):

    # noinspection PyPep8Naming
    def predict(
        self, X: _pd.DataFrame, **predict_params
    ) -> _t.Union[_pd.Series, _pd.DataFrame]:
        return self.final_estimator.predict(self._pre_transform(X), **predict_params)

    # noinspection PyPep8Naming
    def fit_predict(self, X: _pd.DataFrame, y: _pd.Series, **fit_params) -> _pd.Series:
        return self.final_estimator.fit_predict(
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
            return self.final_estimator.score(self._pre_transform(X), y)
        else:
            return self.final_estimator.score(
                self._pre_transform(X), y, sample_weight=sample_weight
            )


class RegressorPipelineDF(
    BaseLearnerPipelineDF[T_FinalRegressorDF],
    _sdf.RegressorDF,
    _t.Generic[T_FinalRegressorDF],
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
        regressor: T_FinalRegressorDF,
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
    def final_estimator(self) -> T_FinalRegressorDF:
        return self.regressor

    @property
    def final_estimator_name(self) -> str:
        return "regressor"


class ClassifierPipelineDF(
    BaseLearnerPipelineDF[T_FinalClassifierDF],
    _sdf.ClassifierDF,
    _t.Generic[T_FinalClassifierDF],
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
        classifier: T_FinalClassifierDF,
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
    def final_estimator(self) -> T_FinalClassifierDF:
        return self.classifier

    @property
    def final_estimator_name(self) -> str:
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
