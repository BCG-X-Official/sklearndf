"""
Core implementation of :mod:`gamma.sklearndf`
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import *

import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)

from gamma.common.fit import FittableMixin, T_Self

log = logging.getLogger(__name__)

__all__ = [
    "BaseEstimatorDF",
    "LearnerDF",
    "ClassifierDF",
    "RegressorDF",
    "TransformerDF",
]

#
# type variables
#

T_EstimatorDF = TypeVar("T_EstimatorDF")

#
# class definitions
#


class BaseEstimatorDF(FittableMixin[pd.DataFrame], metaclass=ABCMeta):
    """
    Mix-in class for scikit-learn estimators with enhanced support for data frames.
    """

    COL_FEATURE_IN = "feature_in"

    def __new__(cls: Type["BaseEstimatorDF"], *args, **kwargs) -> object:
        # make sure this DF estimator also is a subclass of
        if not issubclass(cls, BaseEstimator):
            raise TypeError(
                f"class {cls.__name__} is required to be "
                f"a subclass of {BaseEstimator.__name__}"
            )

        return super().__new__(cls, *args, **kwargs)

    @property
    def native_estimator(self) -> BaseEstimator:
        """
        The native estimator underlying this estimator.

        This can be another estimator that this estimator delegates to, otherwise the
        native estimator is ``self``.
        """

        return cast(BaseEstimator, self)

    # noinspection PyPep8Naming
    @abstractmethod
    def fit(
        self: T_Self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params,
    ) -> T_Self:
        """
        Fit this estimator using the given inputs.

        :param X: a data frame of observations as rows and features as columns
        :param y: an optional series or data frame with one or more outputs
        :param fit_params: additional keyword parameters as required by specific \
            estimator implementations
        :return: ``self``
        """
        pass

    @property
    def features_in(self) -> pd.Index:
        """
        The pandas column index with the names of the features used to fit this
        estimator.

        :raises AttributeError: if this estimator is not fitted
        """
        self._ensure_fitted()
        return self._get_features_in().rename(self.COL_FEATURE_IN)

    @property
    def n_outputs(self) -> int:
        """
        The number of outputs used to fit this estimator.

        :raises AttributeError: if this estimator is not fitted
        """
        self._ensure_fitted()
        return self._get_n_outputs()

    @abstractmethod
    def get_params(self, deep=True) -> Mapping[str, Any]:
        """
        Get the parameters for this estimator.

        :param deep: if ``True``, return the parameters for this estimator, and \
            for any sub-estimators contained in this estimator

        :return: a mapping of parameter names to their values
        """
        pass

    @abstractmethod
    def set_params(self: T_Self, **kwargs) -> T_Self:
        """
        Set the parameters of this estimator.

        Valid parameter keys can be obtained by calling :meth:`.get_params`.

        :returns ``self``
        """
        pass

    def clone(self: T_EstimatorDF) -> T_EstimatorDF:
        """
        Make an unfitted clone of this estimator.

        :return: the unfitted clone
        """
        return clone(self)

    @abstractmethod
    def _get_features_in(self) -> pd.Index:
        # get the input columns as a pandas Index
        pass

    @abstractmethod
    def _get_n_outputs(self) -> int:
        # get the number of outputs this estimator has been fitted to
        pass


class LearnerDF(BaseEstimatorDF, metaclass=ABCMeta):
    """
    Base class for `learners`, i.e. regressors and classifiers
    """

    # noinspection PyPep8Naming
    @abstractmethod
    def predict(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Predict outputs for the given inputs.

        The observations must have the same features as the observations used to fit
        this learner.
        The features can be provided in any order since they are identified by their
        column names.

        :param X: data frame with observations in rows and features in columns
        :param y: a series or data frame with one or more outputs per observation
        :param predict_params: optional keyword parameters as required by specific \
            learner implementations
        """
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def fit_predict(
        self, X: pd.DataFrame, y: pd.Series, **fit_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Fit this learner using the given observations, then predict the outputs.

        :param X: data frame with observations in rows and features in columns
        :param y: a series or data frame with one or more outputs per observation
        :param fit_params: optional keyword parameters as required by specific \
            learner implementations
        """
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def score(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None
    ) -> float:
        """
        Score this learner using the given inputs and outputs.

        :param X: data frame with observations in rows and features in columns
        :param y: a series or data frame with one or more outputs per observation
        :param sample_weight: optional series of scalar weights, for calculating the \
            resulting score as the weighted mean of the scores for the individual \
            predictions
        """
        pass


class TransformerDF(BaseEstimatorDF, TransformerMixin, metaclass=ABCMeta):
    """
    Mix-in class for scikit-learn transformers with enhanced support for data frames.
    """

    COL_FEATURE_OUT = "feature_out"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._features_original = None

    # noinspection PyPep8Naming
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    # noinspection PyPep8Naming
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        return self.fit(X, y, **fit_params).transform(X)

    # noinspection PyPep8Naming
    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    def features_original(self) -> pd.Series:
        """
        Pandas series mapping the output features (the series's index) to the
        original input features (the series' values)
        """
        self._ensure_fitted()
        if self._features_original is None:
            self._features_original = (
                self._get_features_original()
                .rename(self.COL_FEATURE_IN)
                .rename_axis(index=self.COL_FEATURE_OUT)
            )
        return self._features_original

    @property
    def features_out(self) -> pd.Index:
        """
        Pandas column index with the names of the features produced by this transformer
        """
        self._ensure_fitted()
        return self._get_features_out().rename(self.COL_FEATURE_OUT)

    @abstractmethod
    def _get_features_original(self) -> pd.Series:
        """
        :return: a mapping from this transformer's output columns to the original
        columns as a series
        """
        pass

    def _get_features_out(self) -> pd.Index:
        # default behaviour: get index returned by features_original
        return self.features_original.index


class RegressorDF(LearnerDF, RegressorMixin, metaclass=ABCMeta):
    """
    Mix-in class for scikit-learn regressors with enhanced support for data frames.
    """


class ClassifierDF(LearnerDF, ClassifierMixin, metaclass=ABCMeta):
    """
    Mix-in class for scikit-learn classifiers with enhanced support for data frames.
    """

    # noinspection PyPep8Naming
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def predict_log_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def decision_function(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        pass
