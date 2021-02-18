"""
Core implementation of :mod:`sklearndf`
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, List, Mapping, Optional, Sequence, Type, TypeVar, Union, cast

import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)

from pytools.api import AllTracker
from pytools.fit import FittableMixin

log = logging.getLogger(__name__)

__all__ = ["EstimatorDF", "LearnerDF", "ClassifierDF", "RegressorDF", "TransformerDF"]

#
# type variables
#

T_Self = TypeVar("T_Self")
T_EstimatorDF = TypeVar("T_EstimatorDF")


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class EstimatorDF(FittableMixin[pd.DataFrame], metaclass=ABCMeta):
    """
    Base class for augmented scikit-learn `estimators`.

    Provides enhanced support for data frames.
    """

    #: Name assigned to an :class:`~pandas.Index` or a :class:`~pandas.Series`
    #: containing the names of the features used to fit a :class:`.EstimatorDF`.
    #:
    #: See :meth:`.feature_names_in_` and
    #: :meth:`~.TransformerDF.feature_names_original_`.
    COL_FEATURE_IN = "feature_in"

    def __new__(cls: Type["EstimatorDF"], *args, **kwargs) -> object:
        # make sure this DF estimator also is a subclass of
        if not issubclass(cls, BaseEstimator):
            raise TypeError(
                f"class {cls.__name__} is required to be "
                f"a subclass of {BaseEstimator.__name__}"
            )

        return super().__new__(cls)

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

        :param X: input data frame with observations as rows and features as columns
        :param y: an optional series or data frame with one or more outputs
        :param fit_params: additional keyword parameters as required by specific
            estimator implementations
        :return: ``self``
        """
        pass

    @property
    def feature_names_in_(self) -> pd.Index:
        """
        The pandas column index with the names of the features used to fit this
        estimator.

        :raises AttributeError: if this estimator is not fitted
        """
        self._ensure_fitted()
        return self._get_features_in().rename(self.COL_FEATURE_IN)

    @property
    def n_outputs_(self) -> int:
        """
        The number of outputs used to fit this estimator.

        :raises AttributeError: if this estimator is not fitted
        """
        self._ensure_fitted()
        return self._get_n_outputs()

    def get_params(self, deep=True) -> Mapping[str, Any]:
        """
        Get the parameters for this estimator.

        :param deep: if ``True``, return the parameters for this estimator, and
            for any sub-estimators contained in this estimator
        :return: a mapping of parameter names to their values
        """
        # noinspection PyUnresolvedReferences
        return super().get_params(deep=deep)

    def set_params(self: T_Self, **kwargs) -> T_Self:
        """
        Set the parameters of this estimator.

        Valid parameter keys can be obtained by calling :meth:`.get_params`.

        :return: ``self``
        """
        # noinspection PyUnresolvedReferences
        return super().set_params(**kwargs)

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


class LearnerDF(EstimatorDF, metaclass=ABCMeta):
    """
    Base class for augmented scikit-learn `learners`.

    Provides enhanced support for data frames.
    """

    # noinspection PyPep8Naming
    @abstractmethod
    def predict(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Predict outputs for the given inputs.

        The inputs must have the same features as the inputs used to fit
        this learner.
        The features can be provided in any order since they are identified by their
        column names.

        :param X: input data frame with observations as rows and features as columns
        :param predict_params: optional keyword parameters as required by specific
            learner implementations
        :return: predictions per observation as a series, or as a data frame in case
            of multiple outputs
        """
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def fit_predict(
        self, X: pd.DataFrame, y: pd.Series, **fit_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Fit this learner using the given inputs, then predict the outputs.

        :param X: data frame with observations as rows and features as columns
        :param y: a series or data frame with one or more outputs per observation
        :param fit_params: optional keyword parameters as required by specific
            learner implementations
        :return: predictions per observation as a series, or as a data frame in case
            of multiple outputs
        """
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def score(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None
    ) -> float:
        """
        Score this learner using the given inputs and outputs.

        :param X: data frame with observations as rows and features as columns
        :param y: a series or data frame with the true outputs per observation
        :param sample_weight: optional series of scalar weights, for calculating the
            resulting score as the weighted mean of the scores for the individual
            predictions
        """
        pass


class TransformerDF(EstimatorDF, TransformerMixin, metaclass=ABCMeta):
    """
    Base class for augmented scikit-learn `transformers`.

    Provides enhanced support for data frames.
    """

    #: Name assigned to a :class:`~pandas.Index` containing the names of the features
    #: produced by a :class:`.TransformerDF`.
    #:
    #: See :meth:`~.TransformerDF.feature_names_out_` and
    #: :meth:`.feature_names_original_`.
    COL_FEATURE_OUT = "feature_out"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._features_original = None

    @property
    def feature_names_original_(self) -> pd.Series:
        """
        A pandas series, mapping the output features resulting from the transformation
        to the original input features.

        The index of the resulting series consists of the names of the output features;
        the corresponding values are the names of the original input features.
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
    def feature_names_out_(self) -> pd.Index:
        """
        A pandas column index with the names of the features produced by this
        transformer
        """
        self._ensure_fitted()
        return self._get_features_out().rename(self.COL_FEATURE_OUT)

    # noinspection PyPep8Naming
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the given inputs.

        The inputs must have the same features as the inputs used to fit
        this transformer.
        The features can be provided in any order since they are identified by their
        column names.

        :param X: input data frame with observations as rows and features as columns
        :return: the transformed inputs
        """
        pass

    # noinspection PyPep8Naming
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        """
        Fit this transformer using the given inputs, then transform the inputs.

        :param X: input data frame with observations as rows and features as columns
        :param y: an optional series or data frame with one or more outputs
        :param fit_params: additional keyword parameters as required by specific
            transformer implementations
        :return: the transformed inputs
        """
        return self.fit(X, y, **fit_params).transform(X)

    # noinspection PyPep8Naming
    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse-transform the given inputs.

        The inputs must have the same features as the inputs used to fit
        this transformer.
        The features can be provided in any order since they are identified by their
        column names.

        :param X: input data frame with observations as rows and features as columns
        :return: the reverse-transformed inputs
        """
        pass

    @abstractmethod
    def _get_features_original(self) -> pd.Series:
        # return a mapping from this transformer's output columns to the original
        # columns as a series
        pass

    def _get_features_out(self) -> pd.Index:
        # return a pandas index with this transformer's output columns
        # default behaviour: get index returned by feature_names_original_
        return self.feature_names_original_.index


class RegressorDF(LearnerDF, RegressorMixin, metaclass=ABCMeta):
    """
    Base class for augmented scikit-learn `regressors`.

    Provides enhanced support for data frames.
    """


class ClassifierDF(LearnerDF, ClassifierMixin, metaclass=ABCMeta):
    """
    Base class for augmented scikit-learn `classifiers`.

    Provides enhanced support for data frames.
    """

    @property
    @abstractmethod
    def classes_(self) -> Sequence[Any]:
        """
        Get the classes predicted by this classifier.
        By default expects classes as a list-like stored in the `classes_` attribute.

        :return: the classes predicted by this classifier
        """
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def predict_proba(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Predict class probabilities for the given inputs.

        The inputs must have the same features as the inputs used to fit
        this learner.
        The features can be provided in any order since they are identified by their
        column names.

        :param X: input data frame with observations as rows and features as columns
        :param predict_params: optional keyword parameters as required by specific
            learner implementations
        :return: a data frame with observations as rows and classes as columns, and
            values as probabilities per observation and class; for multi-output
            classifiers, a list of one observation/class data frames per output
        """
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def predict_log_proba(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Predict class log-probabilities for the given inputs.

        The inputs must have the same features as the inputs used to fit
        this learner.
        The features can be provided in any order since they are identified by their
        column names.

        :param X: input data frame with observations as rows and features as columns
        :param predict_params: optional keyword parameters as required by specific
            learner implementations
        :return: a data frame with observations as rows and classes as columns, and
            values as log-probabilities per observation and class; for multi-output
            classifiers, a list of one observation/class data frames per output
        """
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def decision_function(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute the decision function for the given inputs.

        The inputs must have the same features as the inputs used to fit
        this learner.
        The features can be provided in any order since they are identified by their
        column names.

        :param X: input data frame with observations as rows and features as columns
        :param predict_params: optional keyword parameters as required by specific
            learner implementations
        :return: a data frame with observations as rows and classes as columns, and
            values as the raw values predicted per observation and class;
            for multi-output classifiers, a list of one observation/class data frames
            per output
        """


__tracker.validate()
