"""
Core implementation of :mod:`sklearndf`
"""
import inspect
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union, cast

import numpy.typing as npt
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils import is_scalar_nan

from pytools.api import AllTracker, inheritdoc
from pytools.expression import Expression, HasExpressionRepr, make_expression
from pytools.expression.atomic import Id

log = logging.getLogger(__name__)

__all__ = [
    "ClassifierDF",
    "ClusterDF",
    "EstimatorDF",
    "LearnerDF",
    "RegressorDF",
    "SupervisedLearnerDF",
    "TransformerDF",
]

#
# type variables
#

T_EstimatorDF = TypeVar("T_EstimatorDF")


#
# Constants
#

UNDEFINED = object()


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="""[see superclass]""")
class EstimatorDF(
    HasExpressionRepr,
    BaseEstimator,  # type: ignore
    metaclass=ABCMeta,
):
    """
    Base class for augmented `scikit-learn` estimators.

    Provides enhanced support for data frames.
    """

    #: Name assigned to an :class:`~pandas.Index` or a :class:`~pandas.Series`
    #: containing the names of the features used to fit a :class:`.EstimatorDF`.
    #:
    #: See :meth:`.feature_names_in_` and
    #: :meth:`~.TransformerDF.feature_names_original_`.
    COL_FEATURE_IN = "feature_in"

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
        self: T_EstimatorDF,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params: Any,
    ) -> T_EstimatorDF:
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
    @abstractmethod
    def is_fitted(self) -> bool:
        """
        ``True`` if this object is fitted, ``False`` otherwise.
        """
        pass

    def ensure_fitted(self) -> None:
        """
        Raise a :class:`~sklearn.exceptions.NotFittedError` if this estimator is not
        fitted.

        :raise sklearn.exceptions.NotFittedError: this estimator is not fitted
        """
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__} is not fitted")

    @property
    def feature_names_in_(self) -> pd.Index:
        """
        The pandas column index with the names of the features used to fit this
        estimator.

        :raises AttributeError: if this estimator is not fitted
        """
        self.ensure_fitted()
        return self._get_features_in().rename(self.COL_FEATURE_IN)

    @property
    def n_features_in_(self) -> int:
        """
        The number of features used to fit this estimator.

        :raises AttributeError: if this estimator is not fitted
        :return: the number of features
        """
        self.ensure_fitted()
        return self._get_n_features_in()

    @property
    def n_outputs_(self) -> int:
        """
        The number of outputs used to fit this estimator.

        :raises AttributeError: if this estimator is not fitted
        """
        self.ensure_fitted()
        return self._get_n_outputs()

    def get_params(self, deep: bool = True) -> Mapping[str, Any]:
        """
        Get the parameters for this estimator.

        :param deep: if ``True``, return the parameters for this estimator, and
            for any sub-estimators contained in this estimator
        :return: a mapping of parameter names to their values
        """
        # noinspection PyUnresolvedReferences
        return cast(Mapping[str, Any], super().get_params(deep=deep))

    def set_params(self: T_EstimatorDF, **params: Any) -> T_EstimatorDF:
        """
        Set the parameters of this estimator.

        Valid parameter keys can be obtained by calling :meth:`.get_params`.

        :param params: the estimator parameters to set
        :return: ``self``
        """
        # noinspection PyUnresolvedReferences
        return super().set_params(**params)  # type: ignore

    def clone(self: T_EstimatorDF) -> T_EstimatorDF:
        """
        Make an unfitted clone of this estimator.

        :return: the unfitted clone
        """
        return cast(T_EstimatorDF, clone(self))

    @abstractmethod
    def _get_features_in(self) -> pd.Index:
        # get the input columns as a pandas Index
        pass

    def _get_n_features_in(self) -> int:
        # get the number of inputs this estimator has been fitted to
        return len(self._get_features_in())

    @abstractmethod
    def _get_n_outputs(self) -> int:
        # get the number of outputs this estimator has been fitted to
        pass

    def to_expression(self) -> Expression:
        """[see superclass]"""

        # create a dictionary with all parameters of the estimator, mapping them
        # to their default values if defined (and otherwise to Signature.empty)
        estimator_parameters = {
            name: parameter.default
            for name, parameter in inspect.signature(
                self.__init__  # type: ignore
            ).parameters.items()
        }

        def _kwarg_to_expression(name: str, value: Any) -> Optional[Expression]:
            expression = make_expression(value)

            # are we dealing with a default value?
            default_value = estimator_parameters.get(name, UNDEFINED)

            if (
                (
                    # there is a parameter with the given name
                    default_value
                    is not UNDEFINED
                )
                and (
                    # the parameter has a default value
                    default_value
                    != inspect.Signature.empty
                )
                and (
                    # if the value is an estimator ...
                    not isinstance(value, BaseEstimator)
                    # ... then it has the same class as the default value
                    or value.__class__ == default_value.__class__
                )
                and (
                    # both value and default value are np.nan ...
                    (is_scalar_nan(value) and is_scalar_nan(default_value))
                    or (
                        # ... or both have the same expression.
                        # We cannot compare for equality since we don't know
                        # if the classes of the values implement this.
                        # Therefore, we compare the expressions but do this last,
                        # as it might be computationally more costly in the
                        # unlikely case that the default value is a very complex
                        # object.
                        expression.eq_(make_expression(default_value))
                    )
                )
            ):
                # we can confirm the given value is the default value - no need
                # to include in the expression representation of this estimator
                return None

            else:
                # custom value: we return the expression
                return expression

        kwarg_expressions: Dict[str, Optional[Expression]] = {
            name: _kwarg_to_expression(name, value)
            for name, value in self.get_params(deep=False).items()
        }

        return Id(type(self))(
            **{
                name: expression
                for name, expression in kwarg_expressions.items()
                if expression is not None
            }
        )


class LearnerDF(EstimatorDF, metaclass=ABCMeta):
    """
    Base class for augmented `scikit-learn` learners.

    Provides enhanced support for data frames.
    """

    # noinspection PyPep8Naming
    @abstractmethod
    def predict(
        self, X: pd.DataFrame, **predict_params: Any
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


class SupervisedLearnerDF(LearnerDF, metaclass=ABCMeta):
    """
    Base class for augmented `scikit-learn` supervised learners.

    Provides enhanced support for data frames.
    """

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
        :return: the score
        """
        pass


class TransformerDF(
    TransformerMixin,  # type: ignore
    EstimatorDF,
    metaclass=ABCMeta,
):
    """
    Base class for augmented `scikit-learn` transformers.

    Provides enhanced support for data frames.
    """

    #: Name assigned to a :class:`~pandas.Index` containing the names of the features
    #: produced by a :class:`.TransformerDF`.
    #:
    #: See :meth:`~.TransformerDF.feature_names_out_` and
    #: :meth:`.feature_names_original_`.
    COL_FEATURE_OUT = "feature_out"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        :param args: transformer parameters
        :param kwargs: transformer keyword parameters
        """
        super().__init__(*args, **kwargs)
        self._features_original = None

    @property
    def feature_names_original_(self) -> pd.Series:
        # noinspection GrazieInspection
        """
        A pandas series, mapping the output features resulting from the transformation
        to the original input features.

        The index of the resulting series consists of the names of the output features;
        the corresponding values are the names of the original input features.
        """
        self.ensure_fitted()
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
        self.ensure_fitted()
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
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params: Any,
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


class RegressorDF(
    RegressorMixin,  # type: ignore
    SupervisedLearnerDF,
    metaclass=ABCMeta,
):
    """
    Base class for augmented `scikit-learn` regressors.

    Provides enhanced support for data frames.
    """

    # noinspection PyPep8Naming
    @abstractmethod
    def score(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None
    ) -> float:
        """[see SupervisedLearnerDF]"""

    # we cannot get the docstring via the @inheritdoc mechanism because
    # RegressorMixin precedes SupervisedLearnerDF in the MRO
    score.__doc__ = SupervisedLearnerDF.score.__doc__


class ClassifierDF(
    ClassifierMixin,  # type: ignore
    SupervisedLearnerDF,
    metaclass=ABCMeta,
):
    """
    Base class for augmented `scikit-learn` classifiers.

    Provides enhanced support for data frames.
    """

    @property
    @abstractmethod
    def classes_(self) -> Union[npt.NDArray[Any], List[npt.NDArray[Any]]]:
        """
        Get the classes predicted by this classifier.

        :return: a numpy array of class labels for single-output problems, or a list
            of such arrays for multi-output problems
        """
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def predict_proba(
        self, X: pd.DataFrame, **predict_params: Any
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
        self, X: pd.DataFrame, **predict_params: Any
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
        self, X: pd.DataFrame, **predict_params: Any
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

    # noinspection PyPep8Naming
    @abstractmethod
    def score(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None
    ) -> float:
        """[see SupervisedLearnerDF]"""

    # we cannot get the docstring via the @inheritdoc mechanism because
    # ClassifierMixin precedes SupervisedLearnerDF in the MRO
    score.__doc__ = SupervisedLearnerDF.score.__doc__


class ClusterDF(
    ClusterMixin,  # type: ignore
    LearnerDF,
    metaclass=ABCMeta,
):
    """
    Base class for augmented `scikit-learn` clusterers.

    Provides enhanced support for data frames.
    """

    @property
    @abstractmethod
    def labels_(self) -> pd.Series:
        # noinspection GrazieInspection
        """
        A pandas series, mapping the index of the input data frame to cluster labels.
        """
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def fit_predict(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_predict_params: Any,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Fit this clusterer using the given inputs, then predict the cluster labels.

        :param X: data frame with observations as rows and features as columns
        :param y: a series or data frame with one or more outputs per observation
        :param fit_predict_params: optional keyword parameters as required by specific
            clusterer implementations
        :return: predicted cluster labels for all observations as a series,
            or as a data frame in case of multiple outputs
        """
        pass


__tracker.validate()
