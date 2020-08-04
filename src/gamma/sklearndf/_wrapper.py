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
Wrappers around scikit-learn estimators.

These mimic the behavior of the wrapped scikit-learn estimator, but only accept and
return data frames (while scikit-learn transformers usually return a numpy arrays, and
may not accept data frames as input).

The wrappers also support the additional column attributes introduced by the
DataFrameEstimators and their generic subclasses including transformers and predictors
"""

import inspect
import logging
import re
from abc import ABCMeta, abstractmethod
from functools import update_wrapper
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
)

from gamma.common.fit import T_Self
from gamma.sklearndf import (
    BaseEstimatorDF,
    BaseLearnerDF,
    ClassifierDF,
    RegressorDF,
    TransformerDF,
)

log = logging.getLogger(__name__)

__all__ = [
    "_BaseEstimatorWrapperDF",
    "_BaseLearnerWrapperDF",
    "_ClassifierWrapperDF",
    "df_estimator",
    "_MetaClassifierWrapperDF",
    "_MetaEstimatorWrapperDF",
    "_MetaRegressorWrapperDF",
    "_RegressorWrapperDF",
    "_TransformerWrapperDF",
    "_StackingEstimatorWrapperDF",
    "_StackingClassifierWrapperDF",
    "_StackingRegressorWrapperDF",
]

#
# type variables
#

T = TypeVar("T")

T_DelegateEstimator = TypeVar("T_DelegateEstimator", bound=BaseEstimator)
T_DelegateTransformer = TypeVar("T_DelegateTransformer", bound=TransformerMixin)
T_DelegateLearner = TypeVar(
    "T_DelegateLearner", bound=Union[RegressorMixin, ClassifierMixin]
)
T_DelegateRegressor = TypeVar("T_DelegateRegressor", bound=RegressorMixin)
T_DelegateClassifier = TypeVar("T_DelegateClassifier", bound=ClassifierMixin)

T_EstimatorWrapperDF = TypeVar("T_EstimatorWrapperDF", bound="_BaseEstimatorWrapperDF")


#
# base wrapper classes
#


class _BaseEstimatorWrapperDF(
    BaseEstimator, BaseEstimatorDF, Generic[T_DelegateEstimator], metaclass=ABCMeta
):
    # todo explain what is the benefit compared to the class BaseEstimatorDF
    """
    Abstract base class that is a wrapper around :class:`sklearn.base.BaseEstimator`.

    Implementations must define a method ``_make_delegate_estimator``.

    """

    def __init__(
        self, *args, _delegate_estimator: Optional[T_DelegateEstimator] = None, **kwargs
    ) -> None:
        if _delegate_estimator is None:
            # create a new delegate estimator with the given parameters
            # noinspection PyProtectedMember
            self._delegate_estimator = type(self)._make_delegate_estimator(
                *args, **kwargs
            )
        else:
            self._delegate_estimator = _delegate_estimator

        self._validate_delegate_estimator()

        self._reset_fit()

    @property
    def delegate_estimator(self) -> T_DelegateEstimator:
        """
        Return the original estimator which this wrapper delegates to.

        :return: the original estimator which this estimator delegates to
        """
        return self._delegate_estimator

    @classmethod
    def from_fitted(
        cls: Type[T_EstimatorWrapperDF],
        estimator: T_DelegateEstimator,
        features_in: pd.Index,
        n_outputs: int,
    ) -> T_EstimatorWrapperDF:
        """
        Make a new wrapped data frame estimator whose delegate is an estimator which
        has already been fitted
        :param estimator: the fitted estimator
        :param features_in: the column names of X used for fitting the estimator
        :param n_outputs: the number of outputs in y used for fitting the estimator
        :return: the wrapped data frame estimator
        """

        class _FittedEstimator(cls):
            def __init__(self) -> None:
                super().__init__(_delegate_estimator=estimator)
                self._features_in = features_in
                self._n_outputs = n_outputs

        return _FittedEstimator()

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        :param deep: if ``True``, return the parameters for this estimator and \
        contained sub-objects that are estimators

        :return: mapping of the parameter names to their values
        """
        return self._delegate_estimator.get_params(deep=deep)

    def set_params(self: T_Self, **kwargs) -> T_Self:
        """
        Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        :returns self
        """

        self: _BaseEstimatorWrapperDF  # support type hinting in PyCharm
        self._delegate_estimator.set_params(**kwargs)
        return self

    # noinspection PyPep8Naming
    def fit(
        self: T_Self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params,
    ) -> T_Self:
        """
        Fit the delegate estimator.

        :param X: feature matrix
        :param y: target as a pandas series or data frame (if multi-output)
        """

        # support type hinting in PyCharm
        self: _BaseEstimatorWrapperDF[T_DelegateEstimator]

        self._reset_fit()

        try:
            self._check_parameter_types(X, y)
            self._fit(X, y, **fit_params)
            self._post_fit(X, y, **fit_params)

        except Exception:
            self._reset_fit()
            raise

        return self

    @classmethod
    @abstractmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> T_DelegateEstimator:
        pass

    def _validate_delegate_estimator(self) -> None:
        pass

    @property
    def is_fitted(self) -> bool:
        """``True`` if this estimator is fitted, else ``False``."""
        return self._features_in is not None

    def _get_features_in(self) -> pd.Index:
        return self._features_in

    def _get_n_outputs(self) -> int:
        return self._n_outputs

    def _reset_fit(self) -> None:
        self._features_in = None
        self._n_outputs = None

    # noinspection PyPep8Naming
    def _fit(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, pd.DataFrame]], **fit_params
    ) -> T_DelegateEstimator:
        # noinspection PyUnresolvedReferences
        return self._delegate_estimator.fit(
            self._convert_X_for_delegate(X),
            self._convert_y_for_delegate(y),
            **fit_params,
        )

    # noinspection PyPep8Naming,PyUnusedLocal
    def _post_fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params,
    ) -> None:
        self._features_in = X.columns.rename(self.COL_FEATURE_IN)
        self._n_outputs = (
            0 if y is None else 1 if isinstance(y, pd.Series) else y.shape[1]
        )

    # noinspection PyPep8Naming
    def _check_parameter_types(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, pd.DataFrame]]
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("arg X must be a DataFrame")
        if self.is_fitted:
            _BaseEstimatorWrapperDF._verify_df(
                df_name="X argument", df=X, expected_columns=self.features_in
            )
        if y is not None and not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError("arg y must be None, or a pandas Series or DataFrame")

    @staticmethod
    def _verify_df(
        df_name: str,
        df: pd.DataFrame,
        expected_columns: pd.Index,
        expected_index: pd.Index = None,
    ) -> None:
        def _compare_labels(axis: str, actual: pd.Index, expected: pd.Index):
            error_message = f"{df_name} data frame does not have expected {axis}"
            missing_columns = expected.difference(actual)
            extra_columns = actual.difference(expected)
            error_detail = []
            if len(actual) != len(expected):
                error_detail.append(
                    f"expected {len(expected)} columns but got {len(actual)}"
                )
                if len(missing_columns) > 0:
                    error_detail.append(
                        f"missing columns: "
                        f"{', '.join(str(item) for item in missing_columns)}"
                    )
                if len(extra_columns) > 0:
                    error_detail.append(
                        f"extra columns: "
                        f"{', '.join(str(item) for item in extra_columns)}"
                    )
                raise ValueError(f"{error_message} ({'; '.join(error_detail)})")

        _compare_labels(axis="columns", actual=df.columns, expected=expected_columns)
        if expected_index is not None:
            _compare_labels(axis="index", actual=df.index, expected=expected_index)

    def _validate_delegate_attribute(self, attribute_name: str) -> None:
        if not hasattr(self.delegate_estimator, attribute_name):
            raise AttributeError(
                f"delegate estimator of type {type(self.delegate_estimator).__name__} "
                f"does not have attribute {attribute_name}"
            )

    # noinspection PyPep8Naming
    def _convert_X_for_delegate(self, X: pd.DataFrame) -> Any:
        if not self.is_fitted:
            return X

        features_in = self._get_features_in()
        if X.columns.is_(features_in):
            return X
        else:
            return X.reindex(columns=features_in, copy=False)

    def _convert_y_for_delegate(
        self, y: Optional[Union[pd.Series, pd.DataFrame]]
    ) -> Any:
        return y

    def __dir__(self) -> Iterable[str]:
        # include non-private attributes of delegate estimator in directory
        return {
            *super().__dir__(),
            *(
                attr
                for attr in self._delegate_estimator.__dir__()
                if not attr.startswith("_")
            ),
        }

    def __getattr__(self, name: str) -> Any:
        # get a non-private attribute of the delegate estimator
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__}.{name}")
        else:
            return getattr(self._delegate_estimator, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # set a public attribute of the delegate estimator
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._delegate_estimator, name, value)


class _TransformerWrapperDF(
    TransformerDF,
    _BaseEstimatorWrapperDF[T_DelegateTransformer],
    Generic[T_DelegateTransformer],
    metaclass=ABCMeta,
):
    """
    Wraps a :class:`sklearn.base.TransformerMixin` and ensures that the X and y
    objects passed and returned are pandas data frames with valid column names.

    Implementations must define ``_make_delegate_estimator`` and
    ``_get_features_original``.

    :param `**args`: positional arguments of scikit-learn transformer to be wrapped
    :param `**kwargs`: keyword arguments  of scikit-learn transformer to be wrapped
    """

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Call the transform method of the delegate transformer
        ``self.delegate_estimator``.

        :param X: data frame to transform
        :return: transformed data frame
        """
        self._check_parameter_types(X, None)

        transformed = self._transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.features_out
        )

    # noinspection PyPep8Naming
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        """Call the ``fit_transform`` method of ``self.delegate_estimator``.

        :param X: data frame to transform
        :param y: series of training targets
        :param fit_params: parameters passed to the fit method of the delegate
                           transformer
        :return: data frame of transformed sample
        """
        self._reset_fit()

        self._check_parameter_types(X, y)

        transformed = self._fit_transform(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.features_out
        )

    # noinspection PyPep8Naming
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inverse transformations in reverse order on the delegate
        transformer.

        All estimators in the pipeline must support ``inverse_transform``.
        :param X: data frame of samples
        :return: data frame of inverse-transformed samples
        """
        self._reset_fit()

        self._check_parameter_types(X, None)

        transformed = self._inverse_transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.features_in
        )

    def _reset_fit(self) -> None:
        try:
            # noinspection PyProtectedMember
            super()._reset_fit()
        finally:
            self._features_original = None

    @staticmethod
    def _transformed_to_df(
        transformed: Union[pd.DataFrame, np.ndarray], index: pd.Index, columns: pd.Index
    ):
        if isinstance(transformed, pd.DataFrame):
            # noinspection PyProtectedMember
            _TransformerWrapperDF._verify_df(
                df_name="transformed",
                df=transformed,
                expected_columns=columns,
                expected_index=index,
            )
            return transformed
        else:
            return pd.DataFrame(data=transformed, index=index, columns=columns)

    # noinspection PyPep8Naming
    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.delegate_estimator.transform(self._convert_X_for_delegate(X))

    # noinspection PyPep8Naming
    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params
    ) -> np.ndarray:
        return self.delegate_estimator.fit_transform(
            self._convert_X_for_delegate(X),
            self._convert_y_for_delegate(y),
            **fit_params,
        )

    # noinspection PyPep8Naming
    def _inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.delegate_estimator.inverse_transform(
            self._convert_X_for_delegate(X)
        )


class _BaseLearnerWrapperDF(
    BaseLearnerDF,
    _BaseEstimatorWrapperDF[T_DelegateLearner],
    Generic[T_DelegateLearner],
    metaclass=ABCMeta,
):
    """
    Base class for sklearn regressors and classifiers that preserve data frames

    :param `**kwargs`: arguments passed to :class:`.BaseEstimatorDF` in ``__init__``
    """

    F_PREDICTION = "prediction"

    # noinspection PyPep8Naming
    def predict(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute the prediction as a series or a data frame.

        For single-output problems, return a series, fro multi-output problems,
        return a data frame.

        :param X: the data frame of features
        :param predict_params: additional arguments passed to the ``predict`` method \
        of the delegate estimator
        :return: the predictions
        """
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_to_series_or_frame(
            X,
            self.delegate_estimator.predict(
                self._convert_X_for_delegate(X), **predict_params
            ),
        )

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.Series:
        """
        Fit and return the predictions.

        :param X: the data frame of features
        :param y: the series of target used to train the model
        :param fit_params: additional arguments passed to the the ``predict`` method
          of the delegate estimator
        :return: series of the predictions for X
        """

        self._reset_fit()

        self._check_parameter_types(X, y)

        # noinspection PyUnresolvedReferences
        result = self._prediction_to_series_or_frame(
            X,
            self.delegate_estimator.fit_predict(
                self._convert_X_for_delegate(X),
                self._convert_y_for_delegate(y),
                **fit_params,
            ),
        )

        self._post_fit(X, y, **fit_params)

        return result

    # noinspection PyPep8Naming
    def score(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None
    ) -> float:
        """
        Return the score of the delegate estimator.

        :param X: data frame of the features, shape = (n_samples, n_features)
        :param y: series of the true targets, shape = (n_samples) or (n_samples, \
        n_outputs)
        :param sample_weight:  array-like, sample weights, shape = (n_sample)
        :return: the score of the model
        """
        self._check_parameter_types(X, y)
        if y is None:
            raise ValueError("arg y must not be None")
        if sample_weight is not None and not isinstance(sample_weight, pd.Series):
            raise TypeError("arg sample_weight must be None or a Series")

        return self.delegate_estimator.score(
            self._convert_X_for_delegate(X),
            self._convert_y_for_delegate(y),
            sample_weight,
        )

    # noinspection PyPep8Naming
    def _prediction_to_series_or_frame(
        self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            # if we already have a series or data frame, check it and return it
            # unchanged
            return y
        elif isinstance(y, np.ndarray):
            if len(y) == len(X):
                # predictions are usually provided as a numpy array the same length as X
                if y.ndim == 1:
                    # single-output predictions yield a numpy array of shape (n_samples)
                    return pd.Series(data=y, name=self.F_PREDICTION, index=X.index)
                if y.ndim == 2:
                    # multi-output predictions yield a numpy array of shape (n_samples,
                    # n_outputs)
                    return pd.DataFrame(data=y, index=X.index)
            raise TypeError(
                f"Unexpected shape of numpy array returned as prediction:" f" {y.shape}"
            )
        raise TypeError(
            f"unexpected data type returned as prediction: " f"{type(y).__name__}"
        )


class _RegressorWrapperDF(
    RegressorDF,
    _BaseLearnerWrapperDF[T_DelegateRegressor],
    Generic[T_DelegateRegressor],
    metaclass=ABCMeta,
):
    """
    Wrapper around sklearn regressors that preserves data frames.
    """


class _ClassifierWrapperDF(
    ClassifierDF,
    _BaseLearnerWrapperDF[T_DelegateClassifier],
    Generic[T_DelegateClassifier],
    metaclass=ABCMeta,
):
    """
    Wrapper around sklearn classifiers that preserves data frames.
    """

    # noinspection PyPep8Naming
    def predict_proba(self, X: pd.DataFrame) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Probability estimates.

        :param X: data frame of features
        :return: the series of probability estimates
        """

        self._ensure_delegate_method("predict_proba")

        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X, self.delegate_estimator.predict_proba(self._convert_X_for_delegate(X))
        )

    # noinspection PyPep8Naming
    def predict_log_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Log of probability estimates.

        :param X: data frame of features
        :return: series of log-probabilities
        """

        self._ensure_delegate_method("predict_log_proba")

        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X,
            self.delegate_estimator.predict_log_proba(self._convert_X_for_delegate(X)),
        )

    # noinspection PyPep8Naming
    def decision_function(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Evaluate the decision function for the samples in X.

        :param X: data frame of features
        :return: data frame of the decision functions of the sample for each class
        """

        self._ensure_delegate_method("decision_function")

        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X,
            self.delegate_estimator.decision_function(self._convert_X_for_delegate(X)),
        )

    def _ensure_delegate_method(self, method: str) -> None:
        if not hasattr(self.delegate_estimator, method):
            raise NotImplementedError(
                f"{type(self.delegate_estimator).__name__} does not implement method "
                f"{method}"
            )

    # noinspection PyPep8Naming
    def _prediction_with_class_labels(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame, list, np.ndarray],
        classes: Optional[Sequence[Any]] = None,
    ) -> Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]:

        if classes is None:
            classes = getattr(self.delegate_estimator, "classes_", None)

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            # if we already have a series or data frame, return it unchanged
            return y
        elif isinstance(y, np.ndarray):
            if len(y) == len(X):
                # predictions of probabilities are usually provided as a NumPy array
                # the same length as X
                if y.ndim == 1:
                    # for a binary classifier, we get a series with probabilities
                    # for the second class
                    return pd.Series(data=y, index=X.index, name=classes[1])
                elif y.ndim == 2:
                    # for a multi-class classifiers, we get a two-dimensional NumPy
                    # array with probabilities for each class
                    return pd.DataFrame(data=y, index=X.index, columns=classes)
            raise TypeError(
                f"ndarray with unexpected shape returned as prediction: " f"{y.shape}"
            )
        else:
            raise TypeError(f"unexpected type or prediction result: {type(y).__name__}")


#
# Meta estimator wrappers
#


class _MetaEstimatorWrapperDF(
    _BaseEstimatorWrapperDF[T_DelegateEstimator],
    MetaEstimatorMixin,
    Generic[T_DelegateEstimator],
    metaclass=ABCMeta,
):
    """
    Abstract base class wrapping around estimators implementing
    :class:`sklearn.base.MetaEstimatorMixin`. A meta-estimator will call the methods
    of the embedded estimator using a modified copy of the ``X`` and ``y`` parameters,
    so we need to make sure that these are converted back to data frames.

    This class covers three cases used in sklearn:
    - one inner estimator in attribute ``estimator`` or ``base_estimator``
    - multiple inner estimators in attribute ``estimators``
    """

    def _validate_delegate_estimator(self) -> None:
        def _unwrap_estimator(estimator: BaseEstimator) -> BaseEstimator:
            return (
                estimator.root_estimator
                if isinstance(estimator, BaseEstimatorDF)
                else estimator
            )

        delegate_estimator = self.delegate_estimator

        if hasattr(delegate_estimator, "estimator"):
            delegate_estimator.estimator = _unwrap_estimator(
                delegate_estimator.estimator
            )
        elif hasattr(delegate_estimator, "base_estimator"):
            delegate_estimator.base_estimator = _unwrap_estimator(
                delegate_estimator.base_estimator
            )
        elif hasattr(delegate_estimator, "estimators"):
            delegate_estimator.estimators = [
                (name, _unwrap_estimator(estimator))
                for name, estimator in delegate_estimator.estimators
            ]


class _MetaClassifierWrapperDF(
    _MetaEstimatorWrapperDF[T_DelegateClassifier],
    _ClassifierWrapperDF,
    Generic[T_DelegateClassifier],
    metaclass=ABCMeta,
):
    """
    Abstract base class wrapping around classifiers implementing
    :class:`sklearn.base.MetaEstimatorMixin`.
    """

    pass


class _MetaRegressorWrapperDF(
    _MetaEstimatorWrapperDF[T_DelegateRegressor],
    _RegressorWrapperDF,
    Generic[T_DelegateRegressor],
    metaclass=ABCMeta,
):
    """
    Abstract base class wrapping around regressors implementing
    :class:`sklearn.base.MetaEstimatorMixin`.
    """

    pass


#
# Stacking Estimator wrappers
#


class _StackingEstimatorWrapperDF(
    _BaseEstimatorWrapperDF[T_DelegateEstimator],
    # note: MetaEstimatorMixin is first public child in inheritance from _BaseStacking
    # MetaEstimatorMixin <--  _BaseHeterogeneousEnsemble <-- _BaseStacking
    MetaEstimatorMixin,
    Generic[T_DelegateEstimator],
    metaclass=ABCMeta,
):
    """
    Abstract base class wrapping around estimators implementing
    :class:`sklearn.ensemble._stacking._BaseStacking`. The stacking estimator will call
    the methods of the embedded estimator using a modified copy of the `X` and `y`
    parameters, so we need to make sure that these are converted back to data frames.

    This class covers the following cases used in sklearn:
    - one (optional) inner estimator in attribute `final_estimator`
    - multiple stacked estimators in attribute `estimators`, as name estimator pairs
    """

    def _validate_delegate_estimator(self) -> None:
        def _unwrap_estimator(estimator: BaseEstimator) -> Optional[BaseEstimator]:
            if estimator is None:
                return None
            else:
                return (
                    estimator.root_estimator
                    if isinstance(estimator, BaseEstimatorDF)
                    else estimator
                )

        delegate_estimator = self.delegate_estimator

        # note: as final_estimator is optional, _unwrap_estimator will return None
        #       attribute "named_estimators_" is constructed based off estimators

        if hasattr(delegate_estimator, "final_estimator"):
            delegate_estimator.final_estimator = _unwrap_estimator(
                delegate_estimator.final_estimator
            )

        if hasattr(delegate_estimator, "estimators"):
            delegate_estimator.estimators = [
                (name, _unwrap_estimator(estimator))
                for name, estimator in delegate_estimator.estimators
            ]


class _StackingClassifierWrapperDF(
    _StackingEstimatorWrapperDF[T_DelegateClassifier],
    _ClassifierWrapperDF,
    Generic[T_DelegateClassifier],
    metaclass=ABCMeta,
):
    """
    Abstract base class wrapping around classifiers implementing
    :class:`sklearn.ensemble._stacking._BaseStacking`.
    """

    pass


class _StackingRegressorWrapperDF(
    _StackingEstimatorWrapperDF[T_DelegateRegressor],
    _RegressorWrapperDF,
    Generic[T_DelegateRegressor],
    metaclass=ABCMeta,
):
    """
    Abstract base class wrapping around regressors implementing
    :class:`sklearn.ensemble._stacking._BaseStacking`.
    """

    pass


#
# decorator for wrapping scikit-learn estimators
#


def df_estimator(
    delegate_estimator: Type[T_DelegateEstimator] = None,
    *,
    df_wrapper_type: Type[
        _BaseEstimatorWrapperDF[T_DelegateEstimator]
    ] = _BaseEstimatorWrapperDF[T_DelegateEstimator],
) -> Union[
    Callable[
        [Type[T_DelegateEstimator]], Type[_BaseEstimatorWrapperDF[T_DelegateEstimator]]
    ],
    Type[_BaseEstimatorWrapperDF[T_DelegateEstimator]],
]:
    """
    Class decorator wrapping a :class:`sklearn.base.BaseEstimator` in a
    :class:`BaseEstimatorWrapperDF`.
    :param delegate_estimator: the estimator class to wrap
    :param df_wrapper_type: optional parameter indicating the \
                              :class:`BaseEstimatorWrapperDF` class to be used for \
                              wrapping; defaults to :class:`BaseEstimatorWrapperDF`
    :return: the resulting ``BaseEstimatorWrapperDF`` with ``delegate_estimator`` as \
             the delegate estimator
    """

    def _decorate(
        decoratee: Type[T_DelegateEstimator],
    ) -> Type[_BaseEstimatorWrapperDF[T_DelegateEstimator]]:

        assert issubclass(df_wrapper_type, _BaseEstimatorWrapperDF)

        # determine the sklearn estimator we are wrapping
        sklearn_estimator_type = _get_base_classes(decoratee)

        # we will add this function to the new DF estimator class as a class method
        def _make_delegate_estimator(_cls, *args, **kwargs) -> T_DelegateEstimator:
            # noinspection PyArgumentList
            return sklearn_estimator_type(*args, **kwargs)

        # dynamically create the wrapper class
        df_estimator_type: Type[_BaseEstimatorWrapperDF[T_DelegateEstimator]] = cast(
            Type[_BaseEstimatorWrapperDF[T_DelegateEstimator]],
            type(
                # preserve the name
                decoratee.__name__,
                # subclass the wrapper type (e.g., BaseEstimatorWrapperDF)
                (df_wrapper_type,),
                {
                    # implement abstract class method _make_delegate_estimator
                    _make_delegate_estimator.__name__: classmethod(
                        _make_delegate_estimator
                    ),
                    # mirror all attributes of the wrapped sklearn class, as long
                    # as they are not inherited from the wrapper base class
                    **_mirror_attributes(delegate_type=sklearn_estimator_type),
                },
            ),
        )

        # add link to the wrapped class, for use in python module 'inspect'
        df_estimator_type.__wrapped__ = sklearn_estimator_type

        # preserve the original module, qualified name, and annotations
        df_estimator_type.__module__ = decoratee.__module__
        df_estimator_type.__qualname__ = decoratee.__qualname__

        # adopt the initializer signature of the wrapped sklearn estimator
        df_estimator_type.__init__ = update_wrapper(
            df_estimator_type.__init__, sklearn_estimator_type.__init__
        )

        # adopt the class docstring of the wrapped sklearn estimator
        _update_class_docstring(df_estimator_type, sklearn_estimator_type)

        return df_estimator_type

    def _mirror_attributes(delegate_type: Type[T_DelegateEstimator]) -> Dict[str, Any]:

        inherit_from_base_wrapper = set(dir(df_wrapper_type))

        new_dict = {
            name: _make_alias(name=name, delegate=member)
            for name, member in vars(delegate_type).items()
            if not (
                member is None
                or name.startswith("__")
                or name in inherit_from_base_wrapper
            )
        }

        return new_dict

    def _make_alias(name: str, delegate: T) -> T:
        def _make_forwarder() -> callable:
            def _forwarder(self, *args, **kwargs) -> Any:
                return delegate(self._delegate, *args, **kwargs)

            return _forwarder

        if inspect.isfunction(delegate):
            return update_wrapper(_make_forwarder(), delegate)
        elif inspect.isdatadescriptor(delegate):
            return property(
                fget=lambda self: delegate.__get__(self._delegate),
                fset=lambda self, value: delegate.__set__(self._delegate, value),
                fdel=lambda self: delegate.__delete__(self._delegate),
                doc=delegate.__doc__,
            )
        else:
            return property(
                fget=lambda self: getattr(self._delegate_estimator, name),
                fset=lambda self, value: setattr(self._delegate_estimator, name, value),
                fdel=lambda self: delattr(self._delegate_estimator, name),
                doc=delegate.__doc__,
            )

    def _update_class_docstring(
        df_estimator_type: Type[_BaseEstimatorWrapperDF[T_DelegateEstimator]],
        sklearn_estimator_type: Type[BaseEstimator],
    ):
        base_doc = sklearn_estimator_type.__doc__
        if base_doc is not None:
            base_doc_lines = _parse_pandas_class_docstring(base_doc)

            # does the pandas docstring start with a tag line?
            tag_line: List[AnyStr] = []
            if len(base_doc_lines) >= 3 and len(base_doc_lines[1]) == 0:
                tag_line.append(base_doc_lines[0])
                del base_doc_lines[:2]

            df_estimator_type.__doc__ = "\n".join(
                [
                    *tag_line,
                    f"""
    .. note::
        This class is a wrapper around class
        :class:`{sklearn_estimator_type.__module__}.{sklearn_estimator_type.__name__}`.
        
        It provides enhanced support for pandas data frames, and otherwise replicates 
        all parameters and behaviours of class
        :class:`~{sklearn_estimator_type.__module__}.{sklearn_estimator_type.__name__}`.
""",
                    *base_doc_lines,
                ]
            )

    def _get_base_classes(decoratee: Type[T_DelegateEstimator]) -> Type[BaseEstimator]:
        base_classes = decoratee.__bases__
        sklearn_base_estimators = [
            base for base in base_classes if issubclass(base, BaseEstimator)
        ]
        if len(sklearn_base_estimators) != 1:
            raise TypeError(
                f"class {decoratee.__name__} must have exactly one base class "
                f"that implements class {BaseEstimator.__name__}"
            )
        sklearn_base_estimator = sklearn_base_estimators[0]
        return sklearn_base_estimator

    def _parse_pandas_class_docstring(pandas_doc: AnyStr) -> List[AnyStr]:
        base_doc_split = re.split(
            r"^\s*((?:\w+\s)*\w+)\s*\n\s*-+\s*$",
            pandas_doc.replace("``", "`"),
            flags=re.MULTILINE,
        )
        doc_head = base_doc_split[0]
        doc_sections = dict(zip(base_doc_split[1::2], base_doc_split[2::2]))
        return [
            *doc_head.split("\n"),
            *_parse_pandas_parameters(
                parameters_section=doc_sections.get("Parameters", "")
            ),
        ]

    def _parse_pandas_parameters(parameters_section: AnyStr) -> List[AnyStr]:
        parameters = re.split(r"\s*\n\s*\n", parameters_section, flags=re.MULTILINE)
        # return [f"{len(parameters)} params"]
        return [
            re.sub(
                r"\s*(\w+)\s*:\s*(.*\S)\s*\n((.*\n)*.*)",
                r":param \1: ``\2``: \3",
                parameter,
            ).replace("\n", " ")
            for parameter in parameters
        ]

    if not issubclass(df_wrapper_type, _BaseEstimatorWrapperDF):
        raise ValueError(
            f"arg df_transformer_type is not a subclass of "
            f"{_BaseEstimatorWrapperDF.__name__}: {df_wrapper_type}"
        )
    if delegate_estimator is None:
        return _decorate
    else:
        return _decorate(delegate_estimator)
