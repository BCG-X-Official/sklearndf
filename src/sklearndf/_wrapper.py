"""
Wrappers around native scikit-learn estimators.

`sklearndf` wrappers accept and return data frames (while scikit-learn transformers
usually return a numpy arrays, and may not accept data frames as input).
Otherwise the wrappers are designed to precisely mirror the API and behavior of the
native estimators they wrap.

The wrappers also implement the additional column attributes introduced by `sklearndf`,
:meth:`~EstimatorDF.feature_names_in_`, :meth:`~TransformerDF.feature_names_out_`, and
:meth:`~TransformerDF.feature_names_original_`.
"""

import inspect
import logging
from abc import ABCMeta, abstractmethod
from functools import update_wrapper
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
)

from pytools.api import inheritdoc, public_module_prefix

from sklearndf import ClassifierDF, EstimatorDF, LearnerDF, RegressorDF, TransformerDF

log = logging.getLogger(__name__)

__all__ = [
    "_EstimatorWrapperDF",
    "_LearnerWrapperDF",
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
T_Self = TypeVar("T_Self")
T_DelegateEstimator = TypeVar("T_DelegateEstimator", bound=BaseEstimator)
T_DelegateTransformer = TypeVar("T_DelegateTransformer", bound=TransformerMixin)
T_DelegateLearner = TypeVar("T_DelegateLearner", RegressorMixin, ClassifierMixin)
T_DelegateRegressor = TypeVar("T_DelegateRegressor", bound=RegressorMixin)
T_DelegateClassifier = TypeVar("T_DelegateClassifier", bound=ClassifierMixin)

# noinspection PyTypeChecker
T_EstimatorWrapperDF = TypeVar("T_EstimatorWrapperDF", bound="_EstimatorWrapperDF")


#
# base wrapper classes
#


@inheritdoc(match="[see superclass]")
class _EstimatorWrapperDF(
    EstimatorDF, BaseEstimator, Generic[T_DelegateEstimator], metaclass=ABCMeta
):
    """
    Base class for wrappers around a delegate :class:`sklearn.base.BaseEstimator`.

    Implementations must define a method :meth:`._make_delegate_estimator`, used to
    instantiate the delegate estimator to be wrapped.
    """

    def __init__(
        self, *args, _delegate_estimator: Optional[T_DelegateEstimator] = None, **kwargs
    ) -> None:
        """
        :param _delegate_estimator: (optional) an estimator to use as the delegate;
            if specified, do not create a new estimator and ignore any other arguments
            passed to this initializer
        :param args: positional arguments to use when initializing a new new delegate
            estimator
        :param kwargs: keyword arguments to use when initializing a new new delegate
            estimator
        """
        super().__init__()

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
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self._features_in is not None

    @property
    def native_estimator(self) -> T_DelegateEstimator:
        """
        The native estimator which this wrapper delegates to.
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
        has already been fitted.

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

    def get_params(self, deep=True) -> Mapping[str, Any]:
        """[see superclass]"""
        return self._delegate_estimator.get_params(deep=deep)

    def set_params(self: T_Self, **kwargs) -> T_Self:
        """[see superclass]"""
        self: _EstimatorWrapperDF  # support type hinting in PyCharm
        self._delegate_estimator.set_params(**kwargs)
        return self

    # noinspection PyPep8Naming
    def fit(
        self: T_Self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params,
    ) -> T_Self:
        """[see superclass]"""

        # support type hinting in PyCharm
        self: _EstimatorWrapperDF[T_DelegateEstimator]

        self._reset_fit()

        try:
            self._check_parameter_types(X, y)
            self._fit(X, y, **fit_params)
            self._post_fit(X, y, **fit_params)

        except Exception as cause:
            self._reset_fit()
            raise self._make_verbose_exception(self.fit.__name__, cause) from cause

        return self

    @classmethod
    @abstractmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> T_DelegateEstimator:
        pass

    def _validate_delegate_estimator(self) -> None:
        pass

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
            _EstimatorWrapperDF._verify_df(
                df_name="X argument", df=X, expected_columns=self.feature_names_in_
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
        if not hasattr(self.native_estimator, attribute_name):
            raise AttributeError(
                f"delegate estimator of type {type(self.native_estimator).__name__} "
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

    def _make_verbose_exception(self, method: str, cause: Exception) -> Exception:
        verbose_message = f"{type(self).__name__}.{method}: {cause}"
        # noinspection PyBroadException
        try:
            return type(cause)(verbose_message)
        except Exception:
            return RuntimeError(verbose_message)

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


@inheritdoc(match="[see superclass]")
class _TransformerWrapperDF(
    TransformerDF,
    _EstimatorWrapperDF[T_DelegateTransformer],
    Generic[T_DelegateTransformer],
    metaclass=ABCMeta,
):
    """
    Base class for wrappers around a delegate transformer.
    """

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """[see superclass]"""
        self._check_parameter_types(X, None)

        transformed = self._transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.feature_names_out_
        )

    # noinspection PyPep8Naming
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        """[see superclass]"""
        self._reset_fit()

        try:
            self._check_parameter_types(X, y)
            transformed = self._fit_transform(X, y, **fit_params)
            self._post_fit(X, y, **fit_params)

        except Exception as cause:
            self._reset_fit()
            raise self._make_verbose_exception(
                self.fit_transform.__name__, cause
            ) from cause

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.feature_names_out_
        )

    # noinspection PyPep8Naming
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """[see superclass]"""
        self._reset_fit()

        self._check_parameter_types(X, None)

        transformed = self._inverse_transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.feature_names_in_
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
        return self.native_estimator.transform(self._convert_X_for_delegate(X))

    # noinspection PyPep8Naming
    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params
    ) -> np.ndarray:
        return self.native_estimator.fit_transform(
            self._convert_X_for_delegate(X),
            self._convert_y_for_delegate(y),
            **fit_params,
        )

    # noinspection PyPep8Naming
    def _inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.native_estimator.inverse_transform(self._convert_X_for_delegate(X))


@inheritdoc(match="[see superclass]")
class _LearnerWrapperDF(
    LearnerDF,
    _EstimatorWrapperDF[T_DelegateLearner],
    Generic[T_DelegateLearner],
    metaclass=ABCMeta,
):
    """
    Base class for wrappers around a delegate learner.
    """

    #: Name of :class:`pd.Series` objects containing the predictions of single-output
    #: learners.
    #:
    #: See :meth:`~.LearnerDF.predict`.
    COL_PREDICTION = "prediction"

    # noinspection PyPep8Naming
    def predict(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_to_series_or_frame(
            X,
            self.native_estimator.predict(
                self._convert_X_for_delegate(X), **predict_params
            ),
        )

    # noinspection PyPep8Naming
    def fit_predict(
        self, X: pd.DataFrame, y: pd.Series, **fit_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""

        self._reset_fit()

        try:
            self._check_parameter_types(X, y)

            # noinspection PyUnresolvedReferences
            result = self._prediction_to_series_or_frame(
                X,
                self.native_estimator.fit_predict(
                    self._convert_X_for_delegate(X),
                    self._convert_y_for_delegate(y),
                    **fit_params,
                ),
            )

            self._post_fit(X, y, **fit_params)

        except Exception as cause:
            self._reset_fit()
            raise self._make_verbose_exception(
                self.fit_predict.__name__, cause
            ) from cause

        return result

    # noinspection PyPep8Naming
    def score(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None
    ) -> float:
        """[see superclass]"""
        self._check_parameter_types(X, y)
        if y is None:
            raise ValueError("arg y must not be None")
        if sample_weight is not None and not isinstance(sample_weight, pd.Series):
            raise TypeError("arg sample_weight must be None or a Series")

        return self.native_estimator.score(
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
                    return pd.Series(data=y, name=self.COL_PREDICTION, index=X.index)
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
    _LearnerWrapperDF[T_DelegateRegressor],
    Generic[T_DelegateRegressor],
    metaclass=ABCMeta,
):
    """
    Base class for wrappers around a delegate learner.
    """


@inheritdoc(match="[see superclass]")
class _ClassifierWrapperDF(
    ClassifierDF,
    _LearnerWrapperDF[T_DelegateClassifier],
    Generic[T_DelegateClassifier],
    metaclass=ABCMeta,
):
    """
    Wrapper around sklearn classifiers that preserves data frames.
    """

    @property
    def classes_(self) -> Sequence[Any]:
        """[see superclass]"""
        self._ensure_fitted()
        # noinspection PyUnresolvedReferences
        return self._delegate_estimator.classes_

    # noinspection PyPep8Naming
    def predict_proba(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """[see superclass]"""

        self._ensure_delegate_method("predict_proba")

        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X,
            self.native_estimator.predict_proba(
                self._convert_X_for_delegate(X), **predict_params
            ),
        )

    # noinspection PyPep8Naming
    def predict_log_proba(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """[see superclass]"""

        self._ensure_delegate_method("predict_log_proba")

        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X,
            self.native_estimator.predict_log_proba(
                self._convert_X_for_delegate(X), **predict_params
            ),
        )

    # noinspection PyPep8Naming
    def decision_function(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""

        self._ensure_delegate_method("decision_function")

        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X,
            self.native_estimator.decision_function(
                self._convert_X_for_delegate(X), **predict_params
            ),
        )

    def _ensure_delegate_method(self, method: str) -> None:
        if not hasattr(self.native_estimator, method):
            raise NotImplementedError(
                f"{type(self.native_estimator).__name__} does not implement method "
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
            classes = getattr(self.native_estimator, "classes_", None)

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
    _EstimatorWrapperDF[T_DelegateEstimator],
    MetaEstimatorMixin,
    Generic[T_DelegateEstimator],
    metaclass=ABCMeta,
):
    """
    Abstract base class wrapping around estimators implementing
    :class:`sklearn.base.MetaEstimatorMixin`. A meta-estimator will call the methods
    of the embedded estimator using a modified copy of the `X` and `y` parameters,
    so we need to make sure that these are converted back to data frames.

    This class covers three cases used in sklearn:
    - one inner estimator in attribute `estimator` or `base_estimator`
    - multiple inner estimators in attribute `estimators`
    """

    def _validate_delegate_estimator(self) -> None:
        def _unwrap_estimator(estimator: BaseEstimator) -> BaseEstimator:
            return (
                estimator.native_estimator
                if isinstance(estimator, EstimatorDF)
                else estimator
            )

        delegate_estimator = self.native_estimator

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
    _EstimatorWrapperDF[T_DelegateEstimator],
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
                    estimator.native_estimator
                    if isinstance(estimator, EstimatorDF)
                    else estimator
                )

        delegate_estimator = self.native_estimator

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
        _EstimatorWrapperDF[T_DelegateEstimator]
    ] = _EstimatorWrapperDF[T_DelegateEstimator],
) -> Union[
    Callable[
        [Type[T_DelegateEstimator]], Type[_EstimatorWrapperDF[T_DelegateEstimator]]
    ],
    Type[_EstimatorWrapperDF[T_DelegateEstimator]],
]:
    """
    Class decorator wrapping a :class:`sklearn.base.BaseEstimator` in a
    :class:`_EstimatorWrapperDF`.

    :param delegate_estimator: the estimator class to wrap
    :param df_wrapper_type: optional parameter indicating the
        :class:`_EstimatorWrapperDF` class to be used for wrapping; defaults to
        :class:`_EstimatorWrapperDF`
    :return: the resulting ``_EstimatorWrapperDF`` with ``native_estimator`` as
        the delegate estimator
    """

    def _decorate(
        decoratee: Type[T_DelegateEstimator],
    ) -> Type[_EstimatorWrapperDF[T_DelegateEstimator]]:

        assert issubclass(df_wrapper_type, _EstimatorWrapperDF)

        # determine the sklearn estimator we are wrapping
        sklearn_native_estimator_type = _get_native_estimator(decoratee=decoratee)

        # determine the module of the wrapper
        sklearndf_wrapper_module = decoratee.__module__

        # we will add this function to the new DF estimator class as a class method
        def _make_delegate_estimator(_cls, *args, **kwargs) -> T_DelegateEstimator:
            # noinspection PyArgumentList
            return sklearn_native_estimator_type(*args, **kwargs)

        _make_delegate_estimator.__module__ = sklearndf_wrapper_module

        # dynamically create the wrapper class
        # noinspection PyTypeChecker
        df_estimator_type: Type[_EstimatorWrapperDF[T_DelegateEstimator]] = type(
            # preserve the name
            decoratee.__name__,
            # subclass the wrapper type (e.g., _EstimatorWrapperDF)
            (df_wrapper_type,),
            {
                # implement abstract class method _make_delegate_estimator
                _make_delegate_estimator.__name__: classmethod(
                    _make_delegate_estimator
                ),
                # mirror all attributes of the wrapped sklearn class, as long
                # as they are not inherited from the wrapper base class
                **_mirror_attributes(
                    delegate_type=sklearn_native_estimator_type,
                    wrapper_module=sklearndf_wrapper_module,
                ),
            },
        )

        # add link to the wrapped class, for use in python module 'inspect'
        df_estimator_type.__wrapped__ = sklearn_native_estimator_type

        # preserve the original module, qualified name, and annotations
        df_estimator_type.__module__ = decoratee.__module__
        df_estimator_type.__qualname__ = decoratee.__qualname__

        # adopt the initializer signature of the wrapped sklearn estimator
        df_estimator_type.__init__ = _update_wrapper(
            wrapper=df_estimator_type.__init__,
            wrapped=sklearn_native_estimator_type.__init__,
            wrapper_module=sklearndf_wrapper_module,
        )
        # but do not keep the docstring of __init__
        df_estimator_type.__init__.__doc__ = None

        # adopt the class docstring of the wrapped sklearn estimator
        _update_class_docstring(
            df_estimator_type=df_estimator_type,
            sklearn_native_estimator_type=sklearn_native_estimator_type,
        )

        return df_estimator_type

    def _mirror_attributes(
        delegate_type: Type[T_DelegateEstimator], wrapper_module: str
    ) -> Dict[str, Any]:

        inherit_from_base_wrapper: Set[str] = set(dir(df_wrapper_type))

        new_dict = {
            name: _make_alias(
                module=wrapper_module,
                name=name,
                delegate_cls=delegate_type,
                delegate=member,
            )
            for name, member in vars(delegate_type).items()
            if not (
                member is None
                or name.startswith("__")
                or name in inherit_from_base_wrapper
            )
        }

        return new_dict

    def _make_alias(module: str, name: str, delegate_cls: type, delegate: T) -> T:
        def _make_forwarder() -> callable:
            def _forwarder(self, *args, **kwargs) -> Any:
                return delegate(self._delegate_estimator, *args, **kwargs)

            return _forwarder

        class_name = _full_name(cls=delegate_cls)
        full_name = f"{class_name}.{name}"

        if inspect.isfunction(delegate):
            function = _update_wrapper(
                wrapper=_make_forwarder(), wrapped=delegate, wrapper_module=module
            )
            docstring = f"See :meth:`{full_name}`"
            function.__doc__ = docstring
            return function
        else:
            docstring = f"See documentation of :class:`{class_name}`."
            if inspect.isdatadescriptor(delegate):
                return property(
                    fget=lambda self: delegate.__get__(self._delegate_estimator),
                    fset=lambda self, value: delegate.__set__(
                        self._delegate_estimator, value
                    ),
                    fdel=lambda self: delegate.__delete__(self._delegate_estimator),
                    doc=docstring,
                )
            else:
                return property(
                    fget=lambda self: getattr(self._delegate_estimator, name),
                    fset=lambda self, value: setattr(
                        self._delegate_estimator, name, value
                    ),
                    fdel=lambda self: delattr(self._delegate_estimator, name),
                    doc=docstring,
                )

    def _update_wrapper(
        wrapper: Any,
        wrapped: Any,
        wrapper_module: str,
    ):
        updated = update_wrapper(
            wrapper, wrapped, assigned=("__name__", "__qualname__", "__annotations__")
        )
        updated.__module__ = wrapper_module
        return updated

    def _update_class_docstring(
        df_estimator_type: Type[_EstimatorWrapperDF[T_DelegateEstimator]],
        sklearn_native_estimator_type: Type[BaseEstimator],
    ):
        base_doc = sklearn_native_estimator_type.__doc__

        if not base_doc:
            return

        base_doc_lines = base_doc.split("\n")

        # use the first paragraph as the tag line
        tag_lines: List[str] = []
        for line in base_doc_lines:
            # end of paragraph reached?
            stripped = line.strip()
            if stripped:
                # no: append line to tag lines
                tag_lines.append(stripped)
            elif tag_lines:
                # empty line and we already have tag lines: stop here
                break

        estimator_name = _full_name(cls=sklearn_native_estimator_type)

        df_estimator_type.__doc__ = "\n".join(
            [
                *tag_lines,
                "",
                (
                    f"""
.. note:: This class is a wrapper around class :class:`{estimator_name}`.
   It provides enhanced support for pandas data frames, and otherwise
   replicates all parameters and behaviours of class :class:`~{estimator_name}`."""
                ),
            ]
        )

    def _full_name(cls: type):
        # get the full name of the object, including the module prefix

        try:
            module_name = cls.__module__
        except AttributeError as e:
            raise RuntimeError(f"cannot get module for {cls}") from e

        module_name = public_module_prefix(module_name)

        return f"{module_name}.{cls.__qualname__}"

    def _get_native_estimator(
        decoratee: Type[T_DelegateEstimator],
    ) -> Type[BaseEstimator]:
        base_classes = decoratee.__bases__
        sklearn_native_estimators = [
            base for base in base_classes if issubclass(base, BaseEstimator)
        ]
        if len(sklearn_native_estimators) != 1:
            raise TypeError(
                f"class {decoratee.__name__} must have exactly one base class "
                f"that implements class {BaseEstimator.__name__}"
            )
        return sklearn_native_estimators[0]

    if not issubclass(df_wrapper_type, _EstimatorWrapperDF):
        raise ValueError(
            f"arg df_transformer_type is not a subclass of "
            f"{_EstimatorWrapperDF.__name__}: {df_wrapper_type}"
        )
    if delegate_estimator is None:
        return _decorate
    else:
        return _decorate(delegate_estimator)
