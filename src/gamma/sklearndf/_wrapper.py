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

import logging
import re
from abc import ABC, abstractmethod, ABCMeta
from functools import wraps
from typing import *
from typing import Type

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
)

from gamma.sklearndf import (
    BaseEstimatorDF,
    BaseLearnerDF,
    ClassifierDF,
    RegressorDF,
    TransformerDF,
)

log = logging.getLogger(__name__)

__all__ = [
    "BaseEstimatorWrapperDF",
    "BaseLearnerWrapperDF",
    "ClassifierWrapperDF",
    "df_estimator",
    "MetaClassifierWrapperDF",
    "MetaEstimatorWrapperDF",
    "MetaRegressorWrapperDF",
    "RegressorWrapperDF",
    "TransformerWrapperDF",
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

T_EstimatorWrapperDF = TypeVar("T_EstimatorWrapperDF", bound="BaseEstimatorWrapperDF")


#
# base wrapper classes
#


class BaseEstimatorWrapperDF(
    BaseEstimator, BaseEstimatorDF, ABC, Generic[T_DelegateEstimator]
):
    # todo explain what is the benefit compared to the class BaseEstimatorDF
    """
    Abstract base class that is a wrapper around :class:`sklearn.base.BaseEstimator`.

    Implementations must define a method ``_make_delegate_estimator``.

    :param `**kwargs`: the arguments passed to the delegate estimator
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._delegate_estimator = type(self)._make_delegate_estimator(*args, **kwargs)
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
    ) -> T_EstimatorWrapperDF:
        """
        Make a new wrapped data frame estimator whose delegate is an estimator which
        has already been fitted
        :param estimator: the fitted estimator
        :param features_in: the column names of X used for fitting the estimator
        :return: the wrapped data frame estimator
        """

        class _FittedEstimator(cls):
            def __init__(self) -> None:
                super().__init__()
                self._features_in = features_in

            @classmethod
            def _make_delegate_estimator(cls, *args, **kwargs) -> T_DelegateEstimator:
                return estimator

        return _FittedEstimator()

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        :param deep: if ``True``, return the parameters for this estimator and \
        contained sub-objects that are estimators

        :return: mapping of the parameter names to their values
        """
        # noinspection PyUnresolvedReferences
        return self._delegate_estimator.get_params(deep=deep)

    def set_params(self: T, **kwargs) -> T:
        """
        Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        :returns self
        """
        # noinspection PyUnresolvedReferences
        self._delegate_estimator.set_params(**kwargs)
        return self

    # noinspection PyPep8Naming
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params,
    ) -> "BaseEstimatorWrapperDF[T_DelegateEstimator]":
        """
        Fit the delegate estimator.

        :param X: feature matrix
        :param y: target as a pandas series or data frame (if multi-output)
        """

        self._reset_fit()

        success = False

        try:
            self._check_parameter_types(X, y)
            self._fit(X, y, **fit_params)
            self._post_fit(X, y, **fit_params)
            success = True

        finally:
            if not success:
                self._reset_fit()

        return self

    @classmethod
    @abstractmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> T_DelegateEstimator:
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
            BaseEstimatorWrapperDF._verify_df(df=X, expected_columns=self.features_in)
        if y is not None and not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError("arg y must be None, or a pandas Series or DataFrame")

    @staticmethod
    def _verify_df(
        df: pd.DataFrame, expected_columns: pd.Index, expected_index: pd.Index = None
    ) -> None:
        def _error_message(axis: str, actual: pd.Index, expected: pd.Index):
            error_message = f"transformed data frame does not have expected {axis}"
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
            if len(error_detail) == 0:
                error_detail = [f"{axis} not in expected order"]
            return f"{error_message} ({'; '.join(error_detail)})"

        if not df.columns.equals(expected_columns):
            raise ValueError(
                _error_message(
                    axis="columns", actual=df.columns, expected=expected_columns
                )
            )
        if expected_index is not None and not df.index.equals(expected_index):
            raise ValueError(
                _error_message(axis="index", actual=df.index, expected=expected_index)
            )

    def _validate_delegate_attribute(self, attribute_name: str) -> None:
        if not hasattr(self.delegate_estimator, attribute_name):
            raise AttributeError(
                f"delegate estimator of type {type(self.delegate_estimator).__name__} "
                f"does not have attribute {attribute_name}"
            )

    # noinspection PyPep8Naming
    @staticmethod
    def _convert_X_for_delegate(X: pd.DataFrame) -> Any:
        return X

    @staticmethod
    def _convert_y_for_delegate(y: Optional[Union[pd.Series, pd.DataFrame]]) -> Any:
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
        # get a public attribute of the delegate estimator
        if name.startswith("_"):
            raise AttributeError(name)
        else:
            return getattr(self._delegate_estimator, name)

    def __setattr__(self, name: str, value: Any) -> Any:
        # set a public attribute of the delegate estimator
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._delegate_estimator, name, value)


class TransformerWrapperDF(
    TransformerDF,
    BaseEstimatorWrapperDF[T_DelegateTransformer],
    ABC,
    Generic[T_DelegateTransformer],
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
            TransformerWrapperDF._verify_df(
                df=transformed, expected_columns=columns, expected_index=index
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


class BaseLearnerWrapperDF(
    BaseLearnerDF,
    BaseEstimatorWrapperDF[T_DelegateLearner],
    ABC,
    Generic[T_DelegateLearner],
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
                # predictions are usually provided as an ndarray of the same length as X
                if y.ndim == 1:
                    # single-output predictions yield an ndarray of shape (n_samples)
                    return pd.Series(data=y, name=self.F_PREDICTION, index=X.index)
                if y.ndim == 2:
                    # multi-output predictions yield an ndarray of shape (n_samples,
                    # n_outputs)
                    return pd.DataFrame(data=y, index=X.index)
            raise TypeError(
                f"Unexpected shape of ndarray returned as prediction:" f" {y.shape}"
            )
        raise TypeError(
            f"unexpected data type returned as prediction: " f"{type(y).__name__}"
        )


class RegressorWrapperDF(
    RegressorDF,
    BaseLearnerWrapperDF[T_DelegateRegressor],
    ABC,
    Generic[T_DelegateRegressor],
):
    """
    Wrapper around sklearn regressors that preserves data frames.
    """


class ClassifierWrapperDF(
    ClassifierDF,
    BaseLearnerWrapperDF[T_DelegateClassifier],
    ABC,
    Generic[T_DelegateClassifier],
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


class MetaEstimatorWrapperDF(
    BaseEstimatorWrapperDF[T_DelegateEstimator],
    MetaEstimatorMixin,
    ABC,
    Generic[T_DelegateEstimator],
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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


class MetaClassifierWrapperDF(
    MetaEstimatorWrapperDF[T_DelegateClassifier],
    ClassifierWrapperDF,
    ABC,
    Generic[T_DelegateClassifier],
):
    """
    Abstract base class wrapping around classifiers implementing
    :class:`sklearn.base.MetaEstimatorMixin`.
    """

    pass


class MetaRegressorWrapperDF(
    MetaEstimatorWrapperDF[T_DelegateRegressor],
    RegressorWrapperDF,
    ABC,
    Generic[T_DelegateRegressor],
):
    """
    Abstract base class wrapping around regressors implementing
    :class:`sklearn.base.MetaEstimatorMixin`.
    """

    pass


#
# decorator for wrapping scikit-learn estimators
#


def df_estimator(
    delegate_estimator: Type[T_DelegateEstimator] = None,
    *,
    df_wrapper_type: Type[
        BaseEstimatorWrapperDF[T_DelegateEstimator]
    ] = BaseEstimatorWrapperDF[T_DelegateEstimator],
) -> Union[
    Callable[
        [Type[T_DelegateEstimator]], Type[BaseEstimatorWrapperDF[T_DelegateEstimator]]
    ],
    Type[BaseEstimatorWrapperDF[T_DelegateEstimator]],
]:
    """
    Class decorator wrapping a :class:`sklearn.base.BaseEstimator` in a
    :class:`BaseEstimatorWrapperDF`.
    :param delegate_estimator: the estimator class to wrap
    :param df_wrapper_type: optional parameter indicating the \
                              :class:`BaseEstimatorWrapperDF` class to be used for \
                              wrapping; defaults to :class:`BaseEstimatorWrapperDF`
    :return: the resulting `BaseEstimatorWrapperDF` with ``delegate_estimator`` as \
             the delegate estimator
    """

    def _decorate(
        decoratee: Type[T_DelegateEstimator]
    ) -> Type[BaseEstimatorWrapperDF[T_DelegateEstimator]]:

        # determine the sklearn estimator we are wrapping

        sklearn_base_estimator, _ = _get_base_classes(decoratee)

        # wrap the delegate estimator

        @wraps(decoratee, updated=())
        class _DataFrameEstimator(df_wrapper_type):
            @classmethod
            def _make_delegate_estimator(cls, *args, **kwargs) -> T_DelegateEstimator:
                # noinspection PyArgumentList
                return sklearn_base_estimator(*args, **kwargs)

        base_doc = sklearn_base_estimator.__doc__
        if base_doc is not None:
            base_doc_lines = _parse_pandas_class_docstring(base_doc)

            # does the pandas docstring start with a tag line?
            tag_line: List[AnyStr] = []
            if len(base_doc_lines) >= 3 and len(base_doc_lines[1]) == 0:
                tag_line.append(base_doc_lines[0])
                del base_doc_lines[:2]

            _DataFrameEstimator.__doc__ = "\n".join(
                [
                    *tag_line,
                    f"""
    .. note::
        This class is a wrapper around class
        :class:`{sklearn_base_estimator.__module__}.{sklearn_base_estimator.__name__}`.
        
        It provides enhanced support for pandas data frames, and otherwise replicates 
        all parameters and behaviours of class
        :class:`~{sklearn_base_estimator.__module__}.{sklearn_base_estimator.__name__}`.
""",
                    *base_doc_lines,
                ]
            )

        return _DataFrameEstimator

    def _get_base_classes(
        decoratee: Type[T_DelegateEstimator]
    ) -> Tuple[Type[BaseEstimator], List[Type[Any]]]:
        base_classes = decoratee.__bases__
        is_sklearn_base_estimator = [
            issubclass(base, BaseEstimator) for base in base_classes
        ]
        non_sklearn_bases = [
            base
            for base, is_sklearn in zip(base_classes, is_sklearn_base_estimator)
            if not is_sklearn
        ]
        sklearn_base_estimators = [
            base
            for base, is_sklearn in zip(base_classes, is_sklearn_base_estimator)
            if is_sklearn
        ]
        if len(sklearn_base_estimators) != 1:
            raise TypeError(
                f"class {decoratee.__name__} must have exactly one base class "
                f"that implements class {BaseEstimator.__name__}"
            )
        sklearn_base_estimator = sklearn_base_estimators[0]
        return sklearn_base_estimator, non_sklearn_bases

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
                r":param \1: `\2`: \3",
                parameter,
            ).replace("\n", " ")
            for parameter in parameters
        ]

    if not issubclass(df_wrapper_type, BaseEstimatorWrapperDF):
        raise ValueError(
            f"arg df_transformer_type is not a subclass of "
            f"{BaseEstimatorWrapperDF.__name__}: {df_wrapper_type}"
        )
    if delegate_estimator is None:
        return _decorate
    else:
        return _decorate(delegate_estimator)
