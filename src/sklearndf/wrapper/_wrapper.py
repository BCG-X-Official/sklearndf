"""
Wrappers around native `scikit-learn` estimators.

`sklearndf` wrappers accept and return data frames (while `scikit-learn` transformers
usually return a numpy arrays, and may not accept data frames as input).
Otherwise the wrappers are designed to precisely mirror the API and behavior of the
native estimators they wrap.

The wrappers also implement the additional column attributes introduced by `sklearndf`,
:meth:`~EstimatorDF.feature_names_in_`, :meth:`~TransformerDF.feature_names_out_`, and
:meth:`~TransformerDF.feature_names_original_`.
"""
import inspect
import logging
import warnings
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
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from weakref import WeakValueDictionary

import numpy as np
import pandas as pd
import sklearn.utils.metaestimators as sklearn_meta
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
)

from pytools.api import AllTracker, inheritdoc, public_module_prefix

from ._adapter import SupervisedLearnerNPDF
from sklearndf import (
    ClassifierDF,
    ClustererDF,
    EstimatorDF,
    LearnerDF,
    RegressorDF,
    SupervisedLearnerDF,
    TransformerDF,
)

log = logging.getLogger(__name__)

__all__ = [
    "ClassifierWrapperDF",
    "EstimatorWrapperDF",
    "EstimatorWrapperDFMeta",
    "LearnerWrapperDF",
    "SupervisedLearnerWrapperDF",
    "MetaEstimatorWrapperDF",
    "RegressorWrapperDF",
    "ClustererWrapperDF",
    "StackingEstimatorWrapperDF",
    "TransformerWrapperDF",
    "make_df_classifier",
    "make_df_estimator",
    "make_df_regressor",
    "make_df_transformer",
    "make_df_clusterer",
]


#
# type variables
#

T = TypeVar("T")
T_NativeEstimator = TypeVar("T_NativeEstimator", bound=BaseEstimator)
T_NativeTransformer = TypeVar("T_NativeTransformer", bound=TransformerMixin)
T_NativeLearner = TypeVar(
    "T_NativeLearner", bound=Union[RegressorMixin, ClassifierMixin, ClusterMixin]
)
T_NativeSupervisedLearner = TypeVar(
    "T_NativeSupervisedLearner", bound=Union[RegressorMixin, ClassifierMixin]
)
T_NativeRegressor = TypeVar("T_NativeRegressor", bound=RegressorMixin)
T_NativeClassifier = TypeVar("T_NativeClassifier", bound=ClassifierMixin)
T_NativeClusterer = TypeVar("T_NativeClusterer", bound=ClusterMixin)

T_EstimatorWrapperDF = TypeVar("T_EstimatorWrapperDF", bound="EstimatorWrapperDF")
T_TransformerWrapperDF = TypeVar("T_TransformerWrapperDF", bound="TransformerWrapperDF")
T_SupervisedLearnerDF = TypeVar("T_SupervisedLearnerDF", bound="SupervisedLearnerDF")
T_RegressorWrapperDF = TypeVar("T_RegressorWrapperDF", bound="RegressorWrapperDF")
T_ClassifierWrapperDF = TypeVar("T_ClassifierWrapperDF", bound="ClassifierWrapperDF")
T_ClustererWrapperDF = TypeVar("T_ClustererWrapperDF", bound="ClustererWrapperDF")

T_StackableSupervisedLearnerDF = TypeVar(
    "T_StackableSupervisedLearnerDF", bound="_StackableSupervisedLearnerDF"
)
T_StackingEstimatorWrapperDF = TypeVar(
    "T_StackingEstimatorWrapperDF", bound="StackingEstimatorWrapperDF"
)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# base wrapper classes
#


class EstimatorWrapperDFMeta(ABCMeta, Generic[T_NativeEstimator]):
    """
    Metaclass of DF wrappers, providing a reference to the type of the wrapped native
    estimator.
    """

    __wrapped__: Type[T_NativeEstimator]

    @property
    def native_estimator_type(cls) -> Type[T_NativeEstimator]:
        """
        The type of native estimator that instances of this wrapper class delegate to.
        """
        return cls.__wrapped__


@inheritdoc(match="[see superclass]")
class EstimatorWrapperDF(
    EstimatorDF, Generic[T_NativeEstimator], metaclass=EstimatorWrapperDFMeta
):
    """
    Base class of DF wrappers for native estimators conforming with the `scikit-learn`
    API.

    Estimator wrapper classes should be created using function
    :func:`.make_df_estimator`.
    """

    __ARG_FITTED_DELEGATE_CONTEXT = "__EstimatorWrapperDF_fitted"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        :param args: positional arguments to use when initializing a new new delegate
            estimator
        :param kwargs: keyword arguments to use when initializing a new new delegate
            estimator
        """
        super().__init__()
        self._features_in: Optional[pd.Index] = None
        self._n_outputs: Optional[int] = None

        # check if a fitted estimator was passed by class method is_fitted
        fitted_delegate_context: Tuple[T_NativeEstimator, pd.Index, int] = kwargs.get(
            EstimatorWrapperDF.__ARG_FITTED_DELEGATE_CONTEXT, None
        )

        if fitted_delegate_context is None:
            # create a new delegate estimator with the given parameters
            # noinspection PyProtectedMember
            _native_estimator = type(self).__wrapped__(*args, **kwargs)
            self._reset_fit()
        else:
            (
                _native_estimator,
                self._features_in,
                self._n_outputs,
            ) = fitted_delegate_context

        self._native_estimator = _native_estimator

        self._validate_delegate_estimator()

    def __new__(cls: Type[T], *args, **kwargs: Any) -> T:
        if not hasattr(cls, "__wrapped__"):
            raise TypeError(f"cannot instantiate abstract wrapper class {cls.__name__}")
        else:
            return cast(Type[EstimatorDF], super()).__new__(cls)

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self._features_in is not None

    @property
    def native_estimator(self) -> T_NativeEstimator:
        """
        The native estimator that this wrapper delegates to.
        """
        return self._native_estimator

    @property
    def feature_names_in_(self) -> pd.Index:
        """[see superclass]"""
        return self._check_feature_names_in(
            super().feature_names_in_, warning_stacklevel=2
        )

    def _check_feature_names_in(
        self, wrapper_feature_names_in: pd.Index, *, warning_stacklevel: int
    ) -> pd.Index:
        # Check that the given feature names are the same as the ingoing feature names
        # recorded by the native estimator, if present. Issue a warning if the feature
        # names differ.
        # Return the same feature names that were passed to this method.
        try:
            _feature_names_in_native_ = self.native_estimator.feature_names_in_
        except AttributeError:
            return wrapper_feature_names_in

        if not np.array_equal(
            wrapper_feature_names_in.values, _feature_names_in_native_
        ):
            warnings.warn(
                "conflicting input feature names: "
                "the input feature names recorded by this estimator are "
                f"{wrapper_feature_names_in}, but the input feature names recorded by "
                f"the wrapped native estimator are {_feature_names_in_native_}",
                stacklevel=warning_stacklevel + 1,
            )
        return wrapper_feature_names_in

    @property
    def _estimator_type(self) -> Optional[str]:
        try:
            # noinspection PyProtectedMember
            return self.native_estimator._estimator_type
        except AttributeError:
            return None

    @classmethod
    def from_fitted(
        cls: Type[T_EstimatorWrapperDF],
        estimator: T_NativeEstimator,
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

        return cls(
            **{
                EstimatorWrapperDF.__ARG_FITTED_DELEGATE_CONTEXT: (
                    estimator,
                    features_in,
                    n_outputs,
                )
            }
        )

    def get_params(self, deep: bool = True) -> Mapping[str, Any]:
        """[see superclass]"""
        return self._native_estimator.get_params(deep=deep)

    def set_params(self: T_EstimatorWrapperDF, **params: Any) -> T_EstimatorWrapperDF:
        """[see superclass]"""
        self._native_estimator.set_params(**params)
        return self

    # noinspection PyPep8Naming
    def fit(
        self: T_EstimatorWrapperDF,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params: Any,
    ) -> T_EstimatorWrapperDF:
        """[see superclass]"""

        self._reset_fit()

        try:
            self._check_parameter_types(X, y)
            self._fit(X, y, **fit_params)
            self._post_fit(X, y, **fit_params)

        except Exception as cause:
            self._reset_fit()
            raise self._make_verbose_exception(self.fit.__name__, cause) from cause

        return self

    def _validate_delegate_estimator(self) -> None:
        pass

    def _get_features_in(self) -> pd.Index:
        assert self._features_in is not None, "estimator is fitted"
        return self._features_in

    def _get_n_outputs(self) -> int:
        assert self._n_outputs is not None, "estimator is fitted"
        return self._n_outputs

    def _reset_fit(self) -> None:
        self._features_in = None
        self._n_outputs = None

    # noinspection PyPep8Naming
    def _fit(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, pd.DataFrame]], **fit_params
    ) -> T_NativeEstimator:
        # noinspection PyUnresolvedReferences
        return self._native_estimator.fit(
            self._prepare_X_for_delegate(X),
            self._prepare_y_for_delegate(y),
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
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        *,
        expected_columns: pd.Index = None,
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("arg X must be a DataFrame")

        if self.is_fitted:
            EstimatorWrapperDF._verify_df(
                df_name="arg X",
                df=X,
                expected_columns=(
                    self.feature_names_in_
                    if expected_columns is None
                    else expected_columns
                ),
            )

        if y is not None and not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError("arg y must be None, or a pandas series or data frame")

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
    def _prepare_X_for_delegate(self, X: pd.DataFrame) -> Any:
        # convert X before passing it to the delegate estimator
        return self._adjust_X_type_for_delegate(self._adjust_X_columns_for_delegate(X))

    def _prepare_y_for_delegate(
        self, y: Optional[Union[pd.Series, pd.DataFrame]]
    ) -> Any:
        return self._adjust_y_type_for_delegate(y)

    # noinspection PyPep8Naming
    def _adjust_X_columns_for_delegate(self, X: pd.DataFrame) -> pd.DataFrame:
        # make sure columns of X are aligned with frame used to fit this estimator

        if not self.is_fitted:
            # return X unchanged if estimator is not fitted yet
            return X

        features_in = self._get_features_in()
        if X.columns.is_(features_in):
            return X
        else:
            return X.reindex(columns=features_in, copy=False)

    # noinspection PyPep8Naming
    def _adjust_X_type_for_delegate(
        self, X: pd.DataFrame, *, to_numpy: Optional[bool] = None
    ) -> Any:
        # convert X before passing it to the delegate estimator
        return X.values if to_numpy else X

    def _adjust_y_type_for_delegate(
        self,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        *,
        to_numpy: Optional[bool] = None,
    ) -> Any:
        # convert y before passing it to the delegate estimator
        return y if y is None or not to_numpy else y.values

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
                for attr in self._native_estimator.__dir__()
                if not attr.startswith("_")
            ),
        }

    def __getattr__(self, name: str) -> Any:
        # get a non-private attribute of the delegate estimator
        if name.startswith("_"):
            # raise attribute error
            self.__getattribute__(name)
        else:
            try:
                return getattr(self._native_estimator, name)
            except AttributeError:
                # raise attribute error
                self.__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        # set a public attribute of the delegate estimator
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._native_estimator, name, value)


@inheritdoc(match="[see superclass]")
class TransformerWrapperDF(
    EstimatorWrapperDF[T_NativeTransformer],
    TransformerDF,
    Generic[T_NativeTransformer],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for native transformers conforming with the `scikit-learn`
    API.

    Transformer wrapper classes should be created using function
    :func:`.make_df_transformer`.
    """

    @property
    def feature_names_out_(self) -> pd.Index:
        """[see superclass]"""
        return self._check_feature_names_out(
            super().feature_names_out_, warning_stacklevel=2
        )

    @property
    def feature_names_original_(self) -> pd.Series:
        """[see superclass]"""
        feature_names_original_ = super().feature_names_original_
        self._check_feature_names_out(
            feature_names_original_.index, warning_stacklevel=2
        )
        return feature_names_original_

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
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params: Any
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
        self._check_parameter_types(X, None, expected_columns=self.feature_names_out_)

        transformed = self._inverse_transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.feature_names_in_
        )

    def _check_feature_names_out(
        self, wrapper_feature_names_out: pd.Index, *, warning_stacklevel: int
    ) -> pd.Index:
        try:
            native_feature_names_out = self.native_estimator.get_feature_names_out()
        except AttributeError:
            return wrapper_feature_names_out
        if not np.all(native_feature_names_out == wrapper_feature_names_out):
            warnings.warn(
                "conflicting output feature names: "
                "the output feature names recorded by this transformer are "
                f"{wrapper_feature_names_out} but the input feature names recorded "
                f"by the wrapped native transformer are {native_feature_names_out}",
                stacklevel=warning_stacklevel + 1,
            )
        return wrapper_feature_names_out

    def _reset_fit(self) -> None:
        try:
            # noinspection PyProtectedMember
            super()._reset_fit()
        finally:
            self._features_original = None

    # noinspection PyPep8Naming
    def _prepare_X_for_delegate(self, X: pd.DataFrame, *, inverse: bool = False) -> Any:
        return self._adjust_X_type_for_delegate(
            self._adjust_X_columns_for_delegate(X, inverse=inverse),
            to_numpy=inverse or None,
        )

    # noinspection PyPep8Naming
    def _adjust_X_columns_for_delegate(
        self, X: pd.DataFrame, *, inverse: Optional[bool] = None
    ) -> pd.DataFrame:
        if inverse:
            # when converting X for an inverse transform, ensure the data frame is
            # aligned with the output features, and convert the data frame to a
            # numpy array
            features_out = self.feature_names_out_
            if X.columns.is_(features_out):
                return X
            else:
                return X.reindex(columns=features_out, copy=False)
        else:
            return super()._adjust_X_columns_for_delegate(X)

    @staticmethod
    def _transformed_to_df(
        transformed: Union[pd.DataFrame, np.ndarray], index: pd.Index, columns: pd.Index
    ):
        if isinstance(transformed, pd.DataFrame):
            # noinspection PyProtectedMember
            TransformerWrapperDF._verify_df(
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
        return self.native_estimator.transform(self._prepare_X_for_delegate(X))

    # noinspection PyPep8Naming
    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params
    ) -> np.ndarray:
        return self.native_estimator.fit_transform(
            self._prepare_X_for_delegate(X),
            self._prepare_y_for_delegate(y),
            **fit_params,
        )

    # noinspection PyPep8Naming
    def _inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
        try:
            inverse_transform_fn = self.native_estimator.inverse_transform
        except AttributeError:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement method inverse_transform()"
            )

        return inverse_transform_fn(self._prepare_X_for_delegate(X, inverse=True))


@inheritdoc(match="[see superclass]")
class LearnerWrapperDF(
    LearnerDF,
    EstimatorWrapperDF[T_NativeLearner],
    Generic[T_NativeLearner],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for native learners conforming with the `scikit-learn`
    API.
    """

    #: Name of :class:`pd.Series` objects containing the predictions of single-output
    #: learners.
    #:
    #: See :meth:`~.LearnerDF.predict`.
    COL_PREDICTION = "prediction"

    # noinspection PyPep8Naming
    def predict(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_to_series_or_frame(
            X,
            self.native_estimator.predict(
                self._prepare_X_for_delegate(X), **predict_params
            ),
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


@inheritdoc(match="[see superclass]")
class SupervisedLearnerWrapperDF(
    SupervisedLearnerDF,
    LearnerWrapperDF[T_NativeSupervisedLearner],
    Generic[T_NativeSupervisedLearner],
    metaclass=ABCMeta,
):
    """
    ...
    """

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
            self._prepare_X_for_delegate(X),
            self._prepare_y_for_delegate(y),
            sample_weight,
        )


class RegressorWrapperDF(
    RegressorDF,
    SupervisedLearnerWrapperDF[T_NativeRegressor],
    Generic[T_NativeRegressor],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for native regressors conforming with the `scikit-learn`
    API.

    Regressor wrapper classes should be created using function
    :func:`.make_df_regressor`.
    """


@inheritdoc(match="[see superclass]")
class ClassifierWrapperDF(
    ClassifierDF,
    SupervisedLearnerWrapperDF[T_NativeClassifier],
    Generic[T_NativeClassifier],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for native classifiers conforming with the `scikit-learn`
    API.

    Classifier wrapper classes should be created using function
    :func:`.make_df_classifier`.
    """

    @property
    def classes_(self) -> Sequence[Any]:
        """[see superclass]"""
        self.ensure_fitted()
        # noinspection PyUnresolvedReferences
        return self._native_estimator.classes_

    # noinspection PyPep8Naming
    def predict_proba(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """[see superclass]"""

        self._ensure_delegate_method("predict_proba")

        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X,
            self.native_estimator.predict_proba(
                self._prepare_X_for_delegate(X), **predict_params
            ),
        )

    # noinspection PyPep8Naming
    def predict_log_proba(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """[see superclass]"""

        self._ensure_delegate_method("predict_log_proba")

        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X,
            self.native_estimator.predict_log_proba(
                self._prepare_X_for_delegate(X), **predict_params
            ),
        )

    # noinspection PyPep8Naming
    def decision_function(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""

        self._ensure_delegate_method("decision_function")

        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X,
            self.native_estimator.decision_function(
                self._prepare_X_for_delegate(X), **predict_params
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
        y: Union[pd.DataFrame, np.ndarray],
        classes: Optional[Sequence[Any]] = None,
    ) -> Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]:

        if classes is None:
            classes = getattr(self.native_estimator, "classes_", None)
            if classes is None:
                classes = pd.RangeIndex(self._get_n_outputs())

        if isinstance(y, pd.DataFrame):
            return y.set_axis(classes, axis=1, inplace=False)
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


# noinspection PyPep8Naming
@inheritdoc(match="[see superclass]")
class ClustererWrapperDF(
    ClustererDF,
    LearnerWrapperDF[T_NativeClusterer],
    Generic[T_NativeClusterer],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for native clusterers conforming with the scikit-learn
    API.

    Clusterer wrapper classes should be created using function
    :func:`.make_df_clusterer`.
    """

    COL_LABELS = "labels"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """[see superclass]"""
        super().__init__(*args, **kwargs)
        self._x_index: Optional[pd.Index] = None

    @property
    def labels_(self) -> pd.Series:
        """[see superclass]"""
        self.ensure_fitted()
        raw_labels = self._native_estimator.labels_

        return pd.Series(data=raw_labels, name=self.COL_LABELS, index=self._x_index)

    def fit_predict(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_predict_params: Any,
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""

        self._reset_fit()

        try:
            self._check_parameter_types(X, y)

            # Ignore a PyCharm warning that is caused by scikit-learn incorrectly
            # omitting optional arguments from the abstract method declaration
            # of ClassifierMixin.fit_predict():
            # noinspection PyArgumentList
            result = self._prediction_to_series_or_frame(
                X,
                self.native_estimator.fit_predict(
                    self._prepare_X_for_delegate(X),
                    self._prepare_y_for_delegate(y),
                    **fit_predict_params,
                ),
            )

            self._post_fit(X, y, **fit_predict_params)

        except Exception as cause:
            self._reset_fit()
            raise self._make_verbose_exception(
                self.fit_predict.__name__, cause
            ) from cause

        return result

    def _post_fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params,
    ) -> None:
        super()._post_fit(X, y, **fit_params)
        self._x_index = X.index

    def _reset_fit(self) -> None:
        super()._reset_fit()
        self._x_index = None


#
# Meta estimator wrappers
#


class MetaEstimatorWrapperDF(
    EstimatorWrapperDF[T_NativeEstimator],
    MetaEstimatorMixin,
    Generic[T_NativeEstimator],
    metaclass=ABCMeta,
):
    """
    Abstract base class of wrappers for estimators implementing
    :class:`sklearn.base.MetaEstimatorMixin`.

    A meta-estimator will call the methods of the embedded estimator using a modified
    copy of the `X` and `y` parameters, so we need to make sure that these are converted
    back to data frames.

    This class covers three variants used in sklearn:

    - one delegate estimator in attribute `estimator`
    - one delegate estimator in attribute `base_estimator`
    - multiple delegate estimators in attribute `estimators`
    """

    def _validate_delegate_estimator(self) -> None:
        def _unwrap_estimator(estimator: BaseEstimator) -> BaseEstimator:
            native_estimator = (
                estimator.native_estimator
                if isinstance(estimator, EstimatorWrapperDF)
                else estimator
            )
            # noinspection PyProtectedMember
            if isinstance(
                native_estimator, (EstimatorDF, sklearn_meta._BaseComposition)
            ) or not isinstance(native_estimator, (RegressorMixin, ClassifierMixin)):
                raise TypeError(
                    "sklearndf meta-estimators only accept simple regressors and "
                    f"classifiers, but got: {type(estimator).__name__}"
                )
            return cast(BaseEstimator, native_estimator)

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


#
# Stacking Estimator wrappers
#

# noinspection PyPep8Naming
@inheritdoc(match="""[see superclass]""")
class StackingEstimatorWrapperDF(
    SupervisedLearnerWrapperDF[T_NativeSupervisedLearner],
    # note: MetaEstimatorMixin is the first public class in the mro of _BaseStacking
    # MetaEstimatorMixin <-- _BaseHeterogeneousEnsemble <-- _BaseStacking
    MetaEstimatorMixin,
    Generic[T_NativeSupervisedLearner],
    metaclass=ABCMeta,
):
    """
    Abstract base class of wrappers for estimators implementing
    :class:`sklearn.ensemble._stacking._BaseStacking`.

    The stacking estimator will delegate to embedded estimators; this wrapper ensures
    the required conversions from and to numpy arrays as the native stacking estimator
    invokes the embedded estimators.
    """

    def fit(
        self: T_StackingEstimatorWrapperDF,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params: Any,
    ) -> T_StackingEstimatorWrapperDF:
        """[see superclass]"""

        class _ColumnNameFn:
            # noinspection PyMethodParameters
            def __call__(self_) -> Sequence[str]:
                return self._get_final_estimator_features_in()

            def __deepcopy__(self, memo=None) -> Any:
                # prevent a deep copy of this callable, to preserve reference to
                # stacking estimator being fitted
                return self

        native: T_NativeSupervisedLearner = self.native_estimator
        estimators: Sequence[Tuple[str, BaseEstimator]] = native.estimators
        final_estimator: BaseEstimator = native.final_estimator

        try:
            native.estimators = [
                (
                    name,
                    self._make_stackable_learner_df(estimator)
                    if isinstance(estimator, SupervisedLearnerDF)
                    else estimator,
                )
                for name, estimator in native.estimators
            ]
            native.final_estimator = self._make_learner_np_df(
                delegate=native.final_estimator or self._make_default_final_estimator(),
                column_names=_ColumnNameFn(),
            )

            # suppress a false warning from PyCharm's type checker
            # noinspection PyTypeChecker
            return super().fit(X, y, **fit_params)

        finally:
            native.estimators = estimators
            native.final_estimator = final_estimator

    @abstractmethod
    def _make_stackable_learner_df(
        self, learner: SupervisedLearnerDF
    ) -> "_StackableSupervisedLearnerDF":
        pass

    @abstractmethod
    def _make_learner_np_df(
        self, delegate: SupervisedLearnerDF, column_names: Callable[[], Sequence[str]]
    ) -> SupervisedLearnerNPDF:
        pass

    def _get_estimators_features_out(self) -> List[str]:
        return [name for name, estimator in self.estimators if estimator != "drop"]

    def _get_final_estimator_features_in(self) -> List[str]:
        names = self._get_estimators_features_out()
        if self.passthrough:
            return [*names, *self.estimators_[0].feature_names_in_]
        else:
            return names


# noinspection PyPep8Naming
@inheritdoc(match="""[see superclass]""")
class _StackableSupervisedLearnerDF(
    SupervisedLearnerDF, Generic[T_SupervisedLearnerDF]
):
    """
    Returns numpy arrays from all prediction functions, instead of pandas series or
    data frames.

    For use in stacking estimators that forward the predictions of multiple learners to
    one final learner.
    """

    def __init__(self, delegate: T_SupervisedLearnerDF) -> None:
        self.delegate = delegate

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.delegate.is_fitted

    def fit(
        self: T_StackableSupervisedLearnerDF,
        X: pd.DataFrame,
        y: np.ndarray = None,
        **fit_params: Any,
    ) -> T_StackableSupervisedLearnerDF:
        """[see superclass]"""
        self.delegate.fit(X, self._convert_y_to_series(X, y), **fit_params)
        return self

    def predict(self, X: pd.DataFrame, **predict_params: Any) -> np.ndarray:
        """[see superclass]"""
        return self.delegate.predict(X, **predict_params).values

    def score(
        self, X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[pd.Series] = None
    ) -> float:
        """[see superclass]"""
        return self.delegate.score(X, self._convert_y_to_series(X, y), sample_weight)

    def _get_features_in(self) -> pd.Index:
        return self.delegate._get_features_in()

    def _get_n_outputs(self) -> int:
        return self.delegate._get_n_outputs()

    @staticmethod
    def _convert_y_to_series(
        X: pd.DataFrame, y: Optional[np.ndarray]
    ) -> Optional[pd.Series]:
        if y is None:
            return y
        if not isinstance(y, np.ndarray):
            raise TypeError(
                f"expected numpy array for arg y but got a {type(y).__name__}"
            )
        if y.ndim != 1:
            raise TypeError(
                f"expected 1-d numpy array for arg y but got a {y.ndim}-d array"
            )
        if len(y) != len(X):
            raise ValueError(
                "args X and y have different lengths: "
                f"len(X)={len(X)} and len(y)={len(y)}"
            )
        return pd.Series(y, index=X.index)

    @staticmethod
    def _convert_prediction_to_numpy(
        prediction: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(prediction, list):
            return [proba.values for proba in prediction]
        else:
            return prediction.values


# noinspection PyPep8Naming
@inheritdoc(match="""[see superclass]""")
class _StackableClassifierDF(_StackableSupervisedLearnerDF[ClassifierDF], ClassifierDF):
    """[see superclass]"""

    @property
    def classes_(self) -> Sequence[Any]:
        """[see superclass]"""
        return self.delegate.classes_

    def predict_proba(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """[see superclass]"""
        return self._convert_prediction_to_numpy(
            self.delegate.predict_proba(X, **predict_params)
        )

    def predict_log_proba(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """[see superclass]"""
        return self._convert_prediction_to_numpy(
            self.delegate.predict_log_proba(X, **predict_params)
        )

    def decision_function(self, X: pd.DataFrame, **predict_params: Any) -> np.ndarray:
        """[see superclass]"""
        return self.delegate.decision_function(X, **predict_params).values


@inheritdoc(match="""[see superclass]""")
class _StackableRegressorDF(_StackableSupervisedLearnerDF[RegressorDF], RegressorDF):
    """[see superclass]"""


#
# wrapper factory methods
#


def make_df_estimator(
    native_estimator: Type[T_NativeEstimator],
    *,
    name: Optional[str] = None,
    base_wrapper: Optional[Type[EstimatorWrapperDF[T_NativeEstimator]]] = None,
    # The true return type is Type[EstimatorWrapperDF[T_NativeEstimator]],
    # but we provide the union with T_NativeTransformer to enable better
    # code completion in IDEs that derive attributes through static code inspection.
) -> Union[Type[EstimatorWrapperDF[T_NativeEstimator]], T_NativeTransformer]:
    """
    Create an augmented version of a given estimator that conforms with the
    `scikit-learn` API.

    The augmented version is realised as a wrapper class that

    - implements enhanced functionality introduced by :class:`.EstimatorDF`
    - adopts all additional methods and attributes from the wrapped native estimator
    - delegates relevant method calls and attribute access to the native estimator,
      thus replicating the original estimator's behaviour except for the enhanced
      functionality introduced by :class:`.EstimatorDF`

    :param native_estimator: the native estimator to be augmented
    :param name: the name of the resulting augmented estimator, defaults to the name
        of the native estimator with "DF" appended
    :param base_wrapper: the wrapper class used to create the augmented version
    :return: the augmented estimator class
    """
    return _wrap(
        native_estimator=native_estimator,
        name=name,
        base_wrapper=base_wrapper,
        native_estimator_bound=BaseEstimator,
        base_wrapper_bound=EstimatorWrapperDF,
    )


def make_df_transformer(
    native_transformer: Type[T_NativeTransformer],
    *,
    name: Optional[str] = None,
    base_wrapper: Type[TransformerWrapperDF[T_NativeTransformer]],
    # The true return type is Type[TransformerWrapperDF[T_NativeEstimator]],
    # but we provide the union with T_NativeTransformer to enable better
    # code completion in IDEs that derive attributes through static code inspection.
) -> Union[Type[TransformerWrapperDF[T_NativeTransformer]], T_NativeTransformer]:
    """
    Create an augmented version of a given transformer that conforms with the
    `scikit-learn` API.

    The augmented version is realised as a wrapper class that

    - implements enhanced functionality introduced by :class:`.TransformerDF`
    - adopts all additional methods and attributes from the wrapped native transformer
    - delegates relevant method calls and attribute access to the native transformer,
      thus replicating the original transformer's behaviour except for the enhanced
      functionality introduced by :class:`.TransformerDF`

    :param native_transformer: the native transformer to be augmented
    :param name: the name of the resulting augmented transformer, defaults to the name
        of the native transformer with "DF" appended
    :param base_wrapper: the wrapper class used to create the augmented version
    :return: the augmented transformer class
    """
    return _wrap(
        native_estimator=native_transformer,
        name=name,
        base_wrapper=base_wrapper,
        native_estimator_bound=TransformerMixin,
        base_wrapper_bound=TransformerWrapperDF,
    )


def make_df_classifier(
    native_classifier: Type[T_NativeClassifier],
    *,
    name: Optional[str] = None,
    base_wrapper: Optional[Type[ClassifierWrapperDF[T_NativeClassifier]]] = None,
    # The true return type is Type[ClassifierWrapperDF[T_NativeEstimator]],
    # but we provide the union with T_NativeClassifier to enable better
    # code completion in IDEs that derive attributes through static code inspection.
) -> Union[Type[ClassifierWrapperDF[T_NativeClassifier]], T_NativeTransformer]:
    """
    Create an augmented version of a given classifier that conforms with the
    `scikit-learn` API.

    The augmented version is realised as a wrapper class that

    - implements enhanced functionality introduced by :class:`.ClassifierDF`
    - adopts all additional methods and attributes from the wrapped native classifier
    - delegates relevant method calls and attribute access to the native classifier,
      thus replicating the original classifier's behaviour except for the enhanced
      functionality introduced by :class:`.ClassifierDF`

    :param native_classifier: the native classifier to be augmented
    :param name: the name of the resulting augmented classifier, defaults to the name
        of the native classifier with "DF" appended
    :param base_wrapper: the wrapper class used to create the augmented version
    :return: the augmented classifier class
    """
    return _wrap(
        native_estimator=native_classifier,
        name=name,
        base_wrapper=base_wrapper,
        native_estimator_bound=ClassifierMixin,
        base_wrapper_bound=ClassifierWrapperDF,
    )


def make_df_regressor(
    native_regressor: Type[T_NativeRegressor],
    *,
    name: Optional[str] = None,
    base_wrapper: Optional[Type[RegressorWrapperDF[T_NativeRegressor]]] = None,
    # The true return type is Type[RegressorWrapperDF[T_NativeEstimator]],
    # but we provide the union with T_NativeRegressor to enable better
    # code completion in IDEs that derive attributes through static code inspection.
) -> Union[Type[RegressorWrapperDF[T_NativeRegressor]], T_NativeTransformer]:
    """
    Create an augmented version of a given regressor that conforms with the
    `scikit-learn` API.

    The augmented version is realised as a wrapper class that

    - implements enhanced functionality introduced by :class:`.RegressorDF`
    - adopts all additional methods and attributes from the wrapped native regressor
    - delegates relevant method calls and attribute access to the native regressor,
      thus replicating the original regressor's behaviour except for the enhanced
      functionality introduced by :class:`.RegressorDF`

    :param native_regressor: the native regressor to be augmented
    :param name: the name of the resulting augmented regressor, defaults to the name
        of the native regressor with "DF" appended
    :param base_wrapper: the wrapper class used to create the augmented version
    :return: the augmented regressor class
    """
    return _wrap(
        native_estimator=native_regressor,
        name=name,
        base_wrapper=base_wrapper,
        native_estimator_bound=RegressorMixin,
        base_wrapper_bound=RegressorWrapperDF,
    )


def make_df_clusterer(
    native_clusterer: Type[T_NativeClusterer],
    *,
    name: Optional[str] = None,
    base_wrapper: Optional[Type[ClustererWrapperDF[T_NativeClusterer]]] = None,
    # The true return type is Type[ClustererWrapperDF[T_NativeEstimator]],
    # but we provide the union with T_NativeClusterer to enable better
    # code completion in IDEs that derive attributes through static code inspection.
) -> Union[Type[ClustererWrapperDF[T_NativeClusterer]], T_NativeClusterer]:
    """
    Create an augmented version of a given clusterer that conforms with the
    scikit-learn API.

    The augmented version is realised as a wrapper class that

    - implements enhanced functionality introduced by :class:`.ClustererDF`
    - adopts all additional methods and attributes from the wrapped native clusterer
    - delegates relevant method calls and attribute access to the native clusterer,
      thus replicating the original clusterer's behaviour except for the enhanced
      functionality introduced by :class:`.ClustererDF`

    :param native_clusterer: the native clusterer to be augmented
    :param name: the name of the resulting augmented clusterer, defaults to the name
        of the native clusterer with "DF" appended
    :param base_wrapper: the wrapper class used to create the augmented version
    :return: the augmented clusterer class
    """
    return _wrap(
        native_estimator=native_clusterer,
        name=name,
        base_wrapper=base_wrapper,
        native_estimator_bound=ClusterMixin,
        base_wrapper_bound=ClustererWrapperDF,
    )


#
# private factory implementation
#

_df_wrapper_classes: Dict[str, Type[EstimatorWrapperDF]] = cast(
    Dict[str, Type[EstimatorWrapperDF]], WeakValueDictionary()
)


def _wrap(
    native_estimator: Type[T_NativeEstimator],
    *,
    name: Optional[str] = None,
    base_wrapper: Optional[Type[EstimatorWrapperDF[T_NativeEstimator]]] = None,
    native_estimator_bound: Optional[Type] = None,
    base_wrapper_bound: Type[T_EstimatorWrapperDF],
) -> Type[T_EstimatorWrapperDF]:
    """
    Class decorator wrapping a :class:`sklearn.base.BaseEstimator` in a
    :class:`EstimatorWrapperDF`.

    :param native_estimator: the native `scikit-learn` estimator class to wrap; must
        be a subclass of :class:`~sklearn.base.BaseEstimator`
    :param name: name of the resulting class
    :param base_wrapper: optional parameter indicating the
        :class:`EstimatorWrapperDF` class to be used for wrapping; defaults to
        :class:`EstimatorWrapperDF`
    :param native_estimator_bound: base class that must be subclassed by arg
        ``native_estimator`` (optional)
    :param base_wrapper_bound: base class that must be subclassed by arg
        ``df_wrapper``
    :return: the resulting ``EstimatorWrapperDF`` with ``native_estimator`` as
        the delegate estimator
    """

    # validate arg native_estimator
    if native_estimator_bound and not issubclass(
        native_estimator, native_estimator_bound
    ):
        raise ValueError(
            f"arg native_estimator is not a subclass of "
            f"{native_estimator_bound.__name__}: {native_estimator}"
        )
    if not issubclass(native_estimator, BaseEstimator):
        raise ValueError(
            f"arg native_estimator is not a subclass of "
            f"{BaseEstimator.__name__}: {native_estimator}"
        )

    # validate arg base_wrapper
    if base_wrapper is None:
        base_wrapper = base_wrapper_bound
    elif not issubclass(base_wrapper, base_wrapper_bound):
        raise ValueError(
            f"arg base_wrapper is not a subclass of "
            f"{base_wrapper_bound.__name__}: {base_wrapper}"
        )

    # use the customized name if given, else append DF to the native name
    name = name or (native_estimator.__name__ + "DF")

    # determine the sklearn estimator we are wrapping
    if name in _df_wrapper_classes:
        raise TypeError(
            "Estimator wrapper type is already defined: "
            f"{name} = {make_df_estimator.__name__}"
            f"({native_estimator.__name__}, "
            f"base_wrapper={base_wrapper.__name__})"
        )

    # dynamically create the wrapper class
    df_wrapper_class = _make_df_wrapper_class(native_estimator, name, base_wrapper)

    # finally, register the newly created class in our global WeakValueDictionary
    _df_wrapper_classes[name] = df_wrapper_class

    return df_wrapper_class


def _make_df_wrapper_class(
    native_estimator: Type[T_NativeEstimator],
    name: str,
    base_wrapper: Type[T_EstimatorWrapperDF],
) -> Type[T_EstimatorWrapperDF]:
    # noinspection PyMissingOrEmptyDocstring
    class WrapperDF(base_wrapper):  # type: ignore
        def __init__(self, *args, **kwargs: Any) -> None:
            # we need to create this __init__ method in order to apply the signature
            # of the native estimator's __init__ method
            super().__init__(*args, **kwargs)

        def __reduce__(
            self,
        ) -> Tuple[
            Callable[[str, Type[BaseEstimator], Type[EstimatorWrapperDF]], object],
            Tuple[str, Type[T_NativeEstimator], Type[T_EstimatorWrapperDF]],
            Dict[str, Any],
        ]:
            # pickling by default does not work for dynamically created classes,
            # so we need to customize it
            return (
                _get_wrapper_instance,
                (name, native_estimator, base_wrapper),
                self.__dict__,
            )

    # set the name
    WrapperDF.__name__ = WrapperDF.__qualname__ = name

    # add link to the wrapped class, for use in python module 'inspect'
    WrapperDF.__wrapped__ = native_estimator

    # set the module to this module's name
    WrapperDF.__module__ = __name__

    # … but do not keep the docstring of __init__
    WrapperDF.__init__.__doc__ = None

    # mirror all attributes of the wrapped sklearn class, as long
    # as they are not inherited from the wrapper base class
    _mirror_attributes(
        wrapper=WrapperDF,
        native_estimator=native_estimator,
        wrapper_module=native_estimator.__module__,
    )

    # adopt the initializer signature of the wrapped sklearn estimator
    _update_wrapper(
        wrapper=WrapperDF.__init__,
        wrapped=native_estimator.__init__,
        wrapper_module=native_estimator.__module__,
        wrapper_parent=name,
    )

    # adopt the class docstring of the wrapped sklearn estimator …
    _update_class_docstring(
        df_estimator_type=WrapperDF,
        sklearn_native_estimator_type=native_estimator,
    )

    return WrapperDF


def _get_wrapper_instance(
    name: str,
    native_estimator: Type[BaseEstimator],
    base_wrapper: Type[EstimatorWrapperDF],
) -> object:
    try:
        wrapper_cls = _df_wrapper_classes[name]
    except KeyError:
        wrapper_cls = make_df_estimator(
            native_estimator=native_estimator,
            name=name,
            base_wrapper=base_wrapper,
        )
    return wrapper_cls.__new__(wrapper_cls)


def _mirror_attributes(
    wrapper: Type[EstimatorWrapperDF[T_NativeEstimator]],
    native_estimator: Type[T_NativeEstimator],
    wrapper_module: str,
) -> None:

    wrapper_name = wrapper.__name__
    wrapper_attributes: Set[str] = set(dir(wrapper))

    for name, member in vars(native_estimator).items():

        if member is None or name.startswith("_") or name in wrapper_attributes:
            continue

        alias = _make_alias(
            module=wrapper_module,
            class_=wrapper_name,
            name=name,
            delegate_cls=native_estimator,
            delegate=member,
        )
        if alias is not None:
            setattr(wrapper, name, alias)


def _make_alias(
    module: str,
    class_: str,
    name: str,
    delegate_cls: type,
    delegate: Union[Callable, Any],
) -> Union[Callable, property, None]:
    def _make_forwarder(delegate: Callable) -> Callable:
        # noinspection PyShadowingNames
        def _forwarder(self, *args, **kwargs: Any) -> Any:
            return delegate(self._native_estimator, *args, **kwargs)

        return _forwarder

    class_name = _full_name(cls=delegate_cls)
    full_name = f"{class_name}.{name}"

    if inspect.isfunction(delegate):
        function = _make_forwarder(delegate)
        _update_wrapper(
            wrapper=function,
            wrapped=delegate,
            wrapper_module=module,
            wrapper_parent=class_,
        )
        function.__doc__ = f"See :meth:`{full_name}`"
        return function

    elif inspect.isdatadescriptor(delegate):
        delegate_prop: property = cast(property, delegate)
        # noinspection PyShadowingNames
        return property(
            fget=lambda self: delegate_prop.__get__(self._native_estimator),
            fset=lambda self, value: delegate_prop.__set__(
                self._native_estimator, value
            ),
            fdel=lambda self: delegate_prop.__delete__(self._native_estimator),
            doc=f"See documentation of :class:`{class_name}`.",
        )

    else:
        return None


def _update_wrapper(
    wrapper: Any,
    wrapped: Any,
    wrapper_module: str,
    wrapper_parent: str,
) -> None:
    updated = update_wrapper(wrapper, wrapped, assigned=("__name__", "__annotations__"))
    updated.__module__ = wrapper_module
    if wrapper_parent:
        updated.__qualname__ = f"{wrapper_parent}.{updated.__name__}"
    else:
        updated.__qualname__ = updated.__name__


def _update_class_docstring(
    df_estimator_type: Type[EstimatorWrapperDF[T_NativeEstimator]],
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
    replicates all parameters and behaviours of class :class:`~{estimator_name}`.
"""
            ),
        ]
    )


def _full_name(cls: type):
    # get the full name of the class, including the module prefix

    try:
        module_name = cls.__module__
    except AttributeError as e:
        raise RuntimeError(f"cannot get module for {cls}") from e

    module_name = public_module_prefix(module_name)

    return f"{module_name}.{cls.__qualname__}"


#
# validate __all__
#

__tracker.validate()
