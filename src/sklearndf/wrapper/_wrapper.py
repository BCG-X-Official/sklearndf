"""
Wrappers around native `scikit-learn` estimators.

`sklearndf` wrappers accept and return data frames (while `scikit-learn` transformers
usually return a numpy arrays, and may not accept data frames as input).
Otherwise, the wrappers are designed to precisely mirror the API and behavior of the
native estimators they wrap.

The wrappers also implement the additional column attributes introduced by `sklearndf`,
:meth:`~EstimatorDF.feature_names_in_`, :meth:`~TransformerDF.feature_names_out_`, and
:meth:`~TransformerDF.feature_names_original_`.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from abc import ABCMeta
from collections.abc import Iterable, Mapping, Sequence
from functools import update_wrapper
from typing import Any, Callable, Generic, Optional, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.utils.metaestimators as sklearn_meta
from scipy import sparse
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
)

from pytools.api import AllTracker, inheritdoc, public_module_prefix

from sklearndf import (
    ClassifierDF,
    ClusterDF,
    EstimatorDF,
    LearnerDF,
    RegressorDF,
    SupervisedLearnerDF,
    TransformerDF,
    __sklearn_1_6__,
    __sklearn_version__,
)

if __sklearn_version__ >= __sklearn_1_6__:
    from sklearn.utils import Tags

log = logging.getLogger(__name__)

__all__ = [
    "ClassifierWrapperDF",
    "ClusterWrapperDF",
    "EstimatorWrapperDF",
    "EstimatorWrapperDFMeta",
    "LearnerWrapperDF",
    "MetaEstimatorWrapperDF",
    "RegressorWrapperDF",
    "SupervisedLearnerWrapperDF",
    "TransformerWrapperDF",
]


#
# type variables
#

T = TypeVar("T")
T_Callable = TypeVar("T_Callable", bound=Callable[..., Any])
T_Target = TypeVar("T_Target", bound=Optional[Union[pd.Series, pd.DataFrame]])

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
T_NativeCluster = TypeVar("T_NativeCluster", bound=ClusterMixin)

T_EstimatorWrapperDF = TypeVar(
    "T_EstimatorWrapperDF", bound="EstimatorWrapperDF[BaseEstimator]"
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

    #: the native class wrapped by the DF wrapper class
    __wrapped__: type[T_NativeEstimator]

    def __new__(
        mcs: type[EstimatorWrapperDFMeta[T_NativeEstimator]],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        native: Optional[T_NativeEstimator] = None,
        **kwargs: Any,
    ) -> EstimatorWrapperDFMeta[T_NativeEstimator]:
        if native in bases:
            raise TypeError(
                f"wrapper class {name} must not state native class "
                f"{native.__name__} as a base class"
            )

        # noinspection PyArgumentList,PyTypeChecker
        cls: EstimatorWrapperDFMeta[T_NativeEstimator] = super().__new__(
            mcs, name, bases, namespace, **kwargs
        )

        if native is None:
            return cls

        wrapper_cls = cast(type[EstimatorWrapperDF[T_NativeEstimator]], cls)

        if not issubclass(native, wrapper_cls.__native_base_class__):
            raise TypeError(
                f"native class {native.__name__} "
                f"cannot be used with wrapper class {wrapper_cls.__name__} "
                f"because it does not implement "
                f"{wrapper_cls.__native_base_class__.__name__}"
            )

        wrapper_cls.__wrapped__ = native
        wrapper_cls.__signature__ = inspect.signature(native)
        wrapper_init = _make_init(wrapper_cls)
        wrapper_init.__signature__ = inspect.signature(native.__init__)  # type: ignore
        setattr(wrapper_cls, "__init__", wrapper_init)
        wrapper_module = namespace["__module__"]

        _mirror_attributes(
            wrapper_class=wrapper_cls,
            native_estimator=native,
            wrapper_module=wrapper_module,
        )
        # adopt the initializer signature of the wrapped sklearn estimator
        _update_wrapper(
            wrapper=wrapper_init,
            wrapped=native.__init__,
            wrapper_module=wrapper_module,
            wrapper_parent=name,
        )
        # adopt the class docstring of the wrapped sklearn estimator
        _update_class_docstring(
            df_estimator_type=wrapper_cls,
            sklearn_native_estimator_type=native,
        )

        return wrapper_cls

    @property
    def native_estimator_type(cls) -> type[T_NativeEstimator]:
        """
        The type of native estimator that instances of this wrapper class delegate to.
        """
        return cls.__wrapped__


def _make_init(cls: type) -> Callable[..., None]:
    def __init__(self: type, *args: Any, **kwargs: Any) -> None:
        """"""
        cast(EstimatorWrapperDF, super(cls, self)).__init__(  # type: ignore
            *args, **kwargs
        )

    return __init__


@inheritdoc(match="[see superclass]")
class EstimatorWrapperDF(
    EstimatorDF, Generic[T_NativeEstimator], metaclass=EstimatorWrapperDFMeta
):
    """
    Base class of DF wrappers for native estimators conforming with the `scikit-learn`
    API.
    """

    __native_base_class__ = BaseEstimator
    __ARG_FITTED_DELEGATE_CONTEXT = "__EstimatorWrapperDF_fitted"

    #: The native estimator that this wrapper delegates to.
    _native_estimator: T_NativeEstimator

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        :param args: positional arguments to use when initializing a new delegate
            estimator
        :param kwargs: keyword arguments to use when initializing a new delegate
            estimator
        """
        super().__init__()
        self._features_in: Optional[pd.Index] = None
        self._outputs: Optional[list[str]] = None

        # check if a fitted estimator was passed by class method is_fitted
        fitted_delegate_context = cast(
            Optional[tuple[T_NativeEstimator, pd.Index, int]],
            kwargs.get(EstimatorWrapperDF.__ARG_FITTED_DELEGATE_CONTEXT, None),
        )

        _native_estimator: T_NativeEstimator

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

    def __new__(
        cls: type[T_EstimatorWrapperDF], *args: Any, **kwargs: Any
    ) -> T_EstimatorWrapperDF:
        try:
            cls.__wrapped__
        except AttributeError:
            raise TypeError(
                f"cannot instantiate wrapper class {cls.__name__}: "
                "need to specify class argument 'native' in class definition"
            )
        else:
            return cast(type[EstimatorDF], super()).__new__(cls)

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

    @property
    def n_features_in_(self) -> int:
        """[see superclass]"""
        return self._check_n_features_in(super().n_features_in_, warning_stacklevel=2)

    def _check_feature_names_in(
        self, wrapper_feature_names_in: pd.Index, *, warning_stacklevel: int
    ) -> pd.Index:
        # Check that the given feature names are the same as the ingoing feature names
        # recorded by the native estimator, if present. Issue a warning if the feature
        # names differ.
        # Return the same feature names that were passed to this method.

        # noinspection PyBroadException
        try:
            feature_names_in_native = self.native_estimator.feature_names_in_
        except Exception:
            return wrapper_feature_names_in

        if not np.array_equal(wrapper_feature_names_in.values, feature_names_in_native):
            warnings.warn(
                "conflicting input feature names: "
                "the input feature names recorded by this estimator are "
                f"{wrapper_feature_names_in}, but the input feature names recorded by "
                f"the wrapped native estimator are {feature_names_in_native}",
                stacklevel=warning_stacklevel + 1,
            )
        return wrapper_feature_names_in

    def _check_n_features_in(
        self, wrapper_n_features: int, *, warning_stacklevel: int
    ) -> int:
        # Check that the given number of features is the same as the number of features
        # recorded by the native estimator, if present. Issue a warning if the number of
        # features differ.
        # Return the same number of features that were passed to this method.

        # noinspection PyBroadException
        try:
            n_features_native = self.native_estimator.n_features_in_
        except Exception:
            return wrapper_n_features

        if wrapper_n_features != n_features_native:
            warnings.warn(
                "conflicting number of features: "
                "the number of features recorded by this estimator is "
                f"{wrapper_n_features}, but the number of features recorded by "
                f"the wrapped native estimator is {n_features_native}",
                stacklevel=warning_stacklevel + 1,
            )
        return wrapper_n_features

    @property
    def _estimator_type(self) -> Optional[str]:
        try:
            # noinspection PyProtectedMember
            return cast(str, self.native_estimator._estimator_type)
        except AttributeError:
            return None

    if __sklearn_version__ >= __sklearn_1_6__:

        def __sklearn_tags__(self) -> Tags:
            return self.native_estimator.__sklearn_tags__()

    @classmethod
    def from_fitted(
        cls: type[T_EstimatorWrapperDF],
        estimator: T_NativeEstimator,
        features_in: pd.Index,
        n_outputs: int,
    ) -> T_EstimatorWrapperDF:
        """
        Make a new wrapped DF estimator, delegating to a given native estimator that
        has already been fitted.

        :param estimator: the fitted native estimator to use as the delegate
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
        return cast(Mapping[str, Any], self._native_estimator.get_params(deep=deep))

    def set_params(self: T_EstimatorWrapperDF, **params: Any) -> T_EstimatorWrapperDF:
        """[see superclass]"""
        self._native_estimator.set_params(**params)
        return self

    # noinspection PyPep8Naming
    def fit(
        self: T_EstimatorWrapperDF,
        X: Union[pd.DataFrame, pd.Series],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params: Any,
    ) -> T_EstimatorWrapperDF:
        """[see superclass]"""

        self._reset_fit()

        try:
            X, y = self._validate_parameter_types(X, y)
            self._fit(X, y, **fit_params)
            self._post_fit(X, y, **fit_params)

        except Exception as cause:
            self._reset_fit()
            raise self._make_verbose_exception(self.fit.__name__, cause) from cause

        return self

    def _validate_delegate_estimator(self) -> None:
        # Called as the last step of the estimator wrapper's constructor.
        # No validation required by default; to be overloaded as needed.
        pass

    def _get_features_in(self) -> pd.Index:
        assert self._features_in is not None, "estimator is fitted"
        return self._features_in

    def _get_outputs(self) -> Optional[list[str]]:
        return self._outputs

    def _reset_fit(self) -> None:
        self._features_in = None
        self._outputs = None

    # noinspection PyPep8Naming
    def _fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        **fit_params: Any,
    ) -> T_NativeEstimator:
        # noinspection PyUnresolvedReferences
        return cast(
            T_NativeEstimator,
            self._native_estimator.fit(
                self._prepare_X_for_delegate(X),
                self._prepare_y_for_delegate(y),
                **fit_params,
            ),
        )

    # noinspection PyPep8Naming,PyUnusedLocal
    def _post_fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params: Any,
    ) -> None:
        self._features_in = X.columns.rename(self.COL_FEATURE)
        if y is None:
            self._outputs = None
        elif isinstance(y, pd.Series):
            self._outputs = [y.name]
        else:
            self._outputs = y.columns.tolist()

    # noinspection PyPep8Naming
    def _validate_parameter_types(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: T_Target,
        *,
        expected_columns: Optional[pd.Index] = None,
    ) -> tuple[pd.DataFrame, T_Target]:
        # Check that the X and y parameters are valid data frames and series,
        # and return X as a data frame and y as a series or data frame.
        #
        # If X is a series, convert it to a data frame with a single column.
        #
        # If expected_columns is not None, check that the columns of X match
        # the expected columns.

        if isinstance(X, pd.Series):
            if X.name is None:
                raise ValueError(
                    "the name of the series passed as arg X must not be None"
                )
            X = X.to_frame()
        elif not isinstance(X, pd.DataFrame):
            raise TypeError("arg X must be a DataFrame or a Series")

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

        return X, y

    @staticmethod
    def _verify_df(
        df_name: str,
        df: pd.DataFrame,
        expected_columns: pd.Index,
        expected_index: Optional[pd.Index] = None,
    ) -> None:
        def _verify_labels(axis: str, actual: pd.Index, expected: pd.Index) -> None:
            missing_columns = expected.difference(actual)
            extra_columns = actual.difference(expected)
            error_detail: list[str] = []

            # check that we have the expected number of columns
            if len(actual) != len(expected):
                error_detail.append(
                    f"expected {len(expected)} elements but got {len(actual)}"
                )

            # check that all the expected columns are in place
            if len(missing_columns) > 0:
                error_detail.append(
                    f"missing elements: "
                    f"{', '.join(str(item) for item in missing_columns)}"
                )

            # check that there are no unexpected columns
            if len(extra_columns) > 0:
                error_detail.append(
                    f"extra elements: "
                    f"{', '.join(str(item) for item in extra_columns)}"
                )

            # raise an exception if we have encountered any errors
            if error_detail:
                raise ValueError(
                    f"{df_name} data frame does not have expected {axis} index "
                    f"({'; '.join(error_detail)})"
                )

        _verify_labels(axis="column", actual=df.columns, expected=expected_columns)
        if expected_index is not None:
            _verify_labels(axis="row", actual=df.index, expected=expected_index)

    def _validate_delegate_attribute(self, attribute_name: str) -> None:
        if not hasattr(self.native_estimator, attribute_name):
            raise AttributeError(
                f"delegate estimator of type {type(self.native_estimator).__name__} "
                f"does not have attribute {attribute_name}"
            )

    # noinspection PyPep8Naming
    def _prepare_X_for_delegate(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, npt.NDArray[Any], sparse.csr_matrix]:
        # convert X before passing it to the delegate estimator
        return self._adjust_X_type_for_delegate(self._adjust_X_columns_for_delegate(X))

    def _prepare_y_for_delegate(
        self, y: Optional[Union[pd.Series, pd.DataFrame]]
    ) -> Optional[Union[pd.Series, pd.DataFrame, npt.NDArray[Any], sparse.csr_matrix]]:
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
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, npt.NDArray[Any], sparse.csr_matrix]:
        # Convert X before passing it to the delegate estimator.
        # By default, does nothing, but can be overridden.
        return X

    def _adjust_y_type_for_delegate(
        self, y: Optional[Union[pd.Series, pd.DataFrame]]
    ) -> Optional[Union[pd.Series, pd.DataFrame, npt.NDArray[Any], sparse.csr_matrix]]:
        # convert y before passing it to the delegate estimator
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
                for attr in self._native_estimator.__dir__()
                if not attr.startswith("_")
            ),
        }

    def __getattr__(self, name: str) -> Any:
        # This method is only called if the attribute name is not found in the
        # instance's dictionary, and __getattribute__() has raised an AttributeError.

        # For private attributes, give up and raise attribute error.
        if name.startswith("_"):
            # The following will raise an AttributeError
            self.__getattribute__(name)
        else:
            # For public attributes, try to get the attribute from the delegate
            # estimator. If the attribute is not found, raise an attribute error.
            try:
                return getattr(self._native_estimator, name)
            except AttributeError:
                # The following will raise an AttributeError
                self.__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        # This method is called whenever an attribute assignment is attempted.

        # For private attributes, set the attribute in this wrapper object.
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            # For public attributes, set the attribute in this wrapper object only
            # if it is already defined. Otherwise, set the attribute in the delegate
            # estimator.

            try:
                self.__getattribute__(name)
            except AttributeError:
                # The attribute is not defined in this wrapper object, so set it in
                # the delegate estimator.
                setattr(self._native_estimator, name, value)
            else:
                # The attribute is defined in this wrapper object, so set it here.
                super().__setattr__(name, value)


@inheritdoc(match="[see superclass]")
class TransformerWrapperDF(
    TransformerDF,
    EstimatorWrapperDF[T_NativeTransformer],
    Generic[T_NativeTransformer],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for native transformers conforming with the `scikit-learn`
    API.
    """

    __native_base_class__ = TransformerMixin

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
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """[see superclass]"""
        X, _ = self._validate_parameter_types(X, None)

        transformed = self._transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.feature_names_out_
        )

    # noinspection PyPep8Naming
    def fit_transform(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[pd.Series] = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        """[see superclass]"""
        self._reset_fit()

        try:
            X, y = self._validate_parameter_types(X, y)
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
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """[see superclass]"""
        X, _ = self._validate_parameter_types(
            X, None, expected_columns=self.feature_names_out_
        )

        transformed = self._inverse_transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.feature_names_in_
        )

    def _check_feature_names_out(
        self, wrapper_feature_names_out: pd.Index, *, warning_stacklevel: int
    ) -> pd.Index:
        # noinspection PyBroadException
        try:
            native_feature_names_out = self.native_estimator.get_feature_names_out(
                self._get_features_in().values
            )
        except Exception:
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
    def _prepare_X_for_delegate(
        self, X: pd.DataFrame, *, inverse: bool = False
    ) -> Union[pd.DataFrame, npt.NDArray[Any]]:
        x_adjusted = self._adjust_X_columns_for_delegate(X, inverse=inverse)
        if inverse:
            # when doing an inverse transform, we need X as a numpy array
            return x_adjusted.values
        else:
            return self._adjust_X_type_for_delegate(x_adjusted)

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
        transformed: Union[pd.DataFrame, npt.NDArray[Any], sparse.spmatrix],
        index: pd.Index,
        columns: pd.Index,
    ) -> pd.DataFrame:
        if isinstance(transformed, pd.DataFrame):
            if transformed.columns.name != EstimatorDF.COL_FEATURE:
                transformed = transformed.rename_axis(columns=EstimatorDF.COL_FEATURE)
            # noinspection PyProtectedMember
            TransformerWrapperDF._verify_df(
                df_name="transformed",
                df=transformed,
                expected_columns=columns,
                expected_index=index,
            )
            return transformed
        elif isinstance(transformed, sparse.spmatrix):
            return pd.DataFrame.sparse.from_spmatrix(
                data=transformed, index=index, columns=columns
            )
        else:
            return pd.DataFrame(data=transformed, index=index, columns=columns)

    # noinspection PyPep8Naming
    def _transform(
        self, X: pd.DataFrame
    ) -> Union[npt.NDArray[Any], sparse.csr_matrix, pd.DataFrame]:
        return cast(
            Union[npt.NDArray[Any], sparse.csr_matrix, pd.DataFrame],
            self.native_estimator.transform(self._prepare_X_for_delegate(X)),
        )

    # noinspection PyPep8Naming
    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params: Any
    ) -> Union[npt.NDArray[Any], sparse.csr_matrix, pd.DataFrame]:
        return cast(
            Union[npt.NDArray[Any], sparse.csr_matrix, pd.DataFrame],
            self.native_estimator.fit_transform(
                self._prepare_X_for_delegate(X),
                self._prepare_y_for_delegate(y),
                **fit_params,
            ),
        )

    # noinspection PyPep8Naming
    def _inverse_transform(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        try:
            inverse_transform_fn = cast(
                Callable[[npt.NDArray[Any]], npt.NDArray[Any]],
                self.native_estimator.inverse_transform,
            )
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
):
    """
    Base class of DF wrappers for native `learners` conforming with the `scikit-learn`
    API.

    Learners in `scikit-learn` typically are regressors, classifiers, or clusterers.
    """

    #: Name of :class:`~pandas.Series` objects containing the predictions of
    #: single-output learners.
    #:
    #: See :meth:`~.LearnerDF.predict`.
    COL_PREDICTION = "prediction"

    # noinspection PyPep8Naming
    def predict(
        self, X: Union[pd.Series, pd.DataFrame], **predict_params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""
        X, _ = self._validate_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_to_series_or_frame(
            X,
            self.native_estimator.predict(
                self._prepare_X_for_delegate(X), **predict_params
            ),
        )

    # noinspection PyPep8Naming
    def _prediction_to_series_or_frame(
        self, X: pd.DataFrame, y: Union[npt.NDArray[Any], pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        if len(y) != len(X):
            raise ValueError(
                f"length of prediction ({len(y)}) does not match length of X ({len(X)})"
            )

        outputs: Optional[list[str]] = self._get_outputs()

        if y.ndim == 1:
            if outputs is None:
                # in case predict() was called without first calling fit(),
                # no outputs are known, so we use the default output name
                outputs = [self.COL_PREDICTION]
            elif len(outputs) != 1:
                raise ValueError(
                    f"expected {len(outputs)} predictions, but got 1 prediction"
                )
        elif y.ndim == 2:
            if outputs is None:
                # in case predict() was called without first calling fit(),
                # no outputs are known, so we use default output names
                outputs = [
                    f"{self.COL_PREDICTION}_{output}" for output in range(y.shape[1])
                ]
            elif y.shape[1] != len(outputs):
                raise ValueError(
                    f"expected {len(outputs)} predictions, "
                    f"but got {y.shape[1]} prediction{'' if y.shape[1] == 1 else 's'}"
                )
        else:
            raise ValueError(
                f"got {y.ndim}-dimensional prediction, "
                f"but expected 1- or 2-dimensional prediction"
            )

        if isinstance(y, pd.Series):
            return y.rename(outputs[0])
        elif isinstance(y, pd.DataFrame):
            return y.set_axis(outputs, axis=1)
        elif isinstance(y, np.ndarray):
            if len(y) == len(X):
                # predictions are usually provided as a numpy array the same length as X
                if y.ndim == 1:
                    # single-output predictions yield a numpy array of shape (n_samples)
                    return pd.Series(data=y, name=outputs[0], index=X.index)
                if y.ndim == 2:
                    # multi-output predictions yield a numpy array of shape (n_samples,
                    # n_outputs)
                    return pd.DataFrame(data=y, columns=outputs, index=X.index)
            raise TypeError(
                f"unexpected shape of numpy array returned as prediction: {y.shape}"
            )
        raise TypeError(
            f"unexpected data type returned as prediction: {type(y).__name__}"
        )


@inheritdoc(match="[see superclass]")
class SupervisedLearnerWrapperDF(
    SupervisedLearnerDF,
    LearnerWrapperDF[T_NativeSupervisedLearner],
    Generic[T_NativeSupervisedLearner],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for native `supervised learners` conforming with the
    `scikit-learn` API.

    Supervised learners in `scikit-learn` typically are regressors or classifiers.
    """

    # noinspection PyPep8Naming
    def score(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> float:
        """[see superclass]"""
        X, y = self._validate_parameter_types(X, y)
        if y is None:
            raise ValueError("arg y must not be None")
        if sample_weight is not None and not isinstance(sample_weight, pd.Series):
            raise TypeError("arg sample_weight must be None or a Series")

        return cast(
            float,
            self.native_estimator.score(
                self._prepare_X_for_delegate(X),
                self._prepare_y_for_delegate(y),
                sample_weight,
            ),
        )


@inheritdoc(match="[see superclass]")
class RegressorWrapperDF(
    RegressorDF,
    SupervisedLearnerWrapperDF[T_NativeRegressor],
    Generic[T_NativeRegressor],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for native regressors conforming with the `scikit-learn`
    API.
    """

    __native_base_class__ = RegressorMixin

    # noinspection PyPep8Naming
    def score(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> float:
        """[see superclass]"""
        return cast(
            float,
            SupervisedLearnerWrapperDF.score(self, X, y, sample_weight=sample_weight),
        )


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
    """

    __native_base_class__ = ClassifierMixin

    def _get_classes(self) -> Union[npt.NDArray[Any], list[npt.NDArray[Any]]]:
        return cast(
            Union[npt.NDArray[Any], list[npt.NDArray[Any]]],
            self._native_estimator.classes_,
        )

    # noinspection PyPep8Naming
    def predict_proba(
        self, X: Union[pd.Series, pd.DataFrame], **predict_params: Any
    ) -> Union[pd.DataFrame, list[pd.DataFrame]]:
        """[see superclass]"""

        self._ensure_delegate_method("predict_proba")

        X, _ = self._validate_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X,
            self.native_estimator.predict_proba(
                self._prepare_X_for_delegate(X), **predict_params
            ),
        )

    # noinspection PyPep8Naming
    def predict_log_proba(
        self, X: Union[pd.Series, pd.DataFrame], **predict_params: Any
    ) -> Union[pd.DataFrame, list[pd.DataFrame]]:
        """[see superclass]"""

        self._ensure_delegate_method("predict_log_proba")

        X, _ = self._validate_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X,
            self.native_estimator.predict_log_proba(
                self._prepare_X_for_delegate(X), **predict_params
            ),
        )

    # noinspection PyPep8Naming
    def decision_function(
        self, X: Union[pd.Series, pd.DataFrame], **predict_params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""

        self._ensure_delegate_method("decision_function")

        X, _ = self._validate_parameter_types(X, None)

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
        prediction: Union[
            pd.Series, pd.DataFrame, list[npt.NDArray[Any]], npt.NDArray[Any]
        ],
        classes: Optional[Sequence[Any]] = None,
    ) -> Union[pd.Series, pd.DataFrame, list[pd.DataFrame]]:
        if classes is None:
            classes = getattr(self.native_estimator, "classes_", None)
            if classes is None:
                classes = pd.RangeIndex(self._get_n_outputs())

        if isinstance(prediction, pd.DataFrame):
            return prediction.set_axis(classes, axis=1)

        elif isinstance(prediction, np.ndarray):
            if len(prediction) == len(X):
                # predictions of probabilities are usually provided as a NumPy array
                # the same length as X
                if prediction.ndim == 1:
                    # for a binary classifier, we get a series with probabilities
                    # for the second class
                    return pd.Series(data=prediction, index=X.index, name=classes[1])
                elif prediction.ndim == 2:
                    # for a multi-class classifiers, we get a two-dimensional NumPy
                    # array with probabilities for each class
                    return pd.DataFrame(data=prediction, index=X.index, columns=classes)
            raise TypeError(
                f"ndarray with unexpected shape returned as prediction: "
                f"{prediction.shape}"
            )

        else:
            raise TypeError(
                f"unexpected type or prediction result: {type(prediction).__name__}"
            )

    # noinspection PyPep8Naming
    def score(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> float:
        """[see superclass]"""
        return cast(
            float,
            SupervisedLearnerWrapperDF.score(self, X, y, sample_weight=sample_weight),
        )


# noinspection PyPep8Naming
@inheritdoc(match="[see superclass]")
class ClusterWrapperDF(
    ClusterDF,
    LearnerWrapperDF[T_NativeCluster],
    Generic[T_NativeCluster],
    metaclass=ABCMeta,
):
    """
    Base class of DF wrappers for native clusterers conforming with the scikit-learn
    API.
    """

    __native_base_class__ = ClusterMixin

    COL_LABELS = "labels"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """[see superclass]"""
        super().__init__(*args, **kwargs)
        self._x_index: Optional[pd.Index] = None

    def _get_labels(self) -> pd.Series:
        return pd.Series(
            data=self._native_estimator.labels_,
            name=self.COL_LABELS,
            index=self._x_index,
        )

    def fit_predict(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_predict_params: Any,
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""

        self._reset_fit()

        try:
            X, y = self._validate_parameter_types(X, y)

            # fitting a clusterer produces a single output column for labels
            self._outputs = [ClusterWrapperDF.COL_LABELS]

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
        **fit_params: Any,
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
    MetaEstimatorMixin,  # type: ignore
    EstimatorWrapperDF[T_NativeEstimator],
    Generic[T_NativeEstimator],
    metaclass=ABCMeta,
):
    """
    Abstract base class of wrappers for estimators implementing
    :class:`sklearn.base.MetaEstimatorMixin`.

    A meta-estimator will call the methods of the embedded estimator using a modified
    copy of the `X` and `y` parameters, so we need to make sure that these are converted
    back to data frames.

    We also ensure that the embedded estimator is a native sci-kit learn estimator.
    If it is a DF estimator, we replace it with its wrapped native estimator and
    re-assign it to the attribute `estimator` or `base_estimator` (depending on the
    meta-estimator).
    In that case, we issue a warning that the wrapped estimator is being used instead
    of the DF version.

    This class covers three variants used in scikit-learn:

    - one delegate estimator in attribute `estimator`
    - one delegate estimator in attribute `base_estimator`
    - multiple delegate estimators in attribute `estimators`
    """

    def _validate_delegate_estimator(self) -> None:
        substituted: list[str] = []

        estimator = getattr(self, "estimator", None)
        if estimator is not None:
            self.estimator = self._native_learner(estimator)
            substituted.append("estimator")

        base_estimator = getattr(self, "base_estimator", None)
        # attribute base_estimator is deprecated as of scikit-learn 1.2, with the
        # default value of "deprecated"
        if base_estimator is not None and base_estimator != "deprecated":
            self.base_estimator = self._native_learner(base_estimator)
            substituted.append("base_estimator")

        estimators = getattr(self, "estimators", None)
        if estimators is not None:
            self.estimators = [
                (name, self._native_learner(estimator))
                for name, estimator in estimators
            ]
            substituted.append("estimators")

        if substituted:
            warnings.warn(
                f"the following attributes of {type(self).__name__} "
                f"have been replaced with their native scikit-learn counterparts: "
                f"{', '.join(substituted)}",
                stacklevel=-2,
            )

    @staticmethod
    def _native_learner(
        estimator_wrapper: BaseEstimator,
    ) -> Union[RegressorMixin, ClassifierMixin]:
        native_estimator: BaseEstimator = (
            estimator_wrapper.native_estimator
            if isinstance(estimator_wrapper, EstimatorWrapperDF)
            else estimator_wrapper
        )
        # noinspection PyProtectedMember
        if isinstance(
            native_estimator, (EstimatorDF, sklearn_meta._BaseComposition)
        ) or not isinstance(native_estimator, (RegressorMixin, ClassifierMixin)):
            raise TypeError(
                "sklearndf meta-estimators only accept simple regressors and "
                f"classifiers, but got: {type(estimator_wrapper).__name__}"
            )
        return native_estimator


#
# private factory implementation
#


def _mirror_attributes(
    wrapper_class: type[EstimatorWrapperDF[T_NativeEstimator]],
    native_estimator: type[T_NativeEstimator],
    wrapper_module: str,
) -> None:
    wrapper_name = wrapper_class.__name__
    wrapper_attributes: set[str] = set(dir(wrapper_class))

    for name, member in vars(native_estimator).items():
        if member is None or name in wrapper_attributes:
            continue

        alias = _make_alias(
            wrapper_module=wrapper_module,
            wrapper_name=wrapper_name,
            name=name,
            delegate_cls=native_estimator,
            delegate=member,
        )

        if alias is not None:
            setattr(wrapper_class, name, alias)


def _make_alias(
    wrapper_module: str, wrapper_name: str, name: str, delegate_cls: type, delegate: Any
) -> Optional[Union[Callable[..., Any], property]]:

    if inspect.isfunction(delegate):
        return _make_method_alias(
            wrapper_module=wrapper_module,
            wrapper_name=wrapper_name,
            name=name,
            delegate_cls=delegate_cls,
            delegate_method=delegate,
        )
    elif inspect.isdatadescriptor(delegate):
        return _make_descriptor_alias(
            delegate_cls=delegate_cls, delegate_descriptor=delegate
        )
    else:
        return None


def _make_method_alias(
    wrapper_module: str,
    wrapper_name: str,
    name: str,
    delegate_cls: type,
    delegate_method: T_Callable,
) -> T_Callable:
    # create a method that forwards calls to a native delegate estimator
    wrapper_method = _make_forwarder(delegate_method)
    _update_wrapper(
        wrapper=wrapper_method,
        wrapped=delegate_method,
        wrapper_module=wrapper_module,
        wrapper_parent=wrapper_name,
    )
    class_name = _full_class_name(cls=delegate_cls)
    wrapper_method.__doc__ = f"See :meth:`{class_name}.{name}`"
    return wrapper_method


def _make_descriptor_alias(delegate_cls: type, delegate_descriptor: Any) -> property:
    # create a property that forwards attribute access to a native delegate estimator
    class_name = _full_class_name(cls=delegate_cls)
    return property(
        fget=lambda self: delegate_descriptor.__get__(self._native_estimator),
        fset=lambda self, value: cast(Callable[..., None], delegate_descriptor.__set__)(
            self._native_estimator, value
        ),
        fdel=lambda self: cast(Callable[..., None], delegate_descriptor.__delete__)(
            self._native_estimator
        ),
        doc=f"See documentation of :class:`{class_name}`.",
    )


def _make_forwarder(delegate_method: T_Callable) -> T_Callable:
    # noinspection PyShadowingNames
    def _forwarder(
        self: EstimatorWrapperDF[BaseEstimator], *args: Any, **kwargs: Any
    ) -> Any:
        return delegate_method(self._native_estimator, *args, **kwargs)

    return cast(T_Callable, _forwarder)


def _update_wrapper(
    wrapper: Any,
    wrapped: Any,
    wrapper_module: str,
    wrapper_parent: str,
) -> None:
    update_wrapper(
        wrapper, wrapped, assigned=("__name__", "__annotations__"), updated=()
    )
    wrapper.__module__ = wrapper_module
    if wrapper_parent:
        wrapper.__qualname__ = f"{wrapper_parent}.{wrapper.__name__}"
    else:
        wrapper.__qualname__ = wrapper.__name__


def _update_class_docstring(
    df_estimator_type: type[EstimatorWrapperDF[T_NativeEstimator]],
    sklearn_native_estimator_type: type[T_NativeEstimator],
) -> None:
    base_doc = sklearn_native_estimator_type.__doc__

    if not base_doc:
        return

    base_doc_lines = base_doc.split("\n")

    # use the first paragraph as the tag line
    tag_lines: list[str] = []
    for line in base_doc_lines:
        # end of paragraph reached?
        stripped = line.strip()
        if stripped:
            # no: append line to tag lines
            tag_lines.append(stripped)
        elif tag_lines:
            # empty line, and we already have tag lines: stop here
            break

    estimator_name = _full_class_name(cls=sklearn_native_estimator_type)

    df_estimator_type.__doc__ = "\n".join(
        [
            *tag_lines,
            "",
            (
                f"""
.. note:: This class is a wrapper around class :class:`{estimator_name}`.
    It provides enhanced support for :mod:`pandas` data frames, and otherwise
    delegates all attribute access and method calls to an associated
    :class:`~{estimator_name}` instance.
"""
            ),
        ]
    )


def _full_class_name(cls: type) -> str:
    # get the full name of the class, including the module prefix

    try:
        module_name = cls.__module__
    except AttributeError as e:
        raise RuntimeError(f"cannot get module for {cls}") from e

    if module_name != "__main__":
        module_name = public_module_prefix(module_name)

    return f"{module_name}.{cls.__qualname__}"


#
# validate __all__
#

__tracker.validate()
