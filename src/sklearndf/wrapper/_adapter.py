"""
Estimator adapter classes to handle numpy arrays in meta-estimators
"""

import logging
from typing import Any, Callable, Generic, List, Optional, Sequence, TypeVar, Union

import numpy as np
import pandas as pd

from pytools.api import AllTracker, inheritdoc

from sklearndf import ClassifierDF, EstimatorDF, LearnerDF, RegressorDF, TransformerDF

log = logging.getLogger(__name__)

__all__ = [
    "ClassifierNPDF",
    "EstimatorNPDF",
    "LearnerNPDF",
    "RegressorNPDF",
    "TransformerNPDF",
]

#
# type variables
#

T_Self = TypeVar("T_Self")
T_DelegateEstimatorDF = TypeVar("T_DelegateEstimatorDF", bound=EstimatorDF)
T_DelegateLearnerDF = TypeVar("T_DelegateLearnerDF", bound=LearnerDF)
T_DelegateClassifierDF = TypeVar("T_DelegateClassifierDF", bound=ClassifierDF)
T_DelegateRegressorDF = TypeVar("T_DelegateRegressorDF", bound=RegressorDF)
T_DelegateTransformerDF = TypeVar("T_DelegateTransformerDF", bound=TransformerDF)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


# noinspection PyPep8Naming
@inheritdoc(match="""[see superclass]""")
class EstimatorNPDF(
    EstimatorDF,
    Generic[T_DelegateEstimatorDF],
):
    """
    An adapter class that wraps around a :class:`.EstimatorDF` and accepts numpy arrays
    for all DF estimator methods that would usually only accept pandas data frames or
    series.

    Converts all numpy arrays to pandas series or data frames before deferring to the
    delegate estimator, and passes through pandas objects unchanged.

    For use in meta-estimators that internally hand numpy arrays on to sub-estimators.
    """

    #: The sklearndf estimator to invoke after transforming the incoming numpy arrays to
    #: pandas data frames or series.
    delegate: T_DelegateEstimatorDF

    #: Column names to use for the pandas data frame derived from the features numpy
    #: array.
    column_names: Optional[Union[Sequence[str], Callable[[], Sequence[str]]]]

    def __init__(
        self,
        delegate: T_DelegateEstimatorDF,
        column_names: Optional[
            Union[Sequence[str], Callable[[], Sequence[str]]]
        ] = None,
    ) -> None:
        """
        :param delegate: the sklearndf estimator to invoke after transforming the
            incoming numpy arrays to pandas data frames or series
        :param column_names: optional column names to use for the pandas data frame
            derived from the features numpy array; passed either as a sequence of
            strings, or as a function that dynamically provides the column names
        """
        super().__init__()
        self.delegate = delegate
        self.column_names = column_names

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.delegate.is_fitted

    def fit(
        self: T_Self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        **fit_params: Any,
    ) -> T_Self:
        """[see superclass]"""
        return self.delegate.fit(
            self._ensure_X_frame(X), self._ensure_y_series_or_frame(y), **fit_params
        )

    def _get_features_in(self) -> pd.Index:
        return self.delegate._get_features_in()

    def _get_n_outputs(self) -> int:
        return self.delegate._get_n_outputs()

    def _ensure_X_frame(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        column_names = self.column_names()
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise TypeError(
                    f"expected 2-dimensional array but got {X.ndim} dimensions"
                )
            if column_names:
                if X.shape[1] != len(column_names):
                    raise ValueError(
                        f"expected {len(column_names)} columns but got {X.shape[1]}"
                    )
                return pd.DataFrame(X, columns=column_names)
            else:
                return pd.DataFrame(X)
        else:
            if column_names and X.columns.to_list() != column_names:
                raise ValueError(
                    f"expected column names {column_names} "
                    f"but got {X.columns.to_list()}"
                )
            return X

    @staticmethod
    def _ensure_y_series_or_frame(
        y: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]]
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                return pd.Series(y)
            elif y.ndim == 2:
                return pd.DataFrame(y)
            else:
                raise TypeError(
                    f"expected 1- or 2-dimensional array but got {y.ndim} dimensions"
                )
        else:
            return y


# noinspection PyPep8Naming
@inheritdoc(match="""[see superclass]""")
class LearnerNPDF(
    EstimatorNPDF[T_DelegateLearnerDF], LearnerDF, Generic[T_DelegateLearnerDF]
):
    """[see superclass]"""

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame], **predict_params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""
        return self.delegate.predict(self._ensure_X_frame(X), **predict_params)

    def fit_predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **fit_params: Any,
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""
        return self.delegate.fit_predict(
            self._ensure_X_frame(X), self._ensure_y_series_or_frame(y), **fit_params
        )

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[pd.Series] = None,
    ) -> float:
        """[see superclass]"""
        return self.delegate.score(
            self._ensure_X_frame(X),
            self._ensure_y_series_or_frame(y),
            sample_weight=sample_weight,
        )


# noinspection PyPep8Naming
@inheritdoc(match="""[see superclass]""")
class ClassifierNPDF(
    LearnerNPDF[T_DelegateClassifierDF],
    ClassifierDF,
    Generic[T_DelegateClassifierDF],
):
    """[see superclass]"""

    @property
    def classes_(self) -> Sequence[Any]:
        """[see superclass]"""
        return self.delegate.classes_

    def predict_proba(
        self, X: Union[np.ndarray, pd.DataFrame], **predict_params: Any
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """[see superclass]"""
        return self.delegate.predict_proba(self._ensure_X_frame(X), **predict_params)

    def predict_log_proba(
        self, X: Union[np.ndarray, pd.DataFrame], **predict_params: Any
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """[see superclass]"""
        return self.delegate.predict_log_proba(
            self._ensure_X_frame(X), **predict_params
        )

    def decision_function(
        self, X: Union[np.ndarray, pd.DataFrame], **predict_params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""
        return self.delegate.decision_function(
            self._ensure_X_frame(X), **predict_params
        )

    def _get_features_in(self) -> pd.Index:
        return self.delegate._get_features_in()

    def _get_n_outputs(self) -> int:
        return self.delegate._get_n_outputs()


# noinspection PyPep8Naming
@inheritdoc(match="""[see superclass]""")
class RegressorNPDF(
    LearnerNPDF[T_DelegateRegressorDF],
    RegressorDF,
    Generic[T_DelegateRegressorDF],
):
    """[see superclass]"""


# noinspection PyPep8Naming
@inheritdoc(match="""[see superclass]""")
class TransformerNPDF(
    EstimatorNPDF[T_DelegateTransformerDF],
    TransformerDF,
    Generic[T_DelegateTransformerDF],
):
    """[see superclass]"""

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """[see superclass]"""
        return self.delegate.predict(self._ensure_X_frame(X))

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """[see superclass]"""
        return self.delegate.inverse_transform(self._ensure_X_frame(X))

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        """[see superclass]"""
        return self.delegate.fit_transform(
            self._ensure_X_frame(X), self._ensure_y_series_or_frame(y), **fit_params
        )

    def _get_features_original(self) -> pd.Series:
        # noinspection PyProtectedMember
        return self.delegate._get_features_original()


#
# validate __all__
#

__tracker.validate()
