"""
Core implementation of :mod:`sklearndf.transformation.extra`
"""

import logging
from abc import ABCMeta
from typing import Optional, TypeVar, Union

import pandas as pd
from boruta import BorutaPy
from sklearn.base import BaseEstimator

from pytools.api import AllTracker, inheritdoc

from ... import TransformerDF
from ..._wrapper import _MetaEstimatorWrapperDF, df_estimator
from .._wrapper import _ColumnSubsetTransformerWrapperDF, _NDArrayTransformerWrapperDF

log = logging.getLogger(__name__)

__all__ = ["OutlierRemoverDF", "BorutaDF"]


#
# type variables
#

T_Self = TypeVar("T_Self")


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="[see superclass]")
class OutlierRemoverDF(TransformerDF, BaseEstimator):
    """
    Remove outliers according to Tukey's method.

    A sample is considered an outlier if it is outside the range
    :math:`[Q_1 - iqr\\_ multiple(Q_3-Q_1), Q_3 + iqr\\_ multiple(Q_3-Q_1)]`
    where :math:`Q_1` and :math:`Q_3` are the lower and upper quartiles.

    :param iqr_multiple: the multiple used to define the range of non-outlier
      samples in the above explanation (defaults to 3.0 as per Tukey's definition of
      far outliers)
    """

    def __init__(self, iqr_multiple: float = 3.0):
        super().__init__()
        if iqr_multiple < 0.0:
            raise ValueError(f"arg iqr_multiple is negative: {iqr_multiple}")
        self.iqr_multiple = iqr_multiple
        self.threshold_low_ = None
        self.threshold_high_ = None
        self._features_original = None

    # noinspection PyPep8Naming
    def fit(
        self: T_Self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params,
    ) -> T_Self:
        """
        Fit the transformer.

        :return: the fitted transformer
        """

        self: OutlierRemoverDF  # support type hinting in PyCharm

        q1: pd.Series = X.quantile(q=0.25)
        q3: pd.Series = X.quantile(q=0.75)
        threshold_iqr: pd.Series = (q3 - q1) * self.iqr_multiple
        self.threshold_low_ = q1 - threshold_iqr
        self.threshold_high_ = q3 + threshold_iqr
        self._features_original = X.columns.to_series()
        return self

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return ``X`` with outliers are replaced by ``NaN``.

        :return: the ``X`` where outliers are replaced by ``NaN``
        """
        return X.where(cond=(X >= self.threshold_low_) & (X <= self.threshold_high_))

    # noinspection PyPep8Naming
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform is not implemented.

        :raises NotImplementedError:
        """
        raise NotImplementedError("inverse transform is not implemented")

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.threshold_low_ is not None

    def _get_features_original(self) -> pd.Series:
        return self._features_original

    def _get_features_in(self) -> pd.Index:
        return self.feature_names_original_.index

    def _get_n_outputs(self) -> int:
        return 0


class _BorutaPyWrapperDF(
    _MetaEstimatorWrapperDF[BorutaPy],
    _NDArrayTransformerWrapperDF[BorutaPy],
    _ColumnSubsetTransformerWrapperDF[BorutaPy],
    metaclass=ABCMeta,
):
    def _get_features_out(self) -> pd.Index:
        return self.feature_names_in_[self.native_estimator.support_]


# noinspection PyAbstractClass,PyUnresolvedReferences
@df_estimator(df_wrapper_type=_BorutaPyWrapperDF)
class BorutaDF(TransformerDF, BorutaPy):
    """
    Feature Selection with the Boruta method with dataframes as input and output.


    Given a dataset and an estimator, the Boruta method selects the features that
    perform better than noise for the problem. See `BorutaPy
    <https://github.com/scikit-learn-contrib/boruta_py>`_.

    :class:`BorutaDF` wraps :class:`BorutaPy
    <https://github.com/scikit-learn-contrib/boruta_py>` with dataframes
    as input and output.

    The parameters are the parameters from the boruta :class:`BorutaPy`. For
    convenience we list below the description of the parameters as they appear in
    https://github.com/scikit-learn-contrib/boruta_py.

    :param estimator: object
        A supervised learning estimator, with a 'fit' method that returns the
        ``feature_importances_`` attribute. Important features must correspond to
        high absolute values in the ``feature_importances_``.
    :param n_estimators: int or string, default = 1000
        If int sets the number of estimators in the chosen ensemble method.
        If 'auto' this is determined automatically based on the size of the
        dataset. The other parameters of the used estimators need to be set
        with initialisation.
    :param perc: int, default = 100
        Instead of the max we use the percentile defined by the user, to pick
        our threshold for comparison between shadow and real features. The max
        tend to be too stringent. This provides a finer control over this. The
        lower perc is the more false positives will be picked as relevant but
        also the less relevant features will be left out. The usual trade-off.
        The default is essentially the vanilla Boruta corresponding to the max.
    :param alpha: float, default = 0.05
        Level at which the corrected p-values will get rejected in both
        correction steps.
    :param two_step: Boolean, default = True
        If you want to use the original implementation of Boruta with Bonferroni
        correction only set this to False.
    :param max_iter: int, default = 100
        The number of maximum iterations to perform.
    :param random_state: int, RandomState instance or None; default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by ``np.random``.
    :param verbose: int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays iteration number
        - 2: which features have been selected already
    """

    pass


__tracker.validate()
