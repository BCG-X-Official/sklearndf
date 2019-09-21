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
Data frame versions of all sklearn regressors
"""
import abc as _abc
import logging as _logging
import typing as _t

import numpy as _np
import pandas as _pd
import sklearn.calibration as _cal
import sklearn.discriminant_analysis as _da
import sklearn.dummy as _dm
import sklearn.ensemble as _ens
import sklearn.gaussian_process as _gp
import sklearn.linear_model as _lm
import sklearn.multiclass as _mc
import sklearn.multioutput as _mo
import sklearn.naive_bayes as _nby
import sklearn.neighbors as _nbr
import sklearn.neural_network as _ann
import sklearn.semi_supervised as _ss
import sklearn.svm as _svm
import sklearn.tree as _tr

import gamma.common as _cm
import gamma.sklearndf as _df
import gamma.sklearndf._wrapper as _wr

log = _logging.getLogger(__name__)

#
# neighbors
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class NearestCentroidDF(_df.ClassifierDF, _nbr.NearestCentroid):
    """
    Wraps :class:`sklearn.neighbors.nearest_centroid.NearestCentroid`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class KNeighborsClassifierDF(_df.ClassifierDF, _nbr.KNeighborsClassifier):
    """
    Wraps :class:`sklearn.neighbors.classification.KNeighborsClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class RadiusNeighborsClassifierDF(_df.ClassifierDF, _nbr.RadiusNeighborsClassifier):
    """
    Wraps :class:`sklearn.neighbors.classification.RadiusNeighborsClassifier`; accepts
    and returns data frames.
    """

    pass


#
# voting
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.MetaClassifierWrapperDF)
class VotingClassifierDF(_df.ClassifierDF, _ens.VotingClassifier):
    """
    Wraps :class:`sklearn.ensemble.voting.VotingClassifier`; accepts and returns data
    frames.
    """

    pass


#
# ensemble
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class RandomForestClassifierDF(_df.ClassifierDF, _ens.RandomForestClassifier):
    """
    Wraps :class:`sklearn.ensemble.forest.RandomForestClassifier`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class ExtraTreesClassifierDF(_df.ClassifierDF, _ens.ExtraTreesClassifier):
    """
    Wraps :class:`sklearn.ensemble.forest.ExtraTreesClassifier`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class GradientBoostingClassifierDF(_df.ClassifierDF, _ens.GradientBoostingClassifier):
    """
    Wraps :class:`sklearn.ensemble.gradient_boosting.GradientBoostingClassifier`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class AdaBoostClassifierDF(_df.ClassifierDF, _ens.AdaBoostClassifier):
    """
    Wraps :class:`sklearn.ensemble.weight_boosting.AdaBoostClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class BaggingClassifierDF(_df.ClassifierDF, _ens.BaggingClassifier):
    """
    Wraps :class:`sklearn.ensemble.bagging.BaggingClassifier`; accepts and returns data
    frames.
    """

    pass


#
# tree
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class DecisionTreeClassifierDF(_df.ClassifierDF, _tr.DecisionTreeClassifier):
    """
    Wraps :class:`sklearn.tree.tree.DecisionTreeClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class ExtraTreeClassifierDF(_df.ClassifierDF, _tr.ExtraTreeClassifier):
    """
    Wraps :class:`sklearn.tree.tree.ExtraTreeClassifier`; accepts and returns data
    frames.
    """

    pass


#
# discriminant analysis
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class QuadraticDiscriminantAnalysisDF(
    _df.ClassifierDF, _da.QuadraticDiscriminantAnalysis
):
    """
    Wraps :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class LinearDiscriminantAnalysisDF(_df.ClassifierDF, _da.LinearDiscriminantAnalysis):
    """
    Wraps :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`; accepts and
    returns data frames.
    """

    pass


#
# naive bayes
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class GaussianNBDF(_df.ClassifierDF, _nby.GaussianNB):
    """
    Wraps :class:`sklearn.naive_bayes.GaussianNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class MultinomialNBDF(_df.ClassifierDF, _nby.MultinomialNB):
    """
    Wraps :class:`sklearn.naive_bayes.MultinomialNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class ComplementNBDF(_df.ClassifierDF, _nby.ComplementNB):
    """
    Wraps :class:`sklearn.naive_bayes.ComplementNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class BernoulliNBDF(_df.ClassifierDF, _nby.BernoulliNB):
    """
    Wraps :class:`sklearn.naive_bayes.BernoulliNB`; accepts and returns data frames.
    """

    pass


#
# calibration
#

# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.MetaClassifierWrapperDF)
class CalibratedClassifierCVDF(_df.ClassifierDF, _cal.CalibratedClassifierCV):
    """
    Wraps :class:`sklearn.calibration.CalibratedClassifierCV`; accepts and returns data
    frames.
    """

    pass


#
# SVM
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class SVCDF(_df.ClassifierDF, _svm.SVC):
    """
    Wraps :class:`sklearn.svm.classes.SVC`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class NuSVCDF(_df.ClassifierDF, _svm.NuSVC):
    """
    Wraps :class:`sklearn.svm.classes.NuSVC`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class LinearSVCDF(_df.ClassifierDF, _svm.LinearSVC):
    """
    Wraps :class:`sklearn.svm.classes.LinearSVC`; accepts and returns data frames.
    """

    pass


#
# dummy
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class DummyClassifierDF(_df.ClassifierDF, _dm.DummyClassifier):
    """
    Wraps :class:`sklearn.dummy.DummyClassifier`; accepts and returns data frames.
    """

    pass


#
# gaussian process
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class GaussianProcessClassifierDF(_df.ClassifierDF, _gp.GaussianProcessClassifier):
    """
    Wraps :class:`sklearn.gaussian_process.gpc.GaussianProcessClassifier`; accepts and
    returns data frames.
    """

    pass


#
# linear model
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class LogisticRegressionDF(_df.ClassifierDF, _lm.LogisticRegression):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegression`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class LogisticRegressionCVDF(_df.ClassifierDF, _lm.LogisticRegressionCV):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegressionCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class LogisticRegressionCVDF(_df.ClassifierDF, _lm.LogisticRegressionCV):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegressionCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class PassiveAggressiveClassifierDF(_df.ClassifierDF, _lm.PassiveAggressiveClassifier):
    """
    Wraps :class:`sklearn.linear_model.passive_aggressive.PassiveAggressiveClassifier`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class PerceptronDF(_df.ClassifierDF, _lm.Perceptron):
    """
    Wraps :class:`sklearn.linear_model.perceptron.Perceptron`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class SGDClassifierDF(_df.ClassifierDF, _lm.SGDClassifier):
    """
    Wraps :class:`sklearn.linear_model.stochastic_gradient.SGDClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class RidgeClassifierDF(_df.ClassifierDF, _lm.RidgeClassifier):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class RidgeClassifierCVDF(_df.ClassifierDF, _lm.RidgeClassifierCV):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeClassifierCV`; accepts and returns
    data frames.
    """

    pass


#
# semi-supervised
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class LabelPropagationDF(_df.ClassifierDF, _ss.LabelPropagation):
    """
    Wraps :class:`sklearn.semi_supervised.label_propagation.LabelPropagation`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class LabelSpreadingDF(_df.ClassifierDF, _ss.LabelSpreading):
    """
    Wraps :class:`sklearn.semi_supervised.label_propagation.LabelSpreading`; accepts and
    returns data frames.
    """

    pass


#
# multi-class
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.MetaClassifierWrapperDF)
class OneVsRestClassifierDF(_df.ClassifierDF, _mc.OneVsRestClassifier):
    """
    Wraps :class:`sklearn.multiclass.OneVsRestClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.MetaClassifierWrapperDF)
class OneVsOneClassifierDF(_df.ClassifierDF, _mc.OneVsOneClassifier):
    """
    Wraps :class:`sklearn.multiclass.OneVsOneClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.MetaClassifierWrapperDF)
class OutputCodeClassifierDF(_df.ClassifierDF, _mc.OutputCodeClassifier):
    """
    Wraps :class:`sklearn.multiclass.OutputCodeClassifier`; accepts and returns data
    frames.
    """

    pass


#
# multi-output
#

# estimators attribute of abstract class MultiOutputEstimator
ATTR_MULTI_OUTPUT_ESTIMATORS = "estimators_"


class _MultiOutputClassifierWrapperDF(
    _wr.MetaClassifierWrapperDF[_mo.MultiOutputClassifier], _abc.ABC
):
    # noinspection PyPep8Naming
    def _prediction_with_class_labels(
        self,
        X: _pd.DataFrame,
        y: _t.Union[_pd.Series, _pd.DataFrame, list, _np.ndarray],
        classes: _t.Optional[_cm.ListLike[_t.Any]] = None,
    ) -> _t.Union[_pd.Series, _pd.DataFrame, _t.List[_pd.DataFrame]]:

        # if we have a multi-output classifier, prediction of probabilities
        # yields a list of NumPy arrays
        if not isinstance(y, list):
            raise ValueError(
                "prediction of multi-output classifier expected to be a list of NumPy "
                f"arrays, but got type {type(y)}"
            )

        delegate_estimator = self.delegate_estimator

        # store the super() object as this is not available within a generator
        sup = super()

        # usually the delegate estimator will provide a list of estimators used
        # to predict each output. If present, use these estimators to get
        # individual class labels for each output; otherwise we cannot assign class
        # labels
        if hasattr(delegate_estimator, ATTR_MULTI_OUTPUT_ESTIMATORS):
            return [
                sup._prediction_with_class_labels(
                    X=X, y=output, classes=getattr(estimator, "classes_", None)
                )
                for estimator, output in zip(
                    getattr(delegate_estimator, ATTR_MULTI_OUTPUT_ESTIMATORS), y
                )
            ]
        else:
            return [
                sup._prediction_with_class_labels(X=X, y=output, classes=None)
                for output in y
            ]


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_MultiOutputClassifierWrapperDF)
class MultiOutputClassifierDF(_df.ClassifierDF, _mo.MultiOutputClassifier):
    """
    Wraps :class:`sklearn.multioutput.MultiOutputClassifier`; accepts and returns data
    frames.
    """

    pass


#
# chaining
#


class _ClassifierChainWrapperDF(
    _wr.MetaClassifierWrapperDF[_mo.ClassifierChain], _abc.ABC
):
    # noinspection PyPep8Naming
    def _prediction_with_class_labels(
        self,
        X: _pd.DataFrame,
        y: _t.Union[_pd.Series, _pd.DataFrame, list, _np.ndarray],
        classes: _t.Optional[_cm.ListLike[_t.Any]] = None,
    ) -> _t.Union[_pd.Series, _pd.DataFrame, _t.List[_pd.DataFrame]]:
        return super()._prediction_with_class_labels(
            X=X, y=y, classes=range(self.n_outputs)
        )


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_ClassifierChainWrapperDF)
class ClassifierChainDF(_df.ClassifierDF, _mo.ClassifierChain):
    """
    Wraps :class:`sklearn.multioutput.ClassifierChain`; accepts and returns data frames.
    """

    pass


#
# neural network
#


# noinspection PyAbstractClass
@_wr.df_estimator(df_wrapper_type=_wr.ClassifierWrapperDF)
class MLPClassifierDF(_df.ClassifierDF, _ann.MLPClassifier):
    """
    Wraps :class:`sklearn.neural_network.multilayer_perceptron.MLPClassifier`; accepts
    and returns data frames.
    """

    pass


#
# export all symbols ending in "DF"
#

__all__ = [sym for sym in dir() if sym.endswith("DF") and not sym.startswith("_")]
