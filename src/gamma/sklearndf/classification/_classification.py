"""
Core implementation of :mod:`gamma.sklearndf.classification`
"""
import abc as _abc
import logging as _logging
import typing as _t

import numpy as np
import pandas as pd
import sklearn

# noinspection PyUnresolvedReferences
from sklearn import (
    calibration,
    discriminant_analysis,
    ensemble,
    gaussian_process,
    linear_model,
    multiclass,
    multioutput,
    naive_bayes,
    neighbors,
    neural_network,
    semi_supervised,
    svm,
    tree,
)

from gamma.sklearndf import ClassifierDF
from gamma.sklearndf._wrapper import (
    ClassifierWrapperDF,
    df_estimator,
    MetaClassifierWrapperDF,
)

log = _logging.getLogger(__name__)

#
# neighbors
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class NearestCentroidDF(ClassifierDF, sklearn.neighbors.NearestCentroid):
    """
    Wraps :class:`sklearn.neighbors.nearest_centroid.NearestCentroid`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class KNeighborsClassifierDF(ClassifierDF, sklearn.neighbors.KNeighborsClassifier):
    """
    Wraps :class:`sklearn.neighbors.classification.KNeighborsClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class RadiusNeighborsClassifierDF(
    ClassifierDF, sklearn.neighbors.RadiusNeighborsClassifier
):
    """
    Wraps :class:`sklearn.neighbors.classification.RadiusNeighborsClassifier`; accepts
    and returns data frames.
    """

    pass


#
# voting
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=MetaClassifierWrapperDF)
class VotingClassifierDF(ClassifierDF, sklearn.ensemble.VotingClassifier):
    """
    Wraps :class:`sklearn.ensemble.voting.VotingClassifier`; accepts and returns data
    frames.
    """

    pass


#
# ensemble
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class RandomForestClassifierDF(ClassifierDF, sklearn.ensemble.RandomForestClassifier):
    """
    Wraps :class:`sklearn.ensemble.forest.RandomForestClassifier`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class ExtraTreesClassifierDF(ClassifierDF, sklearn.ensemble.ExtraTreesClassifier):
    """
    Wraps :class:`sklearn.ensemble.forest.ExtraTreesClassifier`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class GradientBoostingClassifierDF(
    ClassifierDF, sklearn.ensemble.GradientBoostingClassifier
):
    """
    Wraps :class:`sklearn.ensemble.gradient_boosting.GradientBoostingClassifier`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class AdaBoostClassifierDF(ClassifierDF, sklearn.ensemble.AdaBoostClassifier):
    """
    Wraps :class:`sklearn.ensemble.weight_boosting.AdaBoostClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class BaggingClassifierDF(ClassifierDF, sklearn.ensemble.BaggingClassifier):
    """
    Wraps :class:`sklearn.ensemble.bagging.BaggingClassifier`; accepts and returns data
    frames.
    """

    pass


#
# tree
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class DecisionTreeClassifierDF(ClassifierDF, sklearn.tree.DecisionTreeClassifier):
    """
    Wraps :class:`sklearn.tree.tree.DecisionTreeClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class ExtraTreeClassifierDF(ClassifierDF, sklearn.tree.ExtraTreeClassifier):
    """
    Wraps :class:`sklearn.tree.tree.ExtraTreeClassifier`; accepts and returns data
    frames.
    """

    pass


#
# discriminant analysis
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class QuadraticDiscriminantAnalysisDF(
    ClassifierDF, sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
):
    """
    Wraps :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class LinearDiscriminantAnalysisDF(
    ClassifierDF, sklearn.discriminant_analysis.LinearDiscriminantAnalysis
):
    """
    Wraps :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`; accepts and
    returns data frames.
    """

    pass


#
# naive bayes
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class GaussianNBDF(ClassifierDF, sklearn.naive_bayes.GaussianNB):
    """
    Wraps :class:`sklearn.naive_bayes.GaussianNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class MultinomialNBDF(ClassifierDF, sklearn.naive_bayes.MultinomialNB):
    """
    Wraps :class:`sklearn.naive_bayes.MultinomialNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class ComplementNBDF(ClassifierDF, sklearn.naive_bayes.ComplementNB):
    """
    Wraps :class:`sklearn.naive_bayes.ComplementNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class BernoulliNBDF(ClassifierDF, sklearn.naive_bayes.BernoulliNB):
    """
    Wraps :class:`sklearn.naive_bayes.BernoulliNB`; accepts and returns data frames.
    """

    pass


#
# calibration
#

# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=MetaClassifierWrapperDF)
class CalibratedClassifierCVDF(
    ClassifierDF, sklearn.calibration.CalibratedClassifierCV
):
    """
    Wraps :class:`sklearn.calibration.CalibratedClassifierCV`; accepts and returns data
    frames.
    """

    pass


#
# SVM
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class SVCDF(ClassifierDF, sklearn.svm.SVC):
    """
    Wraps :class:`sklearn.svm.classes.SVC`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class NuSVCDF(ClassifierDF, sklearn.svm.NuSVC):
    """
    Wraps :class:`sklearn.svm.classes.NuSVC`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class LinearSVCDF(ClassifierDF, sklearn.svm.LinearSVC):
    """
    Wraps :class:`sklearn.svm.classes.LinearSVC`; accepts and returns data frames.
    """

    pass


#
# gaussian process
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class GaussianProcessClassifierDF(
    ClassifierDF, sklearn.gaussian_process.GaussianProcessClassifier
):
    """
    Wraps :class:`sklearn.gaussian_process.gpc.GaussianProcessClassifier`; accepts and
    returns data frames.
    """

    pass


#
# linear model
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class LogisticRegressionDF(ClassifierDF, sklearn.linear_model.LogisticRegression):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegression`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class LogisticRegressionCVDF(ClassifierDF, sklearn.linear_model.LogisticRegressionCV):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegressionCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class LogisticRegressionCVDF(ClassifierDF, sklearn.linear_model.LogisticRegressionCV):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegressionCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class PassiveAggressiveClassifierDF(
    ClassifierDF, sklearn.linear_model.PassiveAggressiveClassifier
):
    """
    Wraps :class:`sklearn.linear_model.passive_aggressive.PassiveAggressiveClassifier`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class PerceptronDF(ClassifierDF, sklearn.linear_model.Perceptron):
    """
    Wraps :class:`sklearn.linear_model.perceptron.Perceptron`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class SGDClassifierDF(ClassifierDF, sklearn.linear_model.SGDClassifier):
    """
    Wraps :class:`sklearn.linear_model.stochastic_gradient.SGDClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class RidgeClassifierDF(ClassifierDF, sklearn.linear_model.RidgeClassifier):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class RidgeClassifierCVDF(ClassifierDF, sklearn.linear_model.RidgeClassifierCV):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeClassifierCV`; accepts and returns
    data frames.
    """

    pass


#
# semi-supervised
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class LabelPropagationDF(ClassifierDF, sklearn.semi_supervised.LabelPropagation):
    """
    Wraps :class:`sklearn.semi_supervised.label_propagation.LabelPropagation`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class LabelSpreadingDF(ClassifierDF, sklearn.semi_supervised.LabelSpreading):
    """
    Wraps :class:`sklearn.semi_supervised.label_propagation.LabelSpreading`; accepts and
    returns data frames.
    """

    pass


#
# multi-class
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=MetaClassifierWrapperDF)
class OneVsRestClassifierDF(ClassifierDF, sklearn.multiclass.OneVsRestClassifier):
    """
    Wraps :class:`sklearn.multiclass.OneVsRestClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=MetaClassifierWrapperDF)
class OneVsOneClassifierDF(ClassifierDF, sklearn.multiclass.OneVsOneClassifier):
    """
    Wraps :class:`sklearn.multiclass.OneVsOneClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=MetaClassifierWrapperDF)
class OutputCodeClassifierDF(ClassifierDF, sklearn.multiclass.OutputCodeClassifier):
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
    MetaClassifierWrapperDF[sklearn.multioutput.MultiOutputClassifier], _abc.ABC
):
    # noinspection PyPep8Naming
    def _prediction_with_class_labels(
        self,
        X: pd.DataFrame,
        y: _t.Union[pd.Series, pd.DataFrame, list, np.ndarray],
        classes: _t.Optional[_t.Sequence[_t.Any]] = None,
    ) -> _t.Union[pd.Series, pd.DataFrame, _t.List[pd.DataFrame]]:

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
@df_estimator(df_wrapper_type=_MultiOutputClassifierWrapperDF)
class MultiOutputClassifierDF(ClassifierDF, sklearn.multioutput.MultiOutputClassifier):
    """
    Wraps :class:`sklearn.multioutput.MultiOutputClassifier`; accepts and returns data
    frames.
    """

    pass


#
# chaining
#


class _ClassifierChainWrapperDF(
    MetaClassifierWrapperDF[sklearn.multioutput.ClassifierChain], _abc.ABC
):
    # noinspection PyPep8Naming
    def _prediction_with_class_labels(
        self,
        X: pd.DataFrame,
        y: _t.Union[pd.Series, pd.DataFrame, list, np.ndarray],
        classes: _t.Optional[_t.Sequence[_t.Any]] = None,
    ) -> _t.Union[pd.Series, pd.DataFrame, _t.List[pd.DataFrame]]:
        return super()._prediction_with_class_labels(
            X=X, y=y, classes=range(self.n_outputs)
        )


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=_ClassifierChainWrapperDF)
class ClassifierChainDF(ClassifierDF, sklearn.multioutput.ClassifierChain):
    """
    Wraps :class:`sklearn.multioutput.ClassifierChain`; accepts and returns data frames.
    """

    pass


#
# neural network
#


# noinspection PyAbstractClass
@df_estimator(df_wrapper_type=ClassifierWrapperDF)
class MLPClassifierDF(ClassifierDF, sklearn.neural_network.MLPClassifier):
    """
    Wraps :class:`sklearn.neural_network.multilayer_perceptron.MLPClassifier`; accepts
    and returns data frames.
    """

    pass


#
# export all symbols ending in "DF"
#

__all__ = [sym for sym in dir() if sym.endswith("DF") and not sym.startswith("_")]
