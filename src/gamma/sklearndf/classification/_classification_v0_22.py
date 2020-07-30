"""
Additional implementation of :mod:`gamma.sklearndf.classification` loaded
from sklearn 0.22 onwards
"""
import sklearn
from packaging import version

if version.parse(sklearn.__version__) < version.parse("0.22"):
    # do not expose anything
    __all__ = []
else:
    # wrap additional classifiers from sklearn >= 0.22
    import logging

    from sklearn.naive_bayes import CategoricalNB

    from gamma.sklearndf import ClassifierDF
    from gamma.sklearndf._wrapper import (
        ClassifierWrapperDF,
        df_estimator,
    )

    log = logging.getLogger(__name__)

    __all__ = ["CategoricalNBDF"]

    __imported_estimators = {name for name in globals().keys() if name.endswith("DF")}

    #
    # naive bayes
    #

    # noinspection PyAbstractClass
    @df_estimator(df_wrapper_type=ClassifierWrapperDF)
    class CategoricalNBDF(ClassifierDF, CategoricalNB):
        """
        Wraps :class:`sklearn.naive_bayes.CategoricalNB`; accepts and returns data frames.
        """

        pass

    #
    # validate that __all__ comprises all symbols ending in "DF", and no others
    #

    __estimators = [
        sym
        for sym in dir()
        if sym.endswith("DF")
        and sym not in __imported_estimators
        and not sym.startswith("_")
    ]
    if set(__estimators) != set(__all__):
        raise RuntimeError(
            "__all__ does not contain exactly all DF estimators; expected value is:\n"
            f"{__estimators}"
        )
