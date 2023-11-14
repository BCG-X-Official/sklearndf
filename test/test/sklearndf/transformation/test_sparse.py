import pandas as pd
import pytest
from pandas.testing import assert_index_equal, assert_series_equal

from sklearndf._util import sparse_frame_density
from sklearndf.pipeline import FeatureUnionDF, PipelineDF
from sklearndf.transformation import CountVectorizerDF, TfidfTransformerDF


def test_tfidf() -> None:
    # expected results

    word_feature_names = (
        ["and", "document", "first", "here", "is", "it"]
        + ["last", "one", "or", "second", "the", "third", "this"]
        # single-word features
    )
    bigram_feature_names = (
        ["and the", "first document", "here is", "is it", "is the", "is this", "it the"]
        + ["last document", "or is", "second document", "the first", "the last"]
        + ["the second", "the third", "third one", "this the"]
    )

    # create a simple toy corpus, inspired by scikit-learn's documentation

    corpus = pd.Series(
        [
            "Here is the first document.",
            "Here is the second document.",
            "And the third one.",
            "Is this the first document?",
            "The last document?",
            "Or is it the second document?",
        ]
    )
    corpus_named = corpus.rename("document")

    # count the words for every document in the corpus

    word_counter = CountVectorizerDF()

    with pytest.raises(
        ValueError, match="the name of the series passed as arg X must not be None$"
    ):
        word_counter.fit_transform(corpus)

    word_counts_sparse_df = word_counter.fit_transform(corpus_named)

    assert word_counter.feature_names_out_.to_list() == word_feature_names
    assert all(
        isinstance(dtype, pd.SparseDtype) for dtype in word_counts_sparse_df.dtypes
    )

    # compute the tf-idf values for every word in every document

    tfidf = TfidfTransformerDF()
    x_tfidf = tfidf.fit_transform(word_counts_sparse_df)

    assert all(isinstance(dtype, pd.SparseDtype) for dtype in x_tfidf.dtypes)
    assert_index_equal(tfidf.feature_names_out_, word_counts_sparse_df.columns)
    assert_index_equal(tfidf.feature_names_out_, x_tfidf.columns)
    assert sparse_frame_density(x_tfidf) == pytest.approx(0.3589744)

    # count the bigrams for every document in the corpus

    bigram_counter = CountVectorizerDF(analyzer="word", ngram_range=(2, 2))
    x2 = bigram_counter.fit_transform(corpus_named)
    assert bigram_counter.feature_names_out_.to_list() == bigram_feature_names
    assert all(isinstance(dtype, pd.SparseDtype) for dtype in x2.dtypes)

    # create a pipeline that combines the word and bigram counter
    # and computes the tf-idf values for every word and bigram

    vectorize = FeatureUnionDF(
        [
            ("words", word_counter),
            ("bigrams", bigram_counter),
        ]
    )
    pipeline = PipelineDF(
        [
            ("vectorize", vectorize),
            ("tfidf", tfidf),
        ]
    )

    tfidf = pipeline.fit_transform(corpus_named)
    assert all(isinstance(dtype, pd.SparseDtype) for dtype in tfidf.dtypes)
    assert_series_equal(
        pipeline.feature_names_original_,
        pd.Series(
            index=pd.Index(
                [f"words__{name}" for name in word_feature_names]
                + [f"bigrams__{name}" for name in bigram_feature_names],
                name="feature",
            ),
            data="document",  # all features share the same input column, "document"
            name="feature_original",
        ),
    )
