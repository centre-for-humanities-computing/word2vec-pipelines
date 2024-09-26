from typing import Iterable, Literal

import awkward as ak
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin

from skword2vec.streams import deeplist


def count_to_offset(sent_word_counts: list[int]):
    offsets = np.cumsum(sent_word_counts)
    offsets = np.concatenate(([0], offsets))
    return ak.index.Index(offsets)


def build_ragged_array(
    sent_word_counts: list[int],
    doc_sent_counts: list[int],
    embeddings: np.ndarray,
) -> ak.Array:
    offset = count_to_offset(sent_word_counts)
    contents = ak.contents.NumpyArray(embeddings)
    ragged = ak.contents.ListOffsetArray(offset, contents)
    ragged = ak.unflatten(ragged, doc_sent_counts, axis=0)
    return ragged


class StaticWordVectorizer(BaseEstimator, TransformerMixin):
    """Vectorizes text documents with arbitrarily derived
    static word embeddings.

    Parameters
    ----------
    embeddings: array-like of shape (n_vocab, n_components)
        All word embeddings stacked into a matrix, where every row
        corresponds to one word.
    vocab: array-like of shape (n_vocab)
        Vocabulary of the static embedding model.
    oov_strategy: {'drop', 'nan'}, default 'drop'
        Indicates whether you want out-of-vocabulary
        words to have a vector filled with nans or
        drop them.

    Attributes
    ----------
    key_to_index: dict[str, int]
        Mapping of words to their indices in the vocabulary.
    components_: array of shape (n_components, n_vocab)
        Matrix of all embeddings in the model.
        Note that this is the transposed embeddings,
        the reason for this is that topic models and other
        decomposition models in sklearn also have components with
        this kind of shape.
    feature_names_in_: array of shape (n_vocab)
        Vocabulary of the static embedding model.
    """

    def __init__(
        self,
        embeddings: ArrayLike,
        vocab: ArrayLike,
        oov_strategy: Literal["drop", "nan"] = "nan",
    ):
        self.embeddings = np.array(embeddings)
        self.vocab = np.array(vocab)
        self.key_to_index = {
            term: index for index, term in enumerate(self.vocab)
        }
        self.n_vocab, self.n_components = self.embeddings.shape
        self.oov_strategy = oov_strategy

    def fit(self, documents: Iterable[Iterable[Iterable[str]]], y=None):
        """Does nothing, exists for compatibility."""
        return self

    def partial_fit(
        self, documents: Iterable[Iterable[Iterable[str]]], y=None
    ):
        """Does nothing, exists for compatibility."""
        return self

    def transform(
        self, documents: Iterable[Iterable[Iterable[str]]]
    ) -> ak.Array:
        """Infers word vectors for all sentences.

        Parameters
        ----------
        documents: deep iterable with dimensions (documents, sentences, words)
            Words in each sentence in each document.

        Returns
        -------
        awkward.Array with dimensions (documents, sentences, words, dimensions)
            Ragged array of word embeddings in each sentence in each document.
            Note that this array can potentially not be used for other
            applications due to its awkward shape, and you will either
            have to do pooling or padding to turn it into a numpy array.
        """
        documents = deeplist(documents)
        sent_word_counts = []
        doc_sent_counts = []
        words = []
        for doc in documents:
            doc_sent_counts.append(len(doc))  # type: ignore
            for sent in doc:
                sent_word_counts.append(len(sent))  # type: ignore
                words.extend(sent)
        embeddings = np.full((len(words), self.n_components), np.nan)
        for i_word, word in enumerate(words):
            try:
                embeddings[i_word, :] = self.embeddings[
                    self.key_to_index[word]
                ]
            except KeyError:
                continue
        embeddings = build_ragged_array(
            sent_word_counts, doc_sent_counts, embeddings
        )
        if self.oov_strategy == "drop":
            embeddings = ak.nan_to_none(embeddings)
            embeddings = ak.drop_none(embeddings)
        return embeddings

    @property
    def components_(self) -> np.ndarray:
        return np.array(self.embeddings).T

    @property
    def feature_names_in_(self) -> np.ndarray:
        return self.vocab

    @classmethod
    def from_keyed_vectors(
        cls, keyed_vectors, oov_strategy: Literal["drop", "nan"] = "nan"
    ) -> "StaticWordVectorizer":
        """Initializes static embedding vectorizer component from
        Gensim keyed vectors.

        Parameters
        ----------
        keyed_vectors: KeyedVectors
            Keyed vectors from Gensim model.
        oov_strategy: {'drop', 'nan'}, default 'drop'
            Indicates whether you want out-of-vocabulary
            words to have a vector filled with nans or
            drop them.
        """
        vocab = np.array(keyed_vectors.index_to_key)
        embeddings = keyed_vectors.vectors
        return cls(
            vocab=vocab, embeddings=embeddings, oov_strategy=oov_strategy
        )
