from typing import Iterable, Literal

import awkward as ak
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from skword2vec.streams.utils import deeplist, flatten_stream


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


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible Word2Vec model.

    Parameters
    ----------
    n_components: int, default 100
        Vector dimensionality.
    n_jobs: int, default 1
        Number of cores to be used by the model.
    window: int, default 5
        Window size of the Word2Vec model.
    algorithm: {'cbow', 'sg'}, default 'cbow'
        Indicates whether a continuous-bag-of-words or a skip-gram
        model should be trained.
    oov_strategy: {'drop', 'nan'}, default 'drop'
        Indicates whether you want out-of-vocabulary
        words to have a vector filled with nans or
        drop them.
    frozen: bool, default False
        Indicates whether the model should be frozen in the pipeline.
        This can be advantageous when you want to train other models down
        the pipeline from the outputs of a pretrained Word2Vec model.

    **kwargs
        Keyword arguments passed down to Gensim's Word2Vec model.

    Attributes
    ----------
    model: Word2Vec
        Underlying Gensim Word2Vec model.
    loss: list of float
        List of training losses after each call to partial_fit.
    feature_names_in_: array of shape (n_vocab)
        Vocabulary of the Word2Vec model.
    components_: array of shape (n_components, n_vocab)
        Matrix of all embeddings in the model.
    """

    def __init__(
        self,
        n_components: int = 100,
        n_jobs: int = 1,
        window: int = 5,
        algorithm: Literal["cbow", "sg"] = "cbow",
        oov_strategy: Literal["drop", "nan"] = "nan",
        frozen: bool = False,
        **kwargs,
    ):
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.window = window
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.model = None
        self.loss: list[float] = []
        self.oov_strategy = oov_strategy
        self.frozen = frozen

    @classmethod
    def from_gensim(
        cls,
        model: Word2Vec,
        oov_strategy: Literal["drop", "nan"] = "nan",
        frozen: bool = False,
    ):
        """Creates Word2VecVectorizer from the given Gensim Word2Vec model.

        Parameters
        ----------
        model: Word2Vec
            Gensim Word2Vec object.
        oov_strategy: {'drop', 'nan'}, default 'drop'
            Indicates whether you want out-of-vocabulary
            words to have a vector filled with nans or
            drop them.
        frozen: bool, default False
            Indicates whether the model should be frozen in the pipeline.
            This can be advantageous when you want to train other models down
            the pipeline from the outputs of a pretrained Word2Vec model.


        Returns
        -------
        Word2VecVectorizer
            Transformer object with the given Word2Vec model.
        """
        res = cls(
            n_components=model.vector_size,
            window=model.window,
            algorithm="sg" if model.sg else "cbow",
            n_jobs=model.workers,
            oov_strategy=oov_strategy,
            frozen=frozen,
        )
        res.model = model
        return res

    def fit(self, documents: Iterable[Iterable[Iterable[str]]], y=None):
        """Fits a new word2vec model to the given sentences.

        Parameters
        ----------
        documents: deep iterable with dimensions (documents, sentences, words)
            Words in each sentence in each document.
        y: None
            Ignored. Exists for compatibility.

        Returns
        -------
        self
            Fitted model.
        """
        if self.frozen:
            return self
        self.model = Word2Vec(
            vector_size=self.n_components,
            window=self.window,
            sg=int(self.algorithm == "sg"),
            workers=self.n_jobs,
            **self.kwargs,
        )
        self.partial_fit(documents)
        return self

    def partial_fit(
        self, documents: Iterable[Iterable[Iterable[str]]], y=None
    ):
        """Partially fits word2vec model (online fitting).

        Parameters
        ----------
        documents: deep iterable with dimensions (documents, sentences, words)
            Words in each sentence in each document.
        y: None
            Ignored. Exists for compatibility.

        Returns
        -------
        self
            Fitted model.
        """
        if self.frozen:
            return self
        sentences = flatten_stream(documents, axis=1)
        sentences = deeplist(sentences)
        if self.model is None:
            self.model = Word2Vec(
                vector_size=self.n_components,
                window=self.window,
                sg=int(self.algorithm == "sg"),
                workers=self.n_jobs,
                **self.kwargs,
            )
        prev_corpus_count = self.model.corpus_count
        update = prev_corpus_count != 0
        self.model.build_vocab(sentences, update=update)
        self.model.train(
            sentences,
            total_examples=self.model.corpus_count + prev_corpus_count,  # type: ignore
            epochs=1,
            compute_loss=True,
        )
        self.loss.append(self.model.get_latest_training_loss())
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
        if self.model is None:
            raise NotFittedError("Word2Vec model has not been fitted yet.")
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
                embeddings[i_word, :] = self.model.wv[word]
            except KeyError:
                continue
        embeddings = build_ragged_array(
            sent_word_counts, doc_sent_counts, embeddings
        )
        if self.oov_strategy == "drop":
            embeddings = ak.nan_to_none(embeddings)
            embeddings = ak.drop_none(embeddings)
        if self.oov_strategy == "nan":
            embeddings = ak.fill_none(embeddings, np.nan)
        return embeddings

    @property
    def components_(self) -> np.ndarray:
        if self.model is None:
            raise NotFittedError("Word2Vec model has not been fitted yet.")
        return np.array(self.model.wv.vectors).T

    @property
    def feature_names_in_(self) -> np.ndarray:
        if self.model is None:
            raise NotFittedError("Word2Vec model has not been fitted yet.")
        return np.array(self.model.wv.index_to_key)

    @classmethod
    def load(
        cls,
        path: str,
        oov_strategy: Literal["drop", "nan"],
        frozen: bool = False,
    ):
        """Loads model from disk.

        Parameters
        ----------
        path: str
            Path to Word2Vec model.
        oov_strategy: {'drop', 'nan'}, default 'drop'
            Indicates whether you want out-of-vocabulary
            words to have a vector filled with nans or
            drop them.
        frozen: bool, default False
            Indicates whether the model should be frozen in the pipeline.
            This can be advantageous when you want to train other models down
            the pipeline from the outputs of a pretrained Word2Vec model.



        Returns
        -------
        Word2VecVectorizer
            Transformer component.
        """
        model = Word2Vec.load(path)
        res = cls.from_gensim(model, oov_strategy=oov_strategy, frozen=frozen)
        return res

    def save(self, path: str):
        """Saves model to disk.
        Beware that this saves the Gensim model itself, so you will be able
        to load it with Word2Vec.load() too.

        Parameters
        ----------
        path: str
            Path to save Word2Vec model to.
        """
        if self.model is None:
            raise NotFittedError("Word2Vec model has not been fitted yet.")
        self.model.save(path)
