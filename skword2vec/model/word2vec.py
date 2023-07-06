from typing import Iterable, Literal

import numpy as np
from gensim.models import Word2Vec
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class Word2VecTransformer(BaseEstimator, TransformerMixin):
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
        Indicates whether a continusous-bag-of-words or a skip-gram
        model should be trained.
    **kwargs
        Keyword arguments passed down to Gensim's Word2Vec model.

    Attributes
    ----------
    model: Word2Vec
        Underlying Gensim Word2Vec model.
    loss: list of float
        List of training losses after each call to partial_fit.
    """

    def __init__(
        self,
        n_components: int = 100,
        n_jobs: int = 1,
        window: int = 5,
        algorithm: Literal["cbow", "sg"] = "cbow",
        **kwargs,
    ):
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.window = window
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.model = None
        self.loss: list[float] = []

    def fit(self, sentences: Iterable[Iterable[str]], y=None):
        """Fits the word2vec model to the given sentences.

        Parameters
        ----------
        sentences: iterable of iterable of str
            List of sentences as list of tokens.
        y: None
            Ignored. Exists for compatibility.

        Returns
        -------
        self
            Fitted model.
        """
        self.model = Word2Vec(
            vector_size=self.n_components,
            window=self.window,
            sg=int(self.algorithm == "sg"),
            workers=self.n_jobs,
            **self.kwargs,
        )
        self.partial_fit(sentences)
        return self

    def partial_fit(self, sentences: Iterable[Iterable[str]], y=None):
        """Partially fits word2vec model (online fitting).

        Parameters
        ----------
        sentences: iterable of iterable of str
            List of sentences as list of tokens.
        y: None
            Ignored. Exists for compatibility.

        Returns
        -------
        self
            Fitted model.
        """
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
        self,
        sentences: Iterable[Iterable[str]],
        oov_strategy: Literal["drop", "nan"] = "drop",
    ) -> list[list[ArrayLike]]:
        """Infers word vectors for all sentences.

        Parameters
        ----------
        sentences: iterable of iterable of str
            List of sentences as list of tokens.
        oov_strategy: {'drop', 'nan'}, default 'drop'
            Indicates whether you want out-of-vocabulary
            words to have a vector filled with nans or
            drop them.

        Returns
        -------
        self
            Fitted model.
        """
        if self.model is None:
            raise NotFittedError("Word2Vec model has not been fitted yet.")
        res = []
        for sentence in sentences:
            sent_res = []
            for word in sentence:
                try:
                    embedding = self.model.wv[word]
                    sent_res.append(embedding)
                except KeyError:
                    if oov_strategy == "nan":
                        sent_res.append(np.full(self.n_components, np.nan))
            if sent_res or (oov_strategy == "nan"):
                res.append(sent_res)
        return res

    @classmethod
    def load(cls, path: str):
        """Loads model."""
        res = cls()
        res.model = Word2Vec.load(path)
        res.n_components = res.model.vector_size
        return res
