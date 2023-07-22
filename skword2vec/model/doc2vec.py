from typing import Iterable, Literal

import mmh3
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from skword2vec.streams import deeplist, flatten_stream


class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible Doc2Vec model.

    Parameters
    ----------
    n_components: int, default 100
        Vector dimensionality.
    n_jobs: int, default 1
        Number of cores to be used by the model.
    window: int, default 5
        Window size of the Word2Vec model.
    algorithm: {"dm", "dbow"}, "dm"
        Indicates whether distributed memory or distributed
        bag of words should be used as the training algortihm.
    max_docs: int, default 100_000
        Maximum number of document embeddings that should be stored
        in memory. This is mostly useful when streaming
        texts from a large data set and storing vectors for all
        documents would not be feasible.
        Set a smaller value if you want smaller embeddings,
        and set a larger one if you want to keep embeddings for
        all seen documents.
    tagging_scheme: {"hash", "closest"}, default "hash"
        Scheme used for tagging documents after we ran out of
        buckets. If 'hash' the document will be hashed and its
        modulo max_docs will be the tag.
        If "closest" is used, the closest document is chosen as the tag,
        kind of aggregating closer documents together.
        This could result in better inferences, but likely makes training
        a lot slower.
    frozen: bool, default False
        Indicates whether the model should be frozen in the pipeline.
        This can be advantageous when you want to train other models down
        the pipeline from the outputs of a pretrained Doc2Vec model.
    **kwargs
        Keyword arguments passed down to Gensim's Doc2Vec model.

    Attributes
    ----------
    model: Doc2Vec
        Underlying Gensim Doc2Vec model.
    loss: list of float
        List of training losses after each call to partial_fit.
    components_: array of shape (n_components, n_stored_documents)
        Matrix of all stored embeddings in the model.
    """

    def __init__(
        self,
        n_components: int = 100,
        n_jobs: int = 1,
        window: int = 5,
        algorithm: Literal["dm", "dbow"] = "dm",
        max_docs: int = 100_000,
        tagging_scheme: Literal["hash", "closest"] = "hash",
        frozen: bool = False,
        **kwargs,
    ):
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.window = window
        self.kwargs = kwargs
        self.model = None
        self.algorithm = algorithm
        self.max_docs = max_docs
        if tagging_scheme not in ["hash", "closest"]:
            raise ValueError(
                "Tagging scheme should either be 'hash' or 'closest'"
            )
        self.tagging_scheme = tagging_scheme
        self.loss: list[float] = []
        self.frozen = frozen
        self.seen_docs = 0

    @classmethod
    def from_gensim(
        cls,
        model: Doc2Vec,
        max_docs: int = 100_000,
        tagging_scheme: Literal["hash", "closest"] = "hash",
        frozen: bool = False,
        **kwargs,
    ):
        """Creates Doc2VecVectorizer from the given Gensim Doc2Vec model.

        Parameters
        ----------
        model: Doc2Vec
            Gensim Doc2Vec object.
        max_docs: int, default 100_000
            Maximum number of document embeddings that should be stored
            in memory. This is mostly useful when streaming
            texts from a large data set and storing vectors for all
            documents would not be feasible.
            Set a smaller value if you want smaller embeddings,
            and set a larger one if you want to keep embeddings for
            all seen documents.
        tagging_scheme: {"hash", "closest"}, default "hash"
            Scheme used for tagging documents after we ran out of
            buckets. If 'hash' the document will be hashed and its
            modulo max_docs will be the tag.
            If "closest" is used, the closest document is chosen as the tag,
            kind of aggregating closer documents together.
            This could result in better inferences, but likely makes training
            a lot slower.
        frozen: bool, default False
            Indicates whether the model should be frozen in the pipeline.
            This can be advantageous when you want to train other models down
            the pipeline from the outputs of a pretrained Doc2Vec model.
        **kwargs
            Keyword arguments passed down to Gensim's Doc2Vec model.

        Returns
        -------
        Doc2VecVectorizer
            Transformer object with the given Doc2Vec model.
        """
        res = cls(
            n_components=model.vector_size,
            window=model.window,
            n_jobs=model.workers,
            frozen=frozen,
            tagging_scheme=tagging_scheme,
            max_docs=max_docs,
            **kwargs,
        )
        res.model = model
        res.seen_docs = model.dv.vectors.shape[0]
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
        self.model = None
        self.seen_docs = 0
        self.partial_fit(documents)
        return self

    def tag_documents(
        self, documents: list[list[str]]
    ) -> list[TaggedDocument]:
        if self.model is None:
            raise TypeError(
                "You should not call tag_documents"
                "before model is initialised."
            )
        res = []
        for document in documents:
            # While we have available slots we just add new documents to those
            if self.seen_docs < self.max_docs:
                res.append(TaggedDocument(document, [self.seen_docs]))
            else:
                # If we run out, we choose a tag based on a scheme
                if self.tagging_scheme == "hash":
                    # Here we use murmur hash
                    hash = mmh3.hash("".join(document))
                    id = hash % self.max_docs
                    res.append(TaggedDocument(document, [id]))
                elif self.tagging_scheme == "closest":
                    # We obtain the key of the most semantically
                    # similar document and use that.
                    doc_vector = self.model.infer_vector(document)
                    key, similarity = self.model.dv.similar_by_key(
                        doc_vector, topn=1
                    )[0]
                    res.append(TaggedDocument(document, [key]))
                else:
                    raise ValueError(
                        "Tagging scheme should either be 'hash' or 'closest'"
                        f" but {self.tagging_scheme} was provided."
                    )
            self.seen_docs += 1
        return res

    def initialise_model(self, documents: list[list[str]]) -> list[list[str]]:
        # We initialise the model with a set of documents if
        # there are more documents in total than max_docs
        # This is necessary because otherwise if the user provides
        # a set of documents larger than max_docs and the tagging scheme
        # is 'closest' the model training will start before we can
        # assign tags to later documents
        if len(documents) > self.max_docs:
            tagged_docs = [
                TaggedDocument(doc, [i])
                for i, doc in enumerate(documents[: self.max_docs])
            ]
            self.seen_docs = self.max_docs
            documents = documents[self.max_docs :]
        else:
            tagged_docs = None
        self.model = Doc2Vec(
            documents=tagged_docs,
            vector_size=self.n_components,
            window=self.window,
            workers=self.n_jobs,
            **self.kwargs,
        )
        # We return the rest so that the model doesn't run twice over the
        # documents used for initialization
        return documents

    def partial_fit(
        self, documents: Iterable[Iterable[Iterable[str]]], y=None
    ):
        """Partially fits doc2vec model (online fitting).

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
        # Flattens out sentences in a document
        documents: list[list[str]] = deeplist(
            flatten_stream(documents, axis=2)
        )
        if self.model is None:
            documents = self.initialise_model(documents)
        update = self.seen_docs != 0
        # We obtained tagged documents
        tagged_docs = self.tag_documents(documents)
        # Then build vocabulary
        self.model.build_vocab(tagged_docs, update=update)
        self.model.train(
            tagged_docs,
            total_examples=self.seen_docs,
            epochs=1,
            compute_loss=True,
        )
        self.loss.append(self.model.get_latest_training_loss())
        return self

    def transform(
        self, documents: Iterable[Iterable[Iterable[str]]]
    ) -> np.ndarray:
        """Infers vectors for all of the given documents.

        Parameters
        ----------
        documents: deep iterable with dimensions (documents, sentences, words)
            Words in each sentence in each document.
            Sentences exist as a level, so that this is compatible
            with preprocessors and word2vec API, but are ignored.

        Results
        -------
        ndarray of shape (n_documents, n_components)
            Vectors for all given documents.
        """
        if self.model is None:
            raise NotFittedError(
                "Model ha been not fitted, please fit before inference."
            )
        # Flattens out sentences in a document
        documents: list[list[str]] = deeplist(
            flatten_stream(documents, axis=2)
        )
        vectors = [self.model.infer_vector(doc) for doc in documents]
        return np.stack(vectors)

    @property
    def components_(self) -> np.ndarray:
        if self.model is None:
            raise NotFittedError("Model has not been fitted yet.")
        return np.array(self.model.dv.vectors).T

    @classmethod
    def load(
        cls,
        path: str,
        max_docs: int = 100_000,
        tagging_scheme: Literal["hash", "closest"] = "hash",
        frozen: bool = False,
        **kwargs,
    ):
        """Loads model from disk.

        Parameters
        ----------
        path: str
            Path to Doc2Vec model.
        max_docs: int, default 100_000
            Maximum number of document embeddings that should be stored
            in memory. This is mostly useful when streaming
            texts from a large data set and storing vectors for all
            documents would not be feasible.
            Set a smaller value if you want smaller embeddings,
            and set a larger one if you want to keep embeddings for
            all seen documents.
        tagging_scheme: {"hash", "closest"}, default "hash"
            Scheme used for tagging documents after we ran out of
            buckets. If 'hash' the document will be hashed and its
            modulo max_docs will be the tag.
            If "closest" is used, the closest document is chosen as the tag,
            kind of aggregating closer documents together.
            This could result in better inferences, but likely makes training
            a lot slower.
        frozen: bool, default False
            Indicates whether the model should be frozen in the pipeline.
            This can be advantageous when you want to train other models down
            the pipeline from the outputs of a pretrained Doc2Vec model.
        **kwargs
            Keyword arguments passed down to Gensim's Doc2Vec model.



        Returns
        -------
        Doc2VecVectorizer
            Transformer component.
        """
        model: Doc2Vec = Doc2Vec.load(path)  # type: ignore
        res = cls.from_gensim(
            model,
            frozen=frozen,
            tagging_scheme=tagging_scheme,
            max_docs=max_docs,
            **kwargs,
        )
        return res

    def save(self, path: str):
        """Saves model to disk.
        Beware that this saves the Gensim model itself, so you will be able
        to load it with Doc2Vec.load() too.

        Parameters
        ----------
        path: str
            Path to save Doc2Vec model to.
        """
        if self.model is None:
            raise NotFittedError("Doc2Vec model has not been fitted yet.")
        self.model.save(path)
