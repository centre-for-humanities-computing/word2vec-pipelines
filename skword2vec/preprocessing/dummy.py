import importlib
import string
from multiprocessing import Pool
from typing import Iterable, Union

from sklearn.base import BaseEstimator, TransformerMixin

punct_without_dot = "\"#$%&'()*+,-/:<=>@[\\]^_`{|}~"
sentence_seps = "?!.;"
sep_to_dot = str.maketrans(sentence_seps, "." * len(sentence_seps))
punct_to_space = str.maketrans(
    string.punctuation, " " * len(string.punctuation)
)
punct_to_space_keep_dots = str.maketrans(
    punct_without_dot, " " * len(punct_without_dot)
)
digit_to_space = str.maketrans(string.digits, " " * len(string.digits))


class DummyPreprocessor(TransformerMixin, BaseEstimator):
    """Language agnostic dummy preprocessor that is probably orders
    of magnitudes faster than spaCy but also produces results of
    lower quality.

    Parameters
    ----------
    stop_words: str or iterable of str or None, default None
        Words to remove from all texts.
        If a single string, it is interpreted as a language code,
        and stop words are imported from spaCy.
        If its an iterable of strings, every token will be removed that's
        in the list.
        If None, nothing gets removed.
    lowercase: bool, default True
        Inidicates whether tokens should be lowercased.
    remove_digits: bool, default True
        Inidicates whether digits should be removed.
    remove_punctuation: bool, default True
        Indicates whether the component should remove
        punctuation.
    sentencize: bool, default False
        Determines whether the document should be split into sentences.
    n_jobs: int, default 1
        Number of cores to use for preprocessing.
        -1 stands for all cores.
    chunksize: int, default 100
        Chunk size used in imap() for multicore
        processing. Bigger values can result in faster processing
        because less memory has to be copied over to new processes.
        Ignored if n_jobs == 1.
    """

    def __init__(
        self,
        stop_words: Union[str, list[str], None],
        lowercase: bool = True,
        remove_digits: bool = True,
        remove_punctuation: bool = True,
        sentencize: bool = False,
        n_jobs: int = 1,
        chunkize: int = 100,
    ):
        self.sentencize = sentencize
        self.lowercase = lowercase
        self.remove_digits = remove_digits
        self.remove_punctuation = remove_punctuation
        self.n_jobs = n_jobs
        self.chunksize = chunkize
        if isinstance(stop_words, str):
            lang = stop_words
            self.stop_word_set = importlib.import_module(
                f"spacy.lang.{lang}.stop_words"
            ).STOP_WORDS
        elif stop_words is None:
            self.stop_word_set = set()
        else:
            self.stop_word_set = set(stop_words)

    def fit(self, X, y=None):
        """Exists for compatiblity, doesn't do anything."""
        return self

    def partial_fit(self, X, y=None):
        """Exists for compatiblity, doesn't do anything."""
        return self

    def process_string(self, text: str) -> list[list[str]]:
        # Removes digits if asked to
        if self.remove_digits:
            text = text.translate(digit_to_space)
        # If we want to separate sentences we transform
        # all sentence separators into dots
        if self.sentencize:
            text = text.translate(sep_to_dot)
        if self.remove_punctuation:
            if self.sentencize:
                # We remove everything except dots
                # (remember we use those as sentence boundary.)
                text = text.translate(punct_to_space_keep_dots)
            else:
                # We remove all punctuation
                text = text.translate(punct_to_space)
        if self.lowercase:
            text = text.lower()
        text = text.strip()
        # We split on dot
        if self.sentencize:
            sentences = text.split(".")
        else:
            sentences = [text]
        res = []
        for sentence in sentences:
            if not sentence:
                continue
            sentence_tokens = []
            # Tokens are split on whitespace
            for token in sentence.split():
                if token not in self.stop_word_set:
                    sentence_tokens.append(token)
            res.append(sentence_tokens)
        return res

    def transform(self, X: Iterable[str]) -> list[list[list[str]]]:
        """Preprocesses document with a dummy pipeline.

        Parameters
        ----------
        X: iterable of str
            Stream of documents.

        Returns
        -------
        list of list of list of str
            List of documents represented as list of sentences
            represented as lists of tokens.
        """
        if self.n_jobs == 1:
            res = map(self.process_string, X)
        else:
            with Pool(self.n_jobs) as pool:
                res = pool.imap(
                    self.process_string, X, chunksize=self.chunksize
                )
        return list(res)
