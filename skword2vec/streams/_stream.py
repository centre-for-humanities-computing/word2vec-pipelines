import functools
import json
from dataclasses import dataclass
from itertools import islice
from typing import Callable, Iterable, Literal

from sklearn.base import BaseEstimator

from skword2vec.streams.filtering import filter_batches
from skword2vec.streams.utils import (
    chunk,
    deeplist,
    flatten_stream,
    reusable,
    stream_files,
)


@dataclass
class Stream:
    """Utility class for streaming, batching and filtering texts
    from an external source.

    Parameters
    ----------
    iterable: Iterable
        Core iterable object in the stream.
    """

    iterable: Iterable

    def __iter__(self):
        return iter(self.iterable)

    def filter(self, func: Callable, *args, **kwargs):
        """Filters the stream given a function that returns a bool."""

        @functools.wraps(func)
        def _func(elem):
            return func(elem, *args, **kwargs)

        _iterable = reusable(filter)(_func, self.iterable)
        return Stream(_iterable)

    def map(self, func: Callable, *args, **kwargs):
        """Maps a function over the stream."""

        @functools.wraps(func)
        def _func(elem):
            return func(elem, *args, **kwargs)

        _iterable = reusable(map)(_func, self.iterable)
        return Stream(_iterable)

    def pipe(self, func: Callable, *args, **kwargs):
        """Pipes the stream into a function that takes
        the whole stream and returns a new one."""

        @functools.wraps(func)
        def _func(iterable):
            return func(iterable, *args, **kwargs)

        _iterable = reusable(_func)(self.iterable)
        return Stream(_iterable)

    def islice(self, *args):
        """Equivalent to itertools.islice()."""
        return self.pipe(islice, *args)

    def evaluate(self, deep: bool = False):
        """Evaluates the entire iterable and collects it into
        a list.

        Parameters
        ----------
        deep: bool, default False
            Indicates whether nested iterables should be deeply
            evaluated. Uses deeplist() internally.
        """
        if deep:
            _iterable = deeplist(self.iterable)
        else:
            _iterable = list(self.iterable)
        return Stream(_iterable)

    def read_files(
        self,
        lines: bool = True,
        not_found_action: Literal["exception", "none", "drop"] = "exception",
    ):
        """Reads a stream of file paths from disk.

        Parameters
        ----------
        lines: bool, default True
            Indicates whether lines should be streamed or not.
        not_found_action: str, default 'exception'
            Indicates what should be done if a given file is not found.
            'exception' raises an exception,
            'drop' ignores it,
            'none' returns a None for each nonexistent file.
        """
        return self.pipe(
            stream_files,
            lines=lines,
            not_found_action=not_found_action,
        )

    def json(self):
        """Parses a stream of texts into JSON objects."""
        return self.map(json.loads)

    def grab(self, field: str):
        """Grabs one field from a stream of records."""
        return self.map(lambda record: record[field])

    def flatten(self, axis=1):
        """Flattens a nested stream along a given axis."""
        return self.pipe(flatten_stream, axis=axis)

    def chunk(self, size: int):
        """Chunks stream with the given batch size."""
        return self.pipe(chunk, chunk_size=size)

    def filter_batches(self, estimator: BaseEstimator, prefit: bool = True):
        """Filters batches with a scikit-learn compatible
        estimator.

        Parameters
        ----------
        estimator: BaseEstimator
            Scikit-learn estimator to use for filtering the batches.
            Either needs a .predict() or .fit_predict() method.
            Every sample that gets labeled -1 will be removed from the
            batch.
        prefit: bool, default True
            Indicates whether the estimator is prefit.
            If it is .predict() will be used (novelty detection), else
            .fit_predict() will be used (outlier detection).
        """
        return self.pipe(filter_batches, estimator=estimator, prefit=prefit)

    def collect(self, deep: bool = False):
        """Does the same as evaluate()."""
        return self.evaluate(deep)
