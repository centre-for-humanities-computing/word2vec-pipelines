import json
from dataclasses import dataclass
from functools import wraps
from itertools import islice
from typing import Callable, Iterable, Literal

from skword2vec.streams.utils import (
    chunk,
    deeplist,
    flatten_stream,
    reusable,
    stream_files,
)


@dataclass
class Stream:
    iterable: Iterable

    @reusable
    def __iter__(self):
        return self.iterable

    def filter(self, func: Callable, *args, **kwargs):
        @wraps(func)
        def _func(elem):
            return func(elem, *args, **kwargs)

        _iterable = reusable(filter)(_func, self.iterable)
        return Stream(_iterable)

    def map(self, func: Callable, *args, **kwargs):
        @wraps(func)
        def _func(elem):
            return func(elem, *args, **kwargs)

        _iterable = reusable(map)(_func, self.iterable)
        return Stream(_iterable)

    def pipe(self, func: Callable, *args, **kwargs):
        @wraps(func)
        def _func(iterable):
            return func(iterable, *args, **kwargs)

        _iterable = reusable(_func)(self.iterable)
        return Stream(_iterable)

    def islice(self, *args):
        return self.pipe(islice, *args)

    def evaluate(self, deep: bool = False):
        if deep:
            evaluator = deeplist
        else:
            evaluator = list
        return self.pipe(evaluator)

    def read_files(
        self,
        lines: bool = True,
        not_found_action: Literal["exception", "none", "drop"] = "exception",
    ):
        return self.pipe(
            reusable(stream_files),
            lines=lines,
            not_found_action=not_found_action,
        )

    def json(self):
        return self.map(json.loads)

    def grab(self, column: str):
        @reusable
        def _func(iterable):
            for record in iterable:
                yield record[column]

        return self.pipe(_func)

    def flatten(self, axis=1):
        return self.pipe(flatten_stream, axis=axis)

    def chunk(self, size: int):
        return self.pipe(chunk, chunk_size=size)
