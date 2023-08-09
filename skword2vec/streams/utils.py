"""Utilities for streaming data from disk or other sources."""
import functools
import random
from itertools import islice
from typing import Callable, Iterable, List, Literal, Optional, TypeVar


def pipe_streams(*transforms: Callable) -> Callable:
    """Pipes iterator transformations together.

    Parameters
    ----------
    *transforms: Callable
        Generator funcitons that transform an iterable into another iterable.

    Returns
    -------
    Callable
        Generator function composing all of the other ones.
    """

    def _pipe(x: Iterable) -> Iterable:
        for f in transforms:
            x = f(x)
        return x

    return _pipe


def reusable(gen_func: Callable) -> Callable:
    """
    Function decorator that turns your generator function into an
    iterator, thereby making it reusable.

    Parameters
    ----------
    gen_func: Callable
        Generator function, that you want to be reusable

    Returns
    ----------
    _multigen: Callable
        Sneakily created iterator class wrapping the generator function
    """

    @functools.wraps(gen_func, updated=())
    class _multigen:
        def __init__(self, *args, limit=None, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
            self.limit = limit
            # functools.update_wrapper(self, gen_func)

        def __iter__(self):
            if self.limit is not None:
                return islice(
                    gen_func(*self.__args, **self.__kwargs), self.limit
                )
            return gen_func(*self.__args, **self.__kwargs)

    return _multigen


U = TypeVar("U")


def chunk(
    iterable: Iterable[U], chunk_size: int, sample_size: Optional[int] = None
) -> Iterable[List[U]]:
    """
    Generator function that chunks an iterable for you.

    Parameters
    ----------
    iterable: Iterable of T
        The iterable you'd like to chunk.
    chunk_size: int
        The size of chunks you would like to get back
    sample_size: int or None, default None
        If specified the yielded lists will be randomly sampled with the buffer
        with replacement. Sample size determines how big you want
        those lists to be.

    Yields
    ------
    buffer: list of T
        sample_size or chunk_size sized lists chunked from
        the original iterable.
    """
    buffer = []
    for index, elem in enumerate(iterable):
        buffer.append(elem)
        if (index % chunk_size == (chunk_size - 1)) and (index != 0):
            if sample_size is None:
                yield buffer
            else:
                yield random.choices(buffer, k=sample_size)
            buffer = []


def stream_files(
    paths: Iterable[str],
    lines: bool = False,
    not_found_action: Literal["exception", "none", "drop"] = "exception",
) -> Iterable[Optional[str]]:
    """Streams text contents from files on disk.

    Parameters
    ----------
    paths: iterable of str
        Iterable of file paths on disk.
    lines: bool, default False
        Indicates whether you want to get a stream over lines
        or file contents.
    not_found_action: {'exception', 'none', 'drop'}, default 'exception'
        Indicates what should happen if a file was not found.
        'exception' propagates the exception to top level, 'none' yields
        None for each file that fails, 'drop' ignores them completely.

    Yields
    ------
    str or None
        File contents or lines in files if lines is True.
        Can only yield None if not_found_action is 'none'.
    """
    for path in paths:
        try:
            with open(path) as in_file:
                if lines:
                    for line in in_file:
                        yield line
                else:
                    yield in_file.read()
        except FileNotFoundError as e:
            if not_found_action == "exception":
                raise FileNotFoundError(
                    f"Streaming failed as file {path} could not be found"
                ) from e
            elif not_found_action == "none":
                yield None
            elif not_found_action == "drop":
                continue
            else:
                raise ValueError(
                    """Unrecognized `not_found_action`.
                    Please chose one of `"exception", "none", "drop"`"""
                )


def flatten_stream(nested: Iterable, axis: int = 1) -> Iterable:
    """Turns nested stream into a flat stream.
    If multiple levels are nested, the iterable will be flattenned along
    the given axis.

    To match the behaviour of Awkward Array flattening, axis=0 only
    removes None elements from the array along the outermost axis.

    Negative axis values are not yet supported.

    Parameters
    ----------
    nested: iterable
        Iterable of iterables of unknown depth.
    axis: int, default 1
        Axis/level of depth at which the iterable should be flattened.

    Returns
    -------
    iterable
        Iterable with one lower level of nesting.
    """
    if not isinstance(nested, Iterable):
        raise ValueError(
            f"Nesting is too deep, values at level {axis} are not iterables"
        )
    if axis == 0:
        return (elem for elem in nested if elem is not None and (elem != []))
    if axis == 1:
        for sub in nested:
            for elem in sub:
                yield elem
    elif axis > 1:
        for sub in nested:
            yield flatten_stream(sub, axis=axis - 1)
    else:
        raise ValueError("Flattening axis needs to be greater than 0.")


def deeplist(nested) -> list:
    """Recursively turns nested iterable to list.

    Parameters
    ----------
    nested: iterable
        Nested iterable.

    Returns
    -------
    list
        Nested list.
    """
    if not isinstance(nested, Iterable) or isinstance(nested, str):
        return nested  # type: ignore
    else:
        return [deeplist(sub) for sub in nested]
