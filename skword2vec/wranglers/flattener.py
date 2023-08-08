from typing import Iterable, Union

import awkward as ak
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin

from skword2vec.streams.utils import flatten_stream


class StreamFlattener(TransformerMixin, BaseEstimator):
    """Flattens iterable stream along given axis.

    The important distinction between this and ArrayFlattener,
    is that StreamFlattener does computes lazily and creates a generator
    instead of constructing an Awkward Array. This can have a positive
    impact on performance especially when the data structure is
    not a tensor to begin with.

    Parameters
    ----------
    axis: int, default 1
        Axis/level of depth at which the iterable should be flattened.
    """

    def __init__(self, axis: int = 1):
        self.axis = axis

    def transform(self, X: Iterable) -> Iterable:
        """Flattens the given iterable along the axis.

        Parameters
        ----------
        X: nested iterable
            Iterable to flatten.

        Returns
        -------
        iterable
            Iterable that gets flattened along the given axis.
        """
        return flatten_stream(X, axis=self.axis)

    def fit(self, X, y=None):
        """Does nothing, exists for compatibility."""
        return self

    def partial_fit(self, X, y=None):
        """Does nothing, exists for compatibility."""
        return self


class ArrayFlattener(TransformerMixin, BaseEstimator):
    """Pipeline component that flattens an array along a given axis.

    Note that this component does eager evaluation and conversion to
    Akward Array, therefore it might not be ideal for streams.
    We recommend that you use this component when the output of the previous
    component is a tensor/array or Awkward Array.

    Parameters
    ----------
    axis: int, default 1
        Axis/level of depth at which the iterable should be flattened.
    """

    def __init__(self, axis: int = 1):
        self.axis = axis

    def transform(self, X: Union[ArrayLike, ak.Array]) -> ak.Array:
        """Flattens the given array along the axis.

        Parameters
        ----------
        X: ArrayLike or ak.Array
            Array to flatten.

        Returns
        -------
        ak.Array
            Awkward Array, flattened along the given axis.
        """
        return ak.flatten(X, axis=self.axis)

    def fit(self, X, y=None):
        """Does nothing, exists for compatibility."""
        return self

    def partial_fit(self, X, y=None):
        """Does nothing, exists for compatibility."""
        return self
