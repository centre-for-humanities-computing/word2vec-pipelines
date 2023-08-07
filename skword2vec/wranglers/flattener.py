from typing import Iterable

import awkward as ak
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


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

    def transform(self, X: ArrayLike) -> ak.Array:
        """Flattens the given array along the axis.

        Parameters
        ----------
        X: ArrayLike
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
