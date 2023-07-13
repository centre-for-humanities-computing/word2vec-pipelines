from typing import Union

import awkward as ak
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


class Flattener(TransformerMixin, BaseEstimator):
    """Pipeline component that flattens an array along a given axis.
    Note that this component does not accept iterables, only Awkward Arrays
    and ArrayLike objects.

    Parameters
    ----------
    axis: int, default 1
        Axis/level of depth at which the iterable should be flattened.
    """

    def __init__(self, axis: int = 1):
        self.axis = axis

    def transform(self, X: Union[ArrayLike, ak.Array]) -> ak.Array:
        """Flattens the given iterable along the axis of the Flattener.

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
