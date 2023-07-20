from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Pooler(TransformerMixin, BaseEstimator):
    """Pipeline component that pools an array along a given axis.

    Parameters
    ----------
    agg: Callable, default np.nanmean
        Function to pool the results with.
    axis: int, default 1
        Axis/level of depth at which the iterable should be flattened.
        If agg does not accept an axis keyword, this parameter is ignored.
    """

    def __init__(self, agg: Callable = np.nanmean, axis: int = 1):
        self.axis = axis
        self.agg = agg

    def transform(self, X):
        """Pools array along given axis with given aggregator function.

        Parameters
        ----------
        X: ArrayLike
            Array to pool.

        Returns
        -------
        ArrayLike
            Pooled array.
        """
        try:
            return self.agg(X, axis=self.axis)
        except TypeError:
            return self.agg(X)

    def fit(self, X, y=None):
        """Does nothing, exists for compatibility."""
        return self

    def partial_fit(self, X, y=None):
        """Does nothing, exists for compatibility."""
        return self
