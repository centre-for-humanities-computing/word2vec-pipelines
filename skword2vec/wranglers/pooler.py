from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Pooler(TransformerMixin, BaseEstimator):
    """Pipeline component that pools an array along a given axis.

    Parameters
    ----------
    agg: Callable, default np.nanmean
        Function to pool the results with.
    axis: int, default 0
        Axis/level of depth at which the iterable should be flattened.
        If agg does not accept an axis keyword, this parameter is ignored.
    """

    def __init__(self, agg: Callable = np.nanmean, axis: int = 0):
        self.axis = axis
        self.agg = agg

    def transform(self, X):
        try:
            return self.agg(X, axis=self.axis)
        except TypeError:
            return self.agg(X)

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self
