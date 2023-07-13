from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin

from skword2vec.streams import flatten


class Flattener(TransformerMixin, BaseEstimator):
    """Pipeline component that flattens an iterable along a given axis.

    Parameters
    ----------
    axis: int, default 0
        Axis/level of depth at which the iterable should be flattened.
    """

    def __init__(self, axis: int = 0):
        self.axis = axis

    def transform(self, X) -> Iterable:
        return flatten(X, axis=self.axis)

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self
