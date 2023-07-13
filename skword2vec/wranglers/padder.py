import warnings
from typing import Any, Optional

import awkward as ak
from sklearn.base import BaseEstimator, TransformerMixin


class Padder(TransformerMixin, BaseEstimator):
    """Pipeline component that pads a ragged array along a given axis.

    Parameters
    ----------
    axis: int, default 1
        Axis along which the array should be padded.
    target: int or None, default None
        Intended dimensionality along the given axis.
        If not specified, maximum dimensionality is learned
        from the data and is used.
    fill_value: Any, default None
        Value to fill empty slots with.
    """

    def __init__(
        self,
        axis: int = 1,
        target: Optional[int] = None,
        fill_value: Any = None,
    ):
        self.axis = axis
        self.target = target
        self.fill_value = fill_value

    def transform(self, X: ak.Array):
        if self.target is None:
            warnings.warn(
                "Max padding length has not been learned during fitting, "
                "Using maximum length of data along the given dimension."
                " Consider calling fit() to save this in the padder's state."
            )
            target = ak.max(ak.num(X, axis=self.axis))
        else:
            target = self.target
        X = ak.pad_none(X, target=target, axis=self.axis)
        if self.fill_value is not None:
            X = ak.fill_none(X, self.fill_value, axis=self.axis)
        return X

    def fit(self, X: ak.Array, y=None):
        self.target = ak.max(ak.num(X, axis=self.axis))
        return self

    def partial_fit(self, X, y=None):
        target = ak.max(ak.num(X, axis=self.axis))
        if self.target is None:
            self.target = target
        else:
            self.target = max(target, self.target)
        return self
