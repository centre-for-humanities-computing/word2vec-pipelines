from typing import Iterable

from sklearn.base import BaseEstimator


def filter_batches(
    chunks: Iterable[list], estimator: BaseEstimator
) -> Iterable[list]:
    for chunk in chunks:
        predictions = estimator.predict(chunk)  # type: ignore
        passes = predictions != -1
        filtered_chunk = [elem for elem, _pass in zip(chunk, passes) if _pass]
        yield filtered_chunk
