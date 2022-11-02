"""
Auxiliary functions for internal use.
"""

from typing import Any, List, Optional, Union, cast

import numpy.typing as npt
import pandas as pd
from scipy import sparse


def hstack_frames(
    frames: List[Union[npt.NDArray[Any], sparse.spmatrix, pd.DataFrame]],
    *,
    prefixes: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    If only data frames are passed, stack them horizontally.

    :param frames: a list of array-likes
    :param prefixes: an optional list of prefixes to use for the columns of each data
        frame in arg ``frames``; must have the same length as arg ``frames``
    :return: the stacked data frame if all elements of ``frames`` are data frames;
        ``None`` otherwise
    """
    if all(isinstance(frame, pd.DataFrame) for frame in frames):
        # all frames are data frames
        frames = cast(List[pd.DataFrame], frames)
        if prefixes is not None:
            assert len(prefixes) == len(
                frames
            ), "number of prefixes must match number of frames"
            frames = [
                frame.add_prefix(f"{prefix}__")
                for frame, prefix in zip(frames, prefixes)
            ]
        return pd.concat(frames, axis=1)
    else:
        return None


def is_sparse_frame(frame: pd.DataFrame) -> bool:
    """
    Check if a data frame contains sparse columns.

    :param frame: the data frame to check
    :return: ``True`` if the data frame contains sparse columns; ``False`` otherwise
    """

    return any(isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes)


def sparse_frame_density(frame: pd.DataFrame) -> float:
    """
    Compute the density of a data frame.

    The density of a data frame is the average density of its columns.
    The density of a sparse column is the ratio of non-sparse points to total (dense)
    data points.
    The density of a dense column is 1.

    :param frame: a data frame
    :return: the density of the data frame
    """

    def _density(sr: pd.Series) -> float:
        if isinstance(sr.dtype, pd.SparseDtype):
            return cast(float, sr.sparse.density)
        else:
            return 1.0

    return sum(_density(sr) for _, sr in frame.items()) / len(frame.columns)
