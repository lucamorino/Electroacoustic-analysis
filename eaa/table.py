"""Loading an analysis CSV back in, and choosing which descriptors to look at.

A run produces several hundred columns, so both ``visualise.py`` and
``similarity.py`` need the same two things: read the table, and resolve a set of
shell-style patterns (``spectral.*.mean``) into an ordered column list.
"""

from __future__ import annotations

import fnmatch
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Columns that locate a segment rather than describe it.
INDEX_COLUMNS = ["segment", "start", "end", "duration"]

#: A small, mixed default view: level, spectral shape, noisiness, perception.
#: Patterns, so it degrades gracefully when a run did not include a group.
DEFAULT_VIEW = [
    "dynamics.rms.mean",
    "spectral.centroid.mean",
    "spectral.flatness.mean",
    "spectral.flux.mean",
    "noisiness.zcr.mean",
    "psycho.loudness.mean",
    "psycho.sharpness.mean",
    "psycho.roughness.mean",
]

#: Feature patterns used for similarity when the user does not say otherwise.
#: Deliberately timbral: level and duration are excluded, because two passages
#: can be the same gesture played quietly and loudly.
DEFAULT_FEATURES = [
    "spectral.*.mean",
    "spectral.*.stdev",
    "noisiness.*.mean",
    "mfcc.coeffs.mean.*",
    "mfcc.coeffs.stdev.*",
    "barkbands.*.mean",
    "psycho.sharpness.mean",
    "psycho.roughness.mean",
    "psycho.loudness.bark_centroid",
]


def load_table(path: str) -> pd.DataFrame:
    """Read a ``*.segments.csv`` produced by ``analyse.py``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in ("segment", "start", "end") if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} does not look like an analyse.py CSV (missing {missing})"
        )
    if "duration" not in df.columns:
        df["duration"] = df["end"] - df["start"]
    return df.sort_values("start").reset_index(drop=True)


def descriptor_columns(df: pd.DataFrame) -> List[str]:
    """Every column that is a descriptor rather than a segment locator."""
    return [c for c in df.columns if c not in INDEX_COLUMNS]


def select(
    df: pd.DataFrame, patterns: Optional[Sequence[str]], default: Sequence[str]
) -> List[str]:
    """Resolve shell-style patterns to columns, keeping the order asked for.

    An exact column name matches itself; anything else is treated as a glob, so
    ``spectral.*.mean`` and ``psycho.*`` both work.  Unmatched patterns are
    skipped silently -- a view can legitimately name descriptors that a
    particular run did not compute.
    """
    available = descriptor_columns(df)
    chosen: List[str] = []
    for pattern in list(patterns) if patterns else list(default):
        if pattern in available:
            matches = [pattern]
        else:
            matches = sorted(fnmatch.filter(available, pattern))
        for column in matches:
            if column not in chosen:
                chosen.append(column)
    return chosen


def groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Descriptor columns bucketed by their leading namespace, for listings."""
    out: Dict[str, List[str]] = {}
    for column in descriptor_columns(df):
        out.setdefault(column.split(".")[0], []).append(column)
    return out


def feature_matrix(
    df: pd.DataFrame,
    columns: Sequence[str],
    standardize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Build a numeric matrix from the chosen columns.

    Columns that are constant or not finite everywhere are dropped: they carry
    no information about how segments differ, and a zero-variance column would
    divide by zero on standardisation.  Standardising is on by default because
    otherwise a descriptor measured in Hz would swamp one measured in 0..1.
    """
    usable: List[str] = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        if np.ptp(values) == 0:
            continue
        usable.append(column)

    if not usable:
        raise ValueError(
            "no usable feature columns: every candidate was constant or had "
            "missing values (try a wider --features pattern)"
        )

    matrix = df[usable].to_numpy(dtype=float)
    if standardize:
        matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    return matrix, usable


def segment_times(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Segment start and end times as arrays."""
    return df["start"].to_numpy(dtype=float), df["end"].to_numpy(dtype=float)


def step_series(
    starts: np.ndarray, ends: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn per-segment values into a step trace.

    A segment's value describes an *extent* of time, not an instant, so a
    straight line between segment midpoints would imply a smoothness the
    analysis never measured.  Steps are the honest mark.
    """
    x = np.empty(len(starts) * 2, dtype=float)
    y = np.empty(len(starts) * 2, dtype=float)
    x[0::2], x[1::2] = starts, ends
    y[0::2], y[1::2] = values, values
    return x, y
