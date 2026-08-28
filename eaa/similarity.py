"""Which segments resemble each other.

Everything here runs on a standardised feature matrix built from the analysis
CSV: rows are segments, columns are descriptors.  Standardising matters -- a
centroid in Hz and a flatness in 0..1 would otherwise contribute in the ratio
of their units rather than of their information.

Three products, in increasing order of interpretation:

* a **self-similarity matrix**, the raw pairwise picture;
* **clusters**, either a cut through a hierarchy or k-means;
* a **profile per cluster** -- the descriptors on which it departs most from
  the piece as a whole, which is what turns "cluster 3" into something you can
  name.

Built on SciPy, which the analysis side already depends on; scikit-learn is
not required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from .table import feature_matrix

log = logging.getLogger(__name__)

METRICS = ("euclidean", "cosine", "correlation", "cityblock")
LINKAGES = ("ward", "average", "complete", "single")
METHODS = ("hierarchical", "kmeans")


@dataclass
class SimilarityResult:
    features: List[str]
    matrix: np.ndarray            # (n_segments, n_features), standardised
    distances: np.ndarray         # (n, n) square form
    metric: str
    method: str
    labels: np.ndarray            # cluster id per segment, 1-based
    k: int
    silhouette: float
    silhouettes: np.ndarray       # per segment
    linkage: Optional[np.ndarray] = None
    coords: Optional[np.ndarray] = None      # PCA projection (n, 2)
    explained: Optional[np.ndarray] = None   # variance ratio of those axes
    scores: Dict[int, float] = field(default_factory=dict)  # silhouette by k

    @property
    def similarity(self) -> np.ndarray:
        """Distances mapped to 0..1, where 1 is identical."""
        largest = float(self.distances.max())
        if largest <= 0:
            return np.ones_like(self.distances)
        return 1.0 - self.distances / largest


# --------------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------------- #
def distances(
    df: pd.DataFrame, features: Sequence[str], metric: str = "euclidean"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Standardise the chosen columns and compute the pairwise distances."""
    matrix, used = feature_matrix(df, features, standardize=True)
    if metric not in METRICS:
        raise ValueError(f"unknown metric '{metric}'; choose from {list(METRICS)}")
    if len(matrix) < 2:
        raise ValueError("need at least two segments to compare")
    condensed = pdist(matrix, metric=metric)
    if not np.isfinite(condensed).all():
        raise ValueError(
            f"the '{metric}' metric produced undefined distances "
            "(a constant segment?); try --metric euclidean"
        )
    return matrix, squareform(condensed), used


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def kmeans(
    matrix: np.ndarray, k: int, seed: int = 0, iterations: int = 100
) -> np.ndarray:
    """k-means with k-means++ seeding.  Returns 1-based labels."""
    rng = np.random.default_rng(seed)
    n = len(matrix)
    k = max(1, min(k, n))

    # k-means++: each new centre is drawn with probability proportional to the
    # squared distance from the nearest centre already chosen.
    centres = [matrix[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(
            [np.sum((matrix - c) ** 2, axis=1) for c in centres], axis=0
        )
        total = d2.sum()
        if total <= 0:
            centres.append(matrix[rng.integers(n)])
            continue
        centres.append(matrix[rng.choice(n, p=d2 / total)])
    centroids = np.asarray(centres, dtype=float)

    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        assigned = np.argmin(
            ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1
        )
        if np.array_equal(assigned, labels):
            break
        labels = assigned
        for j in range(k):
            members = matrix[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
    return labels + 1


def silhouette(dist: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-segment silhouette from a precomputed distance matrix.

    s = (b - a) / max(a, b), where a is the mean distance to the segment's own
    cluster and b the mean distance to the nearest other cluster.  Singleton
    clusters score 0 by convention.
    """
    unique = np.unique(labels)
    scores = np.zeros(len(labels), dtype=float)
    if len(unique) < 2:
        return scores

    for i in range(len(labels)):
        own = labels == labels[i]
        own_count = own.sum() - 1
        if own_count <= 0:
            continue
        a = dist[i, own].sum() / own_count
        b = min(
            dist[i, labels == other].mean()
            for other in unique
            if other != labels[i]
        )
        if max(a, b) > 0:
            scores[i] = (b - a) / max(a, b)
    return scores


def _label(
    matrix: np.ndarray,
    dist: np.ndarray,
    k: int,
    method: str,
    linkage_matrix: Optional[np.ndarray],
    seed: int,
) -> np.ndarray:
    if method == "kmeans":
        return kmeans(matrix, k, seed=seed)
    return fcluster(linkage_matrix, t=k, criterion="maxclust")


def cluster(
    matrix: np.ndarray,
    dist: np.ndarray,
    method: str = "hierarchical",
    linkage_method: str = "ward",
    metric: str = "euclidean",
    k: Optional[int] = None,
    k_range: Tuple[int, int] = (2, 10),
    seed: int = 0,
) -> Tuple[np.ndarray, int, np.ndarray, Optional[np.ndarray], Dict[int, float]]:
    """Cluster the segments, choosing k by silhouette when it is not given.

    Returns ``(labels, k, per-segment silhouettes, linkage, scores by k)``.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}'; choose from {list(METHODS)}")

    linkage_matrix = None
    if method == "hierarchical":
        if linkage_method == "ward" and metric != "euclidean":
            log.warning(
                "ward linkage is only defined for euclidean distances; "
                "using average linkage with the '%s' metric instead", metric
            )
            linkage_method = "average"
        # Ward is fed the observations; the rest take the condensed distances.
        if linkage_method == "ward":
            linkage_matrix = linkage(matrix, method="ward")
        else:
            linkage_matrix = linkage(squareform(dist, checks=False),
                                     method=linkage_method)

    scores: Dict[int, float] = {}
    if k is None:
        low = max(2, k_range[0])
        high = min(k_range[1], len(matrix) - 1)
        if high < low:
            k, labels = 1, np.ones(len(matrix), dtype=int)
            return labels, 1, np.zeros(len(matrix)), linkage_matrix, scores
        for candidate in range(low, high + 1):
            trial = _label(matrix, dist, candidate, method, linkage_matrix, seed)
            scores[candidate] = float(silhouette(dist, trial).mean())
        k = max(scores, key=scores.get)
        log.info(
            "chose k=%d by silhouette (%s)", k,
            ", ".join(f"k{c}={s:.3f}" for c, s in sorted(scores.items())),
        )

    labels = _label(matrix, dist, k, method, linkage_matrix, seed)
    return labels, int(k), silhouette(dist, labels), linkage_matrix, scores


# --------------------------------------------------------------------------- #
# Projection and reporting
# --------------------------------------------------------------------------- #
def pca(matrix: np.ndarray, components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Project onto the leading principal components (plain SVD)."""
    centred = matrix - matrix.mean(axis=0)
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    components = min(components, vt.shape[0])
    coords = centred @ vt[:components].T
    variance = s ** 2
    explained = variance[:components] / variance.sum() if variance.sum() else variance
    return coords, explained


def neighbours(
    dist: np.ndarray, count: int = 3
) -> List[List[Tuple[int, float]]]:
    """For each segment, the ``count`` closest others, nearest first."""
    out: List[List[Tuple[int, float]]] = []
    for i, row in enumerate(dist):
        order = np.argsort(row)
        picked = [(int(j), float(row[j])) for j in order if j != i][:count]
        out.append(picked)
    return out


def profiles(
    result: SimilarityResult, top: int = 6
) -> Dict[int, List[Tuple[str, float]]]:
    """What each cluster *is*, in descriptor terms.

    The matrix is already standardised over the whole piece, so a cluster's
    mean on a feature is directly "how many standard deviations this group sits
    from the piece average".  Reporting the largest of those turns an anonymous
    cluster id into something nameable: "flat spectrum, low centroid, rough".
    """
    out: Dict[int, List[Tuple[str, float]]] = {}
    for label in np.unique(result.labels):
        members = result.matrix[result.labels == label]
        means = members.mean(axis=0)
        order = np.argsort(-np.abs(means))[:top]
        out[int(label)] = [
            (result.features[i], float(means[i])) for i in order
        ]
    return out


def analyse(
    df: pd.DataFrame,
    features: Sequence[str],
    metric: str = "euclidean",
    method: str = "hierarchical",
    linkage_method: str = "ward",
    k: Optional[int] = None,
    k_range: Tuple[int, int] = (2, 10),
    seed: int = 0,
) -> SimilarityResult:
    """Run the whole comparison and return everything the outputs need."""
    matrix, dist, used = distances(df, features, metric=metric)
    labels, k, sils, linkage_matrix, scores = cluster(
        matrix, dist, method=method, linkage_method=linkage_method,
        metric=metric, k=k, k_range=k_range, seed=seed,
    )
    coords, explained = pca(matrix, 2)
    return SimilarityResult(
        features=used, matrix=matrix, distances=dist, metric=metric,
        method=method, labels=labels, k=k,
        silhouette=float(sils.mean()), silhouettes=sils,
        linkage=linkage_matrix, coords=coords, explained=explained, scores=scores,
    )
