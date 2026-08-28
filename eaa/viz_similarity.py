"""Plots for the segment-comparison results.

Four views, each answering a different question:

* the **self-similarity matrix** -- where does the piece return to itself?
* the **dendrogram** -- what is the nesting of those resemblances?
* the **cluster timeline** -- how is that form laid out in time?
* the **cluster gallery** -- how well separated are the groups really?
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import plotstyle
from .similarity import SimilarityResult
from .viz_mpl import format_time


def _time_edges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Cell edges in seconds, honouring unequal segment durations."""
    return np.append(starts, ends[-1])


def _cluster_colors(result: SimilarityResult, theme: str) -> Dict[int, str]:
    """A colour per cluster, from the fixed categorical order.

    Past the palette's eight slots we do not invent hues: the extra clusters
    share the neutral context colour and are told apart by their printed id.
    """
    palette = plotstyle.palette(theme)["series"]
    ids = sorted(set(int(v) for v in result.labels))
    return {
        cluster: (palette[i] if i < len(palette) else plotstyle.palette(theme)["context"])
        for i, cluster in enumerate(ids)
    }


def _ink_on(hex_color: str, palette: Dict[str, str]) -> str:
    """Pick label ink that contrasts with the block it sits on.

    Some categorical slots (yellow, aqua) are light enough that white text on
    them is unreadable, so the label follows the block's luminance rather than
    the page's.
    """
    def luminance(value: str) -> float:
        r, g, b = (int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))
        channels = [
            c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            for c in (r, g, b)
        ]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    block = luminance(hex_color)
    dark, light = "#0b0b0b", "#ffffff"
    contrast = lambda ink: (
        (max(block, luminance(ink)) + 0.05) / (min(block, luminance(ink)) + 0.05)
    )
    return dark if contrast(dark) >= contrast(light) else light


def plot_ssm(
    result: SimilarityResult,
    df: pd.DataFrame,
    path: str,
    theme: str = "light",
    title: str = "",
) -> str:
    """Self-similarity matrix, with the cluster boundaries drawn over it."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    colors = plotstyle.apply(theme)
    starts = df["start"].to_numpy(dtype=float)
    ends = df["end"].to_numpy(dtype=float)
    edges = _time_edges(starts, ends)

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    mesh = ax.pcolormesh(
        edges, edges, result.similarity,
        cmap=plotstyle.sequential_cmap(theme), shading="flat",
        vmin=0.0, vmax=1.0,
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(False)

    # Where the cluster changes, so the blocks can be read against the form.
    changes = np.flatnonzero(np.diff(result.labels)) + 1
    for index in changes:
        ax.axvline(starts[index], color=colors["surface"], linewidth=1.2, alpha=0.85)
        ax.axhline(starts[index], color=colors["surface"], linewidth=1.2, alpha=0.85)

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(format_time))
    ax.set_xlabel("time (m:ss)")
    ax.set_ylabel("time (m:ss)")
    ax.set_title(title or "self-similarity", loc="left")

    bar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02)
    bar.set_label("similarity  (1 = identical)", color=colors["text_secondary"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=colors["axis"], labelcolor=colors["text_secondary"])

    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_dendrogram(
    result: SimilarityResult, path: str, theme: str = "light", title: str = ""
) -> str:
    """The hierarchy of resemblances, cut at the chosen number of clusters."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

    if result.linkage is None:
        raise ValueError("no linkage available (dendrogram needs --method hierarchical)")

    colors = plotstyle.apply(theme)
    palette = plotstyle.palette(theme)["series"]
    set_link_color_palette(list(palette))

    fig, ax = plt.subplots(figsize=(11, 4.6))
    heights = result.linkage[:, 2]
    threshold = (
        heights[-(result.k - 1)] if 1 < result.k <= len(heights) else 0.0
    )
    dendrogram(
        result.linkage, ax=ax, color_threshold=threshold,
        above_threshold_color=colors["context"], no_labels=len(result.labels) > 60,
        leaf_font_size=7,
    )
    if threshold > 0:
        ax.axhline(threshold, color=colors["muted"], linewidth=1)
        ax.text(
            0.998, threshold, f" cut at k={result.k} ",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color=colors["text_secondary"],
        )
    ax.set_xlabel("segment")
    ax.set_ylabel("linkage distance")
    ax.set_title(title or "segment hierarchy", loc="left")
    # x is a categorical leaf order; a vertical grid over it means nothing.
    ax.grid(False)
    ax.grid(True, axis="y")
    set_link_color_palette(None)

    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_timeline(
    result: SimilarityResult,
    df: pd.DataFrame,
    path: str,
    theme: str = "light",
    title: str = "",
) -> str:
    """The clusters laid out in time -- the piece's form, as the features see it.

    Each block carries its cluster id in text, so identity never rests on
    colour alone (and stays readable when two distant slots end up adjacent).
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    colors = plotstyle.apply(theme)
    cluster_colors = _cluster_colors(result, theme)
    starts = df["start"].to_numpy(dtype=float)
    ends = df["end"].to_numpy(dtype=float)
    span = float(ends[-1] - starts[0])

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 3.4), height_ratios=[1.6, 1], sharex=True
    )

    for i, (start, end) in enumerate(zip(starts, ends)):
        cluster = int(result.labels[i])
        # A 2px surface gap between blocks, rather than a border around them.
        ax.axvspan(start, end, color=cluster_colors[cluster], linewidth=0)
        width = (end - start) / span
        if width > 0.022:
            ax.text(
                0.5 * (start + end), 0.5, str(cluster),
                ha="center", va="center", fontsize=8,
                color=_ink_on(cluster_colors[cluster], colors),
            )
    ax.set_yticks([])
    ax.set_ylabel("cluster")
    ax.grid(False)
    ax.set_title(title or "form by cluster", loc="left")

    # Silhouette underneath: how well each segment sits in its cluster.
    ax2.bar(
        0.5 * (starts + ends), result.silhouettes, width=(ends - starts) * 0.92,
        color=[cluster_colors[int(v)] for v in result.labels], linewidth=0,
    )
    ax2.axhline(0, color=colors["axis"], linewidth=0.8)
    ax2.set_ylabel("silhouette")
    ax2.set_ylim(min(-0.25, float(result.silhouettes.min()) - 0.05), 1.0)
    ax2.grid(False)
    ax2.grid(True, axis="y")
    ax2.xaxis.set_major_formatter(FuncFormatter(format_time))
    ax2.set_xlabel("time (m:ss)")
    ax2.set_xlim(float(starts[0]), float(ends[-1]))

    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_clusters(
    result: SimilarityResult,
    df: pd.DataFrame,
    path: str,
    theme: str = "light",
    title: str = "",
) -> str:
    """Segments in the plane of the two leading principal components.

    With up to three clusters they share one panel and are told apart by
    colour.  Beyond three, arbitrary pairs of colours end up side by side in a
    scatter and no palette ordering keeps every pair separable, so the plot
    facets instead: one panel per cluster, the rest of the piece behind it in
    grey.
    """
    import matplotlib.pyplot as plt

    if result.coords is None:
        raise ValueError("no projection available")

    colors = plotstyle.apply(theme)
    cluster_colors = _cluster_colors(result, theme)
    ids = sorted(set(int(v) for v in result.labels))
    coords = result.coords
    explained = result.explained if result.explained is not None else [0, 0]
    xlabel = f"PC1 ({100 * float(explained[0]):.0f}% of variance)"
    ylabel = f"PC2 ({100 * float(explained[1]):.0f}%)" if len(explained) > 1 else "PC2"

    faceted = plotstyle.facet_instead(len(ids), all_pairs=True)
    if not faceted:
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        for cluster in ids:
            mask = result.labels == cluster
            ax.scatter(
                coords[mask, 0], coords[mask, 1], s=44,
                color=cluster_colors[cluster], label=f"cluster {cluster}",
                edgecolors=colors["surface"], linewidths=1.4, zorder=3,
            )
            ax.annotate(
                str(cluster), coords[mask].mean(axis=0), fontsize=11,
                color=colors["text_primary"], ha="center", va="center", zorder=4,
            )
        ax.legend(loc="best")
        axes = [ax]
    else:
        columns = min(4, len(ids))
        rows = int(np.ceil(len(ids) / columns))
        fig, grid = plt.subplots(
            rows, columns, figsize=(3.1 * columns, 3.0 * rows),
            sharex=True, sharey=True, squeeze=False,
        )
        axes = [a for row in grid for a in row]
        for ax, cluster in zip(axes, ids):
            mask = result.labels == cluster
            ax.scatter(coords[~mask, 0], coords[~mask, 1], s=20,
                       color=colors["context"], linewidths=0, zorder=2)
            ax.scatter(coords[mask, 0], coords[mask, 1], s=34,
                       color=cluster_colors[cluster],
                       edgecolors=colors["surface"], linewidths=1.2, zorder=3)
            count = int(mask.sum())
            ax.set_title(
                f"cluster {cluster}  ({count} segment{'s' if count != 1 else ''})",
                loc="left",
            )
        for ax in axes[len(ids):]:
            ax.set_visible(False)

    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.label_outer()
    fig.suptitle(title or "cluster gallery", x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
