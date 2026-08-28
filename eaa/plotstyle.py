"""One visual system for every plot this repo draws.

Light and dark are both selected palettes -- the dark values are the same hues
re-stepped for a dark surface, not an automatic inversion.  Categorical slots
are used in fixed order and never cycled: past the eighth series the caller
facets or folds instead.
"""

from __future__ import annotations

from typing import Any, Dict, List

LIGHT: Dict[str, Any] = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    # Single-hue blue ramp, light -> dark, for magnitude (heatmaps).
    "sequential": ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                   "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                   "#184f95", "#104281", "#0d366b"],
    "context": "#c9c8c1",   # de-emphasised marks in an emphasis/facet plot
}

DARK: Dict[str, Any] = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"],
    "sequential": ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
                   "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef",
                   "#9ec5f4", "#b7d3f6", "#cde2fb"],
    "context": "#44443f",
}

#: Categorical series cap.  Forms where any two series can end up side by side
#: (scatter, small multiples read against each other) cap lower -- see
#: ``facet_instead``.
MAX_SERIES = 8
MAX_SERIES_ALL_PAIRS = 3


def palette(theme: str = "light") -> Dict[str, Any]:
    return DARK if theme == "dark" else LIGHT


def facet_instead(n_series: int, all_pairs: bool) -> bool:
    """Whether this many series must be faceted rather than colour-coded.

    ``all_pairs`` is True for forms where arbitrary pairs of series appear
    adjacent (scatter plots, cluster timelines).  Those cap at three, because
    beyond that no ordering of the palette keeps every pair separable under
    colour-vision deficiency.
    """
    return n_series > (MAX_SERIES_ALL_PAIRS if all_pairs else MAX_SERIES)


def apply(theme: str = "light") -> Dict[str, Any]:
    """Set matplotlib's rcParams for the chosen theme; returns the palette."""
    import matplotlib as mpl

    colors = palette(theme)
    mpl.rcParams.update({
        "figure.facecolor": colors["page"],
        "savefig.facecolor": colors["page"],
        "axes.facecolor": colors["surface"],
        "axes.edgecolor": colors["axis"],
        "axes.labelcolor": colors["text_secondary"],
        "axes.titlecolor": colors["text_primary"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": colors["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",          # solid hairlines; dashing reads as a threshold
        "xtick.color": colors["muted"],
        "ytick.color": colors["muted"],
        "xtick.labelcolor": colors["text_secondary"],
        "ytick.labelcolor": colors["text_secondary"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": colors["text_primary"],
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })
    return colors


def sequential_cmap(theme: str = "light"):
    """A one-hue light-to-dark colormap for magnitude."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "eaa_sequential", palette(theme)["sequential"]
    )


def series_colors(n: int, theme: str = "light") -> List[str]:
    """The first ``n`` categorical slots, in fixed order."""
    colors = palette(theme)["series"]
    if n > len(colors):
        raise ValueError(
            f"{n} series exceeds the {len(colors)}-slot categorical palette; "
            "facet or fold the tail instead of generating hues"
        )
    return colors[:n]
