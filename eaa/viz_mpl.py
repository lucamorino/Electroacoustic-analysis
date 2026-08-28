"""Matplotlib rendering of a descriptor table.

Two layouts, and the choice between them is not cosmetic:

**Small multiples** (the default) give every descriptor its own panel and its
own y-axis, sharing only the time axis.  This is the only honest way to show a
spectral centroid in Hz beside a roughness in asper -- putting two units on one
pair of axes invents a correlation that is not in the data.

**Overlay** (``--overlay``) puts everything on one axis, but only after
z-scoring every descriptor to a common base, so the shared axis means
"standard deviations from this descriptor's own mean" and the comparison is
legitimate.  Use it to see which descriptors move together.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from . import plotstyle
from .table import segment_times, step_series


def format_time(seconds: float, _pos=None) -> str:
    """Axis ticks as m:ss -- a 40-minute piece in raw seconds is unreadable."""
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


def _z(values: np.ndarray) -> np.ndarray:
    spread = values.std()
    return (values - values.mean()) / spread if spread > 0 else values * 0.0


def _prepare(df: pd.DataFrame, column: str, smooth: int) -> np.ndarray:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if smooth and smooth > 1:
        values = (
            pd.Series(values)
            .rolling(smooth, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
    return values


class DescriptorFigure:
    """A figure of stacked descriptor panels with show/hide checkboxes."""

    def __init__(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        theme: str = "light",
        overlay: bool = False,
        smooth: int = 0,
        title: str = "",
        interactive: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        self.colors = plotstyle.apply(theme)
        self.df = df
        self.columns = list(columns)
        self.overlay = overlay
        self.interactive = interactive
        self.starts, self.ends = segment_times(df)
        self.values = {c: _prepare(df, c, smooth) for c in self.columns}
        self.visible = {c: True for c in self.columns}

        if overlay and len(self.columns) > plotstyle.MAX_SERIES:
            raise ValueError(
                f"--overlay shows at most {plotstyle.MAX_SERIES} descriptors "
                f"({len(self.columns)} selected); narrow the selection or drop "
                "--overlay for small multiples"
            )

        height = 1.15 * (len(self.columns) if not overlay else 4) + 1.4
        self.fig = plt.figure(figsize=(11.5, min(height, 14)))
        self.fig.suptitle(
            title, x=self.left_margin, ha="left", fontsize=11,
            color=self.colors["text_primary"],
        )

        self.axes: List = []
        if overlay:
            ax = self.fig.add_subplot(111)
            self.axes.append(ax)
            self.lines = {}
            for column, color in zip(
                self.columns, plotstyle.series_colors(len(self.columns), theme)
            ):
                x, y = step_series(self.starts, self.ends, _z(self.values[column]))
                (line,) = ax.plot(x, y, color=color, label=column, linewidth=1.6)
                self.lines[column] = line
            ax.set_ylabel("z-score (per descriptor)")
            ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), ncols=1)
        else:
            for column in self.columns:
                ax = self.fig.add_subplot(len(self.columns), 1, len(self.axes) + 1)
                x, y = step_series(self.starts, self.ends, self.values[column])
                ax.plot(x, y, color=self.colors["series"][0], linewidth=1.6)
                ax.set_title(column, loc="left", pad=3)
                ax.yaxis.set_major_locator(MaxNLocator(3))
                self.axes.append(ax)

        for ax in self.axes:
            ax.set_xlim(float(self.starts[0]), float(self.ends[-1]))
            ax.xaxis.set_major_formatter(FuncFormatter(format_time))
        self.axes[-1].set_xlabel("time (m:ss)")

        self._build_checkboxes()
        self._layout()

    # -- layout ---------------------------------------------------------- #
    left_margin = 0.215
    plot_top = 0.93
    plot_bottom = 0.075

    def _build_checkboxes(self) -> None:
        from matplotlib.widgets import CheckButtons

        self.check = None
        if not self.interactive or len(self.columns) < 2:
            return

        # Transparent and clear of the left margin: an opaque panel would paint
        # over the y-tick labels of the plots beside it.
        panel = self.fig.add_axes(
            [0.008, self.plot_bottom, 0.15, self.plot_top - self.plot_bottom]
        )
        panel.set_facecolor("none")
        for spine in panel.spines.values():
            spine.set_visible(False)
        panel.set_xticks([])
        panel.set_yticks([])

        labels = [self._short(c) for c in self.columns]
        self.check = CheckButtons(panel, labels, [True] * len(self.columns))
        for text in self.check.labels:
            text.set_fontsize(7.5)
            text.set_color(self.colors["text_secondary"])
        self.check.on_clicked(self._on_toggle)

    @staticmethod
    def _short(column: str) -> str:
        """Shorten ``spectral.centroid.mean`` to something a 7pt label fits."""
        parts = column.split(".")
        return ".".join(parts[1:]) if len(parts) > 2 else column

    def _on_toggle(self, label: str) -> None:
        for column in self.columns:
            if self._short(column) == label:
                self.visible[column] = not self.visible[column]
                break
        if self.overlay:
            for column, line in self.lines.items():
                line.set_visible(self.visible[column])
        self._layout()
        self.fig.canvas.draw_idle()

    def _layout(self) -> None:
        """Position the visible panels; hidden ones give their space back."""
        left, right = self.left_margin, 0.985 if not self.overlay else 0.80
        top, bottom = self.plot_top, self.plot_bottom

        if self.overlay:
            self.axes[0].set_position([left, bottom, right - left, top - bottom])
            return

        shown = [ax for ax, c in zip(self.axes, self.columns) if self.visible[c]]
        for ax, column in zip(self.axes, self.columns):
            ax.set_visible(self.visible[column])
        if not shown:
            return

        gap = 0.032
        height = (top - bottom - gap * (len(shown) - 1)) / len(shown)
        for i, ax in enumerate(shown):
            y = top - height - i * (height + gap)
            ax.set_position([left, y, right - left, height])
            is_last = i == len(shown) - 1
            ax.set_xlabel("time (m:ss)" if is_last else "")
            ax.tick_params(labelbottom=is_last)

    # -- output ---------------------------------------------------------- #
    def save(self, path: str) -> None:
        self.fig.savefig(path, facecolor=self.fig.get_facecolor())

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()
