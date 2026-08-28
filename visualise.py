#!/usr/bin/env python3
"""Plot an analysis CSV produced by ``analyse.py``.

    ./visualise.py analysis/piece.segments.csv
    ./visualise.py analysis/piece.segments.csv --html            # interactive
    ./visualise.py analysis/piece.segments.csv -d 'psycho.*.mean' --overlay

Descriptors are chosen with shell-style patterns and shown as small multiples,
one panel per descriptor with its own y-axis.  That is deliberate: a spectral
centroid in Hz and a roughness in asper share no scale, and putting them on one
pair of axes would invent a relationship the data does not contain.  Use
``--overlay`` to compare shapes -- it z-scores every descriptor first, so the
shared axis actually means something.

Both output modes let you show and hide descriptors: the matplotlib window has
checkboxes down the left, and ``--html`` writes a self-contained page with
checkboxes, a shared crosshair and a table view.
"""

from __future__ import annotations

import argparse
import os
import sys
import signal
from typing import List, Optional

from eaa import table
from eaa.table import DEFAULT_VIEW


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualise.py",
        description="Plot descriptor trajectories from an analyse.py CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "descriptor patterns are shell-style globs, e.g.\n"
            "  -d 'spectral.*.mean'            every spectral mean\n"
            "  -d 'psycho.loudness.*'          the loudness family\n"
            "  -d dynamics.rms.mean 'mfcc.*'   mix exact names and globs\n"
        ),
    )
    parser.add_argument("csv", nargs="?", help="a *.segments.csv from analyse.py")
    parser.add_argument("-d", "--descriptors", nargs="+", metavar="PATTERN",
                        help="descriptors to plot (default: a mixed overview)")
    parser.add_argument("-l", "--list", action="store_true",
                        help="list the descriptors in the CSV and exit")
    parser.add_argument("--overlay", action="store_true",
                        help="one axis, z-scored, to compare shapes (max 8)")
    parser.add_argument("--smooth", type=int, default=0, metavar="N",
                        help="rolling median over N segments; tames onset-sliced runs")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--html", nargs="?", const="", metavar="PATH",
                        help="write a self-contained interactive page")
    parser.add_argument("--html-offer", nargs="+", metavar="PATTERN",
                        help="descriptors offered as checkboxes in the HTML page "
                             "(default: everything in the CSV)")
    parser.add_argument("--png", nargs="?", const="", metavar="PATH",
                        help="save a static image")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open a window (implied when saving)")
    parser.add_argument("-o", "--out", dest="directory",
                        help="directory for written files (default: beside the CSV)")
    return parser


def _default_path(csv: str, directory: Optional[str], suffix: str) -> str:
    base = os.path.basename(csv)
    for ending in (".segments.csv", ".csv"):
        if base.endswith(ending):
            base = base[: -len(ending)]
            break
    return os.path.join(directory or os.path.dirname(csv) or ".", base + suffix)


def main(argv: Optional[List[str]] = None) -> int:
    # A long --list piped into `head` should end quietly.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.csv:
        parser.error("no CSV given")

    try:
        df = table.load_table(args.csv)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if args.list:
        for group, names in sorted(table.groups(df).items()):
            print(f"{group}  ({len(names)})")
            for name in names:
                print(f"  {name}")
        return 0

    columns = table.select(df, args.descriptors, DEFAULT_VIEW)
    if not columns:
        parser.error(
            "no descriptors matched. Try --list to see what the CSV contains."
        )

    title = os.path.basename(args.csv)
    written: List[str] = []

    if args.html is not None:
        from eaa.viz_html import write_html

        offered = table.select(df, args.html_offer, table.descriptor_columns(df))
        path = args.html or _default_path(args.csv, args.directory, ".descriptors.html")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        write_html(df, offered, path, title=title, initial=columns, smooth=args.smooth)
        written.append(path)

    want_figure = args.png is not None or args.html is None
    if want_figure:
        import matplotlib

        show = not args.no_show and args.png is None and args.html is None
        if not show:
            matplotlib.use("Agg")
        from eaa.viz_mpl import DescriptorFigure

        try:
            figure = DescriptorFigure(
                df, columns, theme=args.theme, overlay=args.overlay,
                smooth=args.smooth, title=title, interactive=show,
            )
        except ValueError as exc:
            parser.error(str(exc))

        if args.png is not None:
            path = args.png or _default_path(args.csv, args.directory, ".descriptors.png")
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            figure.save(path)
            written.append(path)
        elif show:
            print(f"{len(columns)} descriptors — use the checkboxes to show/hide.")
            figure.show()

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
