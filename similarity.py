#!/usr/bin/env python3
"""Find out which segments of a piece resemble each other.

    ./similarity.py analysis/piece.segments.csv
    ./similarity.py analysis/piece.segments.csv --k 5 --metric cosine
    ./similarity.py analysis/piece.segments.csv -f 'mfcc.*' --method kmeans

Reads the CSV written by ``analyse.py``, standardises a chosen set of
descriptors, and reports the piece's internal resemblances three ways: a
self-similarity matrix, a clustering (with the number of clusters chosen by
silhouette unless you fix it), and a profile of what each cluster actually is
in descriptor terms.

The written label track is the useful one to keep: load it beside the audio and
the piece's form is annotated by cluster.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import sys
from typing import List, Optional

import numpy as np

from eaa import similarity as sim
from eaa import table
from eaa.table import DEFAULT_FEATURES

log = logging.getLogger("similarity")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="similarity.py",
        description="Cluster the segments of an analysed piece by resemblance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "notes\n"
            "  Features default to timbral descriptors and deliberately exclude\n"
            "  level and duration, so the same gesture played loudly and quietly\n"
            "  still counts as a resemblance. Pass -f to choose your own.\n"
            "\n"
            "  Without --k the number of clusters is the one with the best mean\n"
            "  silhouette in --k-range. A silhouette near 0 means the clusters\n"
            "  are not really separated -- read it before trusting the labels.\n"
        ),
    )
    parser.add_argument("csv", nargs="?", help="a *.segments.csv from analyse.py")
    parser.add_argument("-f", "--features", nargs="+", metavar="PATTERN",
                        help="descriptor patterns to compare on")
    parser.add_argument("-l", "--list", action="store_true",
                        help="list the descriptors in the CSV and exit")
    parser.add_argument("--metric", choices=list(sim.METRICS), default="euclidean")
    parser.add_argument("--method", choices=list(sim.METHODS), default="hierarchical")
    parser.add_argument("--linkage", choices=list(sim.LINKAGES), default="ward")
    parser.add_argument("-k", "--k", type=int, help="number of clusters (default: auto)")
    parser.add_argument("--k-range", nargs=2, type=int, default=[2, 10],
                        metavar=("LOW", "HIGH"), help="search range when k is auto")
    parser.add_argument("--neighbours", type=int, default=3, metavar="N",
                        help="closest segments to report for each segment")
    parser.add_argument("--seed", type=int, default=0, help="k-means seeding")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-matrix", action="store_true",
                        help="also write the distance matrix as .npy")
    parser.add_argument("-o", "--out", dest="directory",
                        help="output directory (default: beside the CSV)")
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser


def _base(csv_path: str, directory: Optional[str]) -> str:
    name = os.path.basename(csv_path)
    for ending in (".segments.csv", ".csv"):
        if name.endswith(ending):
            name = name[: -len(ending)]
            break
    return os.path.join(directory or os.path.dirname(csv_path) or ".", name)


def _write_clusters(result, df, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["segment", "start", "end", "duration", "cluster", "silhouette",
             "pc1", "pc2"]
        )
        for i in range(len(df)):
            writer.writerow([
                int(df["segment"].iloc[i]),
                round(float(df["start"].iloc[i]), 4),
                round(float(df["end"].iloc[i]), 4),
                round(float(df["duration"].iloc[i]), 4),
                int(result.labels[i]),
                round(float(result.silhouettes[i]), 4),
                round(float(result.coords[i, 0]), 4),
                round(float(result.coords[i, 1]), 4),
            ])


def _write_neighbours(result, df, path: str, count: int) -> None:
    closest = sim.neighbours(result.distances, count)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["segment", "start", "rank", "neighbour", "neighbour_start", "distance"]
        )
        for i, picks in enumerate(closest):
            for rank, (j, distance) in enumerate(picks, start=1):
                writer.writerow([
                    i, round(float(df["start"].iloc[i]), 4), rank, j,
                    round(float(df["start"].iloc[j]), 4), round(distance, 5),
                ])


def _write_labels(result, df, path: str) -> None:
    """Audacity label track named by cluster: the form, ready to audition."""
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(len(df)):
            fh.write(
                f"{float(df['start'].iloc[i]):.6f}\t{float(df['end'].iloc[i]):.6f}\t"
                f"c{int(result.labels[i])}\n"
            )


def _report(result, df) -> None:
    durations = df["duration"].to_numpy(dtype=float)
    total = durations.sum()
    print(f"\n{len(df)} segments, {len(result.features)} features, "
          f"metric={result.metric}, method={result.method}")
    if result.scores:
        best = ", ".join(f"k={k}:{v:+.3f}" for k, v in sorted(result.scores.items()))
        print(f"silhouette by k: {best}")
    print(f"chose k={result.k}, mean silhouette {result.silhouette:+.3f}"
          f"{'  (weak separation)' if result.silhouette < 0.25 else ''}")

    print("\ncluster   segments      time   share   silhouette")
    for cluster in sorted(set(int(v) for v in result.labels)):
        mask = result.labels == cluster
        seconds = float(durations[mask].sum())
        print(f"  c{cluster:<6d} {int(mask.sum()):>8d} {seconds:>9.1f}s "
              f"{100 * seconds / total:>6.1f}% {result.silhouettes[mask].mean():>+11.3f}")

    print("\nwhat each cluster is (standard deviations from the piece average):")
    for cluster, features in sim.profiles(result).items():
        bullets = ", ".join(f"{name} {value:+.2f}" for name, value in features)
        print(f"  c{cluster}: {bullets}")


def main(argv: Optional[List[str]] = None) -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.csv:
        parser.error("no CSV given")

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-7s %(message)s", stream=sys.stderr,
    )

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

    features = table.select(df, args.features, DEFAULT_FEATURES)
    if not features:
        parser.error("no descriptors matched; try --list to see what is available")

    try:
        result = sim.analyse(
            df, features, metric=args.metric, method=args.method,
            linkage_method=args.linkage, k=args.k,
            k_range=(args.k_range[0], args.k_range[1]), seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    base = _base(args.csv, args.directory)
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    written = []

    _write_clusters(result, df, base + ".clusters.csv")
    written.append(base + ".clusters.csv")
    _write_neighbours(result, df, base + ".neighbours.csv", args.neighbours)
    written.append(base + ".neighbours.csv")
    _write_labels(result, df, base + ".clusters.labels.txt")
    written.append(base + ".clusters.labels.txt")
    if args.save_matrix:
        np.save(base + ".distances.npy", result.distances)
        written.append(base + ".distances.npy")

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        from eaa import viz_similarity as viz

        title = os.path.basename(args.csv)
        written.append(viz.plot_ssm(result, df, base + ".ssm.png",
                                    theme=args.theme, title=f"{title} · self-similarity"))
        written.append(viz.plot_timeline(result, df, base + ".timeline.png",
                                         theme=args.theme, title=f"{title} · form by cluster"))
        written.append(viz.plot_clusters(result, df, base + ".clusters.png",
                                         theme=args.theme, title=f"{title} · cluster gallery"))
        if result.linkage is not None:
            written.append(viz.plot_dendrogram(result, base + ".dendrogram.png",
                                               theme=args.theme,
                                               title=f"{title} · segment hierarchy"))

    if not args.quiet:
        _report(result, df)
        print()
        for path in written:
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
