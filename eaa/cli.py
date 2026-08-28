"""Command line front end.

Flags override a config file, which overrides the defaults in
:mod:`eaa.config`.  Everything is optional except the audio file(s):

    analyse.py piece.wav
    analyse.py piece.wav -s onset --onset-method complex -j 8
    analyse.py *.wav --config configs/texture.yaml -o analysis/
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import signal
from typing import List, Optional

from . import __version__
from .config import (
    DEFAULT_PSYCHO_METRICS,
    DEFAULT_STATS,
    ESSENTIA_GROUPS,
    PSYCHO_METRICS,
    AnalysisConfig,
)

AUDIO_EXTENSIONS = (".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3", ".m4a")

log = logging.getLogger("eaa")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyse.py",
        description=(
            "Descriptor extraction for long electroacoustic pieces: Essentia "
            "for the signal descriptors, MoSQITo for the psychoacoustic ones, "
            "with a choice of how the audio gets sliced."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "segmentation methods:\n"
            "  fixed    constant-duration chop (--window / --hop)\n"
            "  onset    Essentia onset detection (--onset-method)\n"
            "  sbic     timbral change points (BIC) -- good for sustained textures\n"
            "  markers  boundaries from a label file (--markers)\n"
        ),
    )
    parser.add_argument("inputs", nargs="*", help="audio files, globs, or directories")
    parser.add_argument("--version", action="version", version=f"eaa {__version__}")
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the available segmentation methods and descriptor groups, then exit",
    )
    parser.add_argument("--config", help="YAML or JSON config file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only segment: report the slicing and write the label track, "
        "no descriptors (use it to audition a segmentation before committing "
        "to a long run)",
    )

    audio = parser.add_argument_group("audio")
    audio.add_argument("--sample-rate", type=int, help="analysis rate (default 44100)")
    audio.add_argument("--downmix", choices=["mix", "left", "right"])
    audio.add_argument("--start", type=float, help="skip to this time (seconds)")
    audio.add_argument("--duration", type=float, help="analyse only this much audio")
    audio.add_argument("--normalize", action="store_true", default=None,
                       help="peak-normalise (invalidates the SPL calibration)")

    seg = parser.add_argument_group("segmentation")
    seg.add_argument("-s", "--segmentation", dest="method",
                     choices=["fixed", "onset", "sbic", "markers"])
    seg.add_argument("--window", type=float, help="fixed chop length (seconds)")
    seg.add_argument("--hop", type=float,
                     help="fixed chop hop; < window gives overlapping frames")
    seg.add_argument("--onset-method",
                     choices=["hfc", "complex", "complex_phase", "flux", "melflux",
                              "rms", "superflux"])
    seg.add_argument("--onset-frame-size", type=int)
    seg.add_argument("--onset-hop-size", type=int)
    seg.add_argument("--onset-alpha", type=float,
                     help="onset peak-picking threshold; lower = more onsets")
    seg.add_argument("--onset-silence-threshold", type=float)
    seg.add_argument("--markers", dest="markers_file",
                     help="label file for --segmentation markers")
    seg.add_argument("--min-duration", type=float,
                     help="merge segments shorter than this")
    seg.add_argument("--max-duration", type=float,
                     help="split segments longer than this (0 disables)")
    seg.add_argument("--pad", type=float,
                     help="grow every segment by this much on both sides")

    ess = parser.add_argument_group("Essentia descriptors")
    ess.add_argument("--groups", nargs="+", metavar="GROUP", choices=list(ESSENTIA_GROUPS),
                     help=f"descriptor groups; available: {', '.join(ESSENTIA_GROUPS)}")
    ess.add_argument("--frame-size", type=int, help="inner analysis frame (samples)")
    ess.add_argument("--hop-size", type=int, help="inner analysis hop (samples)")
    ess.add_argument("--stats", nargs="+", metavar="STAT",
                     help=f"aggregation statistics (default: {' '.join(DEFAULT_STATS)})")
    ess.add_argument("--no-essentia", action="store_true", default=None,
                     help="skip the Essentia descriptors entirely")

    psy = parser.add_argument_group("psychoacoustics (MoSQITo)")
    psy.add_argument("--psycho-metrics", nargs="+", metavar="METRIC",
                     choices=list(PSYCHO_METRICS),
                     help=f"default: {' '.join(DEFAULT_PSYCHO_METRICS)}; "
                          f"available: {', '.join(PSYCHO_METRICS)}")
    psy.add_argument("--no-psycho", action="store_true", default=None,
                     help="skip the psychoacoustic metrics (much faster)")
    psy.add_argument("--spl-full-scale", type=float,
                     help="dB SPL that a full-scale sine represents (default 94)")
    psy.add_argument("--field-type", choices=["free", "diffuse"])
    psy.add_argument("--psycho-max-duration", type=float,
                     help="truncate segments longer than this before the slow metrics")

    out = parser.add_argument_group("output")
    out.add_argument("-o", "--out", dest="directory", help="output directory")
    out.add_argument("--basename", help="basename for the written files")
    out.add_argument("--precision", type=int, help="decimals to round to")
    out.add_argument("--no-csv", action="store_true", default=None)
    out.add_argument("--no-json", action="store_true", default=None)
    out.add_argument("--no-labels", action="store_true", default=None)
    out.add_argument("--label-format", nargs="+", metavar="FORMAT",
                     choices=["audacity", "reaper", "reaper-script"],
                     help="editors to write the segmentation for "
                          "(default: audacity reaper)")
    out.add_argument("--label-markers", action="store_true", default=None,
                     help="write REAPER point markers instead of regions")

    run = parser.add_argument_group("run")
    run.add_argument("-j", "--jobs", type=int,
                     help="worker processes for the per-segment analysis")
    run.add_argument("-q", "--quiet", action="store_true")
    run.add_argument("-v", "--verbose", action="store_true", default=None)

    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    """Defaults <- config file <- command line flags."""
    cfg = AnalysisConfig.from_file(args.config) if args.config else AnalysisConfig()

    def apply(section, mapping) -> None:
        for attribute, value in mapping.items():
            if value is not None:
                setattr(section, attribute, value)

    apply(cfg.audio, {
        "sample_rate": args.sample_rate,
        "downmix": args.downmix,
        "start": args.start,
        "duration": args.duration,
        "normalize": args.normalize,
    })
    apply(cfg.segmentation, {
        "method": args.method,
        "window": args.window,
        "hop": args.hop,
        "onset_method": args.onset_method,
        "onset_frame_size": args.onset_frame_size,
        "onset_hop_size": args.onset_hop_size,
        "onset_alpha": args.onset_alpha,
        "onset_silence_threshold": args.onset_silence_threshold,
        "markers_file": args.markers_file,
        "min_duration": args.min_duration,
        "pad": args.pad,
    })
    if args.max_duration is not None:
        cfg.segmentation.max_duration = args.max_duration or None
    if args.markers_file and args.method is None:
        cfg.segmentation.method = "markers"

    apply(cfg.essentia, {
        "groups": args.groups,
        "frame_size": args.frame_size,
        "hop_size": args.hop_size,
        "stats": args.stats,
    })
    if args.no_essentia:
        cfg.essentia.enabled = False

    apply(cfg.psychoacoustics, {
        "metrics": args.psycho_metrics,
        "spl_full_scale": args.spl_full_scale,
        "field_type": args.field_type,
        "max_duration": args.psycho_max_duration,
    })
    if args.no_psycho:
        cfg.psychoacoustics.enabled = False

    apply(cfg.output, {
        "directory": args.directory,
        "basename": args.basename,
        "precision": args.precision,
        "label_formats": args.label_format,
    })
    if args.label_markers:
        cfg.output.label_regions = False
    if args.no_csv:
        cfg.output.csv = False
    if args.no_json:
        cfg.output.json = False
    if args.no_labels:
        cfg.output.labels = False

    if args.jobs is not None:
        cfg.jobs = args.jobs
    cfg.verbose = bool(args.verbose) and not args.quiet
    return cfg


def expand_inputs(inputs: List[str]) -> List[str]:
    """Resolve files, globs and directories into a sorted list of audio files."""
    paths: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                if name.lower().endswith(AUDIO_EXTENSIONS):
                    paths.append(os.path.join(item, name))
        elif any(ch in item for ch in "*?["):
            paths.extend(sorted(glob.glob(item)))
        else:
            paths.append(item)
    seen = set()
    return [p for p in paths if not (p in seen or seen.add(p))]


def _print_listing() -> None:
    from . import descriptors_psycho, segmentation

    print("segmentation methods:")
    for name in segmentation.available():
        print(f"  {name}")
    print("\nEssentia descriptor groups:")
    for name in ESSENTIA_GROUPS:
        print(f"  {name}")
    print("\npsychoacoustic metrics (MoSQITo):")
    for name in descriptors_psycho.available():
        print(f"  {name}")
    print("\naggregation statistics:")
    print("  " + ", ".join(DEFAULT_STATS) + ", dvar2, dmean2, var, cov, icov")
    from .labels import available as label_formats

    print("\nlabel formats:")
    for name in label_formats():
        print(f"  {name}")


def main(argv: Optional[List[str]] = None) -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        _print_listing()
        return 0
    if not args.inputs:
        parser.error("no input files given (use --list to see what is available)")

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-7s %(message)s",
        stream=sys.stderr,
    )

    try:
        cfg = config_from_args(args)
        cfg.validate()
    except (ValueError, FileNotFoundError, SystemExit) as exc:
        parser.error(str(exc))

    paths = expand_inputs(args.inputs)
    if not paths:
        parser.error("none of the given inputs matched an audio file")

    # Imported late so that --help and --list stay instant.
    from . import audio as audio_mod
    from . import export
    from . import segmentation as seg_mod
    from .pipeline import analyse

    failures = 0
    for path in paths:
        try:
            if args.dry_run:
                _dry_run(path, cfg, audio_mod, seg_mod)
                continue
            result = analyse(path, cfg)
            export.write(result, cfg.output)
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop a batch
            failures += 1
            log.error("%s: %s", path, exc)
            if cfg.verbose:
                import traceback

                traceback.print_exc()
    return 1 if failures else 0


def _dry_run(path: str, cfg: AnalysisConfig, audio_mod, seg_mod) -> None:
    import numpy as np

    audio = audio_mod.load(path, cfg.audio)
    segments = seg_mod.segment(audio, cfg.segmentation)
    durations = np.array([s.duration for s in segments]) if segments else np.zeros(1)
    print(f"{path}: {audio.duration:.2f}s -> {len(segments)} segments "
          f"[{cfg.segmentation.method}]")
    print(f"  duration  mean {durations.mean():.3f}s  median "
          f"{np.median(durations):.3f}s  min {durations.min():.3f}s  "
          f"max {durations.max():.3f}s")
    if cfg.output.labels:
        os.makedirs(cfg.output.directory, exist_ok=True)
        base = cfg.output.basename or os.path.splitext(os.path.basename(path))[0]
        for target in seg_mod.write_labels(
            segments, os.path.join(cfg.output.directory, base),
            offset=audio.offset, formats=cfg.output.label_formats,
            regions=cfg.output.label_regions,
        ):
            print(f"  wrote {target}")
