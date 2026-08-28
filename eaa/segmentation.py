"""Turning a piece into the frames that get analysed.

The point of this module is that *how* the audio is sliced is a first-class,
swappable decision.  A constant chop gives you an even grid you can plot
against time; onset-driven slicing gives you one row per gesture, which is
usually what you want when the material is event-based; SBIC gives you
boundaries where the *timbre* changes, which for sustained textures is often
more musically meaningful than either.

Adding a strategy means writing a function that returns a list of
``(start, end)`` pairs and decorating it with ``@register("name")``.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._essentia import load as _load_essentia
from .audio import Audio
from .config import SegmentationConfig

log = logging.getLogger(__name__)

Bounds = List[Tuple[float, float]]
Strategy = Callable[[Audio, SegmentationConfig], Bounds]

_STRATEGIES: Dict[str, Strategy] = {}


def register(name: str) -> Callable[[Strategy], Strategy]:
    def deco(fn: Strategy) -> Strategy:
        _STRATEGIES[name] = fn
        return fn

    return deco


def available() -> List[str]:
    return sorted(_STRATEGIES)


@dataclass
class Segment:
    """One analysis frame, in seconds relative to the start of the excerpt."""

    index: int
    start: float
    end: float
    label: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def mid(self) -> float:
        return 0.5 * (self.start + self.end)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
@register("fixed")
def _fixed(audio: Audio, cfg: SegmentationConfig) -> Bounds:
    """Constant-duration chop, with optional overlap (``hop < window``)."""
    window = float(cfg.window)
    hop = float(cfg.hop) if cfg.hop else window
    total = audio.duration

    bounds: Bounds = []
    start = 0.0
    while start < total:
        end = start + window
        if end > total:
            if cfg.drop_last_partial and bounds:
                break
            end = total
        bounds.append((start, end))
        if end >= total:
            break
        start += hop
    return bounds


@register("onset")
def _onset(audio: Audio, cfg: SegmentationConfig) -> Bounds:
    """Boundaries at detected onsets; each segment runs onset-to-onset."""
    times = detect_onsets(audio, cfg)
    edges: List[float] = []
    if cfg.onset_include_lead or not times:
        edges.append(0.0)
    edges.extend(t for t in times if 0.0 < t < audio.duration)
    edges.append(audio.duration)
    edges = sorted(set(round(t, 6) for t in edges))
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def detect_onsets(audio: Audio, cfg: SegmentationConfig) -> List[float]:
    """Onset times (seconds) using Essentia."""
    essentia, es = _load_essentia()

    x = audio.samples
    sr = audio.sample_rate

    if cfg.onset_method == "superflux":
        times = es.SuperFluxExtractor(
            sampleRate=sr,
            frameSize=max(cfg.onset_frame_size, 2048),
            hopSize=cfg.onset_hop_size,
        )(x)
        return [float(t) for t in times]

    frame_size = int(cfg.onset_frame_size)
    hop_size = int(cfg.onset_hop_size)
    window = es.Windowing(type="hann")
    fft = es.FFT(size=frame_size)
    c2p = es.CartesianToPolar()
    od = es.OnsetDetection(method=cfg.onset_method, sampleRate=sr)

    curve = []
    for frame in es.FrameGenerator(
        x, frameSize=frame_size, hopSize=hop_size, startFromZero=True
    ):
        mag, phase = c2p(fft(window(frame)))
        curve.append(od(mag, phase))

    if not curve:
        return []

    onsets = es.Onsets(
        alpha=float(cfg.onset_alpha),
        delay=int(cfg.onset_delay),
        frameRate=sr / float(hop_size),
        silenceThreshold=float(cfg.onset_silence_threshold),
    )
    times = onsets(
        essentia.array([curve]), essentia.array([1.0])
    )
    return [float(t) for t in times]


@register("sbic")
def _sbic(audio: Audio, cfg: SegmentationConfig) -> Bounds:
    """Timbral change points via the Bayesian Information Criterion.

    Essentia's ``SBic`` looks for the points at which a sequence of feature
    vectors is better described by two Gaussians than by one -- i.e. where the
    spectral character of the sound changes, regardless of whether anything
    that resembles an attack happened.  For sustained/textural writing this is
    frequently the most useful of the three automatic strategies.
    """
    essentia, es = _load_essentia()

    sr = audio.sample_rate
    frame_size = int(cfg.sbic_frame_size)
    hop_size = int(cfg.sbic_hop_size)

    window = es.Windowing(type="hann")
    spectrum = es.Spectrum(size=frame_size)
    mfcc = es.MFCC(
        inputSize=frame_size // 2 + 1,
        numberCoefficients=int(cfg.sbic_number_coefficients),
        sampleRate=sr,
    )

    features = []
    for frame in es.FrameGenerator(
        audio.samples, frameSize=frame_size, hopSize=hop_size, startFromZero=True
    ):
        _bands, coeffs = mfcc(spectrum(window(frame)))
        features.append(coeffs)

    if len(features) < max(cfg.sbic_size1, cfg.sbic_size2):
        log.warning("too few frames for SBic; falling back to a fixed chop")
        return _fixed(audio, cfg)

    # SBic wants (n_features, n_frames).
    matrix = essentia.array(np.asarray(features, dtype=np.float32).T)
    segmentation = es.SBic(
        cpw=float(cfg.sbic_cpw),
        size1=int(cfg.sbic_size1),
        inc1=int(cfg.sbic_inc1),
        size2=int(cfg.sbic_size2),
        inc2=int(cfg.sbic_inc2),
        minLength=int(cfg.sbic_min_length),
    )(matrix)

    edges = sorted({0.0} | {float(f) * hop_size / sr for f in segmentation})
    edges = [t for t in edges if t < audio.duration] + [audio.duration]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


@register("markers")
def _markers(audio: Audio, cfg: SegmentationConfig) -> Bounds:
    """Boundaries from a label file you wrote yourself.

    Accepts Audacity label tracks (``start<TAB>end<TAB>label``), two-column
    ``start,end`` CSVs, and single-column lists of boundary times.  Times are
    absolute in the file, so they are shifted by ``audio.offset`` to become
    excerpt-relative.  Any label text in the third column is ignored for now.
    """
    path = cfg.markers_file
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"markers file not found: {path}")

    starts: List[float] = []
    ends: List[float] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        for row in csv.reader(fh, delimiter=delimiter):
            row = [c.strip() for c in row if c.strip() != ""]
            if not row or row[0].startswith("#"):
                continue
            try:
                start = float(row[0])
            except ValueError:
                continue  # header line
            starts.append(start)
            if len(row) >= 2:
                try:
                    ends.append(float(row[1]))
                except ValueError:
                    ends.append(float("nan"))  # third-column label, not a time
            else:
                ends.append(float("nan"))

    if not starts:
        raise ValueError(f"no usable times found in {path}")

    offset = audio.offset
    if all(np.isnan(e) for e in ends):
        # A plain list of boundary times.
        edges = sorted({0.0} | {s - offset for s in starts if 0 < s - offset < audio.duration})
        edges.append(audio.duration)
        return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    bounds: Bounds = []
    for s, e in zip(starts, ends):
        s -= offset
        e = (s + 0.0) if np.isnan(e) else e - offset
        s = max(0.0, s)
        e = min(audio.duration, e)
        if e > s:
            bounds.append((s, e))
    return sorted(bounds)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def segment(audio: Audio, cfg: SegmentationConfig) -> List[Segment]:
    """Run the configured strategy and apply the shared duration constraints."""
    try:
        strategy = _STRATEGIES[cfg.method]
    except KeyError:
        raise ValueError(
            f"unknown segmentation method '{cfg.method}'; available: {available()}"
        ) from None

    bounds = strategy(audio, cfg)
    bounds = _apply_padding(bounds, cfg.pad, audio.duration)
    bounds = _merge_short(bounds, cfg.min_duration)
    bounds = _split_long(bounds, cfg.max_duration)

    segments = [
        Segment(index=i, start=round(a, 6), end=round(b, 6))
        for i, (a, b) in enumerate(bounds)
    ]
    log.info(
        "%s segmentation -> %d segments (mean %.2fs, min %.2fs, max %.2fs)",
        cfg.method,
        len(segments),
        float(np.mean([s.duration for s in segments])) if segments else 0.0,
        float(np.min([s.duration for s in segments])) if segments else 0.0,
        float(np.max([s.duration for s in segments])) if segments else 0.0,
    )
    return segments


def _apply_padding(bounds: Bounds, pad: float, total: float) -> Bounds:
    if not pad:
        return bounds
    return [(max(0.0, a - pad), min(total, b + pad)) for a, b in bounds]


def _merge_short(bounds: Bounds, min_duration: float) -> Bounds:
    """Fold too-short segments into the neighbour they touch."""
    if min_duration <= 0 or not bounds:
        return bounds
    out: Bounds = []
    for start, end in bounds:
        if out and (end - start) < min_duration:
            prev_start, _prev_end = out[-1]
            out[-1] = (prev_start, end)
        else:
            out.append((start, end))
    # The very first segment can still be too short; merge it forwards.
    if len(out) > 1 and (out[0][1] - out[0][0]) < min_duration:
        out[1] = (out[0][0], out[1][1])
        out.pop(0)
    return out


def _split_long(bounds: Bounds, max_duration: Optional[float]) -> Bounds:
    """Cut over-long segments into equal parts, so no single row hides minutes."""
    if not max_duration:
        return bounds
    out: Bounds = []
    for start, end in bounds:
        duration = end - start
        if duration <= max_duration:
            out.append((start, end))
            continue
        parts = int(np.ceil(duration / max_duration))
        step = duration / parts
        for k in range(parts):
            out.append((start + k * step, start + (k + 1) * step if k < parts - 1 else end))
    return out


def write_labels(
    segments: Sequence[Segment],
    base: str,
    offset: float = 0.0,
    formats: Sequence[str] = ("audacity",),
    regions: bool = True,
) -> List[str]:
    """Write the slicing to label tracks, so it can be inspected by ear.

    ``base`` is a path without a suffix; each format appends its own.
    """
    from .labels import from_segments
    from .labels import write as write_markers

    return write_markers(
        from_segments(segments, offset=offset), base, formats, regions=regions
    )
