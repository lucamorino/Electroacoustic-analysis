"""Audio loading, level calibration and resampling.

A long piece is loaded once, in full, and every segment is then a cheap view
into that array.  A 60-minute mono file at 44.1 kHz is roughly 640 MB as
float64 / 320 MB as float32, which is what Essentia hands back -- if that is
too much for the machine at hand, analyse the piece in chunks with
``--start`` / ``--duration``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._essentia import load as _load_essentia
from .config import AudioConfig

log = logging.getLogger(__name__)

#: Reference sound pressure, 20 micropascals.
P_REF = 20e-6


@dataclass
class Audio:
    """A loaded piece, plus everything downstream needs to know about it."""

    path: str
    samples: np.ndarray          # mono, float32, normalised (-1..1)
    sample_rate: int
    stereo: Optional[np.ndarray]  # (n, 2) float32 or None
    duration: float
    source_sample_rate: int
    source_channels: int
    offset: float = 0.0           # where in the original file this starts

    def slice(self, start: float, end: float) -> np.ndarray:
        """Return the mono samples between two times (seconds, segment-local)."""
        i0 = max(0, int(round(start * self.sample_rate)))
        i1 = min(len(self.samples), int(round(end * self.sample_rate)))
        if i1 <= i0:
            return np.zeros(0, dtype=np.float32)
        return self.samples[i0:i1]


def load(path: str, cfg: AudioConfig) -> Audio:
    """Load ``path`` according to ``cfg`` using Essentia's loaders."""
    _essentia, es = _load_essentia()

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    # AudioLoader gives us the file's own rate/channel count; we need those for
    # the report, and the stereo signal for EBU R128.
    stereo = None
    src_rate = cfg.sample_rate
    src_channels = 1
    if cfg.keep_stereo:
        try:
            raw, src_rate, src_channels, _md5, _br, _codec = es.AudioLoader(
                filename=path
            )()
            if src_channels >= 2:
                stereo = _resample_stereo(
                    np.asarray(raw, dtype=np.float32), src_rate, cfg.sample_rate
                )
        except Exception as exc:  # noqa: BLE001 - loaders raise plain RuntimeError
            log.warning("could not read %s as stereo (%s); continuing mono", path, exc)

    samples = es.MonoLoader(
        filename=path, sampleRate=cfg.sample_rate, downmix=cfg.downmix
    )()
    samples = np.asarray(samples, dtype=np.float32)

    # Trim to the requested excerpt.
    if cfg.start or cfg.duration is not None:
        i0 = int(round(cfg.start * cfg.sample_rate))
        i1 = len(samples) if cfg.duration is None else i0 + int(
            round(cfg.duration * cfg.sample_rate)
        )
        samples = samples[i0:i1]
        if stereo is not None:
            stereo = stereo[i0:i1]

    if samples.size == 0:
        raise ValueError(f"{path}: no audio in the requested range")

    if cfg.normalize:
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples = (samples / peak).astype(np.float32)
            if stereo is not None:
                stereo = (stereo / peak).astype(np.float32)

    return Audio(
        path=path,
        samples=samples,
        sample_rate=cfg.sample_rate,
        stereo=stereo,
        duration=len(samples) / float(cfg.sample_rate),
        source_sample_rate=int(src_rate),
        source_channels=int(src_channels),
        offset=cfg.start,
    )


def _resample_stereo(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x
    left = resample(x[:, 0], sr_in, sr_out)
    right = resample(x[:, 1], sr_in, sr_out)
    n = min(len(left), len(right))
    return np.stack([left[:n], right[:n]], axis=1).astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample a mono signal.  Uses SciPy when available, Essentia otherwise."""
    if sr_in == sr_out or x.size == 0:
        return np.asarray(x, dtype=np.float32)
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except ImportError:
        _essentia, es = _load_essentia()

        return np.asarray(
            es.Resample(
                inputSampleRate=int(sr_in), outputSampleRate=int(sr_out), quality=1
            )(np.asarray(x, dtype=np.float32)),
            dtype=np.float32,
        )


def to_pascal(x: np.ndarray, spl_full_scale: float) -> np.ndarray:
    """Map normalised samples to pascals for the psychoacoustic models.

    The convention: a full-scale sine (peak 1.0, RMS 1/sqrt(2)) corresponds to
    ``spl_full_scale`` dB SPL.  So the gain is chosen such that

        20 * log10(rms(gain * sine) / 20e-6) == spl_full_scale
    """
    gain = np.sqrt(2.0) * P_REF * (10.0 ** (spl_full_scale / 20.0))
    return np.asarray(x, dtype=np.float64) * gain


def spl(x: np.ndarray, spl_full_scale: float) -> float:
    """Equivalent continuous (unweighted) SPL in dB of a normalised signal.

    Follows the same full-scale convention as :func:`to_pascal`, and reduces to
    ``20*log10(rms * sqrt(2)) + spl_full_scale``.
    """
    if x.size == 0:
        return float("nan")
    rms = float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))
    if rms <= 0:
        return float("-inf")
    return 20.0 * np.log10(rms * np.sqrt(2.0)) + spl_full_scale
