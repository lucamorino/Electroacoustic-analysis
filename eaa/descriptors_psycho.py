"""Psychoacoustic descriptors from MoSQITo.

Where Essentia gives you signal-level descriptions, these are perceptual
models: loudness in sone rather than RMS, sharpness in acum, roughness in
asper.  For texture-based work they tend to track what you actually hear
happening far better than a spectral centroid does.

Two practical points about MoSQITo:

* Its models expect a **calibrated** signal in pascals.  See
  ``PsychoacousticConfig.spl_full_scale`` -- if you do not know the playback
  level, leave the default and read the values comparatively.
* They are specified at **48 kHz**, so segments are resampled first, and they
  are *slow* (roughness especially).  Hence ``min_duration`` / ``max_duration``
  guards and the ``--jobs`` option in the CLI.

Metrics are registered by name; missing ones (MoSQITo's API moves between
versions) degrade to a warning instead of killing the run.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from .audio import resample, spl, to_pascal
from .config import PsychoacousticConfig

log = logging.getLogger(__name__)

MetricFn = Callable[["PsychoacousticExtractor", np.ndarray, int], Dict[str, float]]
_METRICS: Dict[str, MetricFn] = {}


def register(name: str) -> Callable[[MetricFn], MetricFn]:
    def deco(fn: MetricFn) -> MetricFn:
        _METRICS[name] = fn
        return fn

    return deco


def available() -> List[str]:
    return sorted(_METRICS)


def mosqito_available() -> bool:
    try:
        import mosqito  # noqa: F401
    except Exception:  # noqa: BLE001 - it also fails on missing matplotlib
        return False
    return True


class PsychoacousticExtractor:
    """Runs the configured MoSQITo metrics over one segment at a time."""

    def __init__(self, cfg: PsychoacousticConfig, sample_rate: int) -> None:
        self.cfg = cfg
        self.sr = sample_rate
        self._warned: Set[str] = set()
        self._loudness_cache: Optional[tuple] = None

    def describe(self, x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=np.float64)
        duration = x.size / float(self.sr)
        if duration < self.cfg.min_duration:
            # Too short for the models to say anything meaningful.
            return {"psycho.skipped": 1.0}

        if self.cfg.max_duration and duration > self.cfg.max_duration:
            x = x[: int(self.cfg.max_duration * self.sr)]

        # Resample to the rate the models are specified at, then calibrate.
        signal = resample(x.astype(np.float32), self.sr, self.cfg.sample_rate)
        pressure = to_pascal(signal, self.cfg.spl_full_scale)
        fs = int(self.cfg.sample_rate)

        out: Dict[str, float] = {
            "psycho.skipped": 0.0,
            "psycho.spl_dB": spl(signal, self.cfg.spl_full_scale),
            "psycho.analysed_duration": len(signal) / float(fs),
        }

        self._loudness_cache = None  # per-segment; sharpness reuses loudness
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name in self.cfg.metrics:
                fn = _METRICS.get(name)
                if fn is None:
                    self._warn(name, ValueError("metric not implemented"))
                    continue
                try:
                    out.update(fn(self, pressure, fs))
                except Exception as exc:  # noqa: BLE001
                    self._warn(name, exc)
        return out

    # ------------------------------------------------------------------ #
    def _zwicker_loudness(self, pressure: np.ndarray, fs: int):
        """``loudness_zwtv`` result, computed once and shared with sharpness."""
        if self._loudness_cache is None:
            from mosqito.sq_metrics import loudness_zwtv

            self._loudness_cache = loudness_zwtv(
                pressure, fs, field_type=self.cfg.field_type
            )
        return self._loudness_cache

    def percentiles(self, values: np.ndarray, prefix: str) -> Dict[str, float]:
        """Exceedance percentiles: ``N5`` is the value exceeded 5% of the time."""
        out: Dict[str, float] = {}
        values = np.asarray(values, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return out
        for p in self.cfg.percentiles:
            out[f"{prefix}_{p:g}"] = float(np.percentile(values, 100.0 - p))
        return out

    def _warn(self, what: str, exc: BaseException) -> None:
        if what not in self._warned:
            self._warned.add(what)
            log.warning("psychoacoustic metric '%s' failed: %s", what, exc)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@register("loudness")
def _loudness(self: PsychoacousticExtractor, pressure, fs) -> Dict[str, float]:
    """Zwicker time-varying loudness (ISO 532-1), in sone."""
    N, N_spec, bark_axis, _time = self._zwicker_loudness(pressure, fs)
    N = np.asarray(N, dtype=np.float64).ravel()
    out = {
        "psycho.loudness.mean": float(np.mean(N)),
        "psycho.loudness.max": float(np.max(N)),
        "psycho.loudness.std": float(np.std(N)),
    }
    out.update(self.percentiles(N, "psycho.loudness.N"))

    # Where the loudness sits in the spectrum: the centre of gravity of the
    # specific-loudness pattern, in bark.  A compact "perceptual centroid".
    spec = np.asarray(N_spec, dtype=np.float64)
    bark = np.asarray(bark_axis, dtype=np.float64).ravel()
    if spec.ndim == 2 and spec.shape[0] == bark.size:
        mean_pattern = spec.mean(axis=1)
        total = mean_pattern.sum()
        if total > 0:
            centroid = float((mean_pattern * bark).sum() / total)
            out["psycho.loudness.bark_centroid"] = centroid
            spread = float(
                np.sqrt((mean_pattern * (bark - centroid) ** 2).sum() / total)
            )
            out["psycho.loudness.bark_spread"] = spread
    return out


@register("sharpness")
def _sharpness(self: PsychoacousticExtractor, pressure, fs) -> Dict[str, float]:
    """DIN 45692 sharpness, in acum (reuses the loudness computed above)."""
    from mosqito.sq_metrics import sharpness_din_from_loudness

    N, N_spec, _bark, _time = self._zwicker_loudness(pressure, fs)
    S = np.asarray(sharpness_din_from_loudness(N, N_spec), dtype=np.float64).ravel()
    out = {
        "psycho.sharpness.mean": float(np.mean(S)),
        "psycho.sharpness.max": float(np.max(S)),
        "psycho.sharpness.std": float(np.std(S)),
    }
    out.update(self.percentiles(S, "psycho.sharpness.S"))
    return out


@register("roughness")
def _roughness(self: PsychoacousticExtractor, pressure, fs) -> Dict[str, float]:
    """Daniel & Weber roughness, in asper."""
    from mosqito.sq_metrics import roughness_dw

    result = roughness_dw(pressure, fs, overlap=0.5)
    R = np.asarray(result[0], dtype=np.float64).ravel()
    out = {
        "psycho.roughness.mean": float(np.mean(R)),
        "psycho.roughness.max": float(np.max(R)),
        "psycho.roughness.std": float(np.std(R)),
    }
    out.update(self.percentiles(R, "psycho.roughness.R"))
    return out


@register("tonality")
def _tonality(self: PsychoacousticExtractor, pressure, fs) -> Dict[str, float]:
    """ECMA-74 tone-to-noise and prominence ratios, in dB.

    A useful axis for electroacoustic material: how far the sound sits from
    pure noise, measured on emergent tonal components rather than on a
    pitch estimate that noise would make meaningless.
    """
    from mosqito.sq_metrics import pr_ecma_st, tnr_ecma_st

    out: Dict[str, float] = {}
    t_tnr, tnr, _prom, freqs = tnr_ecma_st(pressure, fs, prominence=True)
    out["psycho.tonality.tnr_total_dB"] = float(np.asarray(t_tnr).ravel()[0])
    tnr = np.asarray(tnr, dtype=np.float64).ravel()
    # Always reported, so "no prominent tone" (0) is distinguishable from
    # "the metric did not run" (an absent column).
    out["psycho.tonality.n_tones"] = float(tnr.size)
    if tnr.size:
        out["psycho.tonality.tnr_max_dB"] = float(np.max(tnr))
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    if freqs.size and tnr.size == freqs.size:
        out["psycho.tonality.strongest_tone_Hz"] = float(freqs[int(np.argmax(tnr))])

    t_pr, pr, _prom2, _freqs2 = pr_ecma_st(pressure, fs, prominence=True)
    out["psycho.tonality.pr_total_dB"] = float(np.asarray(t_pr).ravel()[0])
    pr = np.asarray(pr, dtype=np.float64).ravel()
    if pr.size:
        out["psycho.tonality.pr_max_dB"] = float(np.max(pr))
    return out


@register("loudness_ecma")
def _loudness_ecma(self: PsychoacousticExtractor, pressure, fs) -> Dict[str, float]:
    """ECMA-418-2 (Sottek hearing model) loudness.  Slow; off by default."""
    from mosqito.sq_metrics import loudness_ecma

    result = loudness_ecma(pressure, fs)
    values = np.asarray(result[0], dtype=np.float64)
    if values.ndim > 1:
        values = values.sum(axis=0)
    values = values.ravel()
    out = {
        "psycho.loudness_ecma.mean": float(np.mean(values)),
        "psycho.loudness_ecma.max": float(np.max(values)),
    }
    out.update(self.percentiles(values, "psycho.loudness_ecma.N"))
    return out


@register("roughness_ecma")
def _roughness_ecma(self: PsychoacousticExtractor, pressure, fs) -> Dict[str, float]:
    """ECMA-418-2 roughness.  Slow; off by default."""
    from mosqito.sq_metrics import roughness_ecma

    result = roughness_ecma(pressure, fs)
    values = np.asarray(result[0], dtype=np.float64).ravel()
    out = {
        "psycho.roughness_ecma.mean": float(np.mean(values)),
        "psycho.roughness_ecma.max": float(np.max(values)),
    }
    out.update(self.percentiles(values, "psycho.roughness_ecma.R"))
    return out
