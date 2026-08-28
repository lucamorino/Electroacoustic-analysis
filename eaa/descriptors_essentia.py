"""Essentia descriptors, computed per segment.

Each segment is analysed at a finer inner resolution (``frame_size`` /
``hop_size``), and the resulting per-frame values are collapsed into scalars by
:class:`essentia.standard.PoolAggregator`.  So a "descriptor" in the output is
always a pair of *what* was measured and *how it was summarised over the
segment* -- ``spectral.centroid.mean`` and ``spectral.centroid.stdev`` say
different and equally interesting things about a texture.

Descriptors are organised in groups (see ``ESSENTIA_GROUPS`` in
:mod:`eaa.config`) so a run can stay cheap while you decide what matters.
Adding one means writing a ``_frame_<group>`` method and listing the group's
name in the config.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ._essentia import load as _load_essentia
from .config import EssentiaConfig

log = logging.getLogger(__name__)


class EssentiaExtractor:
    """Builds its algorithms once, then describes any number of segments.

    Not picklable (Essentia algorithms hold C++ state), so under ``--jobs > 1``
    each worker process constructs its own -- see :mod:`eaa.pipeline`.
    """

    def __init__(self, cfg: EssentiaConfig, sample_rate: int) -> None:
        _essentia, es = _load_essentia()

        self.cfg = cfg
        self.sr = sample_rate
        self.frame_size = int(cfg.frame_size)
        self.hop_size = int(cfg.hop_size)
        self._warned: Set[str] = set()

        spec_size = self.frame_size // 2 + 1
        nyquist = sample_rate / 2.0

        self.window = es.Windowing(type=cfg.window, size=self.frame_size)
        self.spectrum = es.Spectrum(size=self.frame_size)

        # -- dynamics --------------------------------------------------------
        self.rms = es.RMS()
        self.loudness = es.Loudness()
        self.instant_power = es.InstantPower()

        # -- spectral shape --------------------------------------------------
        self.centroid = es.Centroid(range=nyquist)
        self.central_moments = es.CentralMoments(range=nyquist)
        self.distribution_shape = es.DistributionShape()
        self.rolloff = es.RollOff(cutoff=cfg.rolloff_cutoff, sampleRate=sample_rate)
        self.decrease = es.Decrease(range=nyquist)
        self.flatness = es.Flatness()
        self.crest = es.Crest()
        self.entropy = es.Entropy()
        self.hfc = es.HFC(sampleRate=sample_rate)
        self.spectral_complexity = es.SpectralComplexity(sampleRate=sample_rate)
        self.strong_peak = es.StrongPeak()
        self.flux = es.Flux()
        self.energy_band_ratios = [
            (
                f"{int(lo)}_{int(hi)}",
                es.EnergyBandRatio(
                    sampleRate=sample_rate,
                    startFrequency=float(lo),
                    stopFrequency=min(float(hi), nyquist - 1.0),
                ),
            )
            for lo, hi in cfg.energy_bands
        ]

        # -- noisiness / pitchiness ------------------------------------------
        self.zcr = es.ZeroCrossingRate()
        self.pitch_salience = es.PitchSalience(sampleRate=sample_rate)
        self.pitch_yin = es.PitchYinFFT(
            frameSize=self.frame_size, sampleRate=sample_rate
        )
        self.spectral_peaks = es.SpectralPeaks(
            sampleRate=sample_rate,
            magnitudeThreshold=float(cfg.peaks_magnitude_threshold),
            minFrequency=float(cfg.peaks_min_frequency),
            maxFrequency=min(float(cfg.peaks_max_frequency), nyquist - 1.0),
            maxPeaks=int(cfg.peaks_max),
            orderBy="frequency",
        )
        self.dissonance = es.Dissonance()

        # -- cepstral / band groups ------------------------------------------
        self.mfcc = es.MFCC(
            inputSize=spec_size,
            numberBands=int(cfg.mfcc_bands),
            numberCoefficients=int(cfg.mfcc_coefficients),
            sampleRate=sample_rate,
        )
        self.gfcc = es.GFCC(
            inputSize=spec_size,
            numberBands=int(cfg.gfcc_bands),
            numberCoefficients=int(cfg.gfcc_coefficients),
            sampleRate=sample_rate,
        )
        self.bark_bands = es.BarkBands(
            numberBands=int(cfg.bark_bands), sampleRate=sample_rate
        )
        self.mel_bands = es.MelBands(
            inputSize=spec_size, numberBands=int(cfg.mel_bands), sampleRate=sample_rate
        )
        self.erb_bands = es.ERBBands(
            inputSize=spec_size, numberBands=int(cfg.erb_bands), sampleRate=sample_rate
        )

        # -- harmonic / tonal -------------------------------------------------
        self.harmonic_peaks = es.HarmonicPeaks()
        self.inharmonicity = es.Inharmonicity()
        self.tristimulus = es.Tristimulus()
        self.odd_even = es.OddToEvenHarmonicEnergyRatio()
        self.hpcp = es.HPCP(sampleRate=sample_rate)

        self.aggregator = es.PoolAggregator(defaultStats=list(cfg.stats))
        self.dynamic_complexity = es.DynamicComplexity(sampleRate=sample_rate)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def describe(self, x: np.ndarray) -> Dict[str, float]:
        """Describe one segment of mono audio (normalised float32)."""
        essentia, es = _load_essentia()

        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return {}
        if x.size < self.frame_size:
            # Zero-pad so at least one analysis frame exists.  Very short
            # segments are legitimate under onset segmentation.
            x = np.pad(x, (0, self.frame_size - x.size))

        self.flux.reset()
        groups = set(self.cfg.groups)
        pool = essentia.Pool()
        n_frames = 0

        for frame in es.FrameGenerator(
            x, frameSize=self.frame_size, hopSize=self.hop_size, startFromZero=True
        ):
            windowed = self.window(frame)
            spec = self.spectrum(windowed)
            n_frames += 1

            if "dynamics" in groups:
                self._safe("dynamics", self._frame_dynamics, pool, frame, spec)
            if "spectral" in groups:
                self._safe("spectral", self._frame_spectral, pool, frame, spec)
            if "noisiness" in groups:
                self._safe("noisiness", self._frame_noisiness, pool, frame, spec)
            if "mfcc" in groups:
                self._safe("mfcc", self._frame_mfcc, pool, frame, spec)
            if "gfcc" in groups:
                self._safe("gfcc", self._frame_gfcc, pool, frame, spec)
            if "barkbands" in groups:
                self._safe("barkbands", self._frame_barkbands, pool, frame, spec)
            if "melbands" in groups:
                self._safe("melbands", self._frame_melbands, pool, frame, spec)
            if "erbbands" in groups:
                self._safe("erbbands", self._frame_erbbands, pool, frame, spec)
            if "harmonic" in groups:
                self._safe("harmonic", self._frame_harmonic, pool, frame, spec)
            if "tonal" in groups:
                self._safe("tonal", self._frame_tonal, pool, frame, spec)

        out: Dict[str, float] = {"essentia.n_frames": float(n_frames)}
        out.update(self._aggregate(pool))

        # Segment-level descriptors that need the whole signal at once.
        if "dynamics" in groups:
            try:
                complexity, loudness = self.dynamic_complexity(x)
                out["dynamics.dynamic_complexity"] = float(complexity)
                out["dynamics.loudness_dB"] = float(loudness)
            except Exception as exc:  # noqa: BLE001
                self._warn("dynamic_complexity", exc)
        return out

    def global_descriptors(self, audio) -> Dict[str, float]:
        """Whole-file descriptors: EBU R128 loudness, overall dynamics."""
        _essentia, es = _load_essentia()

        out: Dict[str, float] = {}
        try:
            complexity, loudness = self.dynamic_complexity(
                np.asarray(audio.samples, dtype=np.float32)
            )
            out["global.dynamic_complexity"] = float(complexity)
            out["global.loudness_dB"] = float(loudness)
        except Exception as exc:  # noqa: BLE001
            self._warn("global dynamic_complexity", exc)

        # EBU R128 wants a stereo signal at 44.1 or 48 kHz.
        if audio.stereo is not None and self.sr in (44100, 48000):
            try:
                _momentary, short_term, integrated, lra = es.LoudnessEBUR128(
                    sampleRate=self.sr, startAtZero=True
                )(np.asarray(audio.stereo, dtype=np.float32))
                out["global.loudness_integrated_LUFS"] = float(integrated)
                out["global.loudness_range_LU"] = float(lra)
                if len(short_term):
                    out["global.loudness_short_term_max_LUFS"] = float(
                        np.max(short_term)
                    )
            except Exception as exc:  # noqa: BLE001
                self._warn("LoudnessEBUR128", exc)
        return out

    # ------------------------------------------------------------------ #
    # Per-frame groups
    # ------------------------------------------------------------------ #
    def _frame_dynamics(self, pool, frame, spec) -> None:
        pool.add("dynamics.rms", self.rms(frame))
        pool.add("dynamics.loudness", self.loudness(frame))
        power = float(self.instant_power(frame))
        pool.add("dynamics.power_dB", 10.0 * np.log10(power + 1e-12))

    def _frame_spectral(self, pool, frame, spec) -> None:
        pool.add("spectral.centroid", self.centroid(spec))
        spread, skewness, kurtosis = self.distribution_shape(
            self.central_moments(spec)
        )
        pool.add("spectral.spread", spread)
        pool.add("spectral.skewness", skewness)
        pool.add("spectral.kurtosis", kurtosis)
        pool.add("spectral.rolloff", self.rolloff(spec))
        pool.add("spectral.decrease", self.decrease(spec))
        pool.add("spectral.flatness", self.flatness(spec))
        pool.add("spectral.crest", self.crest(spec))
        pool.add("spectral.entropy", self.entropy(spec))
        pool.add("spectral.hfc", self.hfc(spec))
        pool.add("spectral.complexity", self.spectral_complexity(spec))
        pool.add("spectral.strong_peak", self.strong_peak(spec))
        pool.add("spectral.flux", self.flux(spec))
        for name, algo in self.energy_band_ratios:
            pool.add(f"spectral.energy_band_{name}", algo(spec))

    def _frame_noisiness(self, pool, frame, spec) -> None:
        pool.add("noisiness.zcr", self.zcr(frame))
        pool.add("noisiness.pitch_salience", self.pitch_salience(spec))
        pitch, confidence = self.pitch_yin(spec)
        pool.add("noisiness.f0", pitch)
        pool.add("noisiness.f0_confidence", confidence)
        freqs, mags = self.spectral_peaks(spec)
        pool.add("noisiness.n_peaks", float(len(freqs)))
        if len(freqs) > 1:
            pool.add("noisiness.dissonance", self.dissonance(freqs, mags))

    def _frame_mfcc(self, pool, frame, spec) -> None:
        _bands, coeffs = self.mfcc(spec)
        pool.add("mfcc.coeffs", coeffs)

    def _frame_gfcc(self, pool, frame, spec) -> None:
        _bands, coeffs = self.gfcc(spec)
        pool.add("gfcc.coeffs", coeffs)

    def _frame_barkbands(self, pool, frame, spec) -> None:
        bands = self.bark_bands(spec)
        pool.add("barkbands.energies", bands)
        pool.add("barkbands.flatness", self.flatness(bands))
        pool.add("barkbands.crest", self.crest(bands))

    def _frame_melbands(self, pool, frame, spec) -> None:
        bands = self.mel_bands(spec)
        pool.add("melbands.energies", bands)
        pool.add("melbands.flatness", self.flatness(bands))
        pool.add("melbands.crest", self.crest(bands))

    def _frame_erbbands(self, pool, frame, spec) -> None:
        bands = self.erb_bands(spec)
        pool.add("erbbands.energies", bands)
        pool.add("erbbands.flatness", self.flatness(bands))
        pool.add("erbbands.crest", self.crest(bands))

    def _frame_harmonic(self, pool, frame, spec) -> None:
        pitch, confidence = self.pitch_yin(spec)
        if pitch <= 0:
            return
        freqs, mags = self.spectral_peaks(spec)
        if len(freqs) < 2:
            return
        # HarmonicPeaks rejects a 0 Hz partial.
        if freqs[0] <= 0:
            freqs, mags = freqs[1:], mags[1:]
            if len(freqs) < 2:
                return
        hfreqs, hmags = self.harmonic_peaks(freqs, mags, pitch)
        if len(hfreqs) < 2:
            return
        pool.add("harmonic.inharmonicity", self.inharmonicity(hfreqs, hmags))
        t1, t2, t3 = self.tristimulus(hfreqs, hmags)
        pool.add("harmonic.tristimulus_1", t1)
        pool.add("harmonic.tristimulus_2", t2)
        pool.add("harmonic.tristimulus_3", t3)
        pool.add("harmonic.odd_to_even", self.odd_even(hfreqs, hmags))

    def _frame_tonal(self, pool, frame, spec) -> None:
        freqs, mags = self.spectral_peaks(spec)
        if len(freqs) < 1:
            return
        if freqs[0] <= 0:
            freqs, mags = freqs[1:], mags[1:]
            if len(freqs) < 1:
                return
        profile = self.hpcp(freqs, mags)
        pool.add("tonal.hpcp", profile)
        pool.add("tonal.hpcp_entropy", self.entropy(profile))
        pool.add("tonal.hpcp_crest", self.crest(profile))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _aggregate(self, pool) -> Dict[str, float]:
        """Collapse the frame pool into flat ``name.stat`` scalars."""
        try:
            aggregated = self.aggregator(pool)
        except Exception as exc:  # noqa: BLE001 - empty pool, single frame, ...
            self._warn("PoolAggregator", exc)
            return {}

        out: Dict[str, float] = {}
        for name in aggregated.descriptorNames():
            try:
                value = aggregated[name]
            except Exception:  # noqa: BLE001
                continue
            if isinstance(value, (int, float, np.floating, np.integer)):
                out[name] = float(value)
            else:
                arr = np.asarray(value, dtype=np.float64).ravel()
                width = len(str(max(len(arr) - 1, 0)))
                for i, v in enumerate(arr):
                    out[f"{name}.{i:0{width}d}"] = float(v)
        return out

    def _safe(self, group: str, fn, pool, frame, spec) -> None:
        try:
            fn(pool, frame, spec)
        except Exception as exc:  # noqa: BLE001
            self._warn(group, exc)

    def _warn(self, what: str, exc: BaseException) -> None:
        """Warn once per failing descriptor rather than once per frame."""
        if what not in self._warned:
            self._warned.add(what)
            log.warning("descriptor '%s' failed and will be skipped: %s", what, exc)
