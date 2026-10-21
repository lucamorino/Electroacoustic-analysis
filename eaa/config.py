"""Configuration objects for the analysis pipeline.

Everything the pipeline does is driven by an :class:`AnalysisConfig`.  It can be
built from CLI flags, from a YAML/JSON file, or assembled by hand when the
package is used as a library:

    from eaa.config import AnalysisConfig
    cfg = AnalysisConfig()
    cfg.segmentation.method = "onset"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
@dataclass
class AudioConfig:
    """How the file is read before anything is measured."""

    sample_rate: int = 44100
    #: 'mix' | 'left' | 'right' -- how a multichannel file becomes the mono
    #: signal that the descriptors are computed on.
    downmix: str = "mix"
    #: Analyse only an excerpt (seconds).  ``duration=None`` means "to the end".
    start: float = 0.0
    duration: Optional[float] = None
    #: Keep the stereo signal in memory as well (needed for EBU R128).
    keep_stereo: bool = True
    #: Peak-normalise the mono signal.  Off by default: for electroacoustic
    #: work the absolute level usually carries meaning, and normalising would
    #: invalidate the psychoacoustic calibration.
    normalize: bool = False


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
@dataclass
class SegmentationConfig:
    """How the piece is cut into the frames that get described.

    ``method`` selects one of the strategies registered in
    :mod:`eaa.segmentation`:

    ``fixed``    constant-duration chop, with optional overlap
    ``onset``    boundaries from an Essentia onset detector
    ``sbic``     timbral change points (Bayesian Information Criterion)
    ``markers``  boundaries read from a label file (e.g. Audacity export)
    """

    method: str = "fixed"

    # -- fixed ---------------------------------------------------------------
    window: float = 2.0
    #: ``None`` -> hop == window (contiguous, non-overlapping segments).
    hop: Optional[float] = None
    drop_last_partial: bool = True

    # -- onset ---------------------------------------------------------------
    #: 'hfc' | 'complex' | 'complex_phase' | 'flux' | 'melflux' | 'rms'
    #: or 'superflux' to use Essentia's SuperFluxExtractor directly.
    onset_method: str = "complex"
    onset_frame_size: int = 1024
    onset_hop_size: int = 512
    #: Onsets() peak-picking parameters.
    onset_alpha: float = 0.1
    onset_delay: int = 5
    onset_silence_threshold: float = 0.02
    #: Include the audio before the first onset as its own segment.
    onset_include_lead: bool = True

    # -- sbic ----------------------------------------------------------------
    sbic_frame_size: int = 2048
    sbic_hop_size: int = 1024
    sbic_number_coefficients: int = 13
    sbic_cpw: float = 1.5
    sbic_size1: int = 300
    sbic_inc1: int = 60
    sbic_size2: int = 200
    sbic_inc2: int = 20
    sbic_min_length: int = 10

    # -- markers -------------------------------------------------------------
    #: Path to an Audacity-style label file, or a CSV with start[,end[,label]].
    markers_file: Optional[str] = None

    # -- constraints applied to every strategy -------------------------------
    #: Segments shorter than this are merged into their neighbour.
    min_duration: float = 0.25
    #: Segments longer than this are split into equal parts.  ``None`` = keep.
    max_duration: Optional[float] = 30.0
    #: Grow each segment by this much on both sides (seconds).  Useful with
    #: onset segmentation when you want the attack context of the next event.
    pad: float = 0.0


# --------------------------------------------------------------------------- #
# Essentia descriptors
# --------------------------------------------------------------------------- #
#: Descriptor groups that :mod:`eaa.descriptors_essentia` knows how to compute.
ESSENTIA_GROUPS = (
    "dynamics",      # rms, loudness, instant power, dynamic complexity
    "spectral",      # centroid, spread, skewness, kurtosis, rolloff, flux, ...
    "noisiness",     # zcr, flatness, dissonance, pitch salience/confidence
    "mfcc",          # mel-frequency cepstral coefficients
    "gfcc",          # gammatone-frequency cepstral coefficients
    "barkbands",     # bark band energies + their flatness/crest
    "melbands",
    "erbbands",
    "harmonic",      # inharmonicity, tristimulus, odd/even ratio
    "tonal",         # HPCP profile and its entropy/crest
)

DEFAULT_ESSENTIA_GROUPS = ["dynamics", "spectral", "noisiness", "mfcc", "barkbands"]

#: Statistics used to collapse the per-frame values of a segment into scalars.
DEFAULT_STATS = ["mean", "stdev", "median", "min", "max", "dmean", "dvar"]


@dataclass
class EssentiaConfig:
    enabled: bool = True
    groups: List[str] = field(default_factory=lambda: list(DEFAULT_ESSENTIA_GROUPS))
    #: Inner analysis frame, i.e. the resolution at which descriptors are
    #: computed *inside* each segment before being aggregated.
    frame_size: int = 2048
    hop_size: int = 1024
    window: str = "hann"
    stats: List[str] = field(default_factory=lambda: list(DEFAULT_STATS))
    #: Keep the raw per-frame values in the JSON output (large!).
    keep_frames: bool = False

    mfcc_bands: int = 40
    mfcc_coefficients: int = 13
    gfcc_bands: int = 40
    gfcc_coefficients: int = 13
    bark_bands: int = 27
    mel_bands: int = 40
    erb_bands: int = 40
    rolloff_cutoff: float = 0.85
    #: Peak picking for dissonance / harmonic descriptors.
    peaks_magnitude_threshold: float = 1e-5
    peaks_max: int = 100
    peaks_min_frequency: float = 20.0
    peaks_max_frequency: float = 8000.0
    #: Energy band ratios reported per segment, as (low, high) Hz pairs.
    energy_bands: List[List[float]] = field(
        default_factory=lambda: [[20, 150], [150, 800], [800, 4000], [4000, 20000]]
    )


# --------------------------------------------------------------------------- #
# Psychoacoustics (MoSQITo)
# --------------------------------------------------------------------------- #
#: Metrics implemented in :mod:`eaa.descriptors_psycho`.
PSYCHO_METRICS = (
    "loudness",       # Zwicker time-varying loudness (sone) + N5/N10
    "sharpness",      # DIN 45692 (acum), derived from the loudness above
    "roughness",      # Daniel & Weber (asper)
    "tonality",       # ECMA-74 tone-to-noise ratio + prominence ratio (dB)
    "loudness_ecma",  # ECMA-418-2 (Sottek) loudness -- slow, off by default
    "roughness_ecma", # ECMA-418-2 roughness -- slow, off by default
)

DEFAULT_PSYCHO_METRICS = ["loudness", "sharpness", "roughness", "tonality"]


@dataclass
class PsychoacousticConfig:
    """MoSQITo settings.

    MoSQITo expects a *calibrated* signal in pascals, so we need to know what
    sound pressure level a full-scale sample corresponds to.  ``spl_full_scale``
    is that reference: with the default of 94 dB, a full-scale sine reads
    94 dB SPL (1 Pa RMS).  Set it to whatever your monitoring chain does if you
    want the numbers to mean something in absolute terms; leave it alone if you
    only care about relative comparisons within the piece.
    """

    enabled: bool = True
    metrics: List[str] = field(default_factory=lambda: list(DEFAULT_PSYCHO_METRICS))
    spl_full_scale: float = 94.0
    #: 'free' or 'diffuse' sound field, as per the Zwicker model.
    field_type: str = "free"
    #: MoSQITo's models are specified at 48 kHz; segments are resampled to this.
    sample_rate: int = 48000
    #: Segments shorter than this are skipped (the models need a few frames).
    min_duration: float = 0.3
    #: Segments longer than this are truncated before the (slow) metrics run.
    #: ``None`` disables the cap.
    max_duration: Optional[float] = 20.0
    #: Percentiles of the time-varying curves to report, e.g. N5 = the value
    #: exceeded during 5% of the segment.
    percentiles: List[float] = field(default_factory=lambda: [5.0, 10.0, 50.0, 90.0])


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
@dataclass
class OutputConfig:
    directory: str = "analysis"
    #: Basename for the generated files; defaults to the audio file's stem.
    basename: Optional[str] = None
    csv: bool = True
    json: bool = True
    #: Write the segment boundaries out to an editor, so the segmentation can
    #: be checked (and edited) by ear.
    labels: bool = True
    #: Which editors: 'audacity', 'reaper' (marker/region CSV), and/or
    #: 'reaper-script' (the same as a ReaScript, which also carries colours).
    #: REAPER output is opt-in -- see --reaper / --reaper-script.
    label_formats: List[str] = field(default_factory=lambda: ["audacity"])
    #: Segments become REAPER regions rather than point markers.
    label_regions: bool = True
    #: Round floats to this many decimals in the outputs (None = full precision).
    precision: Optional[int] = 6


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    essentia: EssentiaConfig = field(default_factory=EssentiaConfig)
    psychoacoustics: PsychoacousticConfig = field(default_factory=PsychoacousticConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    #: Number of worker processes used for the per-segment analysis.
    jobs: int = 1
    verbose: bool = False

    # -- (de)serialisation ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisConfig":
        cfg = cls()
        _update_dataclass(cfg, data)
        return cfg

    @classmethod
    def from_file(cls, path: str) -> "AnalysisConfig":
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as fh:
            if ext in (".yaml", ".yml"):
                try:
                    import yaml  # type: ignore
                except ImportError as exc:  # pragma: no cover
                    raise SystemExit(
                        "Reading a YAML config needs PyYAML (pip install pyyaml), "
                        "or use a .json config instead."
                    ) from exc
                data = yaml.safe_load(fh) or {}
            else:
                data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        return cls.from_dict(data)

    def validate(self) -> None:
        seg = self.segmentation
        if seg.window <= 0:
            raise ValueError("segmentation.window must be > 0")
        if seg.hop is not None and seg.hop <= 0:
            raise ValueError("segmentation.hop must be > 0")
        if seg.max_duration is not None and seg.max_duration < seg.min_duration:
            raise ValueError("segmentation.max_duration < min_duration")
        if seg.method == "markers" and not seg.markers_file:
            raise ValueError("segmentation.method='markers' needs markers_file")

        unknown = set(self.essentia.groups) - set(ESSENTIA_GROUPS)
        if unknown:
            raise ValueError(
                f"unknown Essentia group(s): {sorted(unknown)}; "
                f"available: {list(ESSENTIA_GROUPS)}"
            )
        unknown = set(self.psychoacoustics.metrics) - set(PSYCHO_METRICS)
        if unknown:
            raise ValueError(
                f"unknown psychoacoustic metric(s): {sorted(unknown)}; "
                f"available: {list(PSYCHO_METRICS)}"
            )
        from .labels import available as label_formats

        unknown = set(self.output.label_formats) - set(label_formats())
        if unknown:
            raise ValueError(
                f"unknown label format(s): {sorted(unknown)}; "
                f"available: {label_formats()}"
            )
        if self.jobs < 1:
            raise ValueError("jobs must be >= 1")


def _update_dataclass(obj: Any, data: Dict[str, Any]) -> None:
    """Recursively overwrite dataclass fields from a (possibly partial) dict."""
    valid = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise ValueError(
                f"unknown config key '{key}' (expected one of {sorted(valid)})"
            )
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)
