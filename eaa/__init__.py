"""Descriptor extraction for long electroacoustic, texture-based pieces.

Essentia supplies the signal descriptors, MoSQITo the psychoacoustic ones, and
the audio can be sliced either on a constant grid or by onset detection (or by
timbral change, or by your own markers).

    from eaa import AnalysisConfig, analyse
    cfg = AnalysisConfig()
    cfg.segmentation.method = "onset"
    result = analyse("piece.wav", cfg)
"""

__version__ = "0.1.0"

__all__ = ["AnalysisConfig", "analyse", "AnalysisResult", "__version__"]


def __getattr__(name):
    # Lazy so that `import eaa` does not drag in Essentia/MoSQITo.
    if name == "AnalysisConfig":
        from .config import AnalysisConfig

        return AnalysisConfig
    if name in ("analyse", "AnalysisResult"):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
