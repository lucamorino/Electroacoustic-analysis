"""The orchestrator: load -> segment -> describe -> collect.

One row per segment, one column per descriptor.  The per-segment work is
independent, so it can be spread over processes with ``jobs > 1`` -- worth it
as soon as the psychoacoustic metrics are on, since those run at roughly a
third of real time per core.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import audio as audio_mod
from . import segmentation as seg_mod
from .config import AnalysisConfig
from .descriptors_essentia import EssentiaExtractor
from .descriptors_psycho import PsychoacousticExtractor, mosqito_available

log = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    path: str
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    segments: List[seg_mod.Segment]
    rows: List[Dict[str, float]]
    summary: Dict[str, float] = field(default_factory=dict)

    @property
    def columns(self) -> List[str]:
        """Every descriptor name present, in a stable order."""
        seen: Dict[str, None] = {}
        for row in self.rows:
            for key in row:
                seen.setdefault(key, None)
        return list(seen)


# --------------------------------------------------------------------------- #
# Worker-process state (one extractor set per process, built after the fork)
# --------------------------------------------------------------------------- #
_WORKER: Dict[str, Any] = {}


def _init_worker(config_dict: Dict[str, Any], sample_rate: int) -> None:
    cfg = AnalysisConfig.from_dict(config_dict)
    _WORKER["cfg"] = cfg
    _WORKER["essentia"] = (
        EssentiaExtractor(cfg.essentia, sample_rate) if cfg.essentia.enabled else None
    )
    _WORKER["psycho"] = (
        PsychoacousticExtractor(cfg.psychoacoustics, sample_rate)
        if cfg.psychoacoustics.enabled
        else None
    )


def _describe_segment(task) -> Dict[str, float]:
    index, samples = task
    row: Dict[str, float] = {}
    essentia_extractor = _WORKER.get("essentia")
    if essentia_extractor is not None:
        row.update(essentia_extractor.describe(samples))
    psycho_extractor = _WORKER.get("psycho")
    if psycho_extractor is not None:
        row.update(psycho_extractor.describe(samples))
    row["segment"] = float(index)
    return row


# --------------------------------------------------------------------------- #
def analyse(path: str, cfg: AnalysisConfig) -> AnalysisResult:
    """Run the whole pipeline over one audio file."""
    cfg.validate()
    started = time.time()

    log.info("loading %s", path)
    audio = audio_mod.load(path, cfg.audio)
    log.info(
        "%.1f s @ %d Hz (source: %d Hz, %d ch)",
        audio.duration,
        audio.sample_rate,
        audio.source_sample_rate,
        audio.source_channels,
    )

    segments = seg_mod.segment(audio, cfg.segmentation)
    if not segments:
        raise ValueError("segmentation produced no segments")

    if cfg.psychoacoustics.enabled and not mosqito_available():
        log.warning(
            "MoSQITo is not importable; continuing without psychoacoustic "
            "descriptors (pip install mosqito)"
        )
        cfg.psychoacoustics.enabled = False

    rows = _run_segments(audio, segments, cfg)

    # Whole-file descriptors, for context on the per-segment numbers.
    metadata: Dict[str, Any] = {
        "path": os.path.abspath(path),
        "duration": audio.duration,
        "offset": audio.offset,
        "sample_rate": audio.sample_rate,
        "source_sample_rate": audio.source_sample_rate,
        "source_channels": audio.source_channels,
        "n_segments": len(segments),
        "segmentation_method": cfg.segmentation.method,
    }
    if cfg.essentia.enabled:
        try:
            extractor = EssentiaExtractor(cfg.essentia, audio.sample_rate)
            metadata["global"] = extractor.global_descriptors(audio)
        except Exception as exc:  # noqa: BLE001
            log.warning("global descriptors failed: %s", exc)

    metadata["elapsed_seconds"] = round(time.time() - started, 2)
    log.info(
        "analysed %d segments in %.1f s (%.1fx real time)",
        len(segments),
        metadata["elapsed_seconds"],
        audio.duration / max(metadata["elapsed_seconds"], 1e-9),
    )

    result = AnalysisResult(
        path=path,
        config=cfg.to_dict(),
        metadata=metadata,
        segments=segments,
        rows=rows,
    )
    result.summary = summarise(result)
    return result


def _run_segments(
    audio: audio_mod.Audio, segments: List[seg_mod.Segment], cfg: AnalysisConfig
) -> List[Dict[str, float]]:
    tasks = [(s.index, audio.slice(s.start, s.end)) for s in segments]
    total = len(tasks)
    rows: List[Optional[Dict[str, float]]] = [None] * total
    started = time.time()

    if cfg.jobs > 1 and total > 1:
        from concurrent.futures import ProcessPoolExecutor

        log.info("describing %d segments on %d processes", total, cfg.jobs)
        with ProcessPoolExecutor(
            max_workers=cfg.jobs,
            initializer=_init_worker,
            initargs=(cfg.to_dict(), audio.sample_rate),
        ) as pool:
            for done, row in enumerate(pool.map(_describe_segment, tasks), start=1):
                rows[int(row["segment"])] = row
                _progress(done, total, started, cfg.verbose)
    else:
        _init_worker(cfg.to_dict(), audio.sample_rate)
        for done, task in enumerate(tasks, start=1):
            row = _describe_segment(task)
            rows[int(row["segment"])] = row
            _progress(done, total, started, cfg.verbose)

    if cfg.verbose:
        sys.stderr.write("\n")

    # Prepend the segment's position in time to every row.
    out: List[Dict[str, float]] = []
    for segment, row in zip(segments, rows):
        record: Dict[str, float] = {
            "segment": float(segment.index),
            "start": segment.start + audio.offset,
            "end": segment.end + audio.offset,
            "duration": segment.duration,
        }
        record.update({k: v for k, v in (row or {}).items() if k != "segment"})
        out.append(record)
    return out


def _progress(done: int, total: int, started: float, verbose: bool) -> None:
    if not verbose:
        return
    elapsed = time.time() - started
    eta = (elapsed / done) * (total - done)
    sys.stderr.write(
        f"\r  segment {done}/{total}  elapsed {elapsed:5.1f}s  eta {eta:5.1f}s "
    )
    sys.stderr.flush()


def summarise(result: AnalysisResult) -> Dict[str, float]:
    """Collapse the per-segment table into one piece-level profile.

    Segment durations vary (wildly, under onset segmentation), so the means are
    duration-weighted -- otherwise a hundred 40 ms clicks would outvote the
    drone they are sitting on.
    """
    if not result.rows:
        return {}

    skip = {"segment", "start", "end", "duration"}
    weights = np.array([max(r.get("duration", 0.0), 1e-9) for r in result.rows])

    summary: Dict[str, float] = {
        "n_segments": float(len(result.rows)),
        "total_duration": float(np.sum(weights)),
        "segment_duration.mean": float(np.mean(weights)),
        "segment_duration.stdev": float(np.std(weights)),
        "segment_duration.min": float(np.min(weights)),
        "segment_duration.max": float(np.max(weights)),
    }

    for column in result.columns:
        if column in skip:
            continue
        values = np.array(
            [float(r.get(column, np.nan)) for r in result.rows], dtype=np.float64
        )
        mask = np.isfinite(values)
        if not mask.any():
            continue
        v, w = values[mask], weights[mask]
        mean = float(np.sum(v * w) / np.sum(w))
        summary[f"{column}.wmean"] = mean
        summary[f"{column}.wstdev"] = float(
            np.sqrt(np.sum(w * (v - mean) ** 2) / np.sum(w))
        )
    return summary
