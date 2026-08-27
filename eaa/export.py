"""Writing results out: a tidy CSV, a JSON sidecar, and a label track."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

from .config import OutputConfig
from .pipeline import AnalysisResult
from .segmentation import write_labels

log = logging.getLogger(__name__)

#: Columns that identify a segment, always written first and in this order.
LEADING = ["segment", "start", "end", "duration"]


def write(result: AnalysisResult, cfg: OutputConfig) -> List[str]:
    """Write everything ``cfg`` asks for; returns the paths written."""
    os.makedirs(cfg.directory, exist_ok=True)
    base = cfg.basename or os.path.splitext(os.path.basename(result.path))[0]
    written: List[str] = []

    if cfg.csv:
        path = os.path.join(cfg.directory, f"{base}.segments.csv")
        write_csv(result, path, cfg.precision)
        written.append(path)

    if cfg.json:
        path = os.path.join(cfg.directory, f"{base}.analysis.json")
        write_json(result, path, cfg.precision)
        written.append(path)

    if cfg.labels:
        path = os.path.join(cfg.directory, f"{base}.labels.txt")
        write_labels(result.segments, path, offset=result.metadata.get("offset", 0.0))
        written.append(path)

    for path in written:
        log.info("wrote %s", path)
    return written


def write_csv(result: AnalysisResult, path: str, precision: Optional[int]) -> None:
    columns = _ordered_columns(result)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="", extrasaction="ignore")
        writer.writeheader()
        for row in result.rows:
            writer.writerow(
                {k: _fmt(row.get(k), precision) for k in columns if k in row}
            )


def write_json(result: AnalysisResult, path: str, precision: Optional[int]) -> None:
    payload: Dict[str, Any] = {
        "metadata": _round(result.metadata, precision),
        "config": result.config,
        "columns": _ordered_columns(result),
        "summary": _round(result.summary, precision),
        "segments": [_round(row, precision) for row in result.rows],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, allow_nan=True)


def _ordered_columns(result: AnalysisResult) -> List[str]:
    columns = result.columns
    leading = [c for c in LEADING if c in columns]
    rest = sorted(c for c in columns if c not in leading)
    return leading + rest


def _fmt(value: Any, precision: Optional[int]) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return round(value, precision) if precision is not None else value
    return value


def _round(obj: Any, precision: Optional[int]) -> Any:
    if precision is None:
        return obj
    if isinstance(obj, dict):
        return {k: _round(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, precision) for v in obj]
    if isinstance(obj, float) and math.isfinite(obj):
        return round(obj, precision)
    return obj
