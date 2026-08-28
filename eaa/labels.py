"""Writing segment boundaries out to the editors people actually work in.

The point of a label track is to get the analysis back in front of your ears:
load it beside the audio and you can hear what a segmentation or a clustering
is claiming.  Three formats, because the two editors want different things:

``audacity``
    The plain ``start<TAB>end<TAB>name`` label track.  Also what
    ``analyse.py --segmentation markers`` reads back in, so a segmentation you
    corrected by hand can be fed to the next run.

``reaper``
    A comma-delimited marker/region CSV, the layout REAPER's own
    Region/Marker Manager exports and imports: ``#,Name,Start,End,Length,Color``
    with ``R``/``M`` distinguishing regions from markers, times in seconds and
    colours as ``#RRGGBB``.  Import it with the Region/Marker Manager's
    *Import...* button, or the *Markers/Regions: Import markers/regions from
    file* action.

``reaper-script``
    The same content as a ReaScript.  It goes through REAPER's documented API
    rather than a file format, so it is the one to reach for if a REAPER
    version parses the CSV differently than expected -- and unlike the CSV it
    is guaranteed to carry the colours.  Actions ▸ Load ReaScript, then Run.

Segments become **regions** rather than markers by default: a region has an
extent, which is what a segment is, and it shows up as a block on the ruler
you can loop and rename.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: Format name -> (filename suffix, writer)
_FORMATS: Dict[str, Tuple[str, Callable]] = {}


@dataclass(frozen=True)
class Marker:
    """One labelled stretch of time, in seconds from the start of the file."""

    start: float
    end: float
    name: str
    color: Optional[str] = None  # "#RRGGBB"


def register(name: str, suffix: str):
    def deco(fn):
        _FORMATS[name] = (suffix, fn)
        return fn

    return deco


def available() -> List[str]:
    return list(_FORMATS)


def suffix_for(fmt: str) -> str:
    return _FORMATS[fmt][0]


@register("audacity", ".labels.txt")
def write_audacity(markers: Sequence[Marker], path: str, regions: bool = True) -> str:
    """Audacity label track: start, end, name, tab separated."""
    with open(path, "w", encoding="utf-8") as fh:
        for marker in markers:
            fh.write(f"{marker.start:.6f}\t{marker.end:.6f}\t{marker.name}\n")
    return path


@register("reaper", ".reaper.csv")
def write_reaper_csv(markers: Sequence[Marker], path: str, regions: bool = True) -> str:
    """REAPER marker/region CSV.

    Comma-delimited -- REAPER's import expects commas, and a tab-delimited file
    silently fails to load.  Times are written as plain seconds, which is
    unambiguous regardless of the project's ruler setting.
    """
    kind = "R" if regions else "M"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("#,Name,Start,End,Length,Color\n")
        for i, marker in enumerate(markers, start=1):
            name = _csv_field(marker.name)
            if regions:
                fh.write(
                    f"{kind}{i},{name},{marker.start:.6f},{marker.end:.6f},"
                    f"{marker.end - marker.start:.6f},{marker.color or ''}\n"
                )
            else:
                fh.write(
                    f"{kind}{i},{name},{marker.start:.6f},,0,{marker.color or ''}\n"
                )
    return path


@register("reaper-script", ".reaper.lua")
def write_reaper_lua(markers: Sequence[Marker], path: str, regions: bool = True) -> str:
    """A ReaScript that creates the same regions through REAPER's API."""
    rows = []
    for marker in markers:
        red, green, blue = _rgb(marker.color)
        rows.append(
            "  {%.6f, %.6f, %s, %d, %d, %d},"
            % (marker.start, marker.end, _lua_string(marker.name), red, green, blue)
        )

    script = f'''-- Segment {"regions" if regions else "markers"} from Electroacoustic-analysis.
-- Actions > Load ReaScript..., select this file, then Run.
-- Times are seconds from the start of the source file: if the audio does not
-- start at 00:00 on the timeline, set OFFSET to where it does.

local OFFSET = 0.0            -- seconds to add to every position
local CLEAR_EXISTING = false  -- true to remove the project's current markers/regions first

local SEGMENTS = {{
{chr(10).join(rows)}
}}

reaper.Undo_BeginBlock()
reaper.PreventUIRefresh(1)

if CLEAR_EXISTING then
  for i = reaper.CountProjectMarkers(0) - 1, 0, -1 do
    reaper.DeleteProjectMarkerByIndex(0, i)
  end
end

for i = 1, #SEGMENTS do
  local seg = SEGMENTS[i]
  local colour = reaper.ColorToNative(seg[4], seg[5], seg[6]) | 0x1000000
  reaper.AddProjectMarker2(
    0, {str(bool(regions)).lower()}, seg[1] + OFFSET, seg[2] + OFFSET, seg[3], -1, colour
  )
end

reaper.PreventUIRefresh(-1)
reaper.UpdateArrange()
reaper.Undo_EndBlock("Import {len(markers)} analysis {"regions" if regions else "markers"}", -1)
'''
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(script)
    return path


# --------------------------------------------------------------------------- #
def write(
    markers: Sequence[Marker],
    base: str,
    formats: Sequence[str] = ("audacity",),
    regions: bool = True,
) -> List[str]:
    """Write ``markers`` in each requested format; returns the paths written."""
    unknown = [f for f in formats if f not in _FORMATS]
    if unknown:
        raise ValueError(
            f"unknown label format(s) {unknown}; available: {available()}"
        )
    directory = os.path.dirname(os.path.abspath(base))
    if directory:
        os.makedirs(directory, exist_ok=True)

    written = []
    for fmt in formats:
        suffix, writer = _FORMATS[fmt]
        written.append(writer(markers, base + suffix, regions))
    return written


def from_segments(segments, offset: float = 0.0, names=None, colors=None) -> List[Marker]:
    """Build markers from :class:`eaa.segmentation.Segment` objects."""
    out = []
    for i, segment in enumerate(segments):
        name = (
            names[i] if names is not None
            else (segment.label or f"{segment.index:04d}")
        )
        out.append(
            Marker(
                start=segment.start + offset,
                end=segment.end + offset,
                name=str(name),
                color=colors[i] if colors is not None else None,
            )
        )
    return out


def _csv_field(value: str) -> str:
    """Quote a name only when it needs it, the way REAPER's own export does."""
    if any(ch in value for ch in ',"\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def _lua_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _rgb(color: Optional[str]) -> Tuple[int, int, int]:
    """A ``#RRGGBB`` string as an (r, g, b) triple; mid grey when absent."""
    if not color or not color.startswith("#") or len(color) != 7:
        return (128, 128, 128)
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
