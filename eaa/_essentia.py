"""One quiet entry point to Essentia.

Essentia's C++ layer writes an ``[ INFO ] MusicExtractorSVM: ...`` banner to
*stdout* the first time its algorithm registry loads, which would end up mixed
into anything the CLI prints.  Silencing it has to happen between importing
``essentia`` and ``essentia.standard``, so every import goes through here.
"""

from __future__ import annotations

from typing import Any, Tuple

_quiet = True


def set_quiet(quiet: bool) -> None:
    """Let Essentia's own INFO logging through again (before the first load)."""
    global _quiet
    _quiet = quiet


def load() -> Tuple[Any, Any]:
    """Return ``(essentia, essentia.standard)``."""
    import essentia

    if _quiet:
        essentia.log.infoActive = False
    import essentia.standard as es

    return essentia, es
