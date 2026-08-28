#!/usr/bin/env python3
"""Entry point: analyse electroacoustic audio files.

    ./analyse.py piece.wav --segmentation onset -j 8

Equivalent to ``python -m eaa``.  See ``./analyse.py --help``.
"""

import sys

from eaa.cli import main

if __name__ == "__main__":
    sys.exit(main())
