# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Load Orca HLS utilities."""

import sys
from pathlib import Path

_ORCA_HLS_SRC = Path(__file__).resolve().parents[1] / "external/orca-hls-utils/src"
if str(_ORCA_HLS_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCA_HLS_SRC))


def get_scraper():
    from orca_hls_utils import scraper

    return scraper


def get_playlist_utils():
    from orca_hls_utils import playlist_utils

    return playlist_utils


def get_s3_utils():
    from orca_hls_utils import s3_utils

    return s3_utils
