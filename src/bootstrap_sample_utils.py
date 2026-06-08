# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Compatibility helpers for sample-processing functions shared with bootstrap code."""

from bootstrap.src.extract_training_samples import (
    REPO_ROOT,
    SEGMENT_DURATION_SECONDS,
    download_60s_audio,
    generate_uri,
    load_manual_corrections,
    process_sample,
)

__all__ = [
    "REPO_ROOT",
    "SEGMENT_DURATION_SECONDS",
    "download_60s_audio",
    "generate_uri",
    "load_manual_corrections",
    "process_sample",
]
