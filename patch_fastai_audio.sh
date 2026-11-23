#!/bin/bash
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
# Patch the fastai/fastaudio package for Python 3.11+ compatibility.

set -e

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        echo "Error: No Python interpreter found on PATH."
        exit 1
    fi
fi

"$PYTHON_BIN" "$(dirname "$0")/patch_fastai_audio.py"
