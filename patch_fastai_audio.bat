@echo off
REM Copyright (c) PODS-AI contributors
REM SPDX-License-Identifier: MIT
REM
REM Patch the fastai/fastaudio package for Python 3.11+ compatibility.

setlocal

set "PYTHON_CMD=%PYTHON_BIN%"
if not defined PYTHON_CMD set "PYTHON_CMD=%PYTHON%"
if not defined PYTHON_CMD set "PYTHON_CMD=python"

where "%PYTHON_CMD%" >nul 2>nul
if errorlevel 1 set "PYTHON_CMD=py"

where "%PYTHON_CMD%" >nul 2>nul
if errorlevel 1 set "PYTHON_CMD=python3"

where "%PYTHON_CMD%" >nul 2>nul
if errorlevel 1 (
  echo Error: No Python interpreter found on PATH.
  exit /b 1
)

"%PYTHON_CMD%" "%~dp0patch_fastai_audio.py"
exit /b %ERRORLEVEL%
