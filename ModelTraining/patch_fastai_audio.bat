@echo off
REM Copyright (c) PODS-AI contributors
REM SPDX-License-Identifier: MIT
REM
REM Patch fastai_audio for Python 3.11+ compatibility
REM This script applies a fix for dataclass field initialization in the audio package

echo Patching fastai_audio for Python 3.11+ compatibility...

REM Find the site-packages directory
for /f "delims=" %%i in ('python -c "import site; print(site.getsitepackages()[0])"') do set SITE_PACKAGES=%%i

echo Site-packages directory: %SITE_PACKAGES%

REM Check if audio package exists
if not exist "%SITE_PACKAGES%\audio" (
    echo Error: audio package not found in %SITE_PACKAGES%
    exit /b 1
)

REM Apply the patch to audio/data.py
set AUDIO_DATA=%SITE_PACKAGES%\audio\data.py

if not exist "%AUDIO_DATA%" (
    echo Error: %AUDIO_DATA% not found
    exit /b 1
)

echo Patching %AUDIO_DATA%...

REM Create a temporary Python script to do the replacement
python -c "import sys; content = open('%AUDIO_DATA%', 'r', encoding='utf-8').read(); content = content.replace('config:AudioConfig=AudioConfig()', 'config:AudioConfig=None').replace('def __post_init__(self):', 'def __post_init__(self):\n        if self.config is None:\n            self.config = AudioConfig()'); open('%AUDIO_DATA%', 'w', encoding='utf-8').write(content)"

if %ERRORLEVEL% neq 0 (
    echo Error: Failed to apply patch
    exit /b 1
)

echo Verifying patch was applied...
findstr /C:"config:AudioConfig=None" "%AUDIO_DATA%" >nul
if %ERRORLEVEL% neq 0 (
    echo Warning: Patch verification failed - could not find expected patched code
    exit /b 1
)

echo Patch applied successfully!
exit /b 0
