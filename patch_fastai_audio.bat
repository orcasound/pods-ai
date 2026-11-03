@echo off
REM Copyright (c) PODS-AI contributors
REM SPDX-License-Identifier: MIT
REM
REM Patch fastai_audio for Python 3.11+ compatibility
REM Based on https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/patch_fastai_audio.sh

echo Applying Python 3.11 compatibility patch to fastai_audio...

REM Resolve the actual audio/data.py path in whichever site-packages directory contains the package.
for /f "delims=" %%i in ('python -c "import os, sys, sysconfig; paths=[p for p in sys.path if p and os.path.isdir(p)] + [sysconfig.get_paths().get('purelib'), sysconfig.get_paths().get('platlib')]; seen=set();
for p in paths:
    if p and p not in seen:
        seen.add(p)
        c=os.path.join(p,'audio','data.py')
        if os.path.isfile(c):
            print(c)
            raise SystemExit
for p in paths:
    if 'site-packages' in p or 'dist-packages' in p:
        for root, dirs, files in os.walk(p):
            if 'audio' in dirs and os.path.isfile(os.path.join(root,'audio','data.py')):
                print(os.path.join(root,'audio','data.py'))
                raise SystemExit
print('')"') do set AUDIO_DATA=%%i

if "%AUDIO_DATA%"=="" (
    echo Error: Could not locate audio/data.py in any Python site-packages directory.
    echo fastai_audio may not be installed in this environment.
    echo Try: python -m pip install -r requirements.txt
    exit /b 1
)

echo Found audio/data.py at: %AUDIO_DATA%

REM Create a backup
copy "%AUDIO_DATA%" "%AUDIO_DATA%.bak" >nul

echo Patching %AUDIO_DATA%...

python -c "import pathlib, sys; p = pathlib.Path(r'%AUDIO_DATA%'); text = p.read_text(encoding='utf-8'); original = text; text = text.replace('from dataclasses import dataclass, asdict', 'from dataclasses import dataclass, asdict, field') if 'from dataclasses import dataclass, asdict, field' not in text else text; text = text.replace('sg_cfg: SpectrogramConfig = SpectrogramConfig()', 'sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)') if 'field(default_factory=SpectrogramConfig)' not in text else text; assert text != original, 'No compatible patch targets found in audio/data.py'; p.write_text(text, encoding='utf-8')"

if %ERRORLEVEL% neq 0 (
    echo Error: Failed to apply patch
    exit /b 1
)

echo Patch applied successfully!
echo Backup saved to: %AUDIO_DATA%.bak

echo Verifying patch was applied...
findstr /C:"field(default_factory=SpectrogramConfig)" "%AUDIO_DATA%" >nul
if %ERRORLEVEL% neq 0 (
    echo Warning: Patch verification failed - could not find expected patched code
    exit /b 1
)

echo Patch verification successful: field(default_factory=SpectrogramConfig) found
exit /b 0
