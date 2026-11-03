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
if errorlevel 1 (
  set "PYTHON_CMD=py"
  where "%PYTHON_CMD%" >nul 2>nul
  if errorlevel 1 set "PYTHON_CMD=python3"
)

where "%PYTHON_CMD%" >nul 2>nul
if errorlevel 1 (
  echo Error: No Python interpreter found on PATH.
  exit /b 1
)

echo Applying Python 3.11 compatibility patch to fastai_audio...

for /f "delims=" %%i in ('"%PYTHON_CMD%" -c "import os,sys,sysconfig,site; roots=[p for p in sys.path if p and os.path.isdir(p)]; roots += [sysconfig.get_paths().get('purelib'), sysconfig.get_paths().get('platlib')]; roots += getattr(site, 'getsitepackages', lambda: [])(); roots += [getattr(site, 'getusersitepackages', lambda: '')()]; seen=set(); package_names=['audio','fastaudio','fastai_audio']; result='';
for base in roots:
    if not base or base in seen: continue
    seen.add(base)
    for pkg in package_names:
        candidate=os.path.join(base,pkg,'data.py')
        if os.path.isfile(candidate): result=candidate; raise SystemExit
for base in roots:
    if not base: continue
    if 'site-packages' in base or 'dist-packages' in base:
        for root, _, _ in os.walk(base):
            for pkg in package_names:
                candidate=os.path.join(root,pkg,'data.py')
                if os.path.isfile(candidate): result=candidate; raise SystemExit
print(result)"') do set "AUDIO_DATA_FILE=%%i"

if not defined AUDIO_DATA_FILE (
  echo Error: Could not locate the fastai_audio-compatible data.py in any Python site-packages directory.
  echo fastai_audio / fastaudio may not be installed in this environment.
  echo Try: python -m pip install -r requirements.txt
  exit /b 1
)

echo Found patch target at: %AUDIO_DATA_FILE%
copy "%AUDIO_DATA_FILE%" "%AUDIO_DATA_FILE%.bak" >nul

"%PYTHON_CMD%" -c "import pathlib,sys; path=pathlib.Path(sys.argv[1]); text=path.read_text(encoding='utf-8');
if 'from dataclasses import dataclass, asdict, field' in text and 'field(default_factory=SpectrogramConfig)' in text:
    print('Already patched; no changes needed.'); raise SystemExit(0)
original=text
if 'from dataclasses import dataclass, asdict, field' not in text:
    text=text.replace('from dataclasses import dataclass, asdict','from dataclasses import dataclass, asdict, field')
if 'field(default_factory=SpectrogramConfig)' not in text:
    text=text.replace('sg_cfg: SpectrogramConfig = SpectrogramConfig()','sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)')
if text==original: raise SystemExit('No compatible patch targets found in the installed fastai_audio package')
path.write_text(text, encoding='utf-8')" "%AUDIO_DATA_FILE%"

findstr /C:"field(default_factory=SpectrogramConfig)" "%AUDIO_DATA_FILE%" >nul
if errorlevel 1 (
  echo Warning: Patch may not have been applied correctly.
  exit /b 1
)

echo Patch applied successfully!
echo Backup saved to: %AUDIO_DATA_FILE%.bak
echo Patch verification successful: field(default_factory=SpectrogramConfig) found
exit /b 0
