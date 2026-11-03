@echo off
REM Copyright (c) PODS-AI contributors
REM SPDX-License-Identifier: MIT
REM
REM Patch fastai_audio for Python 3.11+ compatibility
REM Based on https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/patch_fastai_audio.sh

echo Applying Python 3.11 compatibility patch to fastai_audio...

set "TEMP_PY=%TEMP%\find_audio_data.py"
> "%TEMP_PY%" (
  echo import os, sys, sysconfig
  echo paths = [p for p in sys.path if p and os.path.isdir(p)] + [sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")]
  echo seen = set()
  echo for p in paths:
  echo     if p and p not in seen:
  echo         seen.add(p)
  echo         c = os.path.join(p, "audio", "data.py")
  echo         if os.path.isfile(c):
  echo             print(c)
  echo             raise SystemExit
  echo for p in paths:
  echo     if "site-packages" in p or "dist-packages" in p:
  echo         for root, dirs, _ in os.walk(p):
  echo             if "audio" in dirs and os.path.isfile(os.path.join(root, "audio", "data.py")):
  echo                 print(os.path.join(root, "audio", "data.py"))
  echo                 raise SystemExit
  echo print("")
)

for /f "delims=" %%i in ('python "%TEMP_PY%"') do set "AUDIO_DATA=%%i"
del "%TEMP_PY%"

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

set "PATCH_PY=%TEMP%\patch_audio_data.py"
> "%PATCH_PY%" (
  echo import pathlib, sys
  echo path = pathlib.Path(sys.argv[1])
  echo text = path.read_text(encoding="utf-8")
  echo original = text
  echo if "from dataclasses import dataclass, asdict, field" not in text:
  echo     text = text.replace(
  echo         "from dataclasses import dataclass, asdict",
  echo         "from dataclasses import dataclass, asdict, field",
  echo     )
  echo if "field(default_factory=SpectrogramConfig)" not in text:
  echo     text = text.replace(
  echo         "sg_cfg: SpectrogramConfig = SpectrogramConfig()",
  echo         "sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)",
  echo     )
  echo if text == original:
  echo     raise SystemExit("No compatible patch targets found in audio/data.py")
  echo path.write_text(text, encoding="utf-8")
)

python "%PATCH_PY%" "%AUDIO_DATA%"
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to apply patch
    del "%PATCH_PY%"
    exit /b 1
)

del "%PATCH_PY%"

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
