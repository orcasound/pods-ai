@echo off
REM Copyright (c) PODS-AI contributors
REM SPDX-License-Identifier: MIT
REM
REM Patch the fastai/fastaudio package for Python 3.11+ compatibility.

echo Applying Python 3.11 compatibility patch to fastai_audio...

set "TEMP_FIND=%TEMP%\patch_fastai_audio_find.py"
> "%TEMP_FIND%" (
    echo import os, sys, sysconfig
    echo roots = [p for p in sys.path if p and os.path.isdir(p)] + [sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")]
    echo seen = set()
    echo package_names = ["audio", "fastaudio", "fastai_audio"]
    echo for base in roots:
    echo     if not base or base in seen: continue
    echo     seen.add(base)
    echo     for pkg in package_names:
    echo         candidate = os.path.join(base, pkg, "data.py")
    echo         if os.path.isfile(candidate):
    echo             print(candidate)
    echo             raise SystemExit
    echo for base in roots:
    echo     if not base: continue
    echo     if "site-packages" in base or "dist-packages" in base:
    echo         for root, dirs, _ in os.walk(base):
    echo             for pkg in package_names:
    echo                 candidate = os.path.join(root, pkg, "data.py")
    echo                 if os.path.isfile(candidate):
    echo                     print(candidate)
    echo                     raise SystemExit
    echo print("")
)

for /f "delims=" %%i in ('python "%TEMP_FIND%"') do set "AUDIO_DATA=%%i"
del "%TEMP_FIND%"

if "%AUDIO_DATA%"=="" (
    echo Error: Could not locate the fastai_audio-compatible data.py in any Python site-packages directory.
    echo fastai_audio / fastaudio may not be installed in this environment.
    echo Try: python -m pip install -r requirements.txt
    exit /b 1
)

echo Found patch target at: %AUDIO_DATA%

copy "%AUDIO_DATA%" "%AUDIO_DATA%.bak" >nul

set "TEMP_PATCH=%TEMP%\patch_fastai_audio_patch.py"
> "%TEMP_PATCH%" (
    echo import pathlib, sys
    echo path = pathlib.Path(sys.argv[1])
    echo text = path.read_text(encoding="utf-8")
    echo if "from dataclasses import dataclass, asdict, field" in text and "field(default_factory=SpectrogramConfig)" in text:
    echo     print("Already patched; no changes needed.")
    echo     raise SystemExit(0)
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
    echo     raise SystemExit("No compatible patch targets found in the installed fastai_audio package")
    echo path.write_text(text, encoding="utf-8")
)

python "%TEMP_PATCH%" "%AUDIO_DATA%"
if %ERRORLEVEL% neq 0 (
    del "%TEMP_PATCH%"
    exit /b %ERRORLEVEL%
)

del "%TEMP_PATCH%"

findstr /C:"field(default_factory=SpectrogramConfig)" "%AUDIO_DATA%" >nul
if %ERRORLEVEL% neq 0 (
    echo Warning: Patch may not have been applied correctly
    exit /b 1
)

echo Patch applied successfully!
echo Backup saved to: %AUDIO_DATA%.bak
echo Patch verification successful: field(default_factory=SpectrogramConfig) found
exit /b 0
