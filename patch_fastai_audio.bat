@echo off
REM Copyright (c) PODS-AI contributors
REM SPDX-License-Identifier: MIT
REM
REM Patch fastai_audio for Python 3.11+ compatibility
REM Based on https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/patch_fastai_audio.sh

echo Applying Python 3.11 compatibility patch to fastai_audio...

python -c "import os, sys, sysconfig, pathlib; paths=[p for p in sys.path if p and os.path.isdir(p)] + [sysconfig.get_paths().get('purelib'), sysconfig.get_paths().get('platlib')]; seen=set(); found='';
for base in paths:
    if not base or base in seen: continue
    seen.add(base)
    candidate=os.path.join(base, 'audio', 'data.py')
    if os.path.isfile(candidate): found=candidate; break
if not found:
    for base in paths:
        if base and ('site-packages' in base or 'dist-packages' in base):
            for root, dirs, _ in os.walk(base):
                candidate=os.path.join(root, 'audio', 'data.py')
                if os.path.isfile(candidate): found=candidate; break
            if found: break
if not found:
    raise SystemExit('Could not locate audio/data.py in any Python site-packages directory. fastai_audio may not be installed in this environment. Try: python -m pip install -r requirements.txt')
print('Found audio/data.py at: ' + found)
path=pathlib.Path(found)
backup=path.with_name(path.name + '.bak')
backup.write_bytes(path.read_bytes())
text=path.read_text(encoding='utf-8')
original=text
if 'from dataclasses import dataclass, asdict, field' not in text:
    text=text.replace('from dataclasses import dataclass, asdict', 'from dataclasses import dataclass, asdict, field')
if 'field(default_factory=SpectrogramConfig)' not in text:
    text=text.replace('sg_cfg: SpectrogramConfig = SpectrogramConfig()', 'sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)')
if text == original:
    raise SystemExit('No compatible patch targets found in audio/data.py')
path.write_text(text, encoding='utf-8')
print('Patch applied successfully!')
print('Backup saved to: ' + str(backup))
if 'field(default_factory=SpectrogramConfig)' not in path.read_text(encoding='utf-8'):
    raise SystemExit('Patch verification failed: field(default_factory=SpectrogramConfig) not found')
print('Patch verification successful: field(default_factory=SpectrogramConfig) found')"

exit /b %ERRORLEVEL%
