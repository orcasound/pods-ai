#!/bin/bash
# Script to patch the fastai/fastaudio Python package for Python 3.11+ compatibility.

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

echo "Applying Python 3.11 compatibility patch to fastai_audio..."

AUDIO_DATA_FILE=$("$PYTHON_BIN" - <<'PY'
import os
import sys
import sysconfig

try:
    import site
except ImportError:
    site = None

roots = [p for p in sys.path if p and os.path.isdir(p)]
roots += [
    sysconfig.get_paths().get('purelib'),
    sysconfig.get_paths().get('platlib'),
]
if site is not None:
    try:
        roots += site.getsitepackages()
    except Exception:
        pass
    try:
        roots += [site.getusersitepackages()]
    except Exception:
        pass
seen = set()
package_names = ['audio', 'fastaudio', 'fastai_audio']

for base in roots:
    if not base or base in seen:
        continue
    seen.add(base)
    for pkg in package_names:
        candidate = os.path.join(base, pkg, 'data.py')
        if os.path.isfile(candidate):
            print(candidate)
            raise SystemExit

for base in roots:
    if not base:
        continue
    if 'site-packages' in base or 'dist-packages' in base:
        for root, _, _ in os.walk(base):
            for pkg in package_names:
                candidate = os.path.join(root, pkg, 'data.py')
                if os.path.isfile(candidate):
                    print(candidate)
                    raise SystemExit
print('')
PY
)

if [ -z "$AUDIO_DATA_FILE" ]; then
    echo "Error: Could not locate the fastai_audio-compatible data.py in any Python site-packages directory."
    echo "fastai_audio / fastaudio may not be installed in this environment."
    echo "Try: python -m pip install -r requirements.txt"
    exit 1
fi

echo "Found patch target at: $AUDIO_DATA_FILE"
cp "$AUDIO_DATA_FILE" "${AUDIO_DATA_FILE}.bak"

"$PYTHON_BIN" - "$AUDIO_DATA_FILE" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding='utf-8')
if 'from dataclasses import dataclass, asdict, field' in text and 'field(default_factory=SpectrogramConfig)' in text:
    print('Already patched; no changes needed.')
    raise SystemExit(0)

original = text
if 'from dataclasses import dataclass, asdict, field' not in text:
    text = text.replace(
        'from dataclasses import dataclass, asdict',
        'from dataclasses import dataclass, asdict, field',
    )

if 'field(default_factory=SpectrogramConfig)' not in text:
    text = text.replace(
        'sg_cfg: SpectrogramConfig = SpectrogramConfig()',
        'sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)',
    )

if text == original:
    raise SystemExit('No compatible patch targets found in the installed fastai_audio package')

path.write_text(text, encoding='utf-8')
PY

if grep -q "field(default_factory=SpectrogramConfig)" "$AUDIO_DATA_FILE"; then
    echo "Patch applied successfully!"
    echo "Backup saved to: ${AUDIO_DATA_FILE}.bak"
    echo "Patch verification successful: field(default_factory=SpectrogramConfig) found"
else
    echo "Warning: Patch may not have been applied correctly"
    exit 1
fi
