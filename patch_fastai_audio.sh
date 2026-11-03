#!/bin/bash
# Script to patch fastai_audio for Python 3.11+ compatibility
# Based on https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/patch_fastai_audio.sh

echo "Applying Python 3.11 compatibility patch to fastai_audio..."

AUDIO_DATA_FILE=$(python3 - <<'PY'
import os
import sys
import sysconfig

paths = [p for p in sys.path if p and os.path.isdir(p)]
paths += [
    sysconfig.get_paths().get('purelib'),
    sysconfig.get_paths().get('platlib'),
]

seen = set()
for base in paths:
    if not base or base in seen:
        continue
    seen.add(base)
    candidate = os.path.join(base, 'audio', 'data.py')
    if os.path.isfile(candidate):
        print(candidate)
        raise SystemExit

for base in paths:
    if not base:
        continue
    if 'site-packages' in base or 'dist-packages' in base:
        for root, dirs, _ in os.walk(base):
            if 'audio' in dirs:
                candidate = os.path.join(root, 'audio', 'data.py')
                if os.path.isfile(candidate):
                    print(candidate)
                    raise SystemExit
print('')
PY
)

if [ -z "$AUDIO_DATA_FILE" ]; then
    echo "Error: Could not locate audio/data.py in any Python site-packages directory."
    echo "fastai_audio may not be installed in this environment."
    echo "Try: python -m pip install -r requirements.txt"
    exit 1
fi

echo "Found audio/data.py at: $AUDIO_DATA_FILE"

cp "$AUDIO_DATA_FILE" "${AUDIO_DATA_FILE}.bak"

python3 - "$AUDIO_DATA_FILE" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding='utf-8')
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
    raise SystemExit('No compatible patch targets found in audio/data.py')

path.write_text(text, encoding='utf-8')
PY

echo "Patch applied successfully!"
echo "Backup saved to: ${AUDIO_DATA_FILE}.bak"

if grep -q "field(default_factory=SpectrogramConfig)" "$AUDIO_DATA_FILE"; then
    echo "✓ Patch verification successful: field(default_factory=SpectrogramConfig) found"
else
    echo "✗ Warning: Patch may not have been applied correctly"
    exit 1
fi
