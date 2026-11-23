#!/usr/bin/env python3
"""Patch fastai_audio's mutable dataclass default for Python 3.11+."""

import importlib.util
import shutil
import site
import sys
import sysconfig
from pathlib import Path

PACKAGE_NAMES = ("audio", "fastaudio", "fastai_audio")


def _find_data_file() -> Path | None:
    roots = [
        Path(entry)
        for entry in (
            *sys.path,
            *site.getsitepackages(),
            site.getusersitepackages(),
            *sysconfig.get_paths().values(),
        )
        if entry
    ]

    candidates = []
    for package_name in PACKAGE_NAMES:
        spec = importlib.util.find_spec(package_name)
        if spec and spec.submodule_search_locations:
            candidates.extend(
                Path(location) / "data.py"
                for location in spec.submodule_search_locations
            )
        candidates.extend(root / package_name / "data.py" for root in roots)
    candidates.extend(
        candidate
        for root in roots
        if any(
            part.lower() in {"site-packages", "dist-packages"}
            for part in root.parts
        )
        for candidate in root.rglob("data.py")
        if candidate.parent.name in PACKAGE_NAMES
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        content = candidate.read_text(encoding="utf-8", errors="ignore")
        if "SpectrogramConfig" in content and "sg_cfg" in content:
            return candidate
    return None


def main() -> int:
    print("Applying Python 3.11 compatibility patch to fastai_audio...")
    data_file = _find_data_file()
    if data_file is None:
        print(
            "Error: Could not locate the fastai_audio-compatible data.py using "
            "the selected Python interpreter."
        )
        print(
            "fastai_audio / fastaudio may not be installed in this environment. "
            "Try: python -m pip install -r requirements.txt"
        )
        return 1

    print(f"Found patch target at: {data_file}")
    text = data_file.read_text(encoding="utf-8")
    if (
        "from dataclasses import dataclass, asdict, field" in text
        and "field(default_factory=SpectrogramConfig)" in text
    ):
        print("Already patched; no changes needed.")
        return 0

    patched = text
    if "from dataclasses import dataclass, asdict, field" not in patched:
        patched = patched.replace(
            "from dataclasses import dataclass, asdict",
            "from dataclasses import dataclass, asdict, field",
        )
    if "field(default_factory=SpectrogramConfig)" not in patched:
        patched = patched.replace(
            "sg_cfg: SpectrogramConfig = SpectrogramConfig()",
            "sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)",
        )

    if patched == text:
        print("Error: No compatible patch targets found in the installed package.")
        return 1

    backup = data_file.with_suffix(data_file.suffix + ".bak")
    shutil.copy2(data_file, backup)
    data_file.write_text(patched, encoding="utf-8")

    verified = data_file.read_text(encoding="utf-8")
    if "field(default_factory=SpectrogramConfig)" not in verified:
        print("Error: Patch verification failed.")
        return 1

    print("Patch applied successfully!")
    print(f"Backup saved to: {backup}")
    print("Patch verification successful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
