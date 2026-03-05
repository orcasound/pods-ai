# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Process WAV files from the signals-humpback submodule and add 2-second
segments to the humpback training samples.

Usage:
    python process_humpback_wavs.py

Reads WAV files from the external/signals-humpback submodule, skips any files
shorter than 2 seconds, and for the rest splits them into 2-second segments
(ignoring any remainder shorter than 2 seconds). The resulting segments are
saved under output_segments/humpback/.
"""
from pathlib import Path
import re
import sys

import ffmpeg

N_SECONDS = 2  # Length of each output segment in seconds.
FILENAME_SAFE_PATTERN = r"[^A-Za-z0-9_-]"  # Characters to replace when sanitizing filenames.
EXTERNAL_HUMPBACK_DIR = Path(__file__).parent.parent.parent / "external" / "signals-humpback"


def process_external_humpback_wavs(external_dir: Path, output_root: Path):
    """
    Process WAV files from an external signals-humpback directory and add 2-second
    segments to the humpback training samples.

    Skips files shorter than 2 seconds. For longer files, extracts each full 2-second
    segment (ignoring any remainder shorter than 2 seconds) and saves to the humpback
    label directory under output_root.

    Parameters:
        external_dir (Path): Root of the external signals-humpback directory.
        output_root (Path): Root directory where label subdirectories will be created.
    """
    label_dir = output_root / "humpback"
    label_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(external_dir.rglob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {external_dir}")
        return

    print(f"Found {len(wav_files)} WAV files in {external_dir}")

    for wav_file in wav_files:
        try:
            probe = ffmpeg.probe(str(wav_file))
            duration = float(probe["format"]["duration"])
        except Exception as e:
            print(f"Warning: Could not probe {wav_file}: {e}")
            continue

        if duration < N_SECONDS:
            print(f"Skipping (too short, {duration:.2f}s): {wav_file.name}")
            continue

        num_segments = int(duration // N_SECONDS)
        # Sanitize the stem for use in output filenames.
        safe_stem = re.sub(FILENAME_SAFE_PATTERN, "_", wav_file.stem)

        for i in range(num_segments):
            offset = i * N_SECONDS
            out_filename = f"signals-humpback_{safe_stem}_{offset:04d}s.wav"
            out_path = label_dir / out_filename

            if out_path.exists():
                print(f"Skipping (already exists): {out_path}")
                continue

            try:
                stream = ffmpeg.input(str(wav_file), ss=offset)
                stream = ffmpeg.output(
                    stream,
                    str(out_path),
                    t=N_SECONDS,
                    acodec="pcm_s16le",
                    ar=44100,
                    ac=1
                )
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                print(f"Created: {out_path}")
            except Exception as e:
                print(f"Warning: Failed to extract segment from {wav_file.name} at offset {offset}s: {e}")


if __name__ == "__main__":
    output_root = Path("output_segments")

    external_dir = EXTERNAL_HUMPBACK_DIR
    if not external_dir.exists():
        print(f"Error: External humpback directory not found at {external_dir}")
        print("Run 'git submodule update --init' to initialize the submodule.")
        sys.exit(1)

    process_external_humpback_wavs(external_dir, output_root)
