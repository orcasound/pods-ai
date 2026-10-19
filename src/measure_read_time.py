#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Simple helper to measure WAV load/decode time by different libraries.

Usage:
  python src/measure_read_time.py [path/to/sample.wav]

If no path is provided the script will try to find the first .wav under
output/testing-wav/ or print an instruction if none are present.

The script prints average wall-clock time for a small number of iterations
for each available loading method (librosa, pydub, torchaudio, soundfile).
"""
import argparse
import glob
import os
import statistics
import sys
import time
from pathlib import Path


def find_default_wav() -> Path | None:
    # Prefer output/testing-wav if present
    candidate_dirs = [Path("output/testing-wav"), Path("output/wav"), Path("output")]
    for d in candidate_dirs:
        if d.exists() and d.is_dir():
            wavs = list(d.rglob("*.wav"))
            if wavs:
                return wavs[0]
    # fallback: search repo tree
    wavs = list(Path(".").rglob("*.wav"))
    return wavs[0] if wavs else None


def time_fn(fn, iterations=3):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def measure_librosa(path: str):
    import librosa

    def _load():
        librosa.load(path, sr=None)

    return time_fn(_load)


def measure_pydub(path: str):
    from pydub import AudioSegment

    def _load():
        AudioSegment.from_wav(path)

    return time_fn(_load)


def measure_torchaudio(path: str):
    try:
        import torchaudio
    except Exception:
        # torchaudio may be patched in model_inference; still attempt import
        raise

    def _load():
        torchaudio.load(path)

    return time_fn(_load)


def measure_soundfile(path: str):
    import soundfile as sf

    def _load():
        sf.read(path)

    return time_fn(_load)


def main():
    parser = argparse.ArgumentParser(description="Measure WAV load/decode time.")
    parser.add_argument("wav_file", nargs="?", help="Path to a WAV file to measure.")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per method")
    args = parser.parse_args()

    wav_path = Path(args.wav_file) if args.wav_file else find_default_wav()
    if not wav_path or not wav_path.exists():
        print("No sample WAV found. Provide a path or place WAVs under output/testing-wav/.")
        print("Examples:")
        print("  python src/measure_read_time.py output/testing-wav/resident/sample.wav")
        sys.exit(2)

    wav_str = str(wav_path)
    print(f"Measuring load time for: {wav_str}")
    methods = []
    results = {}

    # Try librosa
    try:
        mean, sd = measure_librosa(wav_str)
        results['librosa'] = (mean, sd)
        print(f"librosa: {mean:.3f}s (sd {sd:.3f})")
    except Exception as e:
        print(f"librosa: unavailable or failed: {e}")

    # Try pydub
    try:
        mean, sd = measure_pydub(wav_str)
        results['pydub'] = (mean, sd)
        print(f"pydub (AudioSegment.from_wav): {mean:.3f}s (sd {sd:.3f})")
    except Exception as e:
        print(f"pydub: unavailable or failed: {e}")

    # Try torchaudio
    try:
        mean, sd = measure_torchaudio(wav_str)
        results['torchaudio'] = (mean, sd)
        print(f"torchaudio.load: {mean:.3f}s (sd {sd:.3f})")
    except Exception as e:
        print(f"torchaudio: unavailable or failed: {e}")

    # Try soundfile
    try:
        mean, sd = measure_soundfile(wav_str)
        results['soundfile'] = (mean, sd)
        print(f"soundfile.read: {mean:.3f}s (sd {sd:.3f})")
    except Exception as e:
        print(f"soundfile: unavailable or failed: {e}")

    if results:
        # Choose the fastest available as representative r
        fastest = min(results.items(), key=lambda kv: kv[1][0])
        name, (mean, sd) = fastest
        print(f"\nRepresentative read time r = {mean:.3f}s (method: {name})")
    else:
        print("No successful load methods to report.")


if __name__ == '__main__':
    main()
