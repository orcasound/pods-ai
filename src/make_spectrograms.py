# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Script to generate spectrograms from .wav files.

This script traverses the output_segments directory (and its subdirectories),
finds all .wav files saved by download_wavs.py, and creates a spectrogram
(.png file) alongside each .wav file.

Usage:
    python src/make_spectrograms.py
"""
import sys
from pathlib import Path
from typing import List

import spectrogram_visualizer


def find_wav_files(root_dir: Path) -> List[Path]:
    """
    Recursively find all .wav files in a directory and its subdirectories.
    
    Parameters:
        root_dir: Root directory to search.
    
    Returns:
        List[Path]: List of paths to .wav files.
    """
    wav_files = []
    for path in root_dir.rglob("*.wav"):
        if path.is_file():
            wav_files.append(path)
    return wav_files


def process_wav_files(output_root: Path, skip_existing: bool = True):
    """
    Process all .wav files in the output_root directory and generate spectrograms.
    
    Parameters:
        output_root: Root directory containing .wav files in subdirectories.
        skip_existing: If True, skip .wav files that already have a .png spectrogram.
    """
    wav_files = find_wav_files(output_root)
    
    if not wav_files:
        print(f"No .wav files found in {output_root}")
        return
    
    print(f"Found {len(wav_files)} .wav file(s) to process")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for wav_path in wav_files:
        # Check if spectrogram already exists
        png_path = wav_path.with_suffix('.png')
        if skip_existing and png_path.exists():
            print(f"Skipping (spectrogram already exists): {wav_path}")
            skipped_count += 1
            continue
        
        try:
            print(f"Processing: {wav_path}")
            spec_path = spectrogram_visualizer.write_spectrogram(wav_path)
            print(f"  -> Created spectrogram: {spec_path}")
            processed_count += 1
        except Exception as e:
            print(f"  -> ERROR processing {wav_path}: {type(e).__name__}: {str(e)}")
            error_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    output_root = Path("output_segments")
    
    if not output_root.exists():
        print(f"Error: Directory not found: {output_root}")
        print("Please ensure the output_segments directory exists.")
        sys.exit(1)
    
    process_wav_files(output_root)
