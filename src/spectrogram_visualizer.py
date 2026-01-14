# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Spectrogram visualization utilities.

This module provides functions to generate spectrograms from audio files.
The implementation is based on:
https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/src/spectrogram_visualizer.py
"""
import os

import librosa
import matplotlib
matplotlib.use('Agg')  # No pictures displayed
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2


def _create_spectrogram_figure(specshow_data, sr, output_path, x_axis='time', y_axis='hz', fmax=None):
    """
    Helper function to create and save a spectrogram using explicit Figure/Axes objects.
    Closes the figure after saving to prevent memory leaks.
    
    Parameters:
        specshow_data: Spectrogram data to visualize.
        sr: Sample rate of the audio.
        output_path: Path where the spectrogram image will be saved.
        x_axis: Type of x-axis (default: 'time').
        y_axis: Type of y-axis (default: 'hz').
        fmax: Maximum frequency to display (optional).
    """
    # Use explicit figure with size 6.4x4.8 inches at 100 dpi = 640x480 pixels
    fig = plt.figure(frameon=False, figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_position([0., 0., 1., 1.])  # Remove borders
    
    if fmax is not None:
        librosa.display.specshow(specshow_data, sr=sr, x_axis=x_axis, y_axis=y_axis, fmax=fmax, ax=ax)
    else:
        librosa.display.specshow(specshow_data, sr=sr, x_axis=x_axis, y_axis=y_axis, ax=ax)
    
    fig.savefig(output_path, bbox_inches=None, pad_inches=0)
    
    # Close figure to release memory and prevent leaks
    plt.close(fig)


def write_spectrogram(wav_file_path):
    """
    Generate a spectrogram from a .wav file and save it as a .png file.
    
    This function divides the audio into two parts and creates spectrograms
    for each half, then combines them into a single image.
    
    Parameters:
        wav_file_path: Path to the .wav file (str or Path object).
    
    Returns:
        str: Path to the generated spectrogram (.png file).
    """
    wav_file_path = str(wav_file_path)  # Convert Path to str if needed
    
    # Get wav_file_path without extension
    directory_name = os.path.dirname(wav_file_path)
    candidate_name = os.path.basename(wav_file_path)
    candidate_name_without_extension = os.path.splitext(candidate_name)[0]

    spectrogram_name = candidate_name_without_extension + ".png"

    # Temp files that will be deleted
    spec_first_half = os.path.join(directory_name, "firstHalf.png")
    spec_second_half = os.path.join(directory_name, "secondHalf.png")

    # Final spec file
    spec_output_path = os.path.join(directory_name, spectrogram_name)

    # Here, we divide the audio into spectrogram into 2 parts and calculate spectrograms for each half
    y, sr = librosa.load(wav_file_path)
    half_len_y = len(y) // 2
    y_first_half = y[:half_len_y]
    y_second_half = y[half_len_y:]

    X_first_half = librosa.stft(y_first_half)
    Xdb_first_half = librosa.amplitude_to_db(abs(X_first_half))
    _create_spectrogram_figure(Xdb_first_half, sr, spec_first_half, x_axis='time', y_axis='hz')

    X_second_half = librosa.stft(y_second_half)
    Xdb_second_half = librosa.amplitude_to_db(abs(X_second_half))
    _create_spectrogram_figure(Xdb_second_half, sr, spec_second_half, x_axis='time', y_axis='hz')

    # Create canvas to create combined spectrogram
    # Use dtype=np.uint8 to match images read by cv2
    canvas = np.zeros((480, 640 * 2, 3), dtype=np.uint8)

    # Combine spectrograms
    spec1 = cv2.imread(spec_first_half)
    spec2 = cv2.imread(spec_second_half)

    if spec1 is None or spec2 is None:
        raise RuntimeError(f"Failed to load temporary spectrogram images for {wav_file_path}")

    canvas[:, :640, :] = spec1
    canvas[:, 640:, :] = spec2

    # Delete temporary spectrogram files
    os.remove(spec_first_half)
    os.remove(spec_second_half)

    cv2.imwrite(spec_output_path, canvas)

    return spec_output_path
