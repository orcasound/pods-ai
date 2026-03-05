# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import csv
import math
import os
import re
import shutil
import sys
from tempfile import TemporaryDirectory

import ffmpeg
import m3u8
from pytz import timezone

from audio_utils import (
    get_cached_folders,
    get_folders_between_timestamp,
    get_difference_between_times_in_seconds,
    download_from_url
)

PACIFIC_TZ = timezone('US/Pacific')
N_SECONDS = 2  # Create 2-second wav files.
FILENAME_SAFE_PATTERN = r"[^A-Za-z0-9_-]"  # Characters to replace when sanitizing filenames.
EXTERNAL_HUMPBACK_DIR = Path(__file__).parent.parent.parent / "external" / "signals-humpback"

@dataclass
class CSVRow:
    category: str
    node_name: str
    timestamp_pst: str
    uri: str
    description: str
    notes: str

# ============================================================================
# CSV Parsing
# ============================================================================



def parse_csv(csv_path: Path) -> List[CSVRow]:
    """
    Parse a CSV file (detections or training samples) and return a list of CSVRow objects.
    
    Parameters:
        csv_path (Path): Path to the CSV file.
    
    Returns:
        List[CSVRow]: List of parsed CSV rows.
    """
    rows = []
    with open(csv_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip header
        next(csv_reader)
        for row in csv_reader:
            if len(row) >= 6:
                rows.append(CSVRow(
                    category=row[0],
                    node_name=row[1],
                    timestamp_pst=row[2],
                    uri=row[3],
                    description=row[4],
                    notes=row[5]
                ))
    return rows

def parse_timestamp_pst(timestamp_str: str) -> datetime:
    """
    Parse a PST timestamp string in the format YYYY_MM_DD_HH_MM_SS_PST.
    
    Parameters:
        timestamp_str (str): Timestamp string (e.g., "2025_12_24_17_51_23_PST").
    
    Returns:
        datetime: Parsed datetime object with Pacific timezone.
    """
    # Remove _PST suffix if present.
    timestamp_str = timestamp_str.replace('_PST', '')

    # Parse the datetime.
    dt_naive = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")

    # Localize to Pacific timezone.
    dt_aware = PACIFIC_TZ.localize(dt_naive)

    return dt_aware

def download_audio_segment(
    category: str,
    node_name: str,
    timestamp_str: str,
    output_root: Path,
):
    """
    Download a 2-second audio segment for a detection and save it to the appropriate label directory.
    
    This function implements a simplified version of DateRangeHLSStream logic to download
    only a 2-second wav file instead of the full 60-second clip.
    
    Parameters:
        category (str): The label/category for the detection (e.g., "resident", "transient").
        node_name (str): The node name (e.g., "rpi_sunset_bay").
        timestamp_str (str): The detection timestamp in Pacific time.
        output_root (Path): Root directory where label subdirectories and audio files will be saved.
    """
    label_dir = output_root / category
    label_dir.mkdir(parents=True, exist_ok=True)
    timestamp_pst = parse_timestamp_pst(timestamp_str)
    
    # Check if the file already exists.
    node_name_in_filename = node_name.replace("_", "-")
    clipname = f"{node_name_in_filename}_{timestamp_str}"
    wav_filename = f"{clipname}.wav"
    expected_path = label_dir / wav_filename
    if expected_path.exists():
        print(f"Skipping (already exists): {expected_path}")
        return
    
    # Set up S3 bucket and folder information.
    hydrophone_stream_url = 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/' + node_name
    bucket_folder = hydrophone_stream_url.split("https://s3-us-west-2.amazonaws.com/")[1]
    tokens = bucket_folder.split("/")
    s3_bucket = tokens[0]
    folder_name = tokens[1]
    prefix = folder_name + "/hls/"
    
    # Convert timestamps to unix time.
    start_time = timestamp_pst
    end_time = start_time + timedelta(seconds=N_SECONDS)
    start_unix_time = int(start_time.timestamp())
    end_unix_time = int(end_time.timestamp())
    
    # Get all folders from S3 and filter by timestamp.
    try:
        # Use cached folders per node/bucket/prefix to avoid repeated S3 listing calls.
        all_hydrophone_folders = get_cached_folders(s3_bucket, prefix=prefix)
        print(f"Found {len(all_hydrophone_folders)} folders in total for {node_name}")
        
        valid_folders = get_folders_between_timestamp(all_hydrophone_folders, start_unix_time, end_unix_time)
        print(f"Found {len(valid_folders)} folders in date range")
        
        if not valid_folders:
            print(f"Warning: No folders found for timestamp {start_time}")
            return
        
        # Use the first valid folder.
        current_folder = int(valid_folders[0])
        
    except Exception as e:
        print(f"\nERROR: Failed to query S3 bucket.")
        print(f"Details: {e}")
        print(f"Hydrophone: {node_name}")
        print(f"Start time (unix): {start_unix_time}")
        print(f"End time (unix): {end_unix_time}")
        return
 
    # Read the m3u8 file for the current folder.
    stream_url = f"{hydrophone_stream_url}/hls/{current_folder}/live.m3u8"
    
    try:
        stream_obj = m3u8.load(stream_url)
    except Exception as e:
        print(f"ERROR: Failed to load m3u8 file from {stream_url}")
        print(f"Details: {e}")
        return
    
    num_total_segments = len(stream_obj.segments)
    if num_total_segments == 0:
        print(f"ERROR: No segments found in m3u8 file")
        return
    
    # Calculate target duration (average segment duration).
    target_duration_exact = sum(item.duration for item in stream_obj.segments) / num_total_segments
    target_duration = round(target_duration_exact, 1)
    
    # Calculate number of segments needed for N_SECONDS.
    num_segments_needed = math.ceil(N_SECONDS / target_duration)
    
    # Calculate start and end indices based on time since folder start.
    # Note: there's typically a 2-second audio offset.
    audio_offset = 2
    time_since_folder_start_for_start = get_difference_between_times_in_seconds(start_unix_time, current_folder)
    time_since_folder_start_for_start -= audio_offset

    time_since_folder_start_for_end = get_difference_between_times_in_seconds(end_unix_time, current_folder)
    time_since_folder_start_for_end -= audio_offset

    segment_start_index = max(0, math.floor(time_since_folder_start_for_start / target_duration))
    segment_end_index = min(num_total_segments, math.ceil(time_since_folder_start_for_end / target_duration))
    
    if segment_end_index > num_total_segments:
        print(f"ERROR: Not enough segments available. Need {segment_end_index}, but only {num_total_segments} available.")
        return
    
    # Download and process segments.
    try:
        with TemporaryDirectory() as tmp_path:
            os.makedirs(tmp_path, exist_ok=True)
            
            file_names = []
            for i in range(segment_start_index, segment_end_index):
                audio_segment = stream_obj.segments[i]
                base_path = audio_segment.base_uri
                file_name = audio_segment.uri
                audio_url = base_path + file_name
                try:
                    download_from_url(audio_url, tmp_path)
                    file_names.append(file_name)
                except Exception as e:
                    print(f"Warning: Skipping {audio_url}: {e}")
            
            if not file_names:
                print("ERROR: No segments were successfully downloaded")
                return
            
            # Concatenate all .ts files.
            if len(file_names) > 1:
                hls_file = os.path.join(tmp_path, clipname + ".ts")
                with open(hls_file, "wb") as wfd:
                    for f in file_names:
                        with open(os.path.join(tmp_path, f), "rb") as fd:
                            shutil.copyfileobj(fd, wfd)
            else:
                hls_file = os.path.join(tmp_path, file_names[0])
            
            # Convert to wav using ffmpeg, but only extract N_SECONDS starting
            # at the requested timestamp offset inside the concatenated file.
            wav_file_path = os.path.join(label_dir, wav_filename)

            # Compute offset (seconds) into the concatenated .ts where the desired start occurs.
            # time_since_folder_start and target_duration are computed earlier in the function.
            ss_offset = time_since_folder_start_for_start - (segment_start_index * target_duration)
            if ss_offset < 0:
                ss_offset = 0.0

            # Use input seeking (ss on input) and limit duration with t on output.
            stream = ffmpeg.input(hls_file, ss=ss_offset)
            stream = ffmpeg.output(
                stream,
                wav_file_path,
                t=N_SECONDS,
                acodec="pcm_s16le",  # optional: force WAV PCM format
                ar=44100,            # optional: sample rate
                ac=1                 # optional: mono
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            print(f"Downloaded: {wav_file_path}")
            
    except Exception as e:
        print(f"\nWarning: Unable to retrieve audio clip.")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        print(f"Hydrophone: {node_name}")

def process_csv(csv_path: Path, output_root: Path):
    """
    Read the training samples CSV file and download corresponding WAV files.
    
    Parameters:
        csv_path (Path): Path to the training_samples.csv file.
        output_root (Path): Root directory where audio files will be saved in label subdirectories.
    """
    rows = parse_csv(csv_path)
    
    print(f"Found {len(rows)} training samples to process")
    
    for row in rows:
        print(f"Processing: {row.category} - {row.node_name} - {row.timestamp_pst}")
        download_audio_segment(row.category, row.node_name, row.timestamp_pst, output_root)

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
    csv_path = Path("output_segments/training_samples.csv")
    output_root = Path("output_segments")
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run extract_training_samples.py first to generate the training_samples.csv file.")
        sys.exit(1)
    
    process_csv(csv_path, output_root)

    external_dir = EXTERNAL_HUMPBACK_DIR
    if external_dir.exists():
        process_external_humpback_wavs(external_dir, output_root)
    else:
        print(f"Skipping external humpback wavs: {external_dir} not found.")
        print("Run 'git submodule update --init' to initialize the submodule.")
