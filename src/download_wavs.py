# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import csv
import math
import os
import shutil
import sys
from tempfile import TemporaryDirectory

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import ffmpeg
import m3u8
from pytz import timezone
import requests

PACIFIC_TZ = timezone('US/Pacific')
N_SECONDS = 2  # Create 2-second wav files.

@dataclass
class CSVRow:
    category: str
    node_name: str
    timestamp_pst: str
    uri: str
    description: str
    notes: str

# ============================================================================
# S3 Utilities (simplified from orca-hls-utils)
# ============================================================================

def get_all_folders(bucket: str, prefix: str) -> List[str]:
    """
    Get all folder names from an S3 bucket with the given prefix.
    
    Parameters:
        bucket (str): Name of the S3 bucket.
        prefix (str): Prefix to filter objects.
    
    Returns:
        List[str]: List of folder names (without the prefix).
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
    
    all_keys = []
    for page in paginator.paginate(**kwargs):
        try:
            common_prefixes = page["CommonPrefixes"]
            prefixes = [
                prefix["Prefix"].split("/")[-2] for prefix in common_prefixes
            ]
            all_keys.extend(prefixes)
        except KeyError:
            break
    
    return all_keys

def get_folders_between_timestamp(bucket_list: List[str], start_time: int, end_time: int) -> List[int]:
    """
    Filter bucket list to only include folders between start_time and end_time.
    
    Parameters:
        bucket_list (List[str]): List of folder names (as strings).
        start_time (int): Start unix timestamp.
        end_time (int): End unix timestamp.
    
    Returns:
        List[int]: Filtered list of folder names as integers.
    """
    bucket_list = [int(bucket) for bucket in bucket_list]
    start_index = 0
    end_index = len(bucket_list) - 1
    
    while start_index < len(bucket_list) and bucket_list[start_index] < start_time:
        start_index += 1
    
    while end_index >= 0 and bucket_list[end_index] > end_time:
        end_index -= 1
    
    # Include the folder before start_time to ensure we have data.
    return bucket_list[max(0, start_index - 1) : end_index + 1]

# ============================================================================
# DateTime Utilities (simplified from orca-hls-utils)
# ============================================================================

def get_clip_name_from_unix_time(source_guid: str, current_clip_start_time: int) -> tuple:
    """
    Generate a clip name from unix timestamp.
    
    Parameters:
        source_guid (str): Hydrophone identifier.
        current_clip_start_time (int): Unix timestamp.
    
    Returns:
        tuple: (clipname, readable_datetime)
    """
    readable_datetime = datetime.fromtimestamp(int(current_clip_start_time)).strftime("%Y_%m_%d_%H_%M_%S")
    clipname = source_guid + "_" + readable_datetime
    return clipname, readable_datetime

def get_difference_between_times_in_seconds(unix_time1: int, unix_time2: int) -> float:
    """
    Calculate the difference between two unix timestamps in seconds.
    
    Parameters:
        unix_time1 (int): First unix timestamp.
        unix_time2 (int): Second unix timestamp.
    
    Returns:
        float: Difference in seconds.
    """
    dt1 = datetime.fromtimestamp(int(unix_time1))
    dt2 = datetime.fromtimestamp(int(unix_time2))
    return (dt1 - dt2).total_seconds()

# ============================================================================
# Download Utilities (simplified from orca-hls-utils)
# ============================================================================

def download_from_url(dl_url: str, dl_dir: str):
    """
    Download a file from URL to a directory.
    
    Parameters:
        dl_url (str): URL to download from.
        dl_dir (str): Directory to save the file.
    """
    file_name = os.path.basename(dl_url)
    dl_path = os.path.join(dl_dir, file_name)
    
    if os.path.isfile(dl_path):
        return
    
    response = requests.get(dl_url, timeout=30)
    response.raise_for_status()
    
    with open(dl_path, 'wb') as f:
        f.write(response.content)

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
    timestamp_pst: datetime,
    output_root: Path,
):
    """
    Download a 2-second audio segment for a detection and save it to the appropriate label directory.
    
    This function implements a simplified version of DateRangeHLSStream logic to download
    only a 2-second wav file instead of the full 60-second clip.
    
    Parameters:
        category (str): The label/category for the detection (e.g., "resident", "transient").
        node_name (str): The node name (e.g., "rpi_sunset_bay").
        timestamp_pst (datetime): The detection timestamp in Pacific time.
        output_root (Path): Root directory where label subdirectories and audio files will be saved.
    """
    label_dir = output_root / category
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the file already exists.
    duration_s = N_SECONDS
    end_timestamp_pst = timestamp_pst + timedelta(seconds=duration_s)
    end_timestamp_str = end_timestamp_pst.strftime("%Y_%m_%d_%H_%M_%S")
    node_name_in_filename = node_name.replace("_", "-")
    expected_filename = f"{node_name_in_filename}_{end_timestamp_str}_PST.wav"
    expected_path = label_dir / expected_filename
    
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
    end_time = start_time + timedelta(seconds=duration_s)
    start_unix_time = int(start_time.timestamp())
    end_unix_time = int(end_time.timestamp())
    
    # Get all folders from S3 and filter by timestamp.
    try:
        all_hydrophone_folders = get_all_folders(s3_bucket, prefix=prefix)
        print(f"Found {len(all_hydrophone_folders)} folders in total for hydrophone")
        
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
    
    # Generate clip name.
    clipname, clip_start_time = get_clip_name_from_unix_time(
        folder_name.replace("_", "-"), start_unix_time
    )
    
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
    target_duration = sum(item.duration for item in stream_obj.segments) / num_total_segments
    
    # Calculate number of segments needed for N_SECONDS.
    num_segments_needed = math.ceil(N_SECONDS / target_duration)
    
    # Calculate start index based on time since folder start.
    # Note: there's typically a 2-second audio offset.
    audio_offset = 2
    time_since_folder_start = get_difference_between_times_in_seconds(start_unix_time, current_folder)
    time_since_folder_start -= audio_offset
    
    segment_start_index = max(0, math.ceil(time_since_folder_start / target_duration))
    segment_end_index = segment_start_index + num_segments_needed
    
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
            hls_file = os.path.join(tmp_path, clipname + ".ts")
            with open(hls_file, "wb") as wfd:
                for f in file_names:
                    with open(os.path.join(tmp_path, f), "rb") as fd:
                        shutil.copyfileobj(fd, wfd)
            
            # Convert to wav using ffmpeg.
            audio_file = clipname + ".wav"
            wav_file_path = os.path.join(label_dir, audio_file)
            stream = ffmpeg.input(hls_file)
            stream = ffmpeg.output(stream, wav_file_path)
            # Use overwrite_output=True since we already checked for file existence above
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
        timestamp_pst = parse_timestamp_pst(row.timestamp_pst)
        download_audio_segment(row.category, row.node_name, timestamp_pst, output_root)

if __name__ == "__main__":
    csv_path = Path("output_segments/training_samples.csv")
    output_root = Path("output_segments")
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run extract_training_samples.py first to generate the training_samples.csv file.")
        sys.exit(1)
    
    process_csv(csv_path, output_root)
