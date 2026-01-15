#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Extract training samples from detections.csv with the following constraints:
- At least 30 samples per category (or all available if < 30)
- At least one sample per category-node combination
- Prefer tp_machine_only or fp_machine_only notes
- Spread samples evenly across nodes per category
- Minimize total rows while meeting constraints
- For tp_human_only detections: Run model on preceding 60 seconds to find correct timestamp
- For other detections: Subtract 2 seconds from timestamps

For tp_human_only detections, we download 60 seconds of audio preceding the detection
timestamp, run model inference to score each 2-second segment, and use the highest
scoring segment to determine the correct timestamp offset (between 2-60 seconds).

This matches the behavior of aifororcas-livesystem's LiveInferenceOrchestrator.py
which uses DateRangeHLSStream to download audio and returns local_confidences for
each 2-second sample. See:
https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/src/LiveInferenceOrchestrator.py
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pytz import timezone
import os
import math
import shutil
from tempfile import TemporaryDirectory

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import ffmpeg
import m3u8
import requests

from model_inference import get_model_inference

PACIFIC_TZ = timezone('US/Pacific')
PREFERRED_NOTES = {'tp_machine_only', 'fp_machine_only'}
QUALITY_FILTER_TERMS = {'faint', 'distant'}
MIN_SAMPLES_PER_CATEGORY = 30

# Cache for S3 folder listings
_FOLDERS_CACHE = {}


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse a PST timestamp string to datetime object."""
    dt = datetime.strptime(timestamp_str, '%Y_%m_%d_%H_%M_%S_PST')
    return PACIFIC_TZ.localize(dt)


def format_timestamp(dt: datetime) -> str:
    """Format a datetime object to PST timestamp string."""
    return dt.strftime('%Y_%m_%d_%H_%M_%S_PST')


def subtract_two_seconds(timestamp_str: str) -> str:
    """Subtract 2 seconds from a PST timestamp string."""
    dt = parse_timestamp(timestamp_str)
    dt_minus_2 = dt - timedelta(seconds=2)
    return format_timestamp(dt_minus_2)


def load_detections(csv_path: Path) -> List[Dict]:
    """Load all detections from CSV file."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def organize_by_category_node(detections: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Organize detections by category and node.
    Returns: {category: {node: [list of detections]}}
    """
    organized = defaultdict(lambda: defaultdict(list))
    for det in detections:
        category = det['Category']
        node = det['NodeName']
        organized[category][node].append(det)
    return organized


def sort_by_preference(detections: List[Dict]) -> List[Dict]:
    """
    Sort detections by preference:
    1. Preferred notes (tp_machine_only, fp_machine_only) first
    2. Descriptions without quality issues (faint, distant) preferred
    3. Then by timestamp (oldest first)
    """
    def sort_key(det):
        has_preferred_note = det['Notes'] in PREFERRED_NOTES
        description_lower = det.get('Description', '').lower()
        has_quality_issue = any(term in description_lower for term in QUALITY_FILTER_TERMS)
        timestamp = det['Timestamp']
        # Return tuple: (not preferred note, has quality issue, timestamp)
        # This puts preferred notes first, then non-faint/distant descriptions, then by timestamp
        return (not has_preferred_note, has_quality_issue, timestamp)
    
    return sorted(detections, key=sort_key)


def select_training_samples(organized_data: Dict[str, Dict[str, List[Dict]]]) -> List[Dict]:
    """
    Select training samples according to requirements:
    - At least 30 samples per category (or all if < 30)
    - At least one sample per category-node combination
    - Spread evenly across nodes
    - Prefer certain note types
    - Minimize total rows
    """
    selected = []
    
    for category in sorted(organized_data.keys()):
        nodes_data = organized_data[category]
        
        # Sort and prepare detections for each node
        node_detections = {}
        for node in nodes_data.keys():
            node_detections[node] = sort_by_preference(nodes_data[node])
        
        # Calculate target count for this category
        total_available = sum(len(node_detections[node]) for node in node_detections)
        target_count = min(MIN_SAMPLES_PER_CATEGORY, total_available)
        
        # Track how many samples selected per node
        node_counts = defaultdict(int)
        category_samples = []
        
        # Round-robin selection across nodes to ensure even distribution
        # First, ensure at least one sample per node
        for node in sorted(node_detections.keys()):
            if node_detections[node]:
                category_samples.append(node_detections[node][0])
                node_counts[node] = 1
        
        # Continue round-robin to reach target count
        while len(category_samples) < target_count:
            added_this_round = False
            
            # Sort nodes by how many samples they've contributed (fewest first)
            # This ensures even distribution
            nodes_by_count = sorted(node_detections.keys(), 
                                   key=lambda n: node_counts[n])
            
            for node in nodes_by_count:
                if len(category_samples) >= target_count:
                    break
                    
                # Check if this node has more samples to give
                if node_counts[node] < len(node_detections[node]):
                    idx = node_counts[node]
                    category_samples.append(node_detections[node][idx])
                    node_counts[node] += 1
                    added_this_round = True
            
            # If no node could contribute, we've exhausted all samples
            if not added_this_round:
                break
        
        # Add all selected samples for this category
        selected.extend(category_samples)
    
    return selected


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


def get_cached_folders(bucket: str, prefix: str) -> List[str]:
    """
    Return cached folder list for (bucket, prefix). If absent, call get_all_folders()
    to populate the cache, then return the cached value.
    """
    key = f"{bucket}::{prefix}"
    if key not in _FOLDERS_CACHE:
        _FOLDERS_CACHE[key] = get_all_folders(bucket, prefix)
    return _FOLDERS_CACHE[key]


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


def download_60s_audio(node_name: str, timestamp_str: str, tmp_dir: str) -> Optional[str]:
    """
    Download 60 seconds of audio preceding the given timestamp.
    
    Args:
        node_name: The node name (e.g., "rpi_sunset_bay").
        timestamp_str: The detection timestamp in Pacific time.
        tmp_dir: Temporary directory to save the audio file.
    
    Returns:
        Path to the downloaded wav file, or None if download failed.
    """
    timestamp_pst = parse_timestamp(timestamp_str)
    
    # We need 60 seconds of audio ending at the given timestamp
    end_time = timestamp_pst
    start_time = end_time - timedelta(seconds=60)
    
    # Set up S3 bucket and folder information
    hydrophone_stream_url = 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/' + node_name
    bucket_folder = hydrophone_stream_url.split("https://s3-us-west-2.amazonaws.com/")[1]
    tokens = bucket_folder.split("/")
    s3_bucket = tokens[0]
    folder_name = tokens[1]
    prefix = folder_name + "/hls/"
    
    # Convert timestamps to unix time
    start_unix_time = int(start_time.timestamp())
    end_unix_time = int(end_time.timestamp())
    
    try:
        # Use cached folders per node/bucket/prefix
        all_hydrophone_folders = get_cached_folders(s3_bucket, prefix=prefix)
        print(f"  Found {len(all_hydrophone_folders)} folders in total for {node_name}")
        
        valid_folders = get_folders_between_timestamp(all_hydrophone_folders, start_unix_time, end_unix_time)
        print(f"  Found {len(valid_folders)} folders in date range")
        
        if not valid_folders:
            print(f"  Warning: No folders found for timestamp {start_time}")
            return None
        
        # Use the first valid folder
        current_folder = int(valid_folders[0])
        
    except Exception as e:
        print(f"  ERROR: Failed to query S3 bucket: {e}")
        return None
    
    # Read the m3u8 file for the current folder
    stream_url = f"{hydrophone_stream_url}/hls/{current_folder}/live.m3u8"
    
    try:
        stream_obj = m3u8.load(stream_url)
    except Exception as e:
        print(f"  ERROR: Failed to load m3u8 file: {e}")
        return None
    
    num_total_segments = len(stream_obj.segments)
    if num_total_segments == 0:
        print(f"  ERROR: No segments found in m3u8 file")
        return None
    
    # Calculate target duration (average segment duration)
    target_duration_exact = sum(item.duration for item in stream_obj.segments) / num_total_segments
    target_duration = round(target_duration_exact, 1)
    
    # Calculate start and end indices based on time since folder start
    # Note: there's typically a 2-second audio offset
    audio_offset = 2
    time_since_folder_start_for_start = get_difference_between_times_in_seconds(start_unix_time, current_folder)
    time_since_folder_start_for_start -= audio_offset

    time_since_folder_start_for_end = get_difference_between_times_in_seconds(end_unix_time, current_folder)
    time_since_folder_start_for_end -= audio_offset

    segment_start_index = max(0, math.floor(time_since_folder_start_for_start / target_duration))
    segment_end_index = min(num_total_segments, math.ceil(time_since_folder_start_for_end / target_duration))
    
    if segment_end_index > num_total_segments:
        print(f"  ERROR: Not enough segments available")
        return None
    
    # Download and process segments
    try:
        file_names = []
        for i in range(segment_start_index, segment_end_index):
            audio_segment = stream_obj.segments[i]
            base_path = audio_segment.base_uri
            file_name = audio_segment.uri
            audio_url = base_path + file_name
            try:
                download_from_url(audio_url, tmp_dir)
                file_names.append(file_name)
            except Exception as e:
                print(f"  Warning: Skipping {audio_url}: {e}")
        
        if not file_names:
            print("  ERROR: No segments were successfully downloaded")
            return None
        
        # Concatenate all .ts files
        clipname = f"temp_60s_{node_name}_{timestamp_str}"
        if len(file_names) > 1:
            hls_file = os.path.join(tmp_dir, clipname + ".ts")
            with open(hls_file, "wb") as wfd:
                for f in file_names:
                    with open(os.path.join(tmp_dir, f), "rb") as fd:
                        shutil.copyfileobj(fd, wfd)
        else:
            hls_file = os.path.join(tmp_dir, file_names[0])
        
        # Convert to wav using ffmpeg
        wav_file_path = os.path.join(tmp_dir, f"{clipname}.wav")

        # Compute offset (seconds) into the concatenated .ts where the desired start occurs
        ss_offset = time_since_folder_start_for_start - (segment_start_index * target_duration)
        if ss_offset < 0:
            ss_offset = 0.0

        # Use input seeking and limit duration to 60 seconds
        stream = ffmpeg.input(hls_file, ss=ss_offset)
        stream = ffmpeg.output(
            stream,
            wav_file_path,
            t=60,  # 60 seconds
            acodec="pcm_s16le",
            ar=44100,
            ac=1
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        print(f"  Downloaded 60s audio: {wav_file_path}")
        return wav_file_path
        
    except Exception as e:
        print(f"  Warning: Unable to retrieve audio clip: {e}")
        return None


def compute_correct_timestamp_for_tp_human_only(
    sample: Dict, 
    model_inference,
    tmp_dir: str
) -> str:
    """
    Compute the correct timestamp for a tp_human_only detection.
    
    For tp_human_only detections, we download the preceding 60 seconds of audio,
    run the model on it to score each 2-second segment, find the highest scoring
    segment, and adjust the timestamp accordingly.
    
    Args:
        sample: Detection sample dictionary
        model_inference: Model inference instance
        tmp_dir: Temporary directory for audio files
    
    Returns:
        Corrected timestamp string
    """
    node_name = sample['NodeName']
    timestamp_str = sample['Timestamp']
    
    print(f"Computing correct timestamp for tp_human_only: {node_name} - {timestamp_str}")
    
    # Download 60 seconds of audio
    wav_path = download_60s_audio(node_name, timestamp_str, tmp_dir)
    
    if wav_path is None:
        print(f"  Failed to download audio, falling back to 2-second offset")
        return subtract_two_seconds(timestamp_str)
    
    try:
        # Run model inference
        print(f"  Running model inference...")
        prediction_results = model_inference.predict(wav_path)
        
        local_confidences = prediction_results["local_confidences"]
        print(f"  Model returned {len(local_confidences)} segments")
        
        if not local_confidences:
            print(f"  No confidences returned, falling back to 2-second offset")
            return subtract_two_seconds(timestamp_str)
        
        # Find the index of the highest scoring 2-second segment
        max_confidence_idx = local_confidences.index(max(local_confidences))
        max_confidence = local_confidences[max_confidence_idx]
        
        print(f"  Highest score: {max_confidence} at segment {max_confidence_idx}")
        
        # Each segment is 2 seconds, so the offset from the end is:
        # (total_segments - max_confidence_idx - 1) * 2 + 2 seconds
        # This gives us how many seconds before the original timestamp
        num_segments = len(local_confidences)
        seconds_before_end = (num_segments - max_confidence_idx - 1) * 2 + 2
        
        # Ensure offset is between 2 and 60 seconds
        seconds_before_end = max(2, min(60, seconds_before_end))
        
        print(f"  Adjusting timestamp by {seconds_before_end} seconds")
        
        # Subtract the offset from the original timestamp
        dt = parse_timestamp(timestamp_str)
        dt_adjusted = dt - timedelta(seconds=seconds_before_end)
        return format_timestamp(dt_adjusted)
        
    except Exception as e:
        print(f"  Error during inference: {e}")
        print(f"  Falling back to 2-second offset")
        return subtract_two_seconds(timestamp_str)
    finally:
        # Clean up the downloaded audio file
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except:
                pass


def write_training_samples(samples: List[Dict], output_path: Path, model_inference=None):
    """
    Write selected samples to CSV with timestamps adjusted.
    
    Args:
        samples: List of sample dictionaries
        output_path: Path to output CSV file
        model_inference: Optional model inference instance for tp_human_only timestamp correction
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory for audio downloads
    with TemporaryDirectory() as tmp_dir:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Use same columns as detections.csv
            fieldnames = ['Category', 'NodeName', 'Timestamp', 'URI', 'Description', 'Notes']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for sample in samples:
                # Create a copy and adjust timestamp
                output_row = sample.copy()
                
                # For tp_human_only detections, use model-based timestamp correction
                if sample['Notes'] == 'tp_human_only' and model_inference is not None:
                    output_row['Timestamp'] = compute_correct_timestamp_for_tp_human_only(
                        sample, model_inference, tmp_dir
                    )
                else:
                    # For all other detections, subtract 2 seconds
                    output_row['Timestamp'] = subtract_two_seconds(sample['Timestamp'])
                
                writer.writerow(output_row)


def main():
    """Main function to extract training samples."""
    # Paths
    input_path = Path('output_segments/detections.csv')
    output_path = Path('output_segments/training_samples.csv')
    
    print(f"Loading detections from {input_path}...")
    detections = load_detections(input_path)
    print(f"Loaded {len(detections)} detections")
    
    print("\nOrganizing detections by category and node...")
    organized_data = organize_by_category_node(detections)
    
    # Print summary
    for category in sorted(organized_data.keys()):
        total = sum(len(nodes) for nodes in organized_data[category].values())
        print(f"  {category}: {total} detections across {len(organized_data[category])} nodes")
    
    print("\nSelecting training samples...")
    samples = select_training_samples(organized_data)
    
    print(f"\nSelected {len(samples)} training samples")
    
    # Print breakdown by category
    category_counts = defaultdict(int)
    category_node_counts = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        category_counts[sample['Category']] += 1
        category_node_counts[sample['Category']][sample['NodeName']] += 1
    
    for category in sorted(category_counts.keys()):
        print(f"  {category}: {category_counts[category]} samples")
        for node in sorted(category_node_counts[category].keys()):
            print(f"    {node}: {category_node_counts[category][node]}")

    # Print breakdown by type
    type_counts = defaultdict(int)
    type_node_counts = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        type_counts[sample['Notes']] += 1
        type_node_counts[sample['Notes']][sample['NodeName']] += 1
    
    for type in sorted(type_counts.keys()):
        print(f"  {type}: {type_counts[type]} samples")
        for node in sorted(type_node_counts[type].keys()):
            print(f"    {node}: {type_node_counts[type][node]}")

    # Initialize model inference for tp_human_only timestamp correction
    print("\nInitializing model inference for tp_human_only timestamp correction...")
    
    # Check for model configuration from environment variables
    model_type = os.environ.get("MODEL_TYPE", "dummy")
    model_path = os.environ.get("MODEL_PATH", "./model")
    auto_download = os.environ.get("MODEL_AUTO_DOWNLOAD", "false").lower() == "true"
    
    print(f"  Model type: {model_type}")
    if model_type == "fastai":
        print(f"  Model path: {model_path}")
        print(f"  Auto download: {auto_download}")
        print(f"  Note: To use the FastAI model, set environment variables:")
        print(f"    MODEL_TYPE=fastai")
        print(f"    MODEL_PATH=./model")
        print(f"    MODEL_AUTO_DOWNLOAD=true (to download model automatically)")
    
    try:
        model_inference = get_model_inference(
            model_path=model_path if model_type == "fastai" else None,
            model_type=model_type,
            auto_download=auto_download
        )
    except Exception as e:
        print(f"  Warning: Failed to initialize model inference: {e}")
        print(f"  Will fall back to 2-second offset for tp_human_only samples")
        model_inference = None

    print(f"\nWriting training samples to {output_path}...")
    write_training_samples(samples, output_path, model_inference)
    print("Done!")


if __name__ == "__main__":
    main()
