# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import csv
import sys
from urllib.parse import unquote, parse_qs, urlparse

from orca_hls_utils.DateRangeHLSStream import DateRangeHLSStream
from pytz import timezone

PACIFIC_TZ = timezone('US/Pacific')
N_SECONDS = 10  # or whatever

@dataclass
class CSVRow:
    category: str
    node: str
    node_name: str
    timestamp_pst: str
    uri: str

def parse_csv(csv_path: Path) -> List[CSVRow]:
    """
    Parse the detections CSV file and return a list of CSVRow objects.
    
    Parameters:
        csv_path (Path): Path to the detections CSV file.
    
    Returns:
        List[CSVRow]: List of parsed CSV rows.
    """
    rows = []
    with open(csv_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip header
        next(csv_reader)
        for row in csv_reader:
            if len(row) >= 5:
                rows.append(CSVRow(
                    category=row[0],
                    node=row[1],
                    node_name=row[2],
                    timestamp_pst=row[3],
                    uri=row[4]
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
    # Remove _PST suffix if present
    timestamp_str = timestamp_str.replace('_PST', '')
    # Parse the datetime
    dt_naive = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
    # Localize to Pacific timezone
    dt_aware = PACIFIC_TZ.localize(dt_naive)
    return dt_aware

def extract_node_name_from_uri(uri: str) -> str:
    """
    Extract the node name from the Orcasound bouts URI.
    
    Parameters:
        uri (str): URI in the format "https://live.orcasound.net/bouts/new/{node}?time={utc_time}".
    
    Returns:
        str: The node name (e.g., "andrews-bay").
    """
    # Parse the URI to get the path
    parsed = urlparse(uri)
    path_parts = parsed.path.split('/')
    # The node is the last part of the path
    if len(path_parts) >= 4:  # ['', 'bouts', 'new', 'node']
        return path_parts[-1]
    return ""

def download_audio_segment(
    category: str,
    node_name: str,
    timestamp_pst: datetime,
    output_root: Path,
):
    """
    Download an audio segment for a detection and save it to the appropriate label directory.
    
    Parameters:
        category (str): The label/category for the detection (e.g., "resident", "transient").
        node_name (str): The node name (e.g., "rpi_sunset_bay").
        timestamp_pst (datetime): The detection timestamp in Pacific time.
        output_root (Path): Root directory where label subdirectories and audio files will be saved.
    """
    label_dir = output_root / category
    
    hls_polling_interval = 60
    hls_hydrophone_id = node_name
    hydrophone_stream_url = 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/' + hls_hydrophone_id
    
    start_time = timestamp_pst
    duration_s = N_SECONDS
    end_time = start_time + timedelta(seconds=duration_s)
    
    hls_start_time_unix = int(start_time.timestamp())
    hls_end_time_unix = int(end_time.timestamp())
    
    try:
        hls_stream = DateRangeHLSStream(hydrophone_stream_url, hls_polling_interval, hls_start_time_unix, hls_end_time_unix, label_dir, False)
    except IndexError as e:
        print("\nERROR: Failed to initialize DateRangeHLSStream.")
        print("This usually means the S3 folder list is malformed or unsorted.")
        print(f"Details: {e}")
        print(f"Hydrophone: {hls_hydrophone_id}")
        print(f"Start time (unix): {hls_start_time_unix}")
        print(f"End time (unix)  : {hls_end_time_unix}")
        sys.exit(0)
    
    try:
        # DateRangeHLSStream currently requires a naive UTC end time, not one already set to UTC.
        utc_tz = timezone('UTC')
        end_time_utc = end_time.astimezone(utc_tz)
        naive_end_time_utc = end_time_utc.replace(tzinfo=None)
        
        clip_path, start_timestamp, next_clip_end_time = hls_stream.get_next_clip(naive_end_time_utc)
        print(f"Downloaded: {clip_path}")
    except (IndexError, ValueError) as e:
        # Handle case when no audio files exist for the specified time range
        print(f"\nWarning: Unable to retrieve audio clip. This may occur when no audio files exist for the specified time range.")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        print(f"Hydrophone: {hls_hydrophone_id}")

def process_csv(csv_path: Path, output_root: Path):
    """
    Read the detections CSV file and download corresponding WAV files.
    
    Parameters:
        csv_path (Path): Path to the detections.csv file.
        output_root (Path): Root directory where audio files will be saved in label subdirectories.
    """
    rows = parse_csv(csv_path)
    
    print(f"Found {len(rows)} detections to process")
    
    for row in rows:
        print(f"Processing: {row.category} - {row.node} - {row.timestamp_pst}")
        timestamp_pst = parse_timestamp_pst(row.timestamp_pst)
        download_audio_segment(row.category, row.node_name, timestamp_pst, output_root)

if __name__ == "__main__":
    csv_path = Path("output_segments/detections.csv")
    output_root = Path("output_segments")
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run make_csv.py first to generate the detections.csv file.")
        sys.exit(1)
    
    process_csv(csv_path, output_root)
