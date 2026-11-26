# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Common utilities for audio downloading and S3 operations.

This module contains shared functions used by download_wavs.py and potentially
other scripts to avoid code duplication.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import http.client
from pathlib import Path
from typing import List, Optional
import os
import time
import urllib.error

from azure.cosmos import CosmosClient
import requests
try:
    from orcasite_feeds import OrcasiteFeed
except ImportError:  # pragma: no cover
    from src.orcasite_feeds import OrcasiteFeed
from orca_hls import get_scraper, get_s3_utils
from orca_hls_utils.DateRangeHLSStream import DateRangeHLSStream
from pytz import timezone

scraper = get_scraper()
s3_utils = get_s3_utils()

# Simple in-memory cache for S3 folder listings keyed by "bucket::prefix"
_FOLDERS_CACHE = {}
PACIFIC_TZ = timezone('America/Los_Angeles')
UTC_TZ = timezone('UTC')
COSMOS_URL = os.environ.get("COSMOS_URL", "").strip() or "https://aifororcasmetadatastore.documents.azure.com:443/"
COSMOS_KEY = os.environ.get("COSMOS_KEY", "").strip()
COSMOS_DB = os.environ.get("COSMOS_DB", "predictions")
COSMOS_CONTAINER = os.environ.get("COSMOS_CONTAINER", "metadata")


@dataclass
class OrcaHelloDetection:
    id: str
    feed: OrcasiteFeed
    timestamp: Optional[datetime]
    status: str
    confidence: Optional[float] = None
    tags: str = ""
    comments: str = ""
    modelId: Optional[str] = None


# Terms in a detection description that indicate the label cannot be determined with confidence.
SKIP_TERMS = {'?', 'not sure', 'unsure', 'possibl', 'sounds like', 'sounded like', 'ould be'}


def get_all_folders(bucket: str, prefix: str) -> List[str]:
    """
    Get all folder names from an S3 bucket with the given prefix.
    
    Parameters:
        bucket (str): Name of the S3 bucket.
        prefix (str): Prefix to filter objects.
    
    Returns:
        List[str]: List of folder names (without the prefix).
    """
    return s3_utils.get_all_folders(bucket, prefix)


def get_cached_folders(bucket: str, prefix: str) -> List[str]:
    """
    Return cached folder list for (bucket, prefix). If absent, call get_all_folders()
    to populate the cache, then return the cached value.
    """
    key = f"{bucket}::{prefix}"
    if key not in _FOLDERS_CACHE:
        _FOLDERS_CACHE[key] = get_all_folders(bucket, prefix)
    return _FOLDERS_CACHE[key]


def get_difference_between_times_in_seconds(later_unix_time: int, earlier_unix_time: int) -> float:
    """
    Calculate the difference between two unix timestamps in seconds.
    
    Parameters:
        later_unix_time (int): Later unix timestamp.
        earlier_unix_time (int): Earlier unix timestamp.
    
    Returns:
        float: Difference in seconds.
    """
    return later_unix_time - earlier_unix_time


def download_from_url(dl_url: str, dl_dir: str):
    """Download a URL with retries for transient network failures."""
    for attempt in range(3):
        try:
            return scraper.download_from_url(dl_url, dl_dir)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            urllib.error.URLError,
            http.client.IncompleteRead,
        ) as error:
            (Path(dl_dir) / os.path.basename(dl_url)).unlink(missing_ok=True)
            if attempt == 2 or (
                isinstance(error, urllib.error.HTTPError)
                and error.code < 500
                and error.code != 429
            ):
                raise
            time.sleep(1)


def format_timestamp_pacific(dt: datetime) -> str:
    """Format a datetime in Pacific time as ``YYYY_MM_DD_HH_MM_SS_{TZ}``."""
    dt_pacific = dt.astimezone(PACIFIC_TZ)
    return dt_pacific.strftime("%Y_%m_%d_%H_%M_%S_%Z")


def parse_timestamp_pacific(ts_str: str) -> datetime:
    """
    Parse a Pacific timestamp string into a timezone-aware datetime.
    """
    if not ts_str.upper().endswith(("_PST", "_PDT")):
        raise ValueError(f"Timestamp '{ts_str}' must end with '_PST' or '_PDT'")
    body, abbreviation = ts_str.upper().rsplit("_", 1)
    dt_naive = datetime.strptime(body, "%Y_%m_%d_%H_%M_%S")
    return PACIFIC_TZ.localize(dt_naive, is_dst=abbreviation == "PDT")

def get_node_name_for_feed(feed: OrcasiteFeed) -> str:
    """Retrieve the node name associated with an OrcasiteFeed."""
    return feed.node_name


def get_machine_detections(
    feed: OrcasiteFeed,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[OrcaHelloDetection]:
    """
    Retrieve machine detections associated with the given feed's node name.
    """
    if not COSMOS_KEY:
        raise ValueError("COSMOS_KEY environment variable must be set and non-empty to fetch machine detections")

    node_name = get_node_name_for_feed(feed)

    cosmos_client = CosmosClient(COSMOS_URL, credential=COSMOS_KEY)
    cosmos_database = cosmos_client.get_database_client(COSMOS_DB)
    container = cosmos_database.get_container_client(COSMOS_CONTAINER)

    query = """
        SELECT * FROM c
        WHERE c.location.id = @node_name
    """
    params = [{"name": "@node_name", "value": node_name}]
    if start_time:
        query += " AND c.timestamp >= @start_time"
        params.append({"name": "@start_time", "value": start_time.isoformat()})
    if end_time:
        query += " AND c.timestamp <= @end_time"
        params.append({"name": "@end_time", "value": end_time.isoformat()})
    query += " ORDER BY c.timestamp DESC"

    items = container.query_items(
        query=query,
        parameters=params,
        enable_cross_partition_query=True
    )

    results = []
    for item in items:
        found = item.get("SRKWFound", "").lower()
        reviewed = item.get("reviewed", False)
        if reviewed and found == "yes":
            status = "confirmed"
        elif reviewed and found == "no":
            status = "rejected"
        else:
            status = "unreviewed"

        modelId = item.get("modelId")
        modelId = modelId.lower() if isinstance(modelId, str) else ""

        ts_raw = item.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            ts = None

        confidence = item.get("whaleFoundConfidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = None

        results.append(
            OrcaHelloDetection(
                id=item.get("id"),
                feed=feed,
                timestamp=ts,
                status=status,
                confidence=confidence,
                tags=(item.get("tags") or "").strip(),
                comments=(item.get("comments") or "").strip(),
                modelId=modelId,
            )
        )

    return results


def _get_aligned_end_time(timestamp_pst_str: str) -> datetime:
    raw_end = parse_timestamp_pacific(timestamp_pst_str)
    snapped_sec = ((raw_end.second + 9) // 10) * 10
    if snapped_sec == 60:
        # roll over to next minute.
        raw_end = raw_end + timedelta(minutes=1)
        snapped_sec = 0
    return raw_end.replace(second=snapped_sec, microsecond=0)


def download_60s_audio(node_name: str, min_end_timestamp_pst_str: str, tmp_dir: str) -> Optional[str]:
    """
    Download 60 seconds of audio ending at the next 10-second boundary at or after the corrected timestamp.
    """
    # Compute the aligned end time and derive the 60s clip start UTC to reuse
    # the download_audio_from_start_utc implementation which handles the
    # HLS folder/segment logic. This reduces duplication and keeps behavior
    # consistent between callers that request clips by end-time vs start-time.
    end_time = _get_aligned_end_time(min_end_timestamp_pst_str)
    start_time = end_time - timedelta(seconds=60)
    start_time_utc = start_time.astimezone(UTC_TZ)

    return download_audio_from_start_utc(node_name, start_time_utc, tmp_dir, 60)


def download_60s_audio_from_start(
    node_name: str,
    start_timestamp_pst_str: str,
    tmp_dir: str,
) -> Optional[str]:
    """Download 60 seconds of audio from the corrected start timestamp."""
    start_time_utc = parse_timestamp_pacific(start_timestamp_pst_str).astimezone(UTC_TZ)
    return download_audio_from_start_utc(node_name, start_time_utc, tmp_dir, 60)


def format_utc_iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC_TZ).strftime("%Y-%m-%dT%H:%M:%SZ")


def download_audio_from_start_utc(
    node_name: str,
    start_time_utc: datetime,
    tmp_dir: str,
    duration_seconds: float,
) -> Optional[str]:
    """Download a clip beginning at start_time_utc and lasting for duration_seconds."""
    try:
        start_time_utc = (
            start_time_utc.replace(tzinfo=UTC_TZ)
            if start_time_utc.tzinfo is None
            else start_time_utc.astimezone(UTC_TZ)
        )
        start_timestamp = start_time_utc.timestamp()
        stream = DateRangeHLSStream(
            f"https://s3-us-west-2.amazonaws.com/audio-orcasound-net/{node_name}",
            duration_seconds,
            start_timestamp,
            start_timestamp + duration_seconds,
            tmp_dir,
            overwrite_output=True,
        )
        wav_file_path, _, _ = stream.get_next_clip()
        if wav_file_path is None:
            print(f"  Warning: No audio found for {start_time_utc}")
            return None
        print(f"  Downloaded {duration_seconds:g}s audio: {wav_file_path}")
        return wav_file_path
    except Exception as e:
        print(f"  Warning: Unable to retrieve audio clip: {e}")
        return None
