# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import sys

from orca_hls_utils.DateRangeHLSStream import DateRangeHLSStream
from pytz import timezone
import requests
@dataclass
class OrcasiteFeed:
    id: str                     # e.g., "feed_02u8r4EPgmlYQmh6gzlGIL"
    name: str                   # "Beach Camp at Sunset Bay"
    node_name: str              # "rpi_sunset_bay"
    slug: str                   # "sunset-bay"
    bucket: str                 # "audio-orcasound-net"
    bucket_region: str          # "us-west-2"
    visible: bool               # True/False
    location: Tuple[float, float]  # (lat, lng)
    image_url: Optional[str] = None
    cloudfront_url: Optional[str] = None

@dataclass
class OrcasiteDetection:
    id: str
    feed: OrcasiteFeed
    timestamp: datetime          # center or start time (we'll define)
    source: str                  # "machine" | "human"
    category: str                # e.g., "whale", "vessel", "other", "none"
    description: str             # free text; may be ""
    # extra fields as needed

@dataclass
class OrcaHelloDetection:
    id: str
    feed: OrcasiteFeed
    timestamp: datetime
    status: str                  # e.g., "confirmed", "rejected", etc.
    # extra fields as needed

def get_label(
    orcasite_det: OrcasiteDetection,
    orcahello_det: Optional[OrcaHelloDetection],
) -> Optional[str]:
    """
    Returns one of: 'resident', 'transient', 'humpback', 'other', or None (unknown).
    """
    desc = (orcasite_det.description or "").lower()
    cat = (orcasite_det.category or "").lower()

    # 1. Resident
    if cat == "whale":
        if ("resident" in desc) or ("pod" in desc):
            return "resident"
        if orcahello_det and orcahello_det.status.lower() == "confirmed":
            return "resident"

    # 2. Transient
    if cat == "whale":
        if ("bigg" in desc) or ("transient" in desc):
            return "transient"

    # 3. Humpback
    if cat == "whale":
        if "humpback" in desc:
            return "humpback"

    # 4. Other
    if cat != "whale" and orcasite_det.source == "machine":
        return "other"

    # Unknown
    return None

def index_orcahello_by_time(
    detections: List[OrcaHelloDetection],
    max_delta: timedelta = timedelta(minutes=2),
):
    """
    Returns a function that maps (feed, time) -> best matching OrcaHelloDetection or None.
    For simplicity, do a linear scan (optimize later if needed).
    """
    def find_match(feed: OrcasiteFeed, t: datetime) -> Optional[OrcaHelloDetection]:
        best = None
        best_dt = max_delta
        for d in detections:
            if d.feed.id != feed.id:
                continue
            dt = abs(d.timestamp - t)
            if dt <= best_dt:
                best_dt = dt
                best = d
        return best

    return find_match

FIVE_MIN = timedelta(minutes=5)

def is_isolated_human_whale(
    det: OrcasiteDetection,
    all_detections: List[OrcasiteDetection],
) -> bool:
    """
    True if there is NO machine whale detection at same feed within ±5 minutes.
    """
    for d in all_detections:
        if d.feed.id != det.feed.id:
            continue
        if d.source != "machine":
            continue
        if (d.category or "").lower() != "whale":
            continue
        if abs(d.timestamp - det.timestamp) <= FIVE_MIN:
            return False
    return True


@dataclass
class Classification:
    kind: str       # 'tp_human_only', 'tp_machine_only', 'fp_machine_only', 'skip'
    include: bool

def classify_detection(
    det: OrcasiteDetection,
    label: str,
    all_detections: List[OrcasiteDetection],
) -> Classification:
    cat = (det.category or "").lower()
    src = (det.source or "").lower()

    # Skip non-whale human / non-machine things already filtered by label logic
    if label is None:
        return Classification(kind="skip", include=False)

    # False positive, machine-only (label 'other')
    if label == "other" and src == "machine":
        return Classification(kind="fp_machine_only", include=True)

    # True positive, human-only (include)
    if cat == "whale" and src == "human":
        if is_isolated_human_whale(det, all_detections):
            return Classification(kind="tp_human_only", include=True)
        else:
            # There is a nearby machine whale detection → not human-only
            # You can choose to skip or treat differently.
            return Classification(kind="skip", include=False)

    # True positive, machine-only (maybe include)
    if cat == "whale" and src == "machine":
        # You can gate this on some flag if 'maybe include' is optional.
        return Classification(kind="tp_machine_only", include=True)

    # False positive, human-only and true negative are effectively already skipped
    return Classification(kind="skip", include=False)

N_SECONDS = 10  # or whatever

def get_segments_for_detection(det: OrcasiteDetection) -> List[tuple[datetime, int]]:
    """
    Return a list of (start_time, duration_seconds) tuples.
    You can compute start_time as det.timestamp - N/2 etc.
    """
    start_time = det.timestamp  # or det.timestamp - timedelta(seconds=N_SECONDS/2)
    return [(start_time, N_SECONDS)]


def extract_and_save_segments(
    det: OrcasiteDetection,
    label: str,
    segments: List[tuple[datetime, int]],
    output_root: Path,
):
    """
    Extract audio segments for a detection and save them to disk.
    """
    label_dir = output_root / label

    for idx, (start_time, duration_s) in enumerate(segments):
        hls_polling_interval=60
        hls_hydrophone_id=det.feed.node_name
        hydrophone_stream_url = 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/' + hls_hydrophone_id

        pacific = timezone('US/Pacific')
        start_dt_aware = det.timestamp.astimezone(pacific)
        hls_start_time_unix = int(start_dt_aware.timestamp())

        end_time = start_time + timedelta(seconds=duration_s)
        end_dt_aware = end_time.astimezone(pacific)
        hls_end_time_unix = int(end_dt_aware.timestamp())

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
            naive_end_time_utc = end_time.replace(tzinfo=None)

            clip_path, start_timestamp, next_clip_end_time = hls_stream.get_next_clip(naive_end_time_utc)
        except (IndexError, ValueError) as e:
            # Handle case when no audio files exist for the specified time range
            print(f"\nWarning: Unable to retrieve audio clip. This may occur when no audio files exist for the specified time range.")
            print(f"Error details: {type(e).__name__}: {str(e)}")
            print(f"Hydrophone: {hls_hydrophone_id}")

def get_orcasite_feeds() -> List[OrcasiteFeed]:
    """
    Fetch all Orcasite feeds and return them as a list of OrcasiteFeed instances.
    """
    url = "https://live.orcasound.net/api/json/feeds"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        feeds = []
        for item in data.get("data", []):
            attrs = item.get("attributes", {})

            lat = attrs.get("lat_lng", {}).get("lat")
            lng = attrs.get("lat_lng", {}).get("lng")

            feed = OrcasiteFeed(
                id=item.get("id"),
                name=attrs.get("name"),
                node_name=attrs.get("node_name"),
                slug=attrs.get("slug"),
                bucket=attrs.get("bucket"),
                bucket_region=attrs.get("bucket_region"),
                visible=attrs.get("visible", True),
                location=(lat, lng),
                image_url=attrs.get("image_url"),
                cloudfront_url=attrs.get("cloudfront_url"),
            )

            feeds.append(feed)

        return feeds

    except Exception as e:
        print("Error fetching Orcasite feeds:", e)
        return []

def get_orcasite_detections(feed: OrcasiteFeed) -> List[OrcasiteDetection]:
    """
    Fetch Orcasite detections for a specific feed.
    Returns a list of Detection objects.
    """

    # Base endpoint
    base_url = "https://live.orcasound.net/api/json/detections"

    # Fields we want (already URL-encoded in your example)
    fields = (
        "id,source_ip,playlist_timestamp,player_offset,"
        "listener_count,timestamp,description,visible,"
        "source,category,candidate_id,feed_id"
    )

    # Build query parameters
    params = {
        "fields[detection]": fields,
        "filter[feed_id]": feed.id,
        # You *said* "&source]=whale" but that looks like a typo.
        # If you want only whale detections, use:
        # "filter[category]": "whale"
        #
        # But since your sample includes non-whale detections,
        # I will NOT filter category here unless you tell me to.
    }

    try:
        r = requests.get(base_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        dets = []
        for item in data.get("data", []):
            attrs = item.get("attributes", {})

            # Parse timestamp safely
            ts_raw = attrs.get("timestamp")
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                ts = None

            det = OrcasiteDetection(
                id=item.get("id"),
                feed=feed,
                timestamp=ts,
                source=attrs.get("source"),
                category=attrs.get("category"),
                description=attrs.get("description") or "",
            )

            # Only include detections for THIS feed
            if det.feed.id == feed.id:
                dets.append(det)

        return dets

    except Exception as e:
        print(f"Error fetching detections for feed {feed.id}: {e}")
        return []

def get_node_name_for_feed(feed: OrcasiteFeed) -> str:
    return feed.node_name

def get_orcahello_detections(feed: OrcasiteFeed) -> List[OrcaHelloDetection]:
    """
    Fetch OrcaHello detections and filter them to only those belonging
    to the given feed (via node_name match in audioUri).
    """

    node_name = get_node_name_for_feed(feed)  # e.g., "rpi_sunset_bay"

    url = "https://aifororcasdetections.azurewebsites.net/api/detections"
    params = {
        "Page": 1,
        "SortBy": "timestamp",
        "SortOrder": "desc",
        "Timeframe": "1w",
        "Location": "all",
        "RecordsPerPage": 500,   # get more so we don't miss matches
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        results = []

        for item in data:
            audio_uri = item.get("audioUri", "")

            # Filter by node_name inside the audio filename
            if node_name not in audio_uri:
                continue

            found = item.get("found", "").lower()
            reviewed = item.get("reviewed", False)

            if reviewed and found == "yes":
                status = "confirmed"
            elif reviewed and found == "no":
                status = "rejected"
            else:
                status = "unreviewed"

            # Parse timestamp
            ts_raw = item.get("timestamp")
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                ts = None

            det = OrcaHelloDetection(
                id=item.get("id"),
                feed=feed,
                timestamp=ts,
                status=status,
            )

            results.append(det)

        return results

    except Exception as e:
        print(f"Error fetching OrcaHello detections for feed {feed.id}: {e}")
        return []

def process_all_feeds(output_root: Path):
    feeds = get_orcasite_feeds()

    for feed in feeds:
        print(f"Processing feed {feed.id} ({feed.node_name})")
        orcasite_dets = get_orcasite_detections(feed)
        orcahello_dets = get_orcahello_detections(feed)

        # Create time-based matcher for this feed's OrcaHello detections
        find_oh_match = index_orcahello_by_time(orcahello_dets)

        for det in orcasite_dets:
            # Only care about source=machine or human as per your logic
            orcahello_match = None
            if det.source == "machine":
                orcahello_match = find_oh_match(det.feed, det.timestamp)

            label = get_label(det, orcahello_match)
            if label is None:
                # label unknown ⇒ skip
                continue

            classification = classify_detection(det, label, orcasite_dets)
            if not classification.include:
                continue

            segments = get_segments_for_detection(det)
            extract_and_save_segments(det, label, segments, output_root)

if __name__ == "__main__":
    output_root = Path("output_segments")
    process_all_feeds(output_root)
