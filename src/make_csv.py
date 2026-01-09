# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import csv
from urllib.parse import quote

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
    Derives a normalized label for an OrcasiteDetection based on its category, description, and an optional OrcaHelloDetection match.
    
    Parameters:
        orcasite_det (OrcasiteDetection): The detection to label.
        orcahello_det (Optional[OrcaHelloDetection]): An optional time-matched OrcaHello detection used to upgrade some whale labels.
    
    Returns:
        str | None: `'resident'`, `'transient'`, `'humpback'`, or `'other'` when a label can be determined; `None` when the label is unknown.
    """
    desc = (orcasite_det.description or "").lower()
    cat = (orcasite_det.category or "").lower()

    if orcahello_det and orcahello_det.status.lower() == "unreviewed":
        return None

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
    if orcasite_det.source == "machine" and not ("click" in desc or "call" in desc or "whistle" in desc):
        return "other"
    # Unknown
    return None

def index_orcahello_by_time(
    detections: List[OrcaHelloDetection],
    max_delta: timedelta = timedelta(minutes=2),
):
    """
    Returns a function that maps (feed, time) -> best matching OrcaHelloDetection or None.
    Only matches detections whose timestamp is <= t.
    """
    def find_match(feed: OrcasiteFeed, t: datetime) -> Optional[OrcaHelloDetection]:
        best = None
        best_dt = max_delta

        for d in detections:
            if d.feed.id != feed.id:
                continue

            # Only consider detections at or BEFORE t
            if d.timestamp > t:
                continue

            dt = t - d.timestamp  # guaranteed non-negative

            if dt <= best_dt:
                best_dt = dt
                best = d

        return best

    return find_match

FIVE_MIN = timedelta(minutes=5)
PACIFIC_TZ = timezone('US/Pacific')
UTC_TZ = timezone('UTC')

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
    """
    Assigns a Classification to an OrcasiteDetection based on its label, source, category, and nearby detections.
    
    Parameters:
        det (OrcasiteDetection): The detection to classify.
        label (str): The computed label for the detection (e.g., 'resident', 'transient', 'humpback', 'other', or None).
        all_detections (List[OrcasiteDetection]): All detections for the same feed used to determine temporal context.
    
    Returns:
        Classification: An object with `kind` set to one of:
            - 'tp_human_only' for human-sourced whale detections with no nearby machine whale detections,
            - 'tp_machine_only' for machine-sourced whale detections,
            - 'fp_machine_only' for machine-sourced non-whale detections labeled 'other',
            - 'skip' for detections that should be excluded.
        The `include` field indicates whether the detection should be included in downstream processing (`true` if included, `false` otherwise).
    """
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

def get_orcasite_feeds() -> List[OrcasiteFeed]:
    """
    Fetch feeds from the Orcasite API and parse them into a list of OrcasiteFeed objects.
    
    Each feed includes metadata such as id, name, node_name, slug, storage bucket info, visibility, geographic location (latitude, longitude), and optional image and CloudFront URLs. If the API request fails or the response cannot be parsed, an empty list is returned.
    
    Returns:
        List[OrcasiteFeed]: A list of parsed feed objects; empty if fetching or parsing fails.
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
    Fetch detections from the Orcasite API for the given feed.
    
    Parses API response into a list of OrcasiteDetection objects filtered to the provided feed. Timestamps are parsed from ISO strings and will be None if parsing fails. On network or parsing errors the function prints an error and returns an empty list.
    
    Parameters:
        feed (OrcasiteFeed): Feed whose detections should be retrieved and matched by feed.id.
    
    Returns:
        List[OrcasiteDetection]: Detections associated with the specified feed (may be empty).
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
    """
    Retrieve the node name associated with an OrcasiteFeed.
    
    Returns:
        node_name (str): The feed's node_name.
    """
    return feed.node_name

def get_orcahello_name_for_feed(feed: OrcasiteFeed) -> str:
    """
    Retrieve the OrcaHello name associated with an OrcasiteFeed.
    
    Returns:
        name (str): The feed's name in OrcaHello format.
    """
    name = feed.name
    if " at " in name:
        name = name.split(" at ", 1)[1]
    if name == "MaST Center Aquarium":
        return "Mast Center"
    return name

def get_orcahello_detections(feed: OrcasiteFeed) -> List[OrcaHelloDetection]:
    """
    Retrieve OrcaHello detections and return those whose audio URI contains the given feed's node_name.
    
    Fetches recent detections from the OrcaHello API, filters results to entries whose `audioUri` includes the feed's node_name, maps review fields to a `status` of "confirmed", "rejected", or "unreviewed", and parses the detection timestamp when possible.
    
    Implements pagination to fetch all available detections by incrementing the Page parameter until an empty page is returned.
    
    Parameters:
        feed (OrcasiteFeed): Feed whose node_name is used to filter OrcaHello detection audio URIs.
    
    Returns:
        List[OrcaHelloDetection]: A list of detections associated with the feed. Each detection's `timestamp` is a `datetime` or `None` if parsing failed, and `status` is one of `"confirmed"`, `"rejected"`, or `"unreviewed"`.
    """

    node_name = get_node_name_for_feed(feed)  # e.g., "rpi_sunset_bay"

    orcahello_name = get_orcahello_name_for_feed(feed)  # e.g., "Sunset Bay"

    url = "https://aifororcasdetections.azurewebsites.net/api/detections"
    
    # Collect all detections from all pages before filtering
    all_items = []
    page = 1
    
    while True:
        params = {
            "Page": page,
            "SortBy": "timestamp",
            "SortOrder": "desc",
            "Timeframe": "1m",
            "Location": orcahello_name,
            "RecordsPerPage": 50,   # API max is 50
        }
        
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            # If the page is empty, we've fetched all available detections
            if not data:
                break
            
            all_items.extend(data)
            page += 1
            
        except Exception as e:
            print(f"Error fetching OrcaHello detections page {page} for feed {feed.id}: {e}")
            break
    
    # Now filter and process all collected items
    results = []
    
    for item in all_items:
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

def format_timestamp_pst(dt: datetime) -> str:
    """
    Format a datetime object as PST timestamp string in the format YYYY_MM_DD_HH_MM_SS_PST.
    
    Parameters:
        dt (datetime): The datetime to format (should be timezone-aware).
    
    Returns:
        str: Formatted timestamp string (e.g., "2025_12_24_17_51_23_PST").
    """
    dt_pst = dt.astimezone(PACIFIC_TZ)
    return dt_pst.strftime("%Y_%m_%d_%H_%M_%S_PST")

def generate_uri(node: str, dt: datetime) -> str:
    """
    Generate a URI for the Orcasound bouts interface.
    
    Parameters:
        node (str): The node name (e.g., "andrews-bay").
        dt (datetime): The datetime in UTC.
    
    Returns:
        str: URI in the format "https://live.orcasound.net/bouts/new/{node}?time={utc_time}".
    """
    # Ensure the datetime is in UTC
    utc_dt = dt.astimezone(UTC_TZ)
    # Format as ISO 8601 with milliseconds and Z suffix
    time_str = utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    # URL encode the time parameter
    time_encoded = quote(time_str, safe='')
    return f"https://live.orcasound.net/bouts/new/{node}?time={time_encoded}"

def process_all_feeds(output_root: Path):
    feeds = get_orcasite_feeds()
    
    # Create CSV file
    csv_path = output_root / "detections.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Category', 'NodeName', 'Timestamp', 'URI'])

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
                    if orcahello_match is None:
                        print(f"Warning: couldn't find matching OrcaHello detection for {det.feed.slug} {det.timestamp}")
                        continue

                label = get_label(det, orcahello_match)
                if label is None:
                    # label unknown ⇒ skip
                    continue

                classification = classify_detection(det, label, orcasite_dets)
                if not classification.include:
                    continue
                
                # Write to CSV
                category = label
                node_name = det.feed.node_name  # e.g., "rpi_sunset_bay"
                timestamp_pst = format_timestamp_pst(det.timestamp)
                uri = generate_uri(det.feed.slug, det.timestamp)
                csv_writer.writerow([category, node_name, timestamp_pst, uri])

if __name__ == "__main__":
    output_root = Path("output_segments")
    process_all_feeds(output_root)
