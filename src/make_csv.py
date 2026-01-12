# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import csv
from urllib.parse import quote

from pytz import timezone
import argparse
import requests
from azure.cosmos import CosmosClient
import os

COSMOS_URL = os.environ.get("COSMOS_URL", "").strip() or "https://aifororcasmetadatastore.documents.azure.com:443/"
COSMOS_KEY = os.environ.get("COSMOS_KEY", "<your-primary-key>")
COSMOS_DB = os.environ.get("COSMOS_DB", "predictions")
COSMOS_CONTAINER = os.environ.get("COSMOS_CONTAINER", "metadata")
client = CosmosClient(COSMOS_URL, credential=COSMOS_KEY)
database = client.get_database_client(COSMOS_DB)
container = database.get_container_client(COSMOS_CONTAINER)

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
    idempotency_key: str

@dataclass
class OrcaHelloDetection:
    id: str
    feed: OrcasiteFeed
    timestamp: datetime
    status: str                  # e.g., "confirmed", "rejected", etc.

def get_label(
    orcasite_det: OrcasiteDetection,
    orcahello_det: Optional[OrcaHelloDetection],
) -> Optional[str]:
    """
    Derives a normalized label for an OrcasiteDetection based on its category, description, and an optional OrcaHelloDetection match.
    
    Parameters:
        orcasite_det (OrcasiteDetection): The detection to label.
        orcahello_det (Optional[OrcaHelloDetection]): An optional id-matched OrcaHello detection used to upgrade some whale labels.
    
    Returns:
        str | None: `'resident'`, `'transient'`, `'humpback'`, or `'other'` when a label can be determined; `None` when the label is unknown.
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

    if orcahello_det and orcahello_det.status.lower() == "unreviewed":
        return None

    # 4. Other
    if orcasite_det.source == "machine" and not ("click" in desc or "call" in desc or "whistle" in desc):
        return "other"

    # Unknown
    return None

NEAR_MIN = timedelta(minutes=10)
PACIFIC_TZ = timezone('US/Pacific')
UTC_TZ = timezone('UTC')
MAX_DETECTION_PAGES = 1000  # Safety limit to prevent infinite loops (500k detections max)

def is_isolated_human_whale(
    det: OrcasiteDetection,
    all_detections: List[OrcasiteDetection],
) -> bool:
    """
    True if there is NO machine whale detection at same feed within NEAR_MIN minutes.
    """
    for d in all_detections:
        if d.feed.id != det.feed.id:
            continue
        if d.source != "machine":
            continue
        if (d.category or "").lower() != "whale":
            continue
        if abs(d.timestamp - det.timestamp) <= NEAR_MIN:
            return False
    return True

@dataclass
class Classification:
    kind: str       # 'tp_human_only', 'tp_machine_only', 'fp_machine_only', 'tp_both', 'skip'
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
            - 'tp_both' for human-sourced whale detections with a nearby machine whale detection,
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
            return Classification(kind="tp_both", include=True)

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
        "id,playlist_timestamp,player_offset,timestamp,description,"
        "source,category,feed_id,idempotency_key"
    )

    # Build query parameters
    limit = 500
    offset = 0
    params = {
        "page[limit]": limit,
        "fields[detection]": fields,
        "filter[feed_id]": feed.id,
    }

    dets = []
    page_count = 0
    
    try:
        # Loop through all pages until no more data is returned
        while page_count < MAX_DETECTION_PAGES:
            params["page[offset]"] = offset
            
            print(f"Fetching Orcasite page {page_count}...")
            r = requests.get(base_url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            items = data.get("data", [])
            
            # If no data is returned, we've fetched all pages
            if not items:
                print(f"Finished after page {page_count}")
                break

            for item in items:
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
                    idempotency_key=attrs.get("idempotency_key") or "",
                )

                dets.append(det)

            # Increment offset for next page
            offset += limit
            page_count += 1

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

def get_orcahello_detections(feed: OrcasiteFeed) -> List[OrcaHelloDetection]:
    """
    Retrieve OrcaHello detections and return those whose audio URI contains the given feed's node_name.
    
    Parameters:
        feed (OrcasiteFeed): Feed whose node_name is used to filter OrcaHello detection audio URIs.
    
    Returns:
        List[OrcaHelloDetection]: A list of detections associated with the feed. Each detection's `timestamp` is a `datetime` or `None` if parsing failed, and `status` is one of `"confirmed"`, `"rejected"`, or `"unreviewed"`.
    """
    node_name = get_node_name_for_feed(feed)      # e.g., "rpi_sunset_bay"

    # Cosmos DB SQL query
    query = """
        SELECT * FROM c
        WHERE CONTAINS(c.audioUri, @node_name)
        ORDER BY c.timestamp DESC
    """

    params = [
        {"name": "@node_name", "value": node_name}
    ]

    # Cosmos DB returns an iterator that handles pagination internally
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

def process_all_feeds(output_root: Path, feed_filter: Optional[str] = None):
    """
    Generate a consolidated CSV file of selected Orcasite detections, optionally filtered by feed
    and matched with corresponding OrcaHello detections.

    This function retrieves all available Orcasite feeds, optionally filters them by
    a specific node name, fetches detections from both Orcasite and OrcaHello for each
    feed, matches detections in time, classifies them, and writes the resulting
    detections to a CSV file named ``detections.csv`` under the given ``output_root``.

    Parameters:
        output_root (Path): Directory in which the output CSV file ``detections.csv`` will be
            created. The directory (and any missing parents) will be created if it does not exist.
        feed_filter (str | None, optional): If provided, only feeds whose ``node_name`` matches
            this value are processed. If no feed matches the given name, the function logs a
            message and returns without creating a CSV file. Defaults to ``None`` (process all feeds).
    """
    feeds = get_orcasite_feeds()

    if feed_filter:
        feeds = [f for f in feeds if f.node_name == feed_filter]
        if not feeds:
            print(f"No feed found with node_name '{feed_filter}'")
            return
    
    # Collect all detections first before writing to CSV
    all_rows = []
    
    for feed in feeds:
        print(f"Processing feed {feed.id} ({feed.node_name})")
        orcasite_dets = get_orcasite_detections(feed)
        orcahello_dets = get_orcahello_detections(feed)

        for det in orcasite_dets:
            orcahello_match = None
            if det.source == "machine":
                orcahello_match = next((d for d in orcahello_dets if d.id == det.idempotency_key), None)
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
            
            # Collect row data with timestamp for sorting
            category = label
            node_name = det.feed.node_name  # e.g., "rpi_sunset_bay"
            timestamp_pst = format_timestamp_pst(det.timestamp)
            uri = generate_uri(det.feed.slug, det.timestamp)
            all_rows.append((det.timestamp, category, node_name, timestamp_pst, uri, det.description, classification.kind))
    
    # Sort all rows by timestamp (first element of tuple) - oldest first, newest last
    all_rows.sort(key=lambda row: row[0])
    
    # Create CSV file and write sorted rows
    csv_path = output_root / "detections.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Category', 'NodeName', 'Timestamp', 'URI', 'Description', 'Notes'])
        
        # Write sorted rows (exclude the timestamp used for sorting)
        for row in all_rows:
            csv_writer.writerow([row[1], row[2], row[3], row[4], row[5], row[6]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CSV of detections from Orcasite and OrcaHello data"
    )
    parser.add_argument(
        "--feed",
        type=str,
        help="Process only this feed (by node_name, e.g., rpi_sunset_bay)"
    )
    args = parser.parse_args()

    output_root = Path("output_segments")
    process_all_feeds(output_root, feed_filter=args.feed)
