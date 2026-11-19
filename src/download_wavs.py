# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from io import StringIO
from typing import List
import argparse
import csv
import os
import shutil
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote, urlparse, parse_qs

from pytz import timezone
import requests

from add_samples import get_orcasite_feeds
from audio_utils import (
    download_audio_from_start_utc,
    format_timestamp_pacific,
    parse_timestamp_pacific,
)

PACIFIC_TZ = timezone('America/Los_Angeles')
N_SECONDS = 3  # Create 3-second wav files.
TESTING_WINDOW_SECONDS = 60
DEFAULT_DCLDE_MANIFEST = Path("output/csv/dclde_60s_samples.csv")
DEFAULT_TESTING_WAV_ROOT = Path("output/testing-wav")
DEFAULT_TRAINING_CSV_PATH = Path("output/csv/training_3s_samples.csv")
DEFAULT_TESTING_CSV_PATH = Path("output/csv/testing_60s_samples.csv")
# DCLDE 60-second recordings intentionally share the ordinary testing WAV
# directory so existing PODS-AI evaluation tools can consume either manifest.
DEFAULT_DCLDE_WAV_ROOT = DEFAULT_TESTING_WAV_ROOT
ORCAHELLO_ORCASITE_WINDOW_SECONDS = TESTING_WINDOW_SECONDS + 1
DETECTIONS_API_URL = "https://aifororcasdetections.azurewebsites.net/api/detections"
CURRENT_EPOCH_START = datetime.fromisoformat("2025-10-12T14:23:00+00:00")
DETECTIONS_PAGE_SIZE = 50
LEGACY_ORCAHELLO_CLIP_SECONDS = 11
CURRENT_HLS_CLIP_SECONDS = 10
AUDIO_OFFSET_SECONDS = 2
_DETECTIONS_WINDOW_CACHE: dict[tuple[str, str, str], list[dict]] = {}
_CORRECTED_TIMESTAMP_CACHE: dict[tuple[str, str], datetime] = {}

@dataclass
class CSVRow:
    category: str
    node_name: str
    timestamp_pst: str
    uri: str
    description: str
    notes: str
    confidence: str = ""

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
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip header
        next(csv_reader)
        for row in csv_reader:
            if len(row) >= 7:
                rows.append(CSVRow(
                    category=row[0],
                    node_name=row[1],
                    timestamp_pst=row[2],
                    uri=row[3],
                    description=row[4],
                    notes=row[5],
                    confidence=row[6]
                ))
    return rows

def add_seconds_to_timestamp_pst(timestamp_str: str, seconds: int) -> str:
    """
    Add seconds to a PST timestamp string and return the same formatted representation.

    Parameters:
        timestamp_str (str): Timestamp string (e.g., "2025_12_24_17_51_23_PST").
        seconds (int): Number of seconds to add (or subtract if negative).

    Returns:
        str: Adjusted timestamp in the format YYYY_MM_DD_HH_MM_SS_{TZ}.
    """
    adjusted = PACIFIC_TZ.normalize(parse_timestamp_pacific(timestamp_str) + timedelta(seconds=seconds))
    return adjusted.strftime("%Y_%m_%d_%H_%M_%S_%Z")


def _training_window(row: CSVRow) -> tuple[datetime, datetime]:
    start = parse_timestamp_pacific(row.timestamp_pst)
    return start, start + timedelta(seconds=N_SECONDS)


def _testing_window(row: CSVRow) -> tuple[datetime, datetime]:
    """
    Return the full-recording interval represented by a testing row.
    The row is assumed to already be aligned with the OrcaHello
    detection timestamp, so the interval is simply the 60-second
    window starting at the row's timestamp.
    """
    start = parse_timestamp_pacific(row.timestamp_pst)
    return start, start + timedelta(seconds=TESTING_WINDOW_SECONDS)


def _dclde_window(row: CSVRow) -> tuple[datetime, datetime]:
    """Return the full-recording interval represented by a DCLDE row."""
    start = parse_timestamp_pacific(row.timestamp_pst)
    return start, start + timedelta(seconds=TESTING_WINDOW_SECONDS)


def _find_overlaps(rows: list[CSVRow], window_fn, label: str) -> list[str]:
    overlaps = []
    by_node: dict[str, list[tuple[datetime, datetime, CSVRow]]] = {}
    for row in rows:
        start, end = window_fn(row)
        by_node.setdefault(row.node_name, []).append((start, end, row))

    for node_name, windows in by_node.items():
        windows.sort(key=lambda item: item[0])
        _, prev_end, prev_row = windows[0]
        for curr_start, curr_end, curr_row in windows[1:]:
            if curr_start < prev_end:
                overlaps.append(
                    f"{label} overlap at node {node_name}: "
                    f"{prev_row.timestamp_pst} overlaps {curr_row.timestamp_pst}"
                )
            if curr_end > prev_end:
                prev_end, prev_row = curr_end, curr_row
    return overlaps


def _find_cross_overlaps(
    left_rows: list[CSVRow],
    left_window_fn,
    left_label: str,
    right_rows: list[CSVRow],
    right_window_fn,
    right_label: str,
) -> list[str]:
    overlaps = []
    left_by_node: dict[str, list[tuple[datetime, datetime, CSVRow]]] = {}
    right_by_node: dict[str, list[tuple[datetime, datetime, CSVRow]]] = {}

    for row in left_rows:
        start, end = left_window_fn(row)
        left_by_node.setdefault(row.node_name, []).append((start, end, row))
    for row in right_rows:
        start, end = right_window_fn(row)
        right_by_node.setdefault(row.node_name, []).append((start, end, row))

    for node_name in set(left_by_node.keys()) & set(right_by_node.keys()):
        left_windows = sorted(left_by_node[node_name], key=lambda item: item[0])
        right_windows = sorted(right_by_node[node_name], key=lambda item: item[0])
        i = 0
        j = 0
        while i < len(left_windows) and j < len(right_windows):
            left_start, left_end, left_row = left_windows[i]
            right_start, right_end, right_row = right_windows[j]
            if left_start < right_end and right_start < left_end:
                overlaps.append(
                    f"cross-file overlap at node {node_name}: "
                    f"{left_label} {left_row.timestamp_pst} overlaps "
                    f"{right_label} {right_row.timestamp_pst}"
                )
            if left_end <= right_end:
                i += 1
            else:
                j += 1
    return overlaps


def _get_wav_filename(node_name: str, timestamp_pst: str) -> str:
    node_name_in_filename = node_name.replace("_", "-")
    return f"{node_name_in_filename}_{timestamp_pst}.wav"


def _get_relative_wav_path(row: CSVRow) -> Path:
    return Path(row.category) / _get_wav_filename(row.node_name, row.timestamp_pst)


def _audio_start_marker_path(wav_path: Path) -> Path:
    return wav_path.with_name(f"{wav_path.name}.audio-start")


def _has_audio_start_marker(wav_path: Path, audio_start: datetime) -> bool:
    marker_path = _audio_start_marker_path(wav_path)
    if not marker_path.is_file():
        return False
    try:
        return marker_path.read_text(encoding="utf-8").strip() == format_timestamp_pacific(audio_start)
    except OSError:
        return False


def _write_audio_start_marker(wav_path: Path, audio_start: datetime) -> None:
    _audio_start_marker_path(wav_path).write_text(
        format_timestamp_pacific(audio_start), encoding="utf-8"
    )


def _preserve_unverified_wav(wav_path: Path) -> None:
    """Move an unverified WAV aside so it cannot be mistaken for the requested clip."""
    preserved_path = wav_path.with_name(f"{wav_path.name}.unverified")
    suffix = 1
    while preserved_path.exists():
        preserved_path = wav_path.with_name(f"{wav_path.name}.unverified.{suffix}")
        suffix += 1
    os.replace(wav_path, preserved_path)

    marker_path = _audio_start_marker_path(wav_path)
    if marker_path.exists():
        os.replace(marker_path, _audio_start_marker_path(preserved_path))


HUMPBACK_SIGNAL_WAV_PREFIX = "signals-humpback_"


def is_external_humpback_training_wav(relative_path: Path) -> bool:
    """Return True for retained submodule-derived humpback training segments."""
    return (
        len(relative_path.parts) >= 2
        and relative_path.parts[0] == "humpback"
        and relative_path.name.startswith(HUMPBACK_SIGNAL_WAV_PREFIX)
        and relative_path.suffix.lower() == ".wav"
    )


def _copy_wav_from_cache_if_exists(
    expected_path: Path,
    output_root: Path,
    cache_root: Path | None,
    audio_start: datetime | None = None,
) -> bool:
    """
    Copy a WAV file from cache_root into output_root if it exists there.

    Returns True when a cached file is copied, otherwise False.
    """
    if cache_root is None:
        return False
    try:
        relative_path = expected_path.relative_to(output_root)
    except ValueError:
        return False
    source_path = cache_root / relative_path
    if not source_path.exists():
        return False
    if audio_start is not None and not _has_audio_start_marker(source_path, audio_start):
        return False
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, expected_path)
    print(f"Copied from cache: {source_path} -> {expected_path}")
    return True


def delete_stale_wavs(output_root: Path, expected_relative_paths: set[Path]) -> None:
    """Delete WAV files under output_root that are not expected by the current CSV rows."""
    if not output_root.exists():
        return

    deleted_count = 0
    for wav_path in output_root.rglob("*.wav"):
        relative_path = wav_path.relative_to(output_root)
        if relative_path in expected_relative_paths:
            continue
        if is_external_humpback_training_wav(relative_path):
            continue
        wav_path.unlink()
        marker_path = _audio_start_marker_path(wav_path)
        if marker_path.exists():
            marker_path.unlink()
        print(f"Deleted stale wav: {wav_path}")
        deleted_count += 1

    expected_marker_paths = {
        Path(f"{relative_path}.audio-start") for relative_path in expected_relative_paths
    }
    for marker_path in output_root.rglob("*.wav.audio-start"):
        if marker_path.relative_to(output_root) not in expected_marker_paths:
            marker_path.unlink()

    for directory in sorted((path for path in output_root.rglob("*") if path.is_dir()), reverse=True):
        if directory == output_root:
            continue
        if not any(directory.iterdir()):
            directory.rmdir()

    if deleted_count:
        print(f"Deleted {deleted_count} stale wav file(s) from {output_root}")


def validate_no_overlaps(
    training_rows: list[CSVRow],
    testing_rows: list[CSVRow],
    dclde_rows: list[CSVRow] | None = None,
) -> None:
    dclde_rows = dclde_rows or []
    overlaps = []
    if training_rows:
        overlaps.extend(_find_overlaps(training_rows, _training_window, "training"))
    if testing_rows:
        overlaps.extend(_find_overlaps(testing_rows, _testing_window, "testing"))
    if dclde_rows:
        overlaps.extend(_find_overlaps(dclde_rows, _dclde_window, "DCLDE"))
    if training_rows and testing_rows:
        overlaps.extend(
            _find_cross_overlaps(
                training_rows, _training_window, "training",
                testing_rows, _testing_window, "testing",
            )
        )
    if training_rows and dclde_rows:
        overlaps.extend(
            _find_cross_overlaps(
                training_rows, _training_window, "training",
                dclde_rows, _dclde_window, "DCLDE",
            )
        )
    if testing_rows and dclde_rows:
        overlaps.extend(
            _find_cross_overlaps(
                testing_rows, _testing_window, "testing",
                dclde_rows, _dclde_window, "DCLDE",
            )
        )

    if overlaps:
        details = "\n".join(f"  - {overlap}" for overlap in overlaps)
        raise ValueError(f"Detected overlapping sample windows:\n{details}")


def _is_false_positive_testing_row(row: CSVRow) -> bool:
    """Return whether a testing CSV row represents a false positive sample.

    Args:
        row: Parsed testing CSV row.

    Returns:
        True when the row notes indicate an fp_machine* sample.
    """
    return row.notes.startswith("fp_machine")


def _normalize_text(value: str) -> str:
    """Normalize text for case-insensitive detection-comment comparisons.

    Args:
        value: Raw text value from CSV or detections API data.

    Returns:
        A lowercase, whitespace-normalized string.
    """
    return " ".join(value.split()).strip().lower()


def _default_slug_from_node_name(node_name: str) -> str:
    """Derive the default Orcasound URI slug from a CSV node name."""
    return node_name.removeprefix("rpi_").replace("_", "-")


def _default_slug_validation_node_name(row: CSVRow, normalized_node: str) -> str | None:
    """Return the node name to use for URI slug fallback validation."""
    if normalized_node.startswith("rpi_"):
        return normalized_node

    if normalized_node != row.node_name or not row.node_name.startswith("dclde_") or not row.uri:
        return None

    uri_parts = [part for part in urlparse(row.uri).path.split("/") if part]
    try:
        audio_index = uri_parts.index("audio")
    except ValueError:
        return None

    if audio_index + 1 >= len(uri_parts):
        return None

    return uri_parts[audio_index + 1]


def _generate_testing_uri(row: CSVRow, timestamp_pst: str) -> str:
    """Build the Orcasound bouts URI for a testing row using the supplied CSV timestamp.

    Args:
        row: Parsed testing CSV row whose node or base URI is reused.
        timestamp_pst: Testing-row timestamp in repository CSV format.

    Returns:
        Orcasound bouts URI pointing at the supplied timestamp.
    """
    if row.uri:
        base_uri = row.uri.split("?", 1)[0]
    else:
        node_slug = _default_slug_from_node_name(row.node_name)
        base_uri = f"https://live.orcasound.net/bouts/new/{node_slug}"
    timestamp_utc = (
        parse_timestamp_pacific(timestamp_pst).astimezone(dt_timezone.utc)
        - timedelta(seconds=AUDIO_OFFSET_SECONDS)
    )
    encoded_time = quote(timestamp_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z", safe="")
    return f"{base_uri}?time={encoded_time}"


def _build_corrected_testing_row(row: CSVRow, timestamp_pst: str) -> CSVRow:
    """Return a copy of a testing row with an updated timestamp and matching URI.

    Args:
        row: Original parsed testing CSV row.
        timestamp_pst: Replacement timestamp in repository CSV format.

    Returns:
        CSVRow with the updated timestamp and regenerated URI.
    """
    return CSVRow(
        category=row.category,
        node_name=row.node_name,
        timestamp_pst=timestamp_pst,
        uri=_generate_testing_uri(row, timestamp_pst),
        description=row.description,
        notes=row.notes,
        confidence=row.confidence
    )


def _format_testing_row_csv(row: CSVRow) -> str:
    """Serialize a testing row using the same column order as testing_60s_samples.csv.

    Args:
        row: Parsed testing CSV row to serialize.

    Returns:
        One CSV data line with no trailing newline.
    """
    output = StringIO()
    csv.writer(output, lineterminator="").writerow([
        row.category,
        row.node_name,
        row.timestamp_pst,
        row.uri,
        row.description,
        row.notes,
        row.confidence
    ])
    return output.getvalue()


def _fetch_detections_page(node_name: str, start_date: datetime, end_date: datetime, page: int) -> tuple[list[dict], bool]:
    """Fetch one detections API page and return its items plus whether another page exists.

    Args:
        node_name: Hydrophone node name used in the API query.
        start_date: Inclusive Pacific-local lower date bound.
        end_date: Inclusive Pacific-local upper date bound.
        page: 1-based detections API page number.

    Returns:
        Tuple of ``(items, has_next_page)`` for the requested page.
    """
    params = {
        "Page": page,
        "SortBy": "timestamp",
        "SortOrder": "desc",
        "Timeframe": "range",
        "DateFrom": start_date.strftime("%m/%d/%Y"),
        "DateTo": end_date.strftime("%m/%d/%Y"),
        "Location": "all",
        "HydrophoneId": node_name,
        "RecordsPerPage": DETECTIONS_PAGE_SIZE,
        "MinutesPerPage": 0,
    }
    response = requests.get(DETECTIONS_API_URL, params=params, timeout=30)
    response.raise_for_status()
    if not response.text.strip():
        return [], False
    payload = response.json()
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("items")
        items = items if isinstance(items, list) else []
    else:
        items = []

    total_pages_header = response.headers.get("totalAmountPages")
    if total_pages_header is not None:
        try:
            total_pages = int(total_pages_header)
        except ValueError:
            total_pages = None
        else:
            return items, page < total_pages

    return items, len(items) == DETECTIONS_PAGE_SIZE


def _fetch_detections_for_window(node_name: str, start_date: datetime, end_date: datetime) -> list[dict]:
    """Fetch and cache all detections for one hydrophone and inclusive date window.

    Args:
        node_name: Hydrophone node name used in the API query.
        start_date: Inclusive Pacific-local lower date bound.
        end_date: Inclusive Pacific-local upper date bound.

    Returns:
        All detections returned by the paginated API query for that window.
    """
    cache_key = (
        node_name,
        start_date.strftime("%m/%d/%Y"),
        end_date.strftime("%m/%d/%Y"),
    )
    cached = _DETECTIONS_WINDOW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    page = 1
    detections: list[dict] = []
    while True:
        items, has_next_page = _fetch_detections_page(node_name, start_date, end_date, page)
        if not items:
            break
        detections.extend(items)
        if not has_next_page:
            break
        page += 1
    _DETECTIONS_WINDOW_CACHE[cache_key] = detections
    return detections


def _parse_detection_timestamp(detection: dict) -> datetime | None:
    """Parse a detections API timestamp field into a timezone-aware datetime.

    Args:
        detection: One detections API result object.

    Returns:
        Parsed UTC datetime, or None when the timestamp is absent or invalid.
    """
    timestamp = detection.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def _get_corrected_detection_timestamp(node_name: str, detection_timestamp: datetime) -> datetime:
    """Return the corrected 60-second clip start for one detections API timestamp.

    Args:
        node_name: Hydrophone node name used to look up legacy HLS folders.
        detection_timestamp: Timestamp reported by the detections API.

    Returns:
        Corrected clip-start datetime in UTC.
    """
    cache_key = (node_name, detection_timestamp.isoformat())
    cached = _CORRECTED_TIMESTAMP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if detection_timestamp >= CURRENT_EPOCH_START:
        _CORRECTED_TIMESTAMP_CACHE[cache_key] = detection_timestamp
        return detection_timestamp

    from audio_utils import get_cached_folders

    folder_prefix = f"{node_name}/hls/"
    folders = get_cached_folders("audio-orcasound-net", prefix=folder_prefix)
    original_unix_time_seconds = int(detection_timestamp.timestamp())

    folder_time_seconds = 0
    for folder_name in folders:
        try:
            unix_time = int(folder_name)
        except (TypeError, ValueError):
            continue
        if unix_time <= original_unix_time_seconds and unix_time > folder_time_seconds:
            folder_time_seconds = unix_time

    if folder_time_seconds == 0:
        raise ValueError(
            f"Unable to correct legacy detection timestamp for {node_name} at {detection_timestamp.isoformat()}: "
            "no matching S3 HLS folder found."
        )

    original_seconds_into_folder = original_unix_time_seconds - folder_time_seconds
    original_clip_index = original_seconds_into_folder // LEGACY_ORCAHELLO_CLIP_SECONDS
    corrected_seconds_into_folder = (original_clip_index * CURRENT_HLS_CLIP_SECONDS) + AUDIO_OFFSET_SECONDS
    corrected_unix_time_seconds = folder_time_seconds + corrected_seconds_into_folder
    corrected_timestamp = datetime.fromtimestamp(corrected_unix_time_seconds, tz=dt_timezone.utc)
    _CORRECTED_TIMESTAMP_CACHE[cache_key] = corrected_timestamp
    return corrected_timestamp


def _find_matching_detection(row: CSVRow) -> tuple[dict, str]:
    """Find the best matching false-positive detection and return its corrected timestamp.

    Args:
        row: False-positive testing CSV row being validated.

    Returns:
        Tuple of ``(detection, corrected_timestamp_pst)`` for the matching 60-second detection.
    """
    row_timestamp = parse_timestamp_pacific(row.timestamp_pst)
    normalized_description = _normalize_text(row.description)
    search_windows = (1, 7)
    last_candidate_count = 0
    last_start_date = row_timestamp
    last_end_date = row_timestamp
    closest_corrected_timestamp_pst = None
    closest_reviewed_comments = None
    closest_reviewed_delta_seconds = None

    for days in search_windows:
        start_date = row_timestamp - timedelta(days=days)
        end_date = row_timestamp + timedelta(days=days)
        last_start_date = start_date
        last_end_date = end_date
        detections = _fetch_detections_for_window(row.node_name, start_date, end_date)
        candidates = []

        for detection in detections:
            if str(detection.get("found", "")).strip().lower() != "no":
                continue
            if not bool(detection.get("reviewed", False)):
                continue

            comments = str(detection.get("comments") or "").strip()
            if normalized_description and _normalize_text(comments) != normalized_description:
                continue

            detection_timestamp = _parse_detection_timestamp(detection)
            if detection_timestamp is None:
                continue

            corrected_timestamp = _get_corrected_detection_timestamp(row.node_name, detection_timestamp)
            corrected_end_timestamp = corrected_timestamp + timedelta(seconds=ORCAHELLO_ORCASITE_WINDOW_SECONDS)
            delta_seconds = min(
                abs((row_timestamp - corrected_timestamp).total_seconds()),
                abs((row_timestamp - corrected_end_timestamp).total_seconds()),
            )
            if closest_reviewed_delta_seconds is None or delta_seconds < closest_reviewed_delta_seconds:
                closest_reviewed_delta_seconds = delta_seconds
                closest_corrected_timestamp_pst = format_timestamp_pacific(corrected_timestamp)
                closest_reviewed_comments = comments

            if corrected_timestamp <= row_timestamp <= corrected_end_timestamp:
                seconds_from_start = (row_timestamp - corrected_timestamp).total_seconds()
                candidates.append((seconds_from_start, corrected_timestamp, detection))

        last_candidate_count = len(candidates)
        if candidates:
            _, corrected_timestamp, detection = min(candidates, key=lambda item: item[0])
            return detection, format_timestamp_pacific(corrected_timestamp)

    raise ValueError(
        f"Could not find matching OrcaHello false-positive detection for testing row {row!r} "
        f"(timestamp={row.timestamp_pst}, description={row.description!r}, "
        f"search_window={last_start_date.strftime('%m/%d/%Y')}..{last_end_date.strftime('%m/%d/%Y')}, "
        f"candidates={last_candidate_count}, closest_corrected_timestamp={closest_corrected_timestamp_pst!r}, "
        f"closest_comments={closest_reviewed_comments!r})."
    )


def _parse_uri_timestamp_pst(uri: str) -> str:
    """Parse a bouts URI query timestamp and return it in PST CSV format."""
    parsed = urlparse(uri)
    query_params = parse_qs(parsed.query)
    if "time" not in query_params:
        raise ValueError(f"URI missing 'time' query parameter: {uri}")

    time_str = query_params["time"][0]
    try:
        utc_dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        utc_dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    utc_dt = utc_dt.replace(tzinfo=dt_timezone.utc)
    return format_timestamp_pacific(utc_dt)


def validate_aligned_entries(testing_rows: list[CSVRow]) -> None:
    """Verify false-positive testing rows match the timestamps reconstructed from OrcaHello detections.

    Args:
        testing_rows: Parsed testing CSV rows to validate.

    Raises:
        ValueError: If any false-positive row does not align with its matching detection.
    """
    mismatches = []

    for row in testing_rows:
        if not _is_false_positive_testing_row(row):
            continue

        _, corrected_timestamp_pst = _find_matching_detection(row)
        if parse_timestamp_pacific(row.timestamp_pst) != parse_timestamp_pacific(corrected_timestamp_pst):
            corrected_row = _build_corrected_testing_row(row, corrected_timestamp_pst)
            mismatches.append(
                "Unaligned false-positive testing row:\n"
                f"  old testing_row: {_format_testing_row_csv(row)}\n"
                f"  new testing_row: {_format_testing_row_csv(corrected_row)}"
            )

    if mismatches:
        raise ValueError("\n".join(mismatches))


def validate_uri_timestamps(rows: list[CSVRow]) -> None:
    """Verify that the timestamp encoded in each row URI matches its Orcasite time.

    Args:
        rows: Parsed CSV rows to validate.
    Raises:
        ValueError: If any row's URI timestamp does not match the CSV timestamp or the URI cannot be parsed.
    """
    mismatches: list[str] = []

    for row in rows:
        if not row.uri:
            continue
        try:
            uri_timestamp_pst = _parse_uri_timestamp_pst(row.uri)
        except Exception as e:
            mismatches.append(
                f"Unable to parse URI for row {row.node_name} {row.timestamp_pst}: {row.uri} ({type(e).__name__}: {e})"
            )
            continue

        uri_timestamp = parse_timestamp_pacific(uri_timestamp_pst)
        row_timestamp = parse_timestamp_pacific(row.timestamp_pst)
        difference = row_timestamp.astimezone(dt_timezone.utc) - uri_timestamp.astimezone(
            dt_timezone.utc
        )
        if difference != timedelta(seconds=AUDIO_OFFSET_SECONDS):
            mismatches.append(
                "Timestamp mismatch between CSV and URI:\n"
                f"  csv: {row.node_name} {row.timestamp_pst}\n"
                f"  uri: {row.node_name} {uri_timestamp_pst} -> {row.uri}"
            )

    if mismatches:
        raise ValueError("\n".join(mismatches))


def validate_node_slug_in_uri(rows: list[CSVRow]) -> None:
    """Verify that the NodeName in CSV corresponds to the slug present in the URI path.

    Args:
        rows: Parsed CSV rows to validate.

    Raises:
        ValueError: If any row's URI slug does not match the node's expected slug or the URI is malformed.
    """
    mismatches: list[str] = []

    # Load Orcasite feeds once to allow mapping DCLDE-style node names to known feeds.
    feeds = []
    try:
        feeds = get_orcasite_feeds()
    except Exception:
        # If feed lookup fails, we'll fall back to using the raw node_name below.
        feeds = []

    # Build a node_name -> slug map so we do not call get_node_slug per row.
    node_to_slug: dict[str, str] = {f.node_name: f.slug for f in feeds} if feeds else {}

    for row in rows:
        if not row.uri:
            continue

        # Map DCLDE-style or other composite node names to known feed node_name when possible.
        normalized_node = row.node_name
        for feed in feeds:
            try:
                if (feed.node_name and feed.node_name in row.node_name) or (
                    feed.slug and feed.slug.replace('-', '_') in row.node_name
                ):
                    normalized_node = feed.node_name
                    break
            except Exception:
                continue

        expected_slug = node_to_slug.get(normalized_node)
        if expected_slug is None:
            fallback_node = _default_slug_validation_node_name(row, normalized_node)
            if fallback_node is None:
                mismatches.append(
                    f"Unable to look up slug for node {row.node_name} (normalized to {normalized_node})"
                )
                continue
            expected_slug = _default_slug_from_node_name(fallback_node)

        # Ensure the expected slug (e.g., 'orcasound-lab') appears somewhere in the URI.
        # Accept either hyphenated or underscored forms (orcasound-lab OR orcasound_lab).
        alt_slug = expected_slug.replace("-", "_")
        uri_path = urlparse(row.uri).path

        if (expected_slug not in uri_path) and (alt_slug not in uri_path):

            mismatches.append(
                "Node slug not found in URI (accepted forms: hyphen or underscore):\n"
                f"  csv node: {row.node_name} (normalized: {normalized_node}) -> expected slug: {expected_slug}\n"
                f"  uri: {row.uri}"
            )

    if mismatches:
        raise ValueError("\n".join(mismatches))


def download_audio_segment(
    category: str,
    node_name: str,
    timestamp_str: str,
    output_root: Path,
    cache_root: Path | None = None,
    uri: str = "",
):
    """
    Download a 3-second audio segment for a detection and save it to the appropriate label directory.

    Uses the shared HLS downloader to retrieve and crop the requested audio.

    Parameters:
        category (str): The label/category for the detection (e.g., "resident", "transient").
        node_name (str): The node name (e.g., "rpi_sunset_bay").
        timestamp_str (str): The detection timestamp in Pacific time.
        output_root (Path): Root directory where label subdirectories and audio files will be saved.
        uri (str): Orcasite URI associated with the manifest row.
    """
    label_dir = output_root / category
    label_dir.mkdir(parents=True, exist_ok=True)
    timestamp_pst = parse_timestamp_pacific(timestamp_str)
    start_time = PACIFIC_TZ.normalize(timestamp_pst - timedelta(seconds=AUDIO_OFFSET_SECONDS))

    # Check if the file already exists.
    wav_filename = _get_wav_filename(node_name, timestamp_str)
    expected_path = label_dir / wav_filename
    if expected_path.exists():
        if _has_audio_start_marker(expected_path, start_time):
            print(f"Skipping (already exists): {expected_path}")
            return
        _preserve_unverified_wav(expected_path)
    if _copy_wav_from_cache_if_exists(expected_path, output_root, cache_root, audio_start=start_time):
        _write_audio_start_marker(expected_path, start_time)
        return

    start_time_utc = start_time.astimezone(dt_timezone.utc)
    try:
        with TemporaryDirectory() as tmp_path:
            wav_path = download_audio_from_start_utc(
                node_name=node_name,
                start_time_utc=start_time_utc,
                tmp_dir=tmp_path,
                duration_seconds=N_SECONDS,
            )
            if wav_path:
                shutil.copy2(wav_path, expected_path)
                _write_audio_start_marker(expected_path, start_time)
                print(f"Downloaded: {expected_path}")
    except Exception as e:
        print("\nWarning: Unable to retrieve audio clip.")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        print(f"Hydrophone: {node_name}")

def process_csv(csv_path: Path, output_root: Path, cache_root: Path | None = None):
    """
    Read the training samples CSV file and download corresponding WAV files.

    Parameters:
        csv_path (Path): Path to the training_3s_samples.csv file.
        output_root (Path): Root directory where audio files will be saved in label subdirectories.
    """
    rows = parse_csv(csv_path)

    print(f"Found {len(rows)} training samples to process")

    expected_relative_paths: set[Path] = set()
    for row in rows:
        expected_relative_paths.add(_get_relative_wav_path(row))
        print(f"Processing: {row.category} - {row.node_name} - {row.timestamp_pst}")
        download_audio_segment(
            row.category,
            row.node_name,
            row.timestamp_pst,
            output_root,
            cache_root=cache_root,
            uri=row.uri,
        )

    delete_stale_wavs(output_root, expected_relative_paths)


def download_testing_sample(row: CSVRow, output_root: Path, cache_root: Path | None = None):
    """
    Download audio for a testing sample.

    The testing CSV StartTimestamp is the 60-second clip start time.

    Args:
        row: Parsed CSV row describing one testing sample.
        output_root: Root directory where category subdirectories are created.

    Returns:
        None.
    """
    label_dir = output_root / row.category
    label_dir.mkdir(parents=True, exist_ok=True)
    wav_filename = _get_wav_filename(row.node_name, row.timestamp_pst)

    expected_path = label_dir / wav_filename
    download_testing_file(expected_path, output_root, cache_root)


def download_testing_file(expected_path: Path, output_root: Path, cache_root: Path | None = None):
    from add_samples import parse_node_and_timestamp_from_filename

    node_name, timestamp_pst = parse_node_and_timestamp_from_filename(expected_path)
    audio_start = parse_timestamp_pacific(timestamp_pst)
    start_time_utc = audio_start.astimezone(dt_timezone.utc) - timedelta(
        seconds=AUDIO_OFFSET_SECONDS
    )
    requested_audio_start = start_time_utc.astimezone(PACIFIC_TZ)
    if expected_path.exists():
        if _has_audio_start_marker(expected_path, requested_audio_start):
            print(f"  Skipping (already exists): {expected_path}")
            return
        _preserve_unverified_wav(expected_path)
    if _copy_wav_from_cache_if_exists(
        expected_path, output_root, cache_root, audio_start=requested_audio_start
    ):
        _write_audio_start_marker(expected_path, requested_audio_start)
        return

    print(f"  Downloading {TESTING_WINDOW_SECONDS}-second audio from {timestamp_pst}...")

    with TemporaryDirectory() as tmp_dir:
        from audio_utils import download_audio_from_start_utc

        wav_path = download_audio_from_start_utc(
            node_name=node_name,
            start_time_utc=start_time_utc,
            tmp_dir=tmp_dir,
            duration_seconds=TESTING_WINDOW_SECONDS,
        )
        if wav_path is None:
            raise AssertionError(f"Error: Failed to download 60-second clip for {node_name} at {timestamp_pst}")

        # Ensure destination directory exists before moving the downloaded file.
        expected_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(wav_path, expected_path)
        _write_audio_start_marker(expected_path, requested_audio_start)
        print(f"  Downloaded: {expected_path}")


def process_testing_csv(
    csv_path: Path,
    output_root: Path,
    cache_root: Path | None = None,
    cleanup_expected_paths: set[Path] | None = None,
    do_cleanup: bool = True,
):
    """
    Read the testing samples CSV file and download corresponding WAV files.

    Args:
        csv_path: Path to the testing_60s_samples.csv file.
        output_root: Root directory where testing WAV files are saved.
    """
    rows = parse_csv(csv_path)
    print(f"Found {len(rows)} testing samples to process")

    expected_relative_paths: set[Path] = set()
    for row in rows:
        expected_relative_paths.add(_get_relative_wav_path(row))
        print(f"Processing testing sample: {row.category} - {row.node_name} - {row.timestamp_pst} ({row.notes})")
        download_testing_sample(row, output_root, cache_root=cache_root)

    if do_cleanup:
        delete_stale_wavs(
            output_root,
            cleanup_expected_paths if cleanup_expected_paths is not None else expected_relative_paths,
        )


def download_dclde_sample(
    row: CSVRow,
    output_root: Path,
    cache_root: Path | None = None,
) -> None:
    """Download one complete DCLDE WAV directly from the URI in its manifest."""
    label_dir = output_root / row.category
    label_dir.mkdir(parents=True, exist_ok=True)
    expected_path = label_dir / _get_wav_filename(row.node_name, row.timestamp_pst)
    if expected_path.exists() and expected_path.stat().st_size:
        print(f"Skipping (already exists): {expected_path}")
        return
    if _copy_wav_from_cache_if_exists(expected_path, output_root, cache_root):
        return
    if not row.uri:
        raise ValueError("DCLDE manifest row has an empty URI")

    with TemporaryDirectory() as tmp_dir:
        from audio_utils import download_from_url

        download_from_url(row.uri, tmp_dir)
        downloaded_path = Path(tmp_dir) / os.path.basename(row.uri.split("?", 1)[0])
        if not downloaded_path.is_file() or not downloaded_path.stat().st_size:
            raise FileNotFoundError(f"Download did not produce {downloaded_path.name}")
        shutil.move(str(downloaded_path), expected_path)
    print(f"Downloaded DCLDE WAV: {expected_path}")


def process_dclde_csv(
    csv_path: Path,
    output_root: Path,
    cache_root: Path | None = None,
    cleanup_expected_paths: set[Path] | None = None,
    do_cleanup: bool = True,
) -> None:
    """Download all full recordings listed in the optional DCLDE manifest."""
    rows = parse_csv(csv_path)
    print(f"Found {len(rows)} DCLDE Orcasound recordings to process")
    expected_relative_paths = {_get_relative_wav_path(row) for row in rows}
    failures: list[str] = []
    for index, row in enumerate(rows, start=1):
        print(
            f"Processing DCLDE sample {index}/{len(rows)}: "
            f"{row.category} - {row.description}"
        )
        try:
            download_dclde_sample(row, output_root, cache_root=cache_root)
        except Exception as error:
            message = (
                f"row {index + 1} ({row.node_name}, {row.timestamp_pst}): "
                f"{type(error).__name__}: {error}"
            )
            failures.append(message)
            print(f"WARNING: DCLDE download failed for {message}", file=sys.stderr)

    if do_cleanup:
        delete_stale_wavs(
            output_root,
            cleanup_expected_paths if cleanup_expected_paths is not None else expected_relative_paths,
        )
    print(
        f"DCLDE download summary: {len(rows) - len(failures)}/{len(rows)} "
        "recordings available"
    )
    if failures:
        print(f"DCLDE failures: {len(failures)}", file=sys.stderr)


def print_usage():
    """
    Display usage information for this script.
    """
    print("Usage: python download_wavs.py [--validate-only]")
    print()
    print("This script downloads training, testing, and optional DCLDE Orcasound WAVs.")
    print("It reads from:")
    print("  - output/csv/training_3s_samples.csv")
    print("  - output/csv/testing_60s_samples.csv")
    print("  - output/csv/dclde_60s_samples.csv (optional)")
    print()
    print("And saves wav files to:")
    print("  - output/wav/ (training samples)")
    print("  - output/testing-wav/ (testing and DCLDE Orcasound samples)")
    print()
    print("Optional argument:")
    print("  --validate-only: validate CSV overlap rules without downloading WAV files")
    print("Optional environment variables:")
    print("  - WAV_WORKTREE_DIR: root directory containing output/ (default: current directory)")
    print("  - WAV_CACHE_DIR: root directory to copy existing wav files from before downloading")


def run_download_wavs(
    validate_only: bool = False,
    training_csv_path: Path = DEFAULT_TRAINING_CSV_PATH,
    testing_csv_path: Path = DEFAULT_TESTING_CSV_PATH,
    dclde_csv_path: Path = DEFAULT_DCLDE_MANIFEST,
    dclde_output_root: Path | None = None,
) -> None:

    worktree_root = Path(os.getenv("WAV_WORKTREE_DIR", "."))
    training_output_root = worktree_root / "output/wav"
    testing_output_root = worktree_root / DEFAULT_TESTING_WAV_ROOT
    if dclde_output_root is None:
        dclde_output_root = worktree_root / DEFAULT_DCLDE_WAV_ROOT

    cache_root_env = os.getenv("WAV_CACHE_DIR")
    training_cache_root = None
    testing_cache_root = None
    dclde_cache_root = None
    if cache_root_env:
        cache_root = Path(cache_root_env)
        training_cache_root = cache_root / "output/wav"
        testing_cache_root = cache_root / DEFAULT_TESTING_WAV_ROOT
        dclde_cache_root = cache_root / DEFAULT_DCLDE_WAV_ROOT

    if not training_csv_path.exists():
        print(f"Error: CSV file not found at {training_csv_path}")
        print("Please update output/csv/training_3s_samples.csv before running download_wavs.py.")
        sys.exit(1)

    training_rows = parse_csv(training_csv_path)
    testing_rows: list[CSVRow] = []
    if not testing_csv_path.exists():
        print(f"Warning: CSV file not found at {testing_csv_path}")
        print("Skipping testing WAV downloads. Update output/csv/testing_60s_samples.csv to enable testing downloads.")
    else:
        testing_rows = parse_csv(testing_csv_path)

    dclde_rows: list[CSVRow] = []
    if dclde_csv_path.exists():
        dclde_rows = parse_csv(dclde_csv_path)
        if not dclde_rows:
            raise ValueError(f"DCLDE manifest has no usable rows: {dclde_csv_path}")
        missing_uri = sum(not row.uri for row in dclde_rows)
        if missing_uri:
            raise ValueError(f"DCLDE manifest contains {missing_uri} row(s) without a URI")

    # First verify that all testing rows are aligned with OrcaHello timestamps.
    validate_aligned_entries(testing_rows)

    # Now we know the testing row timestamps are directly comparable to training row timestamps,
    # so we can check for overlaps between them.
    # TODO(issue #550): Verify whether dclde_rows have the correct timestamp or need correction first.
    validate_no_overlaps(training_rows, testing_rows, dclde_rows)

    validate_uri_timestamps(training_rows)
    validate_uri_timestamps(testing_rows)

    validate_node_slug_in_uri(testing_rows + dclde_rows)

    if validate_only:
        print("Overlap, aligned-entry, and URI validation completed successfully.")
        if dclde_rows:
            print(f"DCLDE manifest validation completed: {len(dclde_rows)} rows.")
        else:
            print(f"DCLDE manifest not found; validation skipped: {dclde_csv_path}")
        return

    process_csv(training_csv_path, training_output_root, cache_root=training_cache_root)

    testing_expected_paths = {_get_relative_wav_path(row) for row in testing_rows}
    dclde_expected_paths = {_get_relative_wav_path(row) for row in dclde_rows}
    roots_are_shared = testing_output_root.resolve() == dclde_output_root.resolve()
    shared_expected_paths = testing_expected_paths | dclde_expected_paths
    shared_cleanup_is_safe = roots_are_shared and bool(testing_rows) and bool(dclde_rows)

    if testing_rows:
        process_testing_csv(
            testing_csv_path,
            testing_output_root,
            cache_root=testing_cache_root,
            cleanup_expected_paths=(
                shared_expected_paths if shared_cleanup_is_safe else testing_expected_paths
            ),
            do_cleanup=not roots_are_shared or shared_cleanup_is_safe,
        )

    if dclde_rows:
        process_dclde_csv(
            dclde_csv_path,
            dclde_output_root,
            cache_root=dclde_cache_root,
            cleanup_expected_paths=(
                shared_expected_paths if shared_cleanup_is_safe else dclde_expected_paths
            ),
            do_cleanup=not roots_are_shared or shared_cleanup_is_safe,
        )
    else:
        print(f"DCLDE manifest not found; skipping DCLDE downloads: {dclde_csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download PODS-AI training, testing, and optional DCLDE Orcasound WAVs. "
            "Run from the repository root as python src/download_wavs.py."
        )
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--training-csv-path",
        type=Path,
        default=DEFAULT_TRAINING_CSV_PATH,
        help=(
            "Training manifest (default: output/csv/training_3s_samples.csv)."
        ),
    )
    parser.add_argument(
        "--testing-csv-path",
        type=Path,
        default=DEFAULT_TESTING_CSV_PATH,
        help=(
            "Testing manifest (default: output/csv/testing_60s_samples.csv)."
        ),
    )
    parser.add_argument(
        "--dclde-manifest",
        type=Path,
        default=DEFAULT_DCLDE_MANIFEST,
        help=(
            "DCLDE Orcasound manifest (default: output/csv/dclde_60s_samples.csv). "
            "If absent, DCLDE downloading is skipped."
        ),
    )
    parser.add_argument(
        "--dclde-wav-root",
        type=Path,
        default=None,
        help=(
            "Override DCLDE WAV output root "
            "(default: output/testing-wav, shared with Orcasound testing clips)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_download_wavs(
        validate_only=args.validate_only,
        training_csv_path=args.training_csv_path,
        testing_csv_path=args.testing_csv_path,
        dclde_csv_path=args.dclde_manifest,
        dclde_output_root=args.dclde_wav_root,
    )
