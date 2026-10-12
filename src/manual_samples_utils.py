#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Shared helpers for reading and appending manual_samples.csv rows."""

import csv
from audio_utils import parse_timestamp_pst
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

CSV_FIELDNAMES = [
    "Category",
    "NodeName",
    "StartTimestamp",
    "URI",
    "Description",
    "Notes",
    "Confidence",
]


def _sample_start(row: dict) -> datetime:
    """Parse a sample's timestamp in the project's Pacific timezone."""
    timestamp = (row.get("StartTimestamp") or "").strip()
    return parse_timestamp_pst(timestamp)


def _sample_key(row: dict) -> tuple[str, str] | None:
    """Return a normalized node/timestamp key when both fields are present."""
    node_name = (row.get("NodeName") or "").strip().casefold()
    timestamp = (row.get("StartTimestamp") or "").strip()
    if not node_name or not timestamp:
        return None
    return node_name, timestamp


def filter_overlapping_samples(
    rows: list[dict],
    destination_path: Path,
    duration_seconds: int,
) -> tuple[list[dict], int]:
    """Remove samples duplicated by or overlapping the destination CSV.

    Samples use half-open intervals, so adjacent clips are retained. Accepted
    rows are registered immediately to also reject overlaps within ``rows``.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    existing_rows = []
    if destination_path.exists():
        try:
            with open(destination_path, "r", encoding="utf-8") as handle:
                existing_rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error) as exc:
            print(
                f"Warning: Failed to read destination samples from {destination_path}: {exc}"
            )

    uris: set[str] = set()
    sample_keys: set[tuple[str, str]] = set()
    intervals_by_node: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    duration = timedelta(seconds=duration_seconds)

    def register(row: dict) -> None:
        uri = (row.get("URI") or "").strip()
        if uri:
            uris.add(uri)

        key = _sample_key(row)
        if key is None:
            return
        sample_keys.add(key)
        try:
            start = _sample_start(row)
        except ValueError:
            return
        intervals_by_node[key[0]].append((start, start + duration))

    for row in existing_rows:
        register(row)

    accepted_rows = []
    filtered = 0
    for row in rows:
        uri = (row.get("URI") or "").strip()
        key = _sample_key(row)
        duplicate = bool(uri and uri in uris) or bool(key and key in sample_keys)

        if not duplicate and key is not None:
            try:
                start = _sample_start(row)
            except ValueError:
                start = None
            if start is not None:
                end = start + duration
                duplicate = any(
                    max(start, existing_start) < min(end, existing_end)
                    for existing_start, existing_end in intervals_by_node[key[0]]
                )

        if duplicate:
            filtered += 1
            continue

        accepted_rows.append(row)
        register(row)

    return accepted_rows, filtered


def load_existing_uris(manual_samples_path: Path) -> set[str]:
    """Load existing manual-sample URIs from a CSV file path."""
    if not manual_samples_path.exists():
        return set()

    try:
        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return {
                (row.get("URI") or "").strip()
                for row in reader
                if (row.get("URI") or "").strip()
            }
    except (OSError, UnicodeError, csv.Error) as exc:
        print(
            f"Warning: Failed to read existing manual sample URIs from "
            f"{manual_samples_path}: {exc}"
        )
        return set()


def append_manual_samples(
    manual_samples_path: Path,
    rows: list[dict],
    existing_uris: set[str],
) -> tuple[int, int]:
    """Append rows to manual_samples.csv, skipping URIs already present."""
    rows_to_append = []
    duplicates = 0

    for row in rows:
        uri = (row.get("URI") or "").strip()
        if uri in existing_uris:
            duplicates += 1
            continue
        existing_uris.add(uri)
        rows_to_append.append({field: row.get(field, "") for field in CSV_FIELDNAMES})

    if not rows_to_append:
        return 0, duplicates

    manual_samples_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        not manual_samples_path.exists() or manual_samples_path.stat().st_size == 0
    )
    with open(manual_samples_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerows(rows_to_append)

    return len(rows_to_append), duplicates
