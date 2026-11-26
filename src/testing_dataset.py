#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Testing dataset version and optional provenance metadata helpers."""

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_SEMVER = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_DATASET_TAG = re.compile(r"^TestingDataset\.v(?P<version>.+)$")


@dataclass(frozen=True)
class TestingDatasetMetadata:
    """Version and optional source/site metadata for an evaluation dataset."""

    dataset_version: str = "unversioned"
    git_tag: Optional[str] = None
    source_ids: tuple[str, ...] = ()
    hydrophone_sites: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _normalize_version(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    version = value.strip()
    if version.startswith("v"):
        version = version[1:]
    if not _SEMVER.fullmatch(version):
        return None
    return f"v{version}"


def _as_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        candidates = value
    else:
        return ()
    return tuple(
        dict.fromkeys(
            item.strip()
            for item in candidates
            if isinstance(item, str) and item.strip()
        )
    )


def _field_values(data: dict[str, Any], *names: str) -> tuple[str, ...]:
    for name in names:
        if name in data:
            values = _as_values(data[name])
            if values:
                return values
    return ()


def _read_json_metadata(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, [f"could not read dataset metadata {path}: {exc}"]
    if not isinstance(data, dict):
        return {}, [f"dataset metadata {path} must contain a JSON object"]
    return data, []


def _normalized_csv_header(header: Optional[str]) -> str:
    if not header:
        return ""
    return re.sub(r"[^a-z0-9]", "", header.lower())


def _read_manifest_metadata(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            versions: list[str] = []
            tags: list[str] = []
            sources: list[str] = []
            sites: list[str] = []
            for row in reader:
                values = {
                    _normalized_csv_header(key): value for key, value in row.items()
                }
                version_values = _as_values(values.get("datasetversion")) or _as_values(
                    values.get("version")
                )
                versions.extend(version_values)
                tags.extend(_as_values(values.get("gittag")))
                sources.extend(_as_values(values.get("sourceid")))
                site_values = _as_values(values.get("hydrophonesite")) or _as_values(
                    values.get("site")
                )
                sites.extend(site_values)
    except (OSError, UnicodeError, csv.Error) as exc:
        return {}, [f"could not read dataset manifest {path}: {exc}"]
    return {
        "dataset_version": tuple(dict.fromkeys(versions)),
        "git_tag": tuple(dict.fromkeys(tags)),
        "source_id": tuple(dict.fromkeys(sources)),
        "hydrophone_site": tuple(dict.fromkeys(sites)),
    }, []


def _tag_version(tag: Any) -> Optional[str]:
    if not isinstance(tag, str):
        return None
    match = _DATASET_TAG.fullmatch(tag.strip())
    if not match:
        return None
    return _normalize_version(match.group("version"))


def _tag_at_head(repo_path: Optional[Path]) -> Optional[str]:
    command = ["git"]
    if repo_path is not None:
        command.extend(("-C", str(repo_path)))
    command.extend(("tag", "--points-at", "HEAD"))
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    tags = [
        tag.strip()
        for tag in result.stdout.splitlines()
        if _tag_version(tag.strip()) is not None
    ]
    return tags[0] if len(tags) == 1 else None


def load_testing_dataset_metadata(
    metadata_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    repo_path: Optional[Path] = None,
    allow_head_tag: bool = True,
) -> TestingDatasetMetadata:
    """Load optional metadata without making legacy dataset inputs fail.

    JSON sidecar values take precedence over manifest columns. A semantic
    dataset tag at the checked-out commit is used when metadata has no version.

    Args:
        metadata_path: Optional JSON sidecar path.
        manifest_path: Optional CSV manifest path.
        repo_path: Repository path used to inspect tags at the current commit.
        allow_head_tag: Whether to use an exact dataset tag on the current commit.

    Returns:
        Parsed metadata, using ``unversioned`` when no valid version is found.
    """
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    manifest: dict[str, Any] = {}

    if metadata_path is not None and metadata_path.exists():
        metadata, read_warnings = _read_json_metadata(metadata_path)
        warnings.extend(read_warnings)
    if manifest_path is not None and manifest_path.exists():
        manifest, read_warnings = _read_manifest_metadata(manifest_path)
        warnings.extend(read_warnings)

    metadata_tags = _field_values(metadata, "git_tag")
    manifest_tags = manifest.get("git_tag", ())
    tag: Optional[str] = None
    tag_source: Optional[str] = None
    tagged_version: Optional[str] = None
    for candidate_tags, source in (
        (metadata_tags, "metadata"),
        (manifest_tags, "manifest"),
    ):
        if len(candidate_tags) != 1:
            if candidate_tags:
                warnings.append(
                    f"ignoring inconsistent dataset git_tag values in {source}"
                )
            continue
        candidate = candidate_tags[0]
        candidate_version = _tag_version(candidate)
        if candidate_version is None:
            warnings.append(
                f"ignoring invalid dataset git_tag {candidate!r}; expected TestingDataset.v<semver>"
            )
            continue
        tag = candidate
        tag_source = source
        tagged_version = candidate_version
        break

    version: Optional[str] = None
    version_source: Optional[str] = None
    version_fields = (
        (_field_values(metadata, "dataset_version"), "metadata dataset_version"),
        (_field_values(metadata, "version"), "metadata version"),
        (manifest.get("dataset_version", ()), "manifest"),
    )
    for declared_versions, source in version_fields:
        if not declared_versions:
            continue
        if len(declared_versions) != 1:
            warnings.append(f"ignoring inconsistent dataset versions in {source}")
            continue
        version = _normalize_version(declared_versions[0])
        if version is not None:
            version_source = source
            break
        warnings.append(
            f"ignoring invalid dataset version {declared_versions[0]!r}; expected semantic version MAJOR.MINOR.PATCH"
        )

    if tagged_version is not None:
        if (
            version is not None
            and version_source is not None
            and version_source.startswith("metadata")
            and tag_source == "manifest"
            and version != tagged_version
        ):
            warnings.append(
                f"ignoring manifest git_tag {tag} because sidecar dataset version "
                f"{version} has higher precedence"
            )
            tag = None
        else:
            if version is not None and version != tagged_version:
                warnings.append(
                    f"dataset version {version} does not match git tag {tag}; using tag version"
                )
            version = tagged_version
    elif version is None and allow_head_tag:
        tag = _tag_at_head(repo_path)
        version = _tag_version(tag)

    source_ids = _field_values(metadata, "source_id") or manifest.get("source_id", ())
    hydrophone_sites = _field_values(
        metadata, "hydrophone_site", "site"
    ) or manifest.get("hydrophone_site", ())
    return TestingDatasetMetadata(
        dataset_version=version or "unversioned",
        git_tag=tag,
        source_ids=source_ids,
        hydrophone_sites=hydrophone_sites,
        warnings=tuple(warnings),
    )
