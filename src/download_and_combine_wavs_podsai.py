#!/usr/bin/env python3
"""Build a testing_60s_samples-style collection for legacy PODS-AI models.

Primary Orcasound clips are downloaded with download_wavs.py. DCLDE clips are
downloaded one source file at a time, extracted to 60 seconds, and stored in
the exact directory/filename convention expected by compare_models.py.
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

DEFAULT_PRIMARY_MANIFEST = "/content/pods-ai/output/csv/testing_60s_samples.csv"
DEFAULT_SECONDARY_MANIFEST = (
    "/content/pods-ai/output/csv/"
    "orcasound_60s_validation_manifest_no_mixed_extracted.csv"
)
DEFAULT_WAV_ROOT = "/content/pods-ai/output/testing-wav"
DEFAULT_OUTPUT_MANIFEST = "/content/pods-ai/output/csv/testing_60s_samples_combined_dclde.csv"
TESTING_FIELDS = [
    "Category", "NodeName", "Timestamp", "URI",
    "Description", "Notes", "Confidence",
]


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def safe_slug(value: object, max_len: int = 140) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", _clean(value))
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len]


def require_columns(
    fieldnames: Optional[list[str]], required: set[str], manifest: Path
) -> None:
    missing = sorted(required - set(fieldnames or []))
    if missing:
        raise ValueError(f"{manifest} is missing required columns: {missing}")


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def download_primary_manifest_wavs(
    manifest: Path, wav_root: Path, cache_root: Optional[Path] = None
) -> None:
    """Use the repository's existing Orcasound downloader for primary rows only."""
    from download_wavs import download_testing_sample, parse_csv

    rows = parse_csv(manifest)
    wav_root.mkdir(parents=True, exist_ok=True)
    print(f"Primary download: {len(rows)} manifest rows")
    for index, row in enumerate(rows, start=1):
        print(
            f"[primary {index}/{len(rows)}] "
            f"{row.category} - {row.node_name} - {row.timestamp_pst}"
        )
        download_testing_sample(row, wav_root, cache_root=cache_root)


def find_gcs_copy_command() -> Optional[list[str]]:
    if shutil.which("gsutil"):
        return ["gsutil", "-q", "cp"]
    if shutil.which("gcloud"):
        return ["gcloud", "storage", "cp", "--quiet"]
    return None


def download_or_copy_source(source_path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.startswith("gs://"):
        command = find_gcs_copy_command()
        if command is None:
            raise RuntimeError("Neither gsutil nor gcloud is available")
        process = subprocess.run(
            [*command, source_path, str(destination)], capture_output=True, text=True
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"GCS copy failed for {source_path}: {process.stderr.strip()[:2000]}"
            )
    else:
        local_source = Path(source_path)
        if not local_source.is_file():
            raise FileNotFoundError(f"Source audio not found: {source_path}")
        shutil.copy2(local_source, destination)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded source is missing or empty: {destination}")


def extract_wav(
    source_audio: Path,
    output_wav: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start_seconds:.3f}", "-i", str(source_audio),
            "-t", f"{duration_seconds:.3f}", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(output_wav),
        ],
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {process.stderr.strip()[:2000]}")
    if not output_wav.is_file() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create a usable WAV: {output_wav}")


def normalize_podsai_category(
    comparison_label: object,
    primary_label: object,
    background_category: str,
) -> str:
    """Map DCLDE labels to labels that the legacy PODS-AI head predicts."""
    aliases = {
        "resident": "resident", "srkw": "resident", "southern resident": "resident",
        "transient": "transient", "tkw": "transient", "bigg's": "transient",
        "biggs": "transient", "humpback": "humpback", "hw": "humpback",
    }
    for value in (comparison_label, primary_label):
        normalized = _clean(value).casefold()
        if normalized in aliases:
            return aliases[normalized]
    # DCLDE BKG does not identify a distractor subtype. Water is the closest
    # legacy PODS-AI class and can be overridden on the command line.
    return background_category


def primary_wav_path(row: dict[str, str], wav_root: Path) -> Path:
    category = _clean(row.get("Category"))
    node = _clean(row.get("NodeName")).replace("_", "-")
    return wav_root / category / f"{node}_{_clean(row.get('Timestamp'))}.wav"


def secondary_identity(row: dict[str, str]) -> tuple[str, str]:
    provider = safe_slug(row.get("Provider") or "unknown").lower()
    dataset = safe_slug(row.get("Dataset") or "unknown").lower()
    node_name = f"dclde_{provider}_{dataset}"
    timestamp = safe_slug(
        row.get("clip_id") or row.get("manifest_row_id") or row.get("Soundfile")
    )
    if not timestamp:
        raise ValueError("DCLDE row has no usable clip identity")
    return node_name, timestamp


def evaluator_wav_path(
    wav_root: Path, category: str, node_name: str, timestamp: str
) -> Path:
    return wav_root / category / f"{node_name.replace('_', '-')}_{timestamp}.wav"


def copy_primary_rows(
    manifest: Path, wav_root: Path
) -> tuple[list[dict[str, str]], list[str]]:
    fields, rows = read_manifest(manifest)
    require_columns(fields, {"Category", "NodeName", "Timestamp"}, manifest)
    output, missing = [], []
    for row in rows:
        path = primary_wav_path(row, wav_root)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
        else:
            output.append({field: _clean(row.get(field)) for field in TESTING_FIELDS})
    return output, missing


def build_secondary_manifest_row(
    row: dict[str, str], background_category: str
) -> tuple[dict[str, str], str, str]:
    category = normalize_podsai_category(
        row.get("comparison_label"), row.get("primary_label"), background_category
    )
    node_name, timestamp = secondary_identity(row)
    soundfile = _clean(row.get("Soundfile"))
    label_source = _clean(row.get("label_source"))
    primary_label = _clean(row.get("primary_label"))
    notes = "; ".join(
        value for value in (
            "dclde_no_mixed_validation",
            f"primary_label={primary_label}" if primary_label else "",
            f"label_source={label_source}" if label_source else "",
        ) if value
    )
    return ({
        "Category": category,
        "NodeName": node_name,
        "Timestamp": timestamp,
        "URI": _clean(row.get("source_audio_path")),
        "Description": f"DCLDE 60s validation clip from {soundfile}".strip(),
        "Notes": notes,
        "Confidence": "",
    }, node_name, timestamp)


def download_secondary_rows(
    manifest: Path,
    wav_root: Path,
    background_category: str,
    skip_existing: bool,
    max_clips: Optional[int],
) -> tuple[list[dict[str, str]], list[str]]:
    fields, rows = read_manifest(manifest)
    require_columns(fields, {
        "source_audio_path", "window_start_sec", "primary_label", "comparison_label"
    }, manifest)
    if max_clips is not None:
        rows = rows[:max_clips]
    output, failures = [], []
    for index, row in enumerate(rows, start=1):
        try:
            manifest_row, node_name, timestamp = build_secondary_manifest_row(
                row, background_category
            )
            output_wav = evaluator_wav_path(
                wav_root, manifest_row["Category"], node_name, timestamp
            )
            if not (skip_existing and output_wav.is_file() and output_wav.stat().st_size):
                source = _clean(row.get("source_audio_path"))
                if not source:
                    raise ValueError("source_audio_path is empty")
                suffix = Path(source.split("?", 1)[0]).suffix or ".audio"
                with tempfile.TemporaryDirectory(prefix="dclde_podsai_") as temp_dir:
                    local_source = Path(temp_dir) / f"source{suffix}"
                    print(f"[secondary {index}/{len(rows)}] Downloading {source}")
                    download_or_copy_source(source, local_source)
                    start = float(_clean(row.get("window_start_sec")) or 0.0)
                    end = _clean(row.get("window_end_sec"))
                    duration = max(0.001, float(end) - start) if end else 60.0
                    extract_wav(local_source, output_wav, start, min(duration, 60.0))
            else:
                print(f"[secondary {index}/{len(rows)}] Reusing {output_wav}")
            output.append(manifest_row)
        except Exception as error:
            identifier = _clean(row.get("clip_id") or row.get("Soundfile") or index)
            message = f"DCLDE row {index - 1} ({identifier}): {error}"
            print(f"WARNING: {message}", file=sys.stderr)
            failures.append(message)
    return output, failures


def write_testing_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    identities = set()
    for row in rows:
        identity = (row["Category"], row["NodeName"], row["Timestamp"])
        if identity in identities:
            raise ValueError(f"Duplicate evaluator identity: {identity}")
        identities.add(identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TESTING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-manifest", default=DEFAULT_PRIMARY_MANIFEST)
    parser.add_argument("--secondary-manifest", default=DEFAULT_SECONDARY_MANIFEST)
    parser.add_argument("--wav-root", default=DEFAULT_WAV_ROOT)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--primary-cache-root", default=None)
    parser.add_argument("--skip-primary-download", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--max-secondary-clips", type=int, default=None)
    parser.add_argument(
        "--secondary-background-category", default="water",
        help="Legacy label for DCLDE BKG clips (default: water).",
    )
    args = parser.parse_args()
    if args.max_secondary_clips is not None and args.max_secondary_clips <= 0:
        parser.error("--max-secondary-clips must be positive")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required but was not found")
    primary_manifest, secondary_manifest = map(
        Path, (args.primary_manifest, args.secondary_manifest)
    )
    for path in (primary_manifest, secondary_manifest):
        if not path.is_file():
            parser.error(f"manifest not found: {path}")
    wav_root = Path(args.wav_root)
    if not args.skip_primary_download:
        download_primary_manifest_wavs(
            primary_manifest, wav_root,
            cache_root=Path(args.primary_cache_root) if args.primary_cache_root else None,
        )
    primary_rows, missing = copy_primary_rows(primary_manifest, wav_root)
    secondary_rows, failures = download_secondary_rows(
        secondary_manifest, wav_root, args.secondary_background_category,
        not args.no_skip_existing, args.max_secondary_clips,
    )
    combined = [*primary_rows, *secondary_rows]
    output_manifest = Path(args.output_manifest)
    write_testing_manifest(output_manifest, combined)
    counts = {}
    for row in combined:
        counts[row["Category"]] = counts.get(row["Category"], 0) + 1
    print(f"\nCombined legacy manifest: {output_manifest}")
    print(f"Primary clips:             {len(primary_rows):,}")
    print(f"Secondary clips:           {len(secondary_rows):,}")
    print(f"Missing primary clips:     {len(missing):,}")
    print(f"Secondary failures:        {len(failures):,}")
    print(f"Category counts:           {dict(sorted(counts.items()))}")
    return 0 if combined else 1


if __name__ == "__main__":
    raise SystemExit(main())
