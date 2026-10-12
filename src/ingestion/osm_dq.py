from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyogrio

try:
    from src.ingestion.osm_collector import (
        bronze_filename,
        bronze_path,
        load_config,
        snapshot_date,
        validate_config,
        validate_osm_pbf,
    )

except ModuleNotFoundError:
    from osm_collector import (
        bronze_filename,
        bronze_path,
        load_config,
        snapshot_date,
        validate_config,
        validate_osm_pbf,
    )


@dataclass(frozen=True)
class SnapshotDQResult:
    snapshot_date: str
    bronze_exists: bool
    physical_ok: bool
    semantic_ok: bool
    file_bytes: int
    md5: str | None
    layers: tuple[str, ...]
    error: str | None


@dataclass(frozen=True)
class SourceDQResult:
    expected_snapshot_count: int
    present_snapshot_count: int
    valid_snapshot_count: int
    missing_snapshots: tuple[str, ...]
    invalid_snapshots: tuple[str, ...]
    source_coverage_ok: bool


def snapshot_path(
    config: dict[str, Any],
    raw_dir: Path | None = None,
) -> Path:
    if raw_dir is None:
        return bronze_path(config)

    return (
        raw_dir
        / (
            "snapshot_date="
            f"{snapshot_date(config)}"
        )
        / bronze_filename(config)
    )


def inspect_pbf_container(
    path: Path,
) -> tuple[
    bool,
    tuple[str, ...],
    str | None,
]:
    if not path.is_file():
        return (
            False,
            (),
            "Artifact does not exist.",
        )

    if path.stat().st_size <= 0:
        return (
            False,
            (),
            "Artifact is empty.",
        )

    drivers = pyogrio.list_drivers()

    if "OSM" not in drivers:
        return (
            False,
            (),
            "GDAL OSM driver is unavailable.",
        )

    osm_mode = str(
        drivers["OSM"]
    )

    if "r" not in osm_mode:
        return (
            False,
            (),
            (
                "GDAL OSM driver does not "
                "support reading."
            ),
        )

    try:
        layer_info = pyogrio.list_layers(
            path
        )

    except Exception as exc:
        return (
            False,
            (),
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

    layers = tuple(
        str(row[0])
        for row in layer_info
    )

    if not layers:
        return (
            False,
            (),
            (
                "OSM PBF contains no "
                "discoverable layers."
            ),
        )

    return (
        True,
        layers,
        None,
    )


def validate_snapshot_dq(
    config: dict[str, Any],
    raw_dir: Path | None = None,
) -> SnapshotDQResult:
    path = snapshot_path(
        config=config,
        raw_dir=raw_dir,
    )

    snapshot_value = snapshot_date(
        config
    )

    if not path.is_file():
        return SnapshotDQResult(
            snapshot_date=snapshot_value,
            bronze_exists=False,
            physical_ok=False,
            semantic_ok=False,
            file_bytes=0,
            md5=None,
            layers=(),
            error=(
                "Bronze artifact does not exist."
            ),
        )

    file_bytes = path.stat().st_size

    (
        physical_ok,
        physical_layers,
        physical_error,
    ) = inspect_pbf_container(path)

    if not physical_ok:
        return SnapshotDQResult(
            snapshot_date=snapshot_value,
            bronze_exists=True,
            physical_ok=False,
            semantic_ok=False,
            file_bytes=file_bytes,
            md5=None,
            layers=physical_layers,
            error=physical_error,
        )

    try:
        metrics = validate_osm_pbf(
            path=path,
            config=config,
        )

        return SnapshotDQResult(
            snapshot_date=snapshot_value,
            bronze_exists=True,
            physical_ok=True,
            semantic_ok=True,
            file_bytes=int(
                metrics["file_bytes"]
            ),
            md5=str(
                metrics["md5"]
            ),
            layers=tuple(
                str(value)
                for value in metrics["layers"]
            ),
            error=None,
        )

    except Exception as exc:
        return SnapshotDQResult(
            snapshot_date=snapshot_value,
            bronze_exists=True,
            physical_ok=True,
            semantic_ok=False,
            file_bytes=file_bytes,
            md5=None,
            layers=physical_layers,
            error=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


def validate_source_dq(
    config: dict[str, Any],
    raw_dir: Path | None = None,
) -> tuple[
    SnapshotDQResult,
    SourceDQResult,
]:
    result = validate_snapshot_dq(
        config=config,
        raw_dir=raw_dir,
    )

    snapshot_value = snapshot_date(
        config
    )

    present_count = (
        1
        if result.bronze_exists
        else 0
    )

    valid_count = (
        1
        if (
            result.physical_ok
            and result.semantic_ok
        )
        else 0
    )

    missing_snapshots = (
        ()
        if result.bronze_exists
        else (snapshot_value,)
    )

    invalid_snapshots = (
        (snapshot_value,)
        if (
            result.bronze_exists
            and not (
                result.physical_ok
                and result.semantic_ok
            )
        )
        else ()
    )

    source_result = SourceDQResult(
        expected_snapshot_count=1,
        present_snapshot_count=present_count,
        valid_snapshot_count=valid_count,
        missing_snapshots=(
            missing_snapshots
        ),
        invalid_snapshots=(
            invalid_snapshots
        ),
        source_coverage_ok=(
            valid_count == 1
        ),
    )

    return (
        result,
        source_result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "OpenStreetMap Bronze "
            "data-quality validation."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Optional OSM raw directory "
            "override for isolated DQ tests."
        ),
    )

    args = parser.parse_args()

    config = load_config()
    validate_config(config)

    result, source_result = (
        validate_source_dq(
            config=config,
            raw_dir=args.raw_dir,
        )
    )

    print(
        "SNAPSHOT_DATE="
        f"{result.snapshot_date}"
    )

    print(
        "BRONZE_EXISTS="
        f"{result.bronze_exists}"
    )

    print(
        "PHYSICAL_OK="
        f"{result.physical_ok}"
    )

    print(
        "SEMANTIC_OK="
        f"{result.semantic_ok}"
    )

    print(
        "FILE_BYTES="
        f"{result.file_bytes}"
    )

    print(
        "MD5="
        f"{result.md5}"
    )

    print(
        "LAYERS="
        f"{list(result.layers)}"
    )

    print(
        "ERROR="
        f"{result.error}"
    )

    print(
        "EXPECTED_SNAPSHOT_COUNT="
        f"{source_result.expected_snapshot_count}"
    )

    print(
        "PRESENT_SNAPSHOT_COUNT="
        f"{source_result.present_snapshot_count}"
    )

    print(
        "VALID_SNAPSHOT_COUNT="
        f"{source_result.valid_snapshot_count}"
    )

    print(
        "MISSING_SNAPSHOTS="
        f"{list(source_result.missing_snapshots)}"
    )

    print(
        "INVALID_SNAPSHOTS="
        f"{list(source_result.invalid_snapshots)}"
    )

    print(
        "SOURCE_COVERAGE_OK="
        f"{source_result.source_coverage_ok}"
    )

    raise SystemExit(
        0
        if source_result.source_coverage_ok
        else 1
    )


if __name__ == "__main__":
    main()