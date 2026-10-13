from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import yaml

from gsmap_now_collector import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    parse_event_start_time,
    validate_config,
    validate_gsmap_now_netcdf,
)


HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


@dataclass
class ArtifactDQResult:
    path: Path
    filename: str
    event_date: str | None
    event_start_time: str | None
    bronze_exists: bool
    physical_ok: bool
    semantic_ok: bool
    partition_ok: bool
    file_bytes: int
    error: str | None


@dataclass
class SourceDQResult:
    artifact_count: int
    valid_artifact_count: int
    invalid_artifacts: list[str]
    duplicate_event_times: list[str]
    source_coverage_ok: bool


def load_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"GSMaP NOW config does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "GSMaP NOW config must be a YAML mapping."
        )

    validate_config(
        config
    )

    return config


def has_valid_hdf5_container(
    path: Path,
) -> bool:
    if not path.is_file():
        return False

    if path.stat().st_size <= 0:
        return False

    try:
        with path.open("rb") as file:
            signature = file.read(8)

        if signature != HDF5_SIGNATURE:
            return False

        with h5py.File(
            path,
            "r",
        ):
            pass

    except Exception:
        return False

    return True


def inspect_artifact(
    path: Path,
    config: dict[str, Any],
) -> ArtifactDQResult:
    filename = path.name

    bronze_exists = path.is_file()

    file_bytes = (
        path.stat().st_size
        if bronze_exists
        else 0
    )

    physical_ok = (
        has_valid_hdf5_container(
            path
        )
    )

    event_date: str | None = None
    event_start_time: str | None = None
    partition_ok = False
    semantic_ok = False
    error: str | None = None

    try:
        parsed_time = (
            parse_event_start_time(
                filename=filename,
                config=config,
            )
        )

        event_start_time = (
            parsed_time.isoformat()
        )

        event_date = (
            parsed_time
            .date()
            .isoformat()
        )

        expected_partition = (
            f"event_date={event_date}"
        )

        partition_ok = (
            path.parent.name
            == expected_partition
        )

        if not partition_ok:
            raise ValueError(
                "Bronze partition does not "
                "match filename event date: "
                f"expected={expected_partition}, "
                f"actual={path.parent.name}"
            )

        validate_gsmap_now_netcdf(
            path=path,
            filename=filename,
            config=config,
        )

        semantic_ok = True

    except Exception as exc:
        error = (
            f"{type(exc).__name__}: {exc}"
        )

    return ArtifactDQResult(
        path=path,
        filename=filename,
        event_date=event_date,
        event_start_time=event_start_time,
        bronze_exists=bronze_exists,
        physical_ok=physical_ok,
        semantic_ok=semantic_ok,
        partition_ok=partition_ok,
        file_bytes=file_bytes,
        error=error,
    )


def audit_source(
    raw_dir: Path,
    config: dict[str, Any],
) -> tuple[
    list[ArtifactDQResult],
    SourceDQResult,
]:
    artifacts = sorted(
        raw_dir.rglob(
            "gsmap_now_rain.*.nc"
        )
    )

    results = [
        inspect_artifact(
            path=artifact,
            config=config,
        )
        for artifact in artifacts
    ]

    valid_results = [
        result
        for result in results
        if (
            result.bronze_exists
            and result.physical_ok
            and result.semantic_ok
            and result.partition_ok
        )
    ]

    invalid_artifacts = [
        str(result.path)
        for result in results
        if not (
            result.bronze_exists
            and result.physical_ok
            and result.semantic_ok
            and result.partition_ok
        )
    ]

    event_times = [
        result.event_start_time
        for result in results
        if result.event_start_time
        is not None
    ]

    seen: set[str] = set()

    duplicates: set[str] = set()

    for event_time in event_times:
        if event_time in seen:
            duplicates.add(
                event_time
            )

        seen.add(
            event_time
        )

    duplicate_event_times = sorted(
        duplicates
    )

    source_coverage_ok = (
        len(results) > 0
        and len(valid_results)
        == len(results)
        and not invalid_artifacts
        and not duplicate_event_times
    )

    summary = SourceDQResult(
        artifact_count=len(
            results
        ),
        valid_artifact_count=len(
            valid_results
        ),
        invalid_artifacts=invalid_artifacts,
        duplicate_event_times=(
            duplicate_event_times
        ),
        source_coverage_ok=(
            source_coverage_ok
        ),
    )

    return (
        results,
        summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Data-quality audit for "
            "GSMaP NOW Bronze artifacts."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Optional Bronze raw directory "
            "override for isolated tests."
        ),
    )

    args = parser.parse_args()

    config = load_config()

    if args.raw_dir is None:
        raw_dir = (
            PROJECT_ROOT
            / config["bronze"]["raw_dir"]
        )
    else:
        raw_dir = (
            args.raw_dir.resolve()
        )

    results, summary = (
        audit_source(
            raw_dir=raw_dir,
            config=config,
        )
    )

    for result in results:
        print(
            "FILENAME="
            f"{result.filename}"
        )

        print(
            "EVENT_DATE="
            f"{result.event_date}"
        )

        print(
            "EVENT_START_TIME="
            f"{result.event_start_time}"
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
            "PARTITION_OK="
            f"{result.partition_ok}"
        )

        print(
            "FILE_BYTES="
            f"{result.file_bytes}"
        )

        print(
            "ERROR="
            f"{result.error}"
        )

        print("---")

    print(
        "ARTIFACT_COUNT="
        f"{summary.artifact_count}"
    )

    print(
        "VALID_ARTIFACT_COUNT="
        f"{summary.valid_artifact_count}"
    )

    print(
        "INVALID_ARTIFACTS="
        f"{summary.invalid_artifacts}"
    )

    print(
        "DUPLICATE_EVENT_TIMES="
        f"{summary.duplicate_event_times}"
    )

    print(
        "SOURCE_COVERAGE_OK="
        f"{summary.source_coverage_ok}"
    )

    raise SystemExit(
        0
        if summary.source_coverage_ok
        else 1
    )


if __name__ == "__main__":
    main()