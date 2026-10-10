from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from .srtm_collector import (
        PROJECT_ROOT,
        archive_filename,
        bronze_path_for_tile,
        configured_tiles,
        has_valid_zip_signature,
        load_config,
        validate_config,
        validate_srtm_archive,
    )
except ImportError:
    from srtm_collector import (
        PROJECT_ROOT,
        archive_filename,
        bronze_path_for_tile,
        configured_tiles,
        has_valid_zip_signature,
        load_config,
        validate_config,
        validate_srtm_archive,
    )


@dataclass
class TileDQResult:
    tile: str
    bronze_exists: bool
    physical_ok: bool
    semantic_ok: bool
    file_bytes: int
    member_name: str | None
    sample_count: int
    error: str | None


@dataclass
class SourceDQResult:
    expected_tile_count: int
    present_tile_count: int
    valid_tile_count: int
    missing_tiles: tuple[str, ...]
    invalid_tiles: tuple[str, ...]
    source_coverage_ok: bool


def bronze_path(
    tile: str,
    config: dict[str, Any],
    raw_dir: Path | None = None,
) -> Path:
    if raw_dir is None:
        return bronze_path_for_tile(
            tile,
            config,
        )

    return (
        raw_dir
        / f"tile={tile}"
        / archive_filename(tile)
    )


def validate_tile_dq(
    tile: str,
    config: dict[str, Any],
    raw_dir: Path | None = None,
) -> TileDQResult:
    path = bronze_path(
        tile=tile,
        config=config,
        raw_dir=raw_dir,
    )

    if not path.is_file():
        return TileDQResult(
            tile=tile,
            bronze_exists=False,
            physical_ok=False,
            semantic_ok=False,
            file_bytes=0,
            member_name=None,
            sample_count=0,
            error="Bronze artifact does not exist.",
        )

    physical_ok = has_valid_zip_signature(
        path
    )

    if not physical_ok:
        return TileDQResult(
            tile=tile,
            bronze_exists=True,
            physical_ok=False,
            semantic_ok=False,
            file_bytes=path.stat().st_size,
            member_name=None,
            sample_count=0,
            error="Invalid ZIP physical signature.",
        )

    try:
        metrics = validate_srtm_archive(
            path=path,
            tile=tile,
            config=config,
        )

        return TileDQResult(
            tile=tile,
            bronze_exists=True,
            physical_ok=True,
            semantic_ok=bool(
                metrics["semantic_ok"]
            ),
            file_bytes=int(
                metrics["file_bytes"]
            ),
            member_name=str(
                metrics["member_name"]
            ),
            sample_count=int(
                metrics["sample_count"]
            ),
            error=None,
        )

    except Exception as exc:
        return TileDQResult(
            tile=tile,
            bronze_exists=True,
            physical_ok=True,
            semantic_ok=False,
            file_bytes=path.stat().st_size,
            member_name=None,
            sample_count=0,
            error=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


def validate_source_dq(
    config: dict[str, Any],
    raw_dir: Path | None = None,
) -> tuple[
    list[TileDQResult],
    SourceDQResult,
]:
    tiles = configured_tiles(config)

    tile_results = [
        validate_tile_dq(
            tile=tile,
            config=config,
            raw_dir=raw_dir,
        )
        for tile in tiles
    ]

    present_tile_count = sum(
        result.bronze_exists
        for result in tile_results
    )

    valid_tile_count = sum(
        (
            result.bronze_exists
            and result.physical_ok
            and result.semantic_ok
        )
        for result in tile_results
    )

    missing_tiles = tuple(
        result.tile
        for result in tile_results
        if not result.bronze_exists
    )

    invalid_tiles = tuple(
        result.tile
        for result in tile_results
        if (
            result.bronze_exists
            and not (
                result.physical_ok
                and result.semantic_ok
            )
        )
    )

    source_coverage_ok = (
        len(tile_results)
        == len(tiles)
        and present_tile_count
        == len(tiles)
        and valid_tile_count
        == len(tiles)
        and not missing_tiles
        and not invalid_tiles
    )

    source_result = SourceDQResult(
        expected_tile_count=len(tiles),
        present_tile_count=
            present_tile_count,
        valid_tile_count=
            valid_tile_count,
        missing_tiles=missing_tiles,
        invalid_tiles=invalid_tiles,
        source_coverage_ok=
            source_coverage_ok,
    )

    return (
        tile_results,
        source_result,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Data quality validation for "
            "SRTM Bronze source."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Optional alternative raw root "
            "for isolated DQ testing."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config()
    validate_config(config)

    tile_results, source_result = (
        validate_source_dq(
            config=config,
            raw_dir=args.raw_dir,
        )
    )

    for result in tile_results:
        print(f"TILE={result.tile}")
        print(
            f"BRONZE_EXISTS="
            f"{result.bronze_exists}"
        )
        print(
            f"PHYSICAL_OK="
            f"{result.physical_ok}"
        )
        print(
            f"SEMANTIC_OK="
            f"{result.semantic_ok}"
        )
        print(
            f"FILE_BYTES="
            f"{result.file_bytes}"
        )
        print(
            f"MEMBER_NAME="
            f"{result.member_name}"
        )
        print(
            f"SAMPLE_COUNT="
            f"{result.sample_count}"
        )
        print(
            f"ERROR="
            f"{result.error}"
        )

    print(
        "EXPECTED_TILE_COUNT="
        f"{source_result.expected_tile_count}"
    )
    print(
        "PRESENT_TILE_COUNT="
        f"{source_result.present_tile_count}"
    )
    print(
        "VALID_TILE_COUNT="
        f"{source_result.valid_tile_count}"
    )
    print(
        "MISSING_TILES="
        f"{list(source_result.missing_tiles)}"
    )
    print(
        "INVALID_TILES="
        f"{list(source_result.invalid_tiles)}"
    )
    print(
        "SOURCE_COVERAGE_OK="
        f"{source_result.source_coverage_ok}"
    )

    sys.exit(
        0
        if source_result.source_coverage_ok
        else 1
    )


if __name__ == "__main__":
    main()