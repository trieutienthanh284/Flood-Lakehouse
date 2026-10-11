from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rasterio
from rasterio.errors import RasterioIOError

try:
    from src.ingestion.worldcover_collector import (
        bronze_path_for_tile,
        configured_tiles,
        filename_for_tile,
        load_config,
        validate_config,
        validate_worldcover_raster,
    )

except ModuleNotFoundError:
    from worldcover_collector import (
        bronze_path_for_tile,
        configured_tiles,
        filename_for_tile,
        load_config,
        validate_config,
        validate_worldcover_raster,
    )


@dataclass(frozen=True)
class TileDQResult:
    tile: str
    bronze_exists: bool
    physical_ok: bool
    semantic_ok: bool
    file_bytes: int
    classes: tuple[int, ...]
    error: str | None


@dataclass(frozen=True)
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
            tile=tile,
            config=config,
        )

    filename = filename_for_tile(
        tile=tile,
        config=config,
    )

    return (
        raw_dir
        / f"tile={tile}"
        / filename
    )


def has_readable_raster_container(
    path: Path,
) -> bool:
    if not path.is_file():
        return False

    if path.stat().st_size <= 0:
        return False

    try:
        with rasterio.open(path) as src:
            _ = src.driver

        return True

    except (
        RasterioIOError,
        OSError,
    ):
        return False


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
            classes=(),
            error=(
                "Bronze artifact does not exist."
            ),
        )

    file_bytes = path.stat().st_size

    physical_ok = (
        has_readable_raster_container(path)
    )

    if not physical_ok:
        return TileDQResult(
            tile=tile,
            bronze_exists=True,
            physical_ok=False,
            semantic_ok=False,
            file_bytes=file_bytes,
            classes=(),
            error=(
                "Artifact is not a readable "
                "raster container."
            ),
        )

    try:
        metrics = (
            validate_worldcover_raster(
                path=path,
                tile=tile,
                config=config,
            )
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
            classes=tuple(
                int(value)
                for value in metrics[
                    "classes"
                ]
            ),
            error=None,
        )

    except Exception as exc:
        return TileDQResult(
            tile=tile,
            bronze_exists=True,
            physical_ok=True,
            semantic_ok=False,
            file_bytes=file_bytes,
            classes=(),
            error=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


def validate_source_dq(
    config: dict[str, Any],
    raw_dir: Path | None = None,
) -> tuple[
    tuple[TileDQResult, ...],
    SourceDQResult,
]:
    tiles = configured_tiles(config)

    results = tuple(
        validate_tile_dq(
            tile=tile,
            config=config,
            raw_dir=raw_dir,
        )
        for tile in tiles
    )

    missing_tiles = tuple(
        result.tile
        for result in results
        if not result.bronze_exists
    )

    invalid_tiles = tuple(
        result.tile
        for result in results
        if (
            result.bronze_exists
            and not (
                result.physical_ok
                and result.semantic_ok
            )
        )
    )

    present_tile_count = sum(
        1
        for result in results
        if result.bronze_exists
    )

    valid_tile_count = sum(
        1
        for result in results
        if (
            result.physical_ok
            and result.semantic_ok
        )
    )

    source_coverage_ok = (
        valid_tile_count == len(tiles)
    )

    source_result = SourceDQResult(
        expected_tile_count=len(tiles),
        present_tile_count=present_tile_count,
        valid_tile_count=valid_tile_count,
        missing_tiles=missing_tiles,
        invalid_tiles=invalid_tiles,
        source_coverage_ok=(
            source_coverage_ok
        ),
    )

    return (
        results,
        source_result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ESA WorldCover Bronze "
            "data-quality validation."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Optional raw directory override "
            "for isolated DQ testing."
        ),
    )

    args = parser.parse_args()

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
            "CLASSES="
            f"{list(result.classes)}"
        )
        print(
            "ERROR="
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

    raise SystemExit(
        0
        if source_result.source_coverage_ok
        else 1
    )


if __name__ == "__main__":
    main()