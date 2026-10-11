from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import math

import numpy as np
import rasterio

import yaml
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "worldcover.yaml"
)

EXPECTED_DATASET = "ESA WorldCover"
EXPECTED_PRODUCT_YEAR = 2021
EXPECTED_PRODUCT_VERSION = "v200"
EXPECTED_LAYER = "Map"
EXPECTED_ROLE = "static_land_cover"
EXPECTED_CRS = "EPSG:4326"

TILE_RE = re.compile(
    r"^[NS]\d{2}[EW]\d{3}$"
)

CONNECT_TIMEOUT_SECONDS = 15
READ_TIMEOUT_SECONDS = 120
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

def load_config(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"WorldCover config does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            "WorldCover config must be a YAML mapping."
        )

    return config


def validate_config(
    config: dict[str, Any],
) -> None:
    if config.get("dataset") != EXPECTED_DATASET:
        raise ValueError(
            "Unexpected WorldCover dataset."
        )

    if (
        config.get("product_year")
        != EXPECTED_PRODUCT_YEAR
    ):
        raise ValueError(
            "Unexpected WorldCover product year."
        )

    if (
        config.get("product_version")
        != EXPECTED_PRODUCT_VERSION
    ):
        raise ValueError(
            "Unexpected WorldCover product version."
        )

    if config.get("layer") != EXPECTED_LAYER:
        raise ValueError(
            "Unexpected WorldCover layer."
        )

    if config.get("role") != EXPECTED_ROLE:
        raise ValueError(
            "Unexpected WorldCover role."
        )

    spatial = config.get("spatial")

    if not isinstance(spatial, dict):
        raise ValueError(
            "Missing spatial config."
        )

    if spatial.get("crs") != EXPECTED_CRS:
        raise ValueError(
            "Unexpected WorldCover CRS."
        )

    tiles = spatial.get("tiles")

    if (
        not isinstance(tiles, list)
        or not tiles
    ):
        raise ValueError(
            "WorldCover tiles must be a non-empty list."
        )

    if len(tiles) != len(set(tiles)):
        raise ValueError(
            "Duplicate WorldCover tile IDs."
        )

    for tile in tiles:
        if (
            not isinstance(tile, str)
            or TILE_RE.fullmatch(tile) is None
        ):
            raise ValueError(
                f"Invalid WorldCover tile ID: {tile!r}"
            )

    tile_size = spatial.get(
        "tile_size_degrees"
    )

    pixel_size = spatial.get(
        "pixel_size_degrees"
    )

    if not isinstance(
        tile_size,
        (int, float),
    ):
        raise ValueError(
            "Invalid tile_size_degrees."
        )

    if not isinstance(
        pixel_size,
        (int, float),
    ):
        raise ValueError(
            "Invalid pixel_size_degrees."
        )

    if tile_size <= 0 or pixel_size <= 0:
        raise ValueError(
            "Spatial sizes must be positive."
        )

    expected_pixels = round(
        tile_size / pixel_size
    )

    validation = config.get(
        "validation"
    )

    if not isinstance(validation, dict):
        raise ValueError(
            "Missing validation config."
        )

    raster = validation.get("raster")

    if not isinstance(raster, dict):
        raise ValueError(
            "Missing raster validation config."
        )

    if raster.get("driver") != "GTiff":
        raise ValueError(
            "Unexpected raster driver."
        )

    if (
        raster.get("width")
        != expected_pixels
    ):
        raise ValueError(
            "Raster width does not match "
            "tile/pixel-size contract."
        )

    if (
        raster.get("height")
        != expected_pixels
    ):
        raise ValueError(
            "Raster height does not match "
            "tile/pixel-size contract."
        )

    if raster.get("band_count") != 1:
        raise ValueError(
            "WorldCover Map must have one band."
        )

    if raster.get("dtype") != "uint8":
        raise ValueError(
            "Unexpected WorldCover dtype."
        )

    if raster.get("nodata_value") != 0:
        raise ValueError(
            "Unexpected WorldCover NoData value."
        )

    if raster.get("tiled") is not True:
        raise ValueError(
            "WorldCover raster must be tiled."
        )

    if raster.get("block_width") != 1024:
        raise ValueError(
            "Unexpected raster block width."
        )

    if raster.get("block_height") != 1024:
        raise ValueError(
            "Unexpected raster block height."
        )

    if (
        str(
            raster.get(
                "compression",
                "",
            )
        ).lower()
        != "deflate"
    ):
        raise ValueError(
            "Unexpected raster compression."
        )

    classes = validation.get(
        "classes"
    )

    if not isinstance(classes, dict):
        raise ValueError(
            "Missing class validation config."
        )

    expected_classes = {
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        95,
        100,
    }

    allowed_values = classes.get(
        "allowed_values"
    )

    if not isinstance(
        allowed_values,
        list,
    ):
        raise ValueError(
            "allowed_values must be a list."
        )

    actual_classes = {
        int(value)
        for value in allowed_values
    }

    if actual_classes != expected_classes:
        raise ValueError(
            "Unexpected WorldCover class set."
        )

    labels = classes.get("labels")

    if not isinstance(labels, dict):
        raise ValueError(
            "WorldCover labels must be a mapping."
        )

    label_classes = {
        int(value)
        for value in labels.keys()
    }

    if label_classes != expected_classes:
        raise ValueError(
            "WorldCover labels do not cover "
            "the expected class set."
        )

    access = config.get("access")

    if not isinstance(access, dict):
        raise ValueError(
            "Missing access config."
        )

    base_url = access.get(
        "base_url"
    )

    filename_template = access.get(
        "filename_template"
    )

    if (
        not isinstance(base_url, str)
        or not base_url.startswith(
            "https://"
        )
    ):
        raise ValueError(
            "Invalid WorldCover base URL."
        )

    if (
        not isinstance(
            filename_template,
            str,
        )
        or "{tile}"
        not in filename_template
    ):
        raise ValueError(
            "WorldCover filename template "
            "must contain {tile}."
        )

    bronze = config.get("bronze")

    if not isinstance(bronze, dict):
        raise ValueError(
            "Missing Bronze config."
        )

    if bronze.get("source") != "worldcover":
        raise ValueError(
            "Unexpected Bronze source."
        )

    if (
        bronze.get("partition_key")
        != "tile"
    ):
        raise ValueError(
            "WorldCover Bronze must partition "
            "by tile."
        )

    if (
        bronze.get("staging_suffix")
        != ".part"
    ):
        raise ValueError(
            "Unexpected staging suffix."
        )

    required_paths = (
        "root",
        "staging_dir",
        "raw_dir",
    )

    for key in required_paths:
        value = bronze.get(key)

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"Invalid Bronze path: {key}"
            )


def configured_tiles(
    config: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(
        config["spatial"]["tiles"]
    )


def validate_tile(
    tile: str,
    config: dict[str, Any],
) -> str:
    if TILE_RE.fullmatch(tile) is None:
        raise ValueError(
            f"Invalid tile format: {tile!r}"
        )

    if tile not in configured_tiles(config):
        raise ValueError(
            f"Tile is not configured "
            f"for current AOI: {tile}"
        )

    return tile


def filename_for_tile(
    tile: str,
    config: dict[str, Any],
) -> str:
    validate_tile(
        tile=tile,
        config=config,
    )

    return config["access"][
        "filename_template"
    ].format(
        tile=tile
    )


def remote_url_for_tile(
    tile: str,
    config: dict[str, Any],
) -> str:
    filename = filename_for_tile(
        tile=tile,
        config=config,
    )

    base_url = config[
        "access"
    ]["base_url"].rstrip("/")

    return f"{base_url}/{filename}"


def staging_path_for_tile(
    tile: str,
    config: dict[str, Any],
) -> Path:
    filename = filename_for_tile(
        tile=tile,
        config=config,
    )

    staging_dir = (
        PROJECT_ROOT
        / config["bronze"]["staging_dir"]
    )

    suffix = config[
        "bronze"
    ]["staging_suffix"]

    return (
        staging_dir
        / f"{filename}{suffix}"
    )


def bronze_path_for_tile(
    tile: str,
    config: dict[str, Any],
) -> Path:
    filename = filename_for_tile(
        tile=tile,
        config=config,
    )

    raw_dir = (
        PROJECT_ROOT
        / config["bronze"]["raw_dir"]
    )

    return (
        raw_dir
        / f"tile={tile}"
        / filename
    )

def tile_bounds_from_id(
    tile: str,
    config: dict[str, Any],
) -> tuple[float, float, float, float]:
    validate_tile(
        tile=tile,
        config=config,
    )

    lat_sign = 1 if tile[0] == "N" else -1
    lon_sign = 1 if tile[3] == "E" else -1

    lat = lat_sign * int(tile[1:3])
    lon = lon_sign * int(tile[4:7])

    tile_size = float(
        config["spatial"]["tile_size_degrees"]
    )

    left = float(lon)
    bottom = float(lat)
    right = left + tile_size
    top = bottom + tile_size

    return (
        left,
        bottom,
        right,
        top,
    )


def validate_worldcover_raster(
    path: Path,
    tile: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    validate_tile(
        tile=tile,
        config=config,
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"WorldCover artifact does not exist: {path}"
        )

    file_bytes = path.stat().st_size

    if file_bytes <= 0:
        raise ValueError(
            "WorldCover artifact is empty."
        )

    raster_cfg = config[
        "validation"
    ]["raster"]

    allowed_classes = {
        int(value)
        for value in config[
            "validation"
        ]["classes"]["allowed_values"]
    }

    expected_bounds = tile_bounds_from_id(
        tile=tile,
        config=config,
    )

    expected_resolution = float(
        config["spatial"][
            "pixel_size_degrees"
        ]
    )

    observed_classes: set[int] = set()

    with rasterio.open(path) as src:
        if src.driver != raster_cfg["driver"]:
            raise ValueError(
                f"Unexpected driver: {src.driver}"
            )

        if str(src.crs) != config["spatial"]["crs"]:
            raise ValueError(
                f"Unexpected CRS: {src.crs}"
            )

        if src.width != raster_cfg["width"]:
            raise ValueError(
                f"Unexpected raster width: {src.width}"
            )

        if src.height != raster_cfg["height"]:
            raise ValueError(
                f"Unexpected raster height: {src.height}"
            )

        if src.count != raster_cfg["band_count"]:
            raise ValueError(
                f"Unexpected band count: {src.count}"
            )

        if src.dtypes != (
            raster_cfg["dtype"],
        ):
            raise ValueError(
                f"Unexpected dtype: {src.dtypes}"
            )

        if src.nodata != float(
            raster_cfg["nodata_value"]
        ):
            raise ValueError(
                f"Unexpected NoData value: {src.nodata}"
            )

        if src.is_tiled is not True:
            raise ValueError(
                "WorldCover raster is not tiled."
            )

        expected_block = (
            raster_cfg["block_height"],
            raster_cfg["block_width"],
        )

        if src.block_shapes != [
            expected_block
        ]:
            raise ValueError(
                f"Unexpected block shape: "
                f"{src.block_shapes}"
            )

        compression = (
            src.compression.value.lower()
            if src.compression is not None
            else None
        )

        if (
            compression
            != str(
                raster_cfg["compression"]
            ).lower()
        ):
            raise ValueError(
                f"Unexpected compression: "
                f"{compression}"
            )

        xres, yres = src.res

        if not math.isclose(
            xres,
            expected_resolution,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Unexpected x resolution: {xres}"
            )

        if not math.isclose(
            yres,
            expected_resolution,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Unexpected y resolution: {yres}"
            )

        actual_bounds = (
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
        )

        for actual, expected in zip(
            actual_bounds,
            expected_bounds,
        ):
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "Raster bounds do not match "
                    f"tile {tile}: "
                    f"{actual_bounds}"
                )

        for _, window in src.block_windows(1):
            block = src.read(
                1,
                window=window,
            )

            observed_classes.update(
                int(value)
                for value in np.unique(block)
            )

            unexpected = (
                observed_classes
                - allowed_classes
            )

            if unexpected:
                raise ValueError(
                    "Unexpected WorldCover "
                    f"class values: "
                    f"{sorted(unexpected)}"
                )

    return {
        "file_bytes": file_bytes,
        "driver": raster_cfg["driver"],
        "crs": config["spatial"]["crs"],
        "width": raster_cfg["width"],
        "height": raster_cfg["height"],
        "band_count": raster_cfg["band_count"],
        "dtype": raster_cfg["dtype"],
        "nodata": raster_cfg["nodata_value"],
        "classes": sorted(
            observed_classes
        ),
        "physical_ok": True,
        "semantic_ok": True,
    }

def build_http_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset(
            {
                "GET",
                "HEAD",
            }
        ),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    return session

def download_to_staging(
    tile: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    validate_tile(
        tile=tile,
        config=config,
    )

    url = remote_url_for_tile(
        tile=tile,
        config=config,
    )

    staging_path = staging_path_for_tile(
        tile=tile,
        config=config,
    )

    staging_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if staging_path.exists():
        staging_path.unlink()

    session = build_http_session()

    try:
        with session.get(
            url,
            stream=True,
            timeout=(
                CONNECT_TIMEOUT_SECONDS,
                READ_TIMEOUT_SECONDS,
            ),
        ) as response:
            response.raise_for_status()

            with staging_path.open(
                "wb"
            ) as f:
                for chunk in response.iter_content(
                    chunk_size=DOWNLOAD_CHUNK_BYTES
                ):
                    if chunk:
                        f.write(chunk)

        if not staging_path.is_file():
            raise RuntimeError(
                "WorldCover staging artifact "
                "was not created."
            )

        if staging_path.stat().st_size <= 0:
            raise RuntimeError(
                "WorldCover staging artifact "
                "is empty."
            )

        metrics = validate_worldcover_raster(
            path=staging_path,
            tile=tile,
            config=config,
        )

        return metrics

    except Exception:
        if staging_path.exists():
            staging_path.unlink()

        raise

    finally:
        session.close()

def promote_to_bronze(
    tile: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    validate_tile(
        tile=tile,
        config=config,
    )

    staging_path = staging_path_for_tile(
        tile=tile,
        config=config,
    )

    bronze_path = bronze_path_for_tile(
        tile=tile,
        config=config,
    )

    if not staging_path.is_file():
        raise FileNotFoundError(
            "WorldCover staging artifact "
            f"does not exist: {staging_path}"
        )

    if bronze_path.exists():
        raise FileExistsError(
            "WorldCover Bronze artifact "
            f"already exists: {bronze_path}"
        )

    staging_metrics = (
        validate_worldcover_raster(
            path=staging_path,
            tile=tile,
            config=config,
        )
    )

    staging_bytes = (
        staging_path.stat().st_size
    )

    bronze_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_anchor = (
        staging_path.resolve().anchor.lower()
    )

    bronze_anchor = (
        bronze_path.parent.resolve()
        .anchor.lower()
    )

    if staging_anchor != bronze_anchor:
        raise RuntimeError(
            "Staging and Bronze destination "
            "must be on the same filesystem "
            "for atomic promotion."
        )

    os.replace(
        staging_path,
        bronze_path,
    )

    if not bronze_path.is_file():
        raise RuntimeError(
            "Bronze artifact was not created "
            "after atomic promotion."
        )

    if staging_path.exists():
        raise RuntimeError(
            "Staging artifact still exists "
            "after atomic promotion."
        )

    bronze_bytes = (
        bronze_path.stat().st_size
    )

    if bronze_bytes != staging_bytes:
        raise RuntimeError(
            "Bronze artifact byte size changed "
            "during promotion."
        )

    bronze_metrics = (
        validate_worldcover_raster(
            path=bronze_path,
            tile=tile,
            config=config,
        )
    )

    if (
        bronze_metrics["file_bytes"]
        != staging_metrics["file_bytes"]
    ):
        raise RuntimeError(
            "Validated byte size changed "
            "during promotion."
        )

    return bronze_metrics

def get_valid_existing_bronze(
    tile: str,
    config: dict[str, Any],
) -> Path | None:
    bronze_path = bronze_path_for_tile(
        tile=tile,
        config=config,
    )

    if not bronze_path.is_file():
        return None

    validate_worldcover_raster(
        path=bronze_path,
        tile=tile,
        config=config,
    )

    return bronze_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ESA WorldCover ingestion collector."
        )
    )

    parser.add_argument(
        "--tile",
        required=True,
        help=(
            "Configured WorldCover tile ID, "
            "for example N18E105."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download the configured WorldCover "
            "tile into staging and validate it."
        ),
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Atomically promote a validated "
            "staging artifact into Bronze raw."
        ),
    )

    args = parser.parse_args()

    config = load_config()
    validate_config(config)

    tile = validate_tile(
        tile=args.tile,
        config=config,
    )

    if args.download or args.promote:
        existing_bronze = (
            get_valid_existing_bronze(
                tile=tile,
                config=config,
            )
        )

        if existing_bronze is not None:
            stale_staging = (
                staging_path_for_tile(
                    tile=tile,
                    config=config,
                )
            )

            if stale_staging.exists():
                stale_staging.unlink()

            print("BRONZE_EXISTS=True")
            print("BRONZE_VALID=True")
            print(
                "BRONZE_ACTION=SKIP_EXISTING"
            )
            print(
                "BRONZE_FILE="
                f"{existing_bronze}"
            )

            return

    print("CONFIG_VALID=True")
    print(f"TILE={tile}")
    print(
        "FILENAME="
        f"{filename_for_tile(tile, config)}"
    )
    print(
        "REMOTE_URL="
        f"{remote_url_for_tile(tile, config)}"
    )
    print(
        "STAGING_PATH="
        f"{staging_path_for_tile(tile, config)}"
    )
    print(
        "BRONZE_PATH="
        f"{bronze_path_for_tile(tile, config)}"
    )
    if args.download:
        metrics = download_to_staging(
            tile=tile,
            config=config,
        )

        print("DOWNLOAD_OK=True")
        print(
            "STAGING_FILE="
            f"{staging_path_for_tile(tile, config)}"
        )
        print(
            "FILE_BYTES="
            f"{metrics['file_bytes']}"
        )
        print(
            "PHYSICAL_OK="
            f"{metrics['physical_ok']}"
        )
        print(
            "SEMANTIC_OK="
            f"{metrics['semantic_ok']}"
        )
        print(
            "CLASSES="
            f"{metrics['classes']}"
        )
    if args.promote:
        metrics = promote_to_bronze(
            tile=tile,
            config=config,
        )

        print("PROMOTE_OK=True")
        print(
            "BRONZE_FILE="
            f"{bronze_path_for_tile(tile, config)}"
        )
        print(
            "FILE_BYTES="
            f"{metrics['file_bytes']}"
        )
        print(
            "PHYSICAL_OK="
            f"{metrics['physical_ok']}"
        )
        print(
            "SEMANTIC_OK="
            f"{metrics['semantic_ok']}"
        )
        print(
            "CLASSES="
            f"{metrics['classes']}"
        )

if __name__ == "__main__":
    main()