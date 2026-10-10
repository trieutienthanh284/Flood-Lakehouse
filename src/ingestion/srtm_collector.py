from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

import struct
from zipfile import BadZipFile, ZipFile, is_zipfile

import os

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "srtm.yaml"

EXPECTED_DATASET = "SRTMGL1.003"
EXPECTED_ROLE = "static_elevation_dem"
EXPECTED_CRS = "EPSG:4326"

TILE_PATTERN = re.compile(
    r"^[NS]\d{2}[EW]\d{3}$"
)

EARTHDATA_BASE_URL = (
    "https://data.lpdaac.earthdatacloud.nasa.gov/"
    "lp-prod-protected"
)

DEFAULT_CONNECT_TIMEOUT_SECONDS = 20
DEFAULT_READ_TIMEOUT_SECONDS = 120
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

def load_config(
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"SRTM config does not exist: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            "SRTM config must be a YAML mapping."
        )

    return config


def validate_config(
    config: dict[str, Any],
) -> None:
    dataset = config.get("dataset")
    role = config.get("role")

    if dataset != EXPECTED_DATASET:
        raise ValueError(
            f"Unexpected dataset: {dataset!r}"
        )

    if role != EXPECTED_ROLE:
        raise ValueError(
            f"Unexpected role: {role!r}"
        )

    spatial = config.get("spatial")
    if not isinstance(spatial, dict):
        raise ValueError(
            "Missing spatial config."
        )

    if spatial.get("crs") != EXPECTED_CRS:
        raise ValueError(
            f"Unexpected CRS: {spatial.get('crs')!r}"
        )

    tiles = spatial.get("tiles")

    if not isinstance(tiles, list) or not tiles:
        raise ValueError(
            "spatial.tiles must be a non-empty list."
        )

    if len(tiles) != len(set(tiles)):
        raise ValueError(
            "Duplicate SRTM tiles found in config."
        )

    for tile in tiles:
        if not isinstance(tile, str):
            raise ValueError(
                f"Tile must be a string: {tile!r}"
            )

        if TILE_PATTERN.fullmatch(tile) is None:
            raise ValueError(
                f"Invalid SRTM tile ID: {tile!r}"
            )

    validation = config.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(
            "Missing validation config."
        )

    zip_cfg = validation.get("zip")
    hgt_cfg = validation.get("hgt")

    if not isinstance(zip_cfg, dict):
        raise ValueError(
            "Missing validation.zip config."
        )

    if not isinstance(hgt_cfg, dict):
        raise ValueError(
            "Missing validation.hgt config."
        )

    if zip_cfg.get("expected_member_count") != 1:
        raise ValueError(
            "SRTM ZIP must contain exactly one member."
        )

    rows = hgt_cfg.get("rows")
    columns = hgt_cfg.get("columns")
    sample_count = hgt_cfg.get("sample_count")
    bytes_per_sample = hgt_cfg.get(
        "bytes_per_sample"
    )
    expected_bytes = hgt_cfg.get(
        "expected_uncompressed_bytes"
    )

    integer_fields = {
        "rows": rows,
        "columns": columns,
        "sample_count": sample_count,
        "bytes_per_sample": bytes_per_sample,
        "expected_uncompressed_bytes":
            expected_bytes,
    }

    for name, value in integer_fields.items():
        if not isinstance(value, int):
            raise ValueError(
                f"validation.hgt.{name} "
                "must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"validation.hgt.{name} "
                "must be positive."
            )

    if rows * columns != sample_count:
        raise ValueError(
            "HGT sample_count does not match "
            "rows * columns."
        )

    if sample_count * bytes_per_sample != expected_bytes:
        raise ValueError(
            "HGT byte size does not match "
            "sample_count * bytes_per_sample."
        )

    if hgt_cfg.get("dtype") != "int16":
        raise ValueError(
            "SRTM HGT dtype must be int16."
        )

    if hgt_cfg.get("byte_order") != "big_endian":
        raise ValueError(
            "SRTM HGT byte_order must be big_endian."
        )

    if hgt_cfg.get(
        "spatial_resolution_arcseconds"
    ) != 1:
        raise ValueError(
            "SRTMGL1 spatial resolution "
            "must be 1 arc-second."
        )

    bronze = config.get("bronze")
    if not isinstance(bronze, dict):
        raise ValueError(
            "Missing bronze config."
        )

    if bronze.get("source") != "srtm":
        raise ValueError(
            "bronze.source must be 'srtm'."
        )

    if bronze.get("partition_key") != "tile":
        raise ValueError(
            "bronze.partition_key must be 'tile'."
        )

    if bronze.get("staging_suffix") != ".part":
        raise ValueError(
            "bronze.staging_suffix must be '.part'."
        )

    required_paths = (
        "root",
        "staging_dir",
        "raw_dir",
    )

    for key in required_paths:
        value = bronze.get(key)

        if not isinstance(value, str) or not value:
            raise ValueError(
                f"bronze.{key} must be "
                "a non-empty string."
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
) -> None:
    if TILE_PATTERN.fullmatch(tile) is None:
        raise ValueError(
            f"Invalid SRTM tile ID: {tile!r}"
        )

    if tile not in configured_tiles(config):
        raise ValueError(
            f"Tile {tile!r} is not configured "
            "for the Hanoi AOI."
        )


def archive_filename(
    tile: str,
) -> str:
    return (
        f"{tile}.SRTMGL1.hgt.zip"
    )


def hgt_member_name(
    tile: str,
    config: dict[str, Any],
) -> str:
    template = config[
        "validation"
    ]["zip"]["member_name_template"]

    return template.format(
        tile=tile,
    )


def staging_path_for_tile(
    tile: str,
    config: dict[str, Any],
) -> Path:
    validate_tile(
        tile,
        config,
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
        / f"{archive_filename(tile)}{suffix}"
    )


def bronze_path_for_tile(
    tile: str,
    config: dict[str, Any],
) -> Path:
    validate_tile(
        tile,
        config,
    )

    raw_dir = (
        PROJECT_ROOT
        / config["bronze"]["raw_dir"]
    )

    return (
        raw_dir
        / f"tile={tile}"
        / archive_filename(tile)
    )

def has_valid_zip_signature(
    path: Path,
) -> bool:
    if not path.is_file():
        return False

    if path.stat().st_size <= 0:
        return False

    return is_zipfile(path)


def validate_srtm_archive(
    path: Path,
    tile: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    validate_tile(
        tile,
        config,
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"SRTM archive does not exist: {path}"
        )

    file_bytes = path.stat().st_size

    if file_bytes <= 0:
        raise ValueError(
            "SRTM archive is empty."
        )

    if not is_zipfile(path):
        raise ValueError(
            "SRTM artifact is not a valid ZIP archive."
        )

    zip_cfg = config[
        "validation"
    ]["zip"]

    hgt_cfg = config[
        "validation"
    ]["hgt"]

    expected_member_count = zip_cfg[
        "expected_member_count"
    ]

    expected_member_name = hgt_member_name(
        tile,
        config,
    )

    expected_bytes = hgt_cfg[
        "expected_uncompressed_bytes"
    ]

    expected_sample_count = hgt_cfg[
        "sample_count"
    ]

    rows = hgt_cfg["rows"]
    columns = hgt_cfg["columns"]
    bytes_per_sample = hgt_cfg[
        "bytes_per_sample"
    ]

    try:
        with ZipFile(path, "r") as zf:
            members = zf.infolist()

            if len(members) != expected_member_count:
                raise ValueError(
                    "Unexpected ZIP member count: "
                    f"{len(members)}"
                )

            member = members[0]

            if member.filename != expected_member_name:
                raise ValueError(
                    "Unexpected HGT member name: "
                    f"{member.filename!r}"
                )

            if member.file_size != expected_bytes:
                raise ValueError(
                    "Unexpected HGT uncompressed size: "
                    f"{member.file_size}"
                )

            if zip_cfg.get(
                "require_crc_check",
                False,
            ):
                bad_member = zf.testzip()

                if bad_member is not None:
                    raise ValueError(
                        "ZIP CRC validation failed for "
                        f"{bad_member!r}"
                    )

            raw = zf.read(
                member.filename
            )

    except BadZipFile as exc:
        raise ValueError(
            "Invalid SRTM ZIP archive."
        ) from exc

    if len(raw) != expected_bytes:
        raise ValueError(
            "Decoded HGT byte length does not "
            "match validation contract."
        )

    if len(raw) % bytes_per_sample != 0:
        raise ValueError(
            "HGT byte length is not divisible "
            "by bytes_per_sample."
        )

    actual_sample_count = (
        len(raw) // bytes_per_sample
    )

    if actual_sample_count != expected_sample_count:
        raise ValueError(
            "Unexpected HGT sample count: "
            f"{actual_sample_count}"
        )

    if rows * columns != actual_sample_count:
        raise ValueError(
            "HGT grid dimensions do not match "
            "sample count."
        )

    first_index = 0
    center_index = (
        (rows // 2) * columns
        + (columns // 2)
    )
    last_index = (
        actual_sample_count - 1
    )

    def decode_sample(
        sample_index: int,
    ) -> int:
        offset = (
            sample_index
            * bytes_per_sample
        )

        return struct.unpack_from(
            ">h",
            raw,
            offset,
        )[0]

    first_sample = decode_sample(
        first_index
    )

    center_sample = decode_sample(
        center_index
    )

    last_sample = decode_sample(
        last_index
    )

    return {
        "file_bytes": file_bytes,
        "member_count": len(members),
        "member_name": member.filename,
        "member_compressed_bytes":
            member.compress_size,
        "member_uncompressed_bytes":
            member.file_size,
        "member_crc32":
            f"{member.CRC:08X}",
        "sample_count":
            actual_sample_count,
        "rows": rows,
        "columns": columns,
        "first_sample":
            first_sample,
        "center_sample":
            center_sample,
        "last_sample":
            last_sample,
        "physical_ok": True,
        "semantic_ok": True,
    }

def load_earthdata_token(
    env_path: Path | None = None,
) -> str:
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"

    load_dotenv(
        dotenv_path=env_path,
    )

    token = os.getenv(
        "EARTHDATA_TOKEN"
    )

    if not token:
        raise RuntimeError(
            "EARTHDATA_TOKEN is missing."
        )

    return token


def remote_url_for_tile(
    tile: str,
    config: dict[str, Any],
) -> str:
    validate_tile(
        tile,
        config,
    )

    dataset = config["dataset"]
    archive = archive_filename(tile)

    return (
        f"{EARTHDATA_BASE_URL}/"
        f"{dataset}/"
        f"{tile}.SRTMGL1.hgt/"
        f"{archive}"
    )


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
            {"GET", "HEAD"}
        ),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    return session


def download_to_staging(
    tile: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    validate_tile(
        tile,
        config,
    )

    token = load_earthdata_token()

    url = remote_url_for_tile(
        tile,
        config,
    )

    staging_path = staging_path_for_tile(
        tile,
        config,
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
            headers={
                "Authorization":
                    f"Bearer {token}",
            },
            stream=True,
            allow_redirects=True,
            timeout=(
                DEFAULT_CONNECT_TIMEOUT_SECONDS,
                DEFAULT_READ_TIMEOUT_SECONDS,
            ),
        ) as response:
            response.raise_for_status()

            with staging_path.open(
                "wb",
            ) as f:
                for chunk in response.iter_content(
                    chunk_size=DOWNLOAD_CHUNK_SIZE,
                ):
                    if chunk:
                        f.write(chunk)

        if not staging_path.is_file():
            raise RuntimeError(
                "Staging file was not created."
            )

        if staging_path.stat().st_size <= 0:
            raise RuntimeError(
                "Downloaded staging file is empty."
            )

        metrics = validate_srtm_archive(
            path=staging_path,
            tile=tile,
            config=config,
        )

        return {
            "tile": tile,
            "url": url,
            "staging_path":
                str(staging_path),
            "downloaded_bytes":
                staging_path.stat().st_size,
            **metrics,
        }

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
        tile,
        config,
    )

    staging_path = staging_path_for_tile(
        tile,
        config,
    )

    bronze_path = bronze_path_for_tile(
        tile,
        config,
    )

    if not staging_path.is_file():
        raise FileNotFoundError(
            "Validated staging artifact "
            f"does not exist: {staging_path}"
        )

    if bronze_path.exists():
        raise FileExistsError(
            "Bronze artifact already exists: "
            f"{bronze_path}"
        )

    staging_bytes = (
        staging_path.stat().st_size
    )

    staging_metrics = validate_srtm_archive(
        path=staging_path,
        tile=tile,
        config=config,
    )

    bronze_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        staging_path.resolve().anchor.lower()
        != bronze_path.resolve().anchor.lower()
    ):
        raise RuntimeError(
            "Atomic promotion requires staging "
            "and Bronze to be on the same volume."
        )

    os.replace(
        staging_path,
        bronze_path,
    )

    if not bronze_path.is_file():
        raise RuntimeError(
            "Bronze artifact does not exist "
            "after promotion."
        )

    if staging_path.exists():
        raise RuntimeError(
            "Staging artifact still exists "
            "after promotion."
        )

    bronze_bytes = (
        bronze_path.stat().st_size
    )

    if bronze_bytes != staging_bytes:
        raise RuntimeError(
            "Bronze artifact size changed "
            "during promotion."
        )

    bronze_metrics = validate_srtm_archive(
        path=bronze_path,
        tile=tile,
        config=config,
    )

    return {
        "tile": tile,
        "staging_path":
            str(staging_path),
        "bronze_path":
            str(bronze_path),
        "staging_bytes":
            staging_bytes,
        "bronze_bytes":
            bronze_bytes,
        "staging_semantic_ok":
            staging_metrics["semantic_ok"],
        "bronze_physical_ok":
            bronze_metrics["physical_ok"],
        "bronze_semantic_ok":
            bronze_metrics["semantic_ok"],
        "member_name":
            bronze_metrics["member_name"],
        "sample_count":
            bronze_metrics["sample_count"],
    }

def get_valid_existing_bronze(
    tile: str,
    config: dict[str, Any],
) -> Path | None:
    validate_tile(
        tile,
        config,
    )

    bronze_path = bronze_path_for_tile(
        tile,
        config,
    )

    if not bronze_path.is_file():
        return None

    metrics = validate_srtm_archive(
        path=bronze_path,
        tile=tile,
        config=config,
    )

    if not (
        metrics["physical_ok"]
        and metrics["semantic_ok"]
    ):
        raise RuntimeError(
            "Existing Bronze artifact "
            "failed validation."
        )

    return bronze_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SRTMGL1 V003 ingestion collector."
        )
    )

    parser.add_argument(
        "--tile",
        required=True,
        help=(
            "Configured SRTM tile ID, "
            "for example N20E105."
        ),
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download and validate the tile "
            "into Bronze staging only."
        ),
    )

    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Atomically promote a validated "
            "staging tile into Bronze raw."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config()
    validate_config(config)

    tile = args.tile.strip().upper()

    validate_tile(
        tile,
        config,
    )

    staging_path = staging_path_for_tile(
        tile,
        config,
    )

    bronze_path = bronze_path_for_tile(
        tile,
        config,
    )

    member_name = hgt_member_name(
        tile,
        config,
    )

    print("CONFIG_OK=True")
    print(
        f"DATASET={config['dataset']}"
    )
    print(
        f"TILE={tile}"
    )
    print(
        f"MEMBER_NAME={member_name}"
    )
    print(
        f"STAGING_PATH={staging_path}"
    )
    print(
        f"BRONZE_PATH={bronze_path}"
    )
    if args.download or args.promote:
        existing_bronze = (
            get_valid_existing_bronze(
                tile=tile,
                config=config,
            )
        )

        if existing_bronze is not None:
            if staging_path.exists():
                staging_path.unlink()

            print("BRONZE_EXISTS=True")
            print("BRONZE_VALID=True")
            print(
                "BRONZE_ACTION=SKIP_EXISTING"
            )
            print(
                f"BRONZE_FILE="
                f"{existing_bronze}"
            )

            return

    if args.download:
        result = download_to_staging(
            tile=tile,
            config=config,
        )

        print("DOWNLOAD_OK=True")
        print(
            f"DOWNLOADED_BYTES="
            f"{result['downloaded_bytes']}"
        )
        print(
            f"PHYSICAL_OK="
            f"{result['physical_ok']}"
        )
        print(
            f"SEMANTIC_OK="
            f"{result['semantic_ok']}"
        )
        print(
            f"MEMBER_NAME="
            f"{result['member_name']}"
        )
        print(
            f"SAMPLE_COUNT="
            f"{result['sample_count']}"
        )

    if args.promote:
        result = promote_to_bronze(
            tile=tile,
            config=config,
        )

        print("PROMOTE_OK=True")
        print(
            f"BRONZE_FILE="
            f"{result['bronze_path']}"
        )
        print(
            f"BRONZE_BYTES="
            f"{result['bronze_bytes']}"
        )
        print(
            f"BRONZE_PHYSICAL_OK="
            f"{result['bronze_physical_ok']}"
        )
        print(
            f"BRONZE_SEMANTIC_OK="
            f"{result['bronze_semantic_ok']}"
        )
        print(
            f"MEMBER_NAME="
            f"{result['member_name']}"
        )
        print(
            f"SAMPLE_COUNT="
            f"{result['sample_count']}"
        )


if __name__ == "__main__":
    main()