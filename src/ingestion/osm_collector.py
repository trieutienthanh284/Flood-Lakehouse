from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any
import hashlib
import os
import pyogrio
import yaml
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "osm.yaml"
)


EXPECTED_DATASET = "OpenStreetMap"
EXPECTED_PROVIDER = "Geofabrik"
EXPECTED_REGION = "Vietnam"

EXPECTED_ROLE = (
    "near_static_urban_vector"
)

EXPECTED_CONTAINER = "OSM PBF"
EXPECTED_EXTENSION = ".osm.pbf"

EXPECTED_DOWNSTREAM_GROUPS = [
    "roads",
    "waterways",
    "water_bodies",
]

MD5_RE = re.compile(
    r"^[0-9a-fA-F]{32}$"
)

DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}$"
)

CONNECT_TIMEOUT_SECONDS = 15
READ_TIMEOUT_SECONDS = 120

DOWNLOAD_CHUNK_BYTES = (
    1024 * 1024
)

def load_config(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"OSM config does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "OSM config must be a YAML mapping."
        )

    return config


def validate_config(
    config: dict[str, Any],
) -> None:
    if (
        config.get("dataset")
        != EXPECTED_DATASET
    ):
        raise ValueError(
            "Unexpected OSM dataset."
        )

    if (
        config.get("provider")
        != EXPECTED_PROVIDER
    ):
        raise ValueError(
            "Unexpected OSM provider."
        )

    if (
        config.get("region")
        != EXPECTED_REGION
    ):
        raise ValueError(
            "Unexpected OSM region."
        )

    if (
        config.get("role")
        != EXPECTED_ROLE
    ):
        raise ValueError(
            "Unexpected OSM role."
        )

    snapshot = config.get("snapshot")

    if not isinstance(snapshot, dict):
        raise ValueError(
            "Missing OSM snapshot config."
        )

    snapshot_date_value = snapshot.get(
        "snapshot_date"
    )

    if not isinstance(
        snapshot_date_value,
        str,
    ):
        raise ValueError(
            "snapshot_date must be a string."
        )

    if not DATE_RE.fullmatch(
        snapshot_date_value
    ):
        raise ValueError(
            "Invalid snapshot_date format."
        )

    snapshot_compact = (
        snapshot_date_value.replace("-", "")
    )

    expected_bronze_filename = (
        f"vietnam-{snapshot_compact}.osm.pbf"
    )

    if (
        snapshot.get("bronze_filename")
        != expected_bronze_filename
    ):
        raise ValueError(
            "Bronze filename does not match "
            "snapshot_date."
        )

    remote_filename = snapshot.get(
        "remote_filename"
    )

    if (
        not isinstance(
            remote_filename,
            str,
        )
        or not remote_filename.endswith(
            EXPECTED_EXTENSION
        )
    ):
        raise ValueError(
            "Invalid remote OSM PBF filename."
        )

    bronze_filename_value = snapshot.get(
        "bronze_filename"
    )

    if (
        not isinstance(
            bronze_filename_value,
            str,
        )
        or not bronze_filename_value.endswith(
            EXPECTED_EXTENSION
        )
    ):
        raise ValueError(
            "Invalid Bronze OSM PBF filename."
        )

    expected_length = snapshot.get(
        "expected_content_length"
    )

    if (
        not isinstance(
            expected_length,
            int,
        )
        or expected_length <= 0
    ):
        raise ValueError(
            "expected_content_length "
            "must be positive."
        )

    expected_md5 = snapshot.get(
        "expected_md5"
    )

    if (
        not isinstance(
            expected_md5,
            str,
        )
        or not MD5_RE.fullmatch(
            expected_md5
        )
    ):
        raise ValueError(
            "Invalid expected_md5."
        )

    access = config.get("access")

    if not isinstance(access, dict):
        raise ValueError(
            "Missing OSM access config."
        )

    for key in (
        "pbf_url",
        "md5_url",
    ):
        value = access.get(key)

        if (
            not isinstance(value, str)
            or not value.startswith(
                "https://"
            )
        ):
            raise ValueError(
                f"Invalid access URL: {key}"
            )

    file_format = config.get("format")

    if not isinstance(
        file_format,
        dict,
    ):
        raise ValueError(
            "Missing OSM format config."
        )

    if (
        file_format.get("container")
        != EXPECTED_CONTAINER
    ):
        raise ValueError(
            "Unexpected OSM container."
        )

    if (
        file_format.get("extension")
        != EXPECTED_EXTENSION
    ):
        raise ValueError(
            "Unexpected OSM extension."
        )

    semantic = config.get(
        "semantic_scope"
    )

    if not isinstance(
        semantic,
        dict,
    ):
        raise ValueError(
            "Missing semantic_scope."
        )

    if (
        semantic.get(
            "downstream_groups"
        )
        != EXPECTED_DOWNSTREAM_GROUPS
    ):
        raise ValueError(
            "Unexpected downstream groups."
        )

    bronze = config.get("bronze")

    if not isinstance(bronze, dict):
        raise ValueError(
            "Missing Bronze config."
        )

    if bronze.get("source") != "osm":
        raise ValueError(
            "Unexpected Bronze source."
        )

    if (
        bronze.get("partition_key")
        != "snapshot_date"
    ):
        raise ValueError(
            "Unexpected partition key."
        )

    if (
        bronze.get("staging_suffix")
        != ".part"
    ):
        raise ValueError(
            "Unexpected staging suffix."
        )

    if (
        bronze.get(
            "raw_artifact_policy"
        )
        != "keep_original_pbf_bytes"
    ):
        raise ValueError(
            "Unexpected raw artifact policy."
        )

    for key in (
        "root",
        "staging_dir",
        "raw_dir",
    ):
        value = bronze.get(key)

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"Invalid Bronze path: {key}"
            )

    validation = config.get(
        "validation"
    )

    if not isinstance(
        validation,
        dict,
    ):
        raise ValueError(
            "Missing validation config."
        )

    required_flags = (
        "require_non_empty",
        "require_content_length_match",
        "require_md5_match",
        "require_pbf_readable",
    )

    for key in required_flags:
        if validation.get(key) is not True:
            raise ValueError(
                f"Validation flag "
                f"must be true: {key}"
            )


def snapshot_date(
    config: dict[str, Any],
) -> str:
    return str(
        config["snapshot"][
            "snapshot_date"
        ]
    )


def remote_pbf_url(
    config: dict[str, Any],
) -> str:
    return str(
        config["access"]["pbf_url"]
    )


def remote_md5_url(
    config: dict[str, Any],
) -> str:
    return str(
        config["access"]["md5_url"]
    )

def parse_remote_md5(
    text: str,
) -> str:
    parts = text.strip().split()

    if not parts:
        raise ValueError(
            "Remote MD5 manifest is empty."
        )

    md5_value = parts[0].strip()

    if not MD5_RE.fullmatch(md5_value):
        raise ValueError(
            "Remote MD5 manifest contains "
            "an invalid checksum."
        )

    return md5_value.lower()

def verify_remote_snapshot(
    config: dict[str, Any],
) -> dict[str, Any]:
    pbf_url = remote_pbf_url(config)
    md5_url = remote_md5_url(config)

    expected_bytes = int(
        config["snapshot"][
            "expected_content_length"
        ]
    )

    expected_md5 = str(
        config["snapshot"][
            "expected_md5"
        ]
    ).lower()

    head_response = requests.head(
        pbf_url,
        timeout=30,
        allow_redirects=True,
    )

    head_response.raise_for_status()

    content_length_text = (
        head_response.headers.get(
            "Content-Length"
        )
    )

    if content_length_text is None:
        raise RuntimeError(
            "Remote PBF does not provide "
            "Content-Length."
        )

    remote_bytes = int(
        content_length_text
    )

    md5_response = requests.get(
        md5_url,
        timeout=30,
    )

    md5_response.raise_for_status()

    remote_md5 = parse_remote_md5(
        md5_response.text
    )

    length_match = (
        remote_bytes == expected_bytes
    )

    md5_match = (
        remote_md5 == expected_md5
    )

    if not length_match or not md5_match:
        raise RuntimeError(
            "Remote OSM snapshot has drifted "
            "from the pinned config: "
            f"expected_bytes={expected_bytes}, "
            f"remote_bytes={remote_bytes}, "
            f"expected_md5={expected_md5}, "
            f"remote_md5={remote_md5}"
        )

    return {
        "remote_bytes": remote_bytes,
        "remote_md5": remote_md5,
        "length_match": True,
        "md5_match": True,
        "last_modified": (
            head_response.headers.get(
                "Last-Modified"
            )
        ),
    }

def bronze_filename(
    config: dict[str, Any],
) -> str:
    return str(
        config["snapshot"][
            "bronze_filename"
        ]
    )


def staging_path(
    config: dict[str, Any],
) -> Path:
    bronze = config["bronze"]

    directory = (
        PROJECT_ROOT
        / bronze["staging_dir"]
    )

    filename = (
        bronze_filename(config)
        + bronze["staging_suffix"]
    )

    return directory / filename


def bronze_path(
    config: dict[str, Any],
) -> Path:
    bronze = config["bronze"]

    directory = (
        PROJECT_ROOT
        / bronze["raw_dir"]
        / (
            "snapshot_date="
            f"{snapshot_date(config)}"
        )
    )

    return (
        directory
        / bronze_filename(config)
    )

def calculate_md5(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.md5()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()

def validate_osm_pbf(
    path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"OSM PBF does not exist: {path}"
        )

    file_bytes = path.stat().st_size

    if file_bytes <= 0:
        raise ValueError(
            "OSM PBF artifact is empty."
        )

    expected_bytes = int(
        config["snapshot"][
            "expected_content_length"
        ]
    )

    if file_bytes != expected_bytes:
        raise ValueError(
            "OSM PBF byte size mismatch: "
            f"expected={expected_bytes}, "
            f"actual={file_bytes}"
        )

    actual_md5 = calculate_md5(path)

    expected_md5 = str(
        config["snapshot"]["expected_md5"]
    ).lower()

    if actual_md5.lower() != expected_md5:
        raise ValueError(
            "OSM PBF MD5 mismatch: "
            f"expected={expected_md5}, "
            f"actual={actual_md5}"
        )

    drivers = pyogrio.list_drivers()

    if "OSM" not in drivers:
        raise RuntimeError(
            "GDAL OSM driver is unavailable."
        )

    osm_driver_mode = drivers["OSM"]

    if "r" not in osm_driver_mode:
        raise RuntimeError(
            "GDAL OSM driver does not "
            "support reading."
        )

    try:
        layers = pyogrio.list_layers(path)

    except Exception as exc:
        raise ValueError(
            "OSM PBF is not readable "
            "by GDAL/pyogrio."
        ) from exc

    layer_names = [
        str(row[0])
        for row in layers
    ]

    if not layer_names:
        raise ValueError(
            "OSM PBF contains no "
            "discoverable OSM layers."
        )

    expected_core_layers = {
        "points",
        "lines",
        "multipolygons",
    }

    missing_core_layers = (
        expected_core_layers
        - set(layer_names)
    )

    if missing_core_layers:
        raise ValueError(
            "OSM PBF is missing expected "
            "GDAL OSM layers: "
            f"{sorted(missing_core_layers)}"
        )

    return {
        "file_bytes": file_bytes,
        "md5": actual_md5,
        "pbf_readable": True,
        "osm_driver_mode": osm_driver_mode,
        "layers": layer_names,
    }

def build_http_session(
) -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "GET",
            "HEAD",
        ],
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
    config: dict[str, Any],
) -> dict[str, Any]:
    remote_metrics = (
        verify_remote_snapshot(config)
    )

    destination = staging_path(config)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        destination.unlink()

    session = build_http_session()

    try:
        with session.get(
            remote_pbf_url(config),
            stream=True,
            timeout=(
                CONNECT_TIMEOUT_SECONDS,
                READ_TIMEOUT_SECONDS,
            ),
        ) as response:
            response.raise_for_status()

            with destination.open(
                "wb"
            ) as file:
                for chunk in (
                    response.iter_content(
                        chunk_size=(
                            DOWNLOAD_CHUNK_BYTES
                        )
                    )
                ):
                    if not chunk:
                        continue

                    file.write(chunk)

        if not destination.is_file():
            raise RuntimeError(
                "OSM staging artifact "
                "was not created."
            )

        if (
            destination.stat().st_size
            <= 0
        ):
            raise RuntimeError(
                "OSM staging artifact "
                "is empty."
            )

        metrics = validate_osm_pbf(
            path=destination,
            config=config,
        )

        return {
            **metrics,
            "remote_bytes": (
                remote_metrics[
                    "remote_bytes"
                ]
            ),
            "remote_md5": (
                remote_metrics[
                    "remote_md5"
                ]
            ),
        }

    except Exception:
        if destination.exists():
            destination.unlink()

        raise

    finally:
        session.close()

def promote_to_bronze(
    config: dict[str, Any],
) -> dict[str, Any]:
    source = staging_path(config)
    destination = bronze_path(config)

    if not source.is_file():
        raise FileNotFoundError(
            "OSM staging artifact "
            f"does not exist: {source}"
        )

    if destination.exists():
        raise FileExistsError(
            "OSM Bronze artifact "
            f"already exists: {destination}"
        )

    staging_metrics = validate_osm_pbf(
        path=source,
        config=config,
    )

    staging_bytes = source.stat().st_size

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_anchor = (
        source.resolve().anchor.lower()
    )

    destination_anchor = (
        destination.parent.resolve()
        .anchor.lower()
    )

    if source_anchor != destination_anchor:
        raise RuntimeError(
            "OSM staging and Bronze "
            "destination must be on the "
            "same filesystem for atomic "
            "promotion."
        )

    os.replace(
        source,
        destination,
    )

    if not destination.is_file():
        raise RuntimeError(
            "OSM Bronze artifact was not "
            "created after promotion."
        )

    if source.exists():
        raise RuntimeError(
            "OSM staging artifact still "
            "exists after promotion."
        )

    bronze_bytes = (
        destination.stat().st_size
    )

    if bronze_bytes != staging_bytes:
        raise RuntimeError(
            "OSM artifact byte size changed "
            "during promotion."
        )

    bronze_metrics = validate_osm_pbf(
        path=destination,
        config=config,
    )

    if (
        bronze_metrics["file_bytes"]
        != staging_metrics["file_bytes"]
    ):
        raise RuntimeError(
            "Validated OSM byte size changed "
            "during promotion."
        )

    if (
        bronze_metrics["md5"]
        != staging_metrics["md5"]
    ):
        raise RuntimeError(
            "OSM MD5 changed during "
            "promotion."
        )

    return bronze_metrics

def get_valid_existing_bronze(
    config: dict[str, Any],
) -> Path | None:
    path = bronze_path(config)

    if not path.is_file():
        return None

    validate_osm_pbf(
        path=path,
        config=config,
    )

    return path

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Geofabrik OpenStreetMap "
            "Bronze collector."
        )
    )

    parser.add_argument(
        "--verify-remote",
        action="store_true",
        help=(
            "Verify that the mutable Geofabrik "
            "latest endpoint still matches "
            "the pinned snapshot contract."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download the pinned Geofabrik "
            "OSM snapshot into Bronze staging "
            "and validate it."
        ),
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Atomically promote a validated "
            "OSM staging artifact into "
            "Bronze raw."
        ),
    )

    args = parser.parse_args()

    config = load_config()

    validate_config(config)

    if args.download or args.promote:
        existing_bronze = (
            get_valid_existing_bronze(
                config
            )
        )

        if existing_bronze is not None:
            stale_staging = staging_path(
                config
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

    print(
        "DATASET="
        f"{config['dataset']}"
    )

    print(
        "PROVIDER="
        f"{config['provider']}"
    )

    print(
        "REGION="
        f"{config['region']}"
    )

    print(
        "SNAPSHOT_DATE="
        f"{snapshot_date(config)}"
    )

    print(
        "REMOTE_PBF_URL="
        f"{remote_pbf_url(config)}"
    )

    print(
        "REMOTE_MD5_URL="
        f"{remote_md5_url(config)}"
    )

    print(
        "EXPECTED_CONTENT_LENGTH="
        f"{config['snapshot']['expected_content_length']}"
    )

    print(
        "EXPECTED_MD5="
        f"{config['snapshot']['expected_md5']}"
    )

    print(
        "STAGING_PATH="
        f"{staging_path(config)}"
    )

    print(
        "BRONZE_PATH="
        f"{bronze_path(config)}"
    )

    if args.verify_remote:
        metrics = verify_remote_snapshot(
            config
        )

        print("REMOTE_VERIFY_OK=True")

        print(
            "REMOTE_BYTES="
            f"{metrics['remote_bytes']}"
        )

        print(
            "REMOTE_MD5="
            f"{metrics['remote_md5']}"
        )

        print(
            "LENGTH_MATCH="
            f"{metrics['length_match']}"
        )

        print(
            "MD5_MATCH="
            f"{metrics['md5_match']}"
        )

        print(
            "REMOTE_LAST_MODIFIED="
            f"{metrics['last_modified']}"
        )
    if args.download:
        metrics = download_to_staging(
            config
        )

        print("DOWNLOAD_OK=True")

        print(
            "STAGING_FILE="
            f"{staging_path(config)}"
        )

        print(
            "FILE_BYTES="
            f"{metrics['file_bytes']}"
        )

        print(
            "MD5="
            f"{metrics['md5']}"
        )

        print(
            "PBF_READABLE="
            f"{metrics['pbf_readable']}"
        )

        print(
            "OSM_DRIVER_MODE="
            f"{metrics['osm_driver_mode']}"
        )

        print(
            "LAYERS="
            f"{metrics['layers']}"
        )
    if args.promote:
        metrics = promote_to_bronze(
            config
        )

        print("PROMOTE_OK=True")

        print(
            "BRONZE_FILE="
            f"{bronze_path(config)}"
        )

        print(
            "FILE_BYTES="
            f"{metrics['file_bytes']}"
        )

        print(
            "MD5="
            f"{metrics['md5']}"
        )

        print(
            "PBF_READABLE="
            f"{metrics['pbf_readable']}"
        )

        print(
            "OSM_DRIVER_MODE="
            f"{metrics['osm_driver_mode']}"
        )

        print(
            "LAYERS="
            f"{metrics['layers']}"
        )

if __name__ == "__main__":
    main()