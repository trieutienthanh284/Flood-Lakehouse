from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import os
from ftplib import FTP
import yaml

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "gsmap_now.yaml"
)

EXPECTED_DATASET = "GSMaP NOW"
EXPECTED_PROVIDER = "JAXA EORC"
EXPECTED_ROLE = "realtime_rainfall_candidate"

EXPECTED_PRODUCT_FORMAT = "NetCDF-4"
EXPECTED_CONTAINER = "HDF5"
EXPECTED_PRODUCT_NUMBER = 4

EXPECTED_LATEST_DIRECTORY = "/now/latest"

EXPECTED_TEMPORAL_RESOLUTION_MINUTES = 60
EXPECTED_UPDATE_FREQUENCY_MINUTES = 30
EXPECTED_TIMEZONE = "UTC"
EXPECTED_TIME_SEMANTICS = "start_time"

EXPECTED_LATITUDE_SIZE = 1800
EXPECTED_LONGITUDE_SIZE = 3600

EXPECTED_LATITUDE_RESOLUTION = 0.1
EXPECTED_LONGITUDE_RESOLUTION = 0.1

EXPECTED_RAIN_DATASET = "hourlyPrecipRate"
EXPECTED_GAUGE_DATASET = "hourlyPrecipRateGC"

EXPECTED_RAIN_LONG_NAME = "precip_now"
EXPECTED_GAUGE_LONG_NAME = "precip_gauge_now"

EXPECTED_RAIN_UNIT = "mm/h"
EXPECTED_FILL_VALUE = -9999.9

EXPECTED_ROOT_DATASETS = {
    "Time",
    "Latitude",
    "Longitude",
    "hourlyPrecipRate",
    "hourlyPrecipRateGC",
}


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

    return config

def load_ftp_credentials(
    config: dict[str, Any],
) -> tuple[str, str, str]:
    env_path = (
        PROJECT_ROOT
        / ".env"
    )

    load_dotenv(
        dotenv_path=env_path,
        override=False,
    )

    access = config["access"]

    host = os.getenv(
        access["host_env"]
    )

    user = os.getenv(
        access["user_env"]
    )

    password = os.getenv(
        access["password_env"]
    )

    if not host:
        raise RuntimeError(
            "Missing GSMaP FTP host."
        )

    if not user:
        raise RuntimeError(
            "Missing GSMaP FTP username."
        )

    if not password:
        raise RuntimeError(
            "Missing GSMaP FTP password."
        )

    return (
        host,
        user,
        password,
    )

def validate_config(
    config: dict[str, Any],
) -> None:
    if config.get("dataset") != EXPECTED_DATASET:
        raise ValueError(
            "Unexpected GSMaP NOW dataset."
        )

    if config.get("provider") != EXPECTED_PROVIDER:
        raise ValueError(
            "Unexpected GSMaP NOW provider."
        )

    if config.get("role") != EXPECTED_ROLE:
        raise ValueError(
            "Unexpected GSMaP NOW role."
        )

    product = config.get("product")

    if not isinstance(product, dict):
        raise ValueError(
            "Missing product configuration."
        )

    if (
        product.get("format")
        != EXPECTED_PRODUCT_FORMAT
    ):
        raise ValueError(
            "Unexpected GSMaP NOW product format."
        )

    if (
        product.get("container")
        != EXPECTED_CONTAINER
    ):
        raise ValueError(
            "Unexpected GSMaP NOW container."
        )

    if (
        product.get("product_number")
        != EXPECTED_PRODUCT_NUMBER
    ):
        raise ValueError(
            "Unexpected GSMaP NOW product number."
        )

    access = config.get("access")

    if not isinstance(access, dict):
        raise ValueError(
            "Missing access configuration."
        )

    if access.get("protocol") != "ftp":
        raise ValueError(
            "GSMaP NOW access protocol must be FTP."
        )

    if access.get("auth") != "environment":
        raise ValueError(
            "GSMaP NOW credentials must come "
            "from environment variables."
        )

    if (
        access.get("latest_directory")
        != EXPECTED_LATEST_DIRECTORY
    ):
        raise ValueError(
            "Unexpected GSMaP NOW latest directory."
        )

    for key in [
        "host_env",
        "user_env",
        "password_env",
    ]:
        value = access.get(key)

        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Invalid access environment key: {key}"
            )

    temporal = config.get("temporal")

    if not isinstance(temporal, dict):
        raise ValueError(
            "Missing temporal configuration."
        )

    if (
        temporal.get("timezone")
        != EXPECTED_TIMEZONE
    ):
        raise ValueError(
            "GSMaP NOW timezone must be UTC."
        )

    if (
        temporal.get(
            "temporal_resolution_minutes"
        )
        != EXPECTED_TEMPORAL_RESOLUTION_MINUTES
    ):
        raise ValueError(
            "Unexpected temporal resolution."
        )

    if (
        temporal.get(
            "update_frequency_minutes"
        )
        != EXPECTED_UPDATE_FREQUENCY_MINUTES
    ):
        raise ValueError(
            "Unexpected update frequency."
        )

    if (
        temporal.get(
            "filename_time_semantics"
        )
        != EXPECTED_TIME_SEMANTICS
    ):
        raise ValueError(
            "Filename timestamp must represent "
            "event start time."
        )

    filename = config.get("filename")

    if not isinstance(filename, dict):
        raise ValueError(
            "Missing filename configuration."
        )

    regex_text = filename.get("regex")

    if (
        not isinstance(regex_text, str)
        or not regex_text
    ):
        raise ValueError(
            "Invalid filename regex."
        )

    regex = re.compile(regex_text)

    sample = (
        "gsmap_now_rain."
        "20261013.0830.nc"
    )

    match = regex.fullmatch(sample)

    if match is None:
        raise ValueError(
            "Filename regex does not match "
            "canonical GSMaP NOW filename."
        )

    if (
        match.group("date") != "20261013"
        or match.group("time") != "0830"
    ):
        raise ValueError(
            "Filename regex capture groups "
            "are invalid."
        )

    spatial = config.get("spatial")

    if not isinstance(spatial, dict):
        raise ValueError(
            "Missing spatial configuration."
        )

    if (
        spatial.get("latitude_size")
        != EXPECTED_LATITUDE_SIZE
    ):
        raise ValueError(
            "Unexpected latitude size."
        )

    if (
        spatial.get("longitude_size")
        != EXPECTED_LONGITUDE_SIZE
    ):
        raise ValueError(
            "Unexpected longitude size."
        )

    if abs(
        float(
            spatial.get(
                "latitude_resolution_degrees"
            )
        )
        - EXPECTED_LATITUDE_RESOLUTION
    ) > 1e-12:
        raise ValueError(
            "Unexpected latitude resolution."
        )

    if abs(
        float(
            spatial.get(
                "longitude_resolution_degrees"
            )
        )
        - EXPECTED_LONGITUDE_RESOLUTION
    ) > 1e-12:
        raise ValueError(
            "Unexpected longitude resolution."
        )

    if spatial.get("latitude_order") != "ascending":
        raise ValueError(
            "Latitude order must be ascending."
        )

    if spatial.get("longitude_order") != "ascending":
        raise ValueError(
            "Longitude order must be ascending."
        )

    variables = config.get("variables")

    if not isinstance(variables, dict):
        raise ValueError(
            "Missing variables configuration."
        )

    rainfall = variables.get("rainfall")
    gauge = variables.get(
        "rainfall_gauge_calibrated"
    )

    if not isinstance(rainfall, dict):
        raise ValueError(
            "Missing rainfall variable contract."
        )

    if not isinstance(gauge, dict):
        raise ValueError(
            "Missing gauge rainfall variable contract."
        )

    if (
        rainfall.get("dataset_name")
        != EXPECTED_RAIN_DATASET
    ):
        raise ValueError(
            "Unexpected rainfall dataset name."
        )

    if (
        rainfall.get("long_name")
        != EXPECTED_RAIN_LONG_NAME
    ):
        raise ValueError(
            "Unexpected rainfall long_name."
        )

    if (
        rainfall.get("units")
        != EXPECTED_RAIN_UNIT
    ):
        raise ValueError(
            "Unexpected rainfall unit."
        )

    if abs(
        float(
            rainfall.get("fill_value")
        )
        - EXPECTED_FILL_VALUE
    ) > 1e-6:
        raise ValueError(
            "Unexpected rainfall fill value."
        )

    if (
        gauge.get("dataset_name")
        != EXPECTED_GAUGE_DATASET
    ):
        raise ValueError(
            "Unexpected gauge rainfall dataset."
        )

    if (
        gauge.get("long_name")
        != EXPECTED_GAUGE_LONG_NAME
    ):
        raise ValueError(
            "Unexpected gauge rainfall long_name."
        )

    validation = config.get("validation")

    if not isinstance(validation, dict):
        raise ValueError(
            "Missing validation configuration."
        )

    required = set(
        validation.get(
            "required_root_datasets",
            [],
        )
    )

    if required != EXPECTED_ROOT_DATASETS:
        raise ValueError(
            "Unexpected required root datasets."
        )

    bronze = config.get("bronze")

    if not isinstance(bronze, dict):
        raise ValueError(
            "Missing Bronze configuration."
        )

    if bronze.get("source") != "gsmap_now":
        raise ValueError(
            "Unexpected Bronze source."
        )

    if (
        bronze.get("partition_key")
        != "event_date"
    ):
        raise ValueError(
            "Bronze partition key must be event_date."
        )

    if bronze.get("staging_suffix") != ".part":
        raise ValueError(
            "Unexpected staging suffix."
        )

    for key in [
        "root",
        "staging_dir",
        "raw_dir",
    ]:
        value = bronze.get(key)

        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Invalid Bronze path: {key}"
            )


def filename_regex(
    config: dict[str, Any],
) -> re.Pattern[str]:
    return re.compile(
        config["filename"]["regex"]
    )


def parse_event_start_time(
    filename: str,
    config: dict[str, Any],
) -> datetime:
    match = filename_regex(
        config
    ).fullmatch(filename)

    if match is None:
        raise ValueError(
            "Filename does not match "
            "GSMaP NOW contract: "
            f"{filename}"
        )

    timestamp_text = (
        match.group("date")
        + "."
        + match.group("time")
    )

    timestamp_format = config[
        "temporal"
    ][
        "filename_timestamp_format"
    ]

    event_time = datetime.strptime(
        timestamp_text,
        timestamp_format,
    )

    return event_time.replace(
        tzinfo=timezone.utc,
    )


def event_date_for_filename(
    filename: str,
    config: dict[str, Any],
) -> str:
    event_time = parse_event_start_time(
        filename=filename,
        config=config,
    )

    return event_time.date().isoformat()


def staging_path_for_filename(
    filename: str,
    config: dict[str, Any],
) -> Path:
    staging_dir = (
        PROJECT_ROOT
        / config["bronze"]["staging_dir"]
    )

    suffix = config[
        "bronze"
    ][
        "staging_suffix"
    ]

    return (
        staging_dir
        / f"{filename}{suffix}"
    )


def bronze_path_for_filename(
    filename: str,
    config: dict[str, Any],
) -> Path:
    event_date = event_date_for_filename(
        filename=filename,
        config=config,
    )

    raw_dir = (
        PROJECT_ROOT
        / config["bronze"]["raw_dir"]
    )

    return (
        raw_dir
        / f"event_date={event_date}"
        / filename
    )

def discover_latest_rain_netcdf(
    config: dict[str, Any],
) -> dict[str, Any]:
    host, user, password = (
        load_ftp_credentials(
            config
        )
    )

    latest_directory = config[
        "access"
    ][
        "latest_directory"
    ]

    ftp = FTP()

    try:
        ftp.connect(
            host=host,
            port=21,
            timeout=30,
        )

        ftp.login(
            user=user,
            passwd=password,
        )

        ftp.cwd(
            latest_directory
        )

        filenames: list[str] = []

        for entry in ftp.nlst():
            try:
                parse_event_start_time(
                    filename=entry,
                    config=config,
                )
            except ValueError:
                continue

            filenames.append(
                entry
            )

        if not filenames:
            raise RuntimeError(
                "No valid GSMaP NOW rain "
                "NetCDF files found."
            )

        latest_filename = max(
            filenames,
            key=lambda name: (
                parse_event_start_time(
                    filename=name,
                    config=config,
                )
            ),
        )

        event_time = (
            parse_event_start_time(
                filename=latest_filename,
                config=config,
            )
        )

        # JAXA FTP requires binary mode
        # before SIZE is accepted.
        ftp.voidcmd(
            "TYPE I"
        )

        remote_bytes = ftp.size(
            latest_filename
        )

        modified_response = (
            ftp.sendcmd(
                f"MDTM {latest_filename}"
            )
        )

        if (
            remote_bytes is None
            or remote_bytes <= 0
        ):
            raise RuntimeError(
                "Latest GSMaP NOW NetCDF "
                "has invalid remote size."
            )

        return {
            "filename": latest_filename,
            "event_start_time": event_time,
            "remote_bytes": remote_bytes,
            "remote_mdtm": modified_response,
            "candidate_count": len(
                filenames
            ),
        }

    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()
def download_latest_to_staging(
    config: dict[str, Any],
) -> dict[str, Any]:
    latest = discover_latest_rain_netcdf(
        config
    )

    filename = latest[
        "filename"
    ]

    expected_remote_bytes = latest[
        "remote_bytes"
    ]

    existing = (
        get_valid_existing_bronze(
            filename=filename,
            config=config,
        )
    )

    if existing is not None:
        return {
            "action": "SKIP_EXISTING",
            "filename": filename,
            "event_start_time": existing[
                "event_start_time"
            ],
            "remote_bytes": (
                expected_remote_bytes
            ),
            "local_bytes": existing[
                "file_bytes"
            ],
            "remote_mdtm": latest[
                "remote_mdtm"
            ],
            "staging_path": (
                staging_path_for_filename(
                    filename=filename,
                    config=config,
                )
            ),
            "bronze_path": existing[
                "bronze_path"
            ],
            "physical_ok": existing[
                "physical_ok"
            ],
            "semantic_ok": existing[
                "semantic_ok"
            ],
            "rain_valid_count": None,
            "gauge_valid_count": None,
        }

    staging_path = (
        staging_path_for_filename(
            filename=filename,
            config=config,
        )
    )

    staging_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if staging_path.exists():
        staging_path.unlink()

    host, user, password = (
        load_ftp_credentials(
            config
        )
    )

    latest_directory = config[
        "access"
    ][
        "latest_directory"
    ]

    ftp = FTP()

    try:
        ftp.connect(
            host=host,
            port=21,
            timeout=30,
        )

        ftp.login(
            user=user,
            passwd=password,
        )

        ftp.cwd(
            latest_directory
        )

        ftp.voidcmd(
            "TYPE I"
        )

        remote_bytes = ftp.size(
            filename
        )

        if remote_bytes is None:
            raise RuntimeError(
                "FTP SIZE returned no value "
                "for GSMaP NOW artifact."
            )

        if (
            remote_bytes
            != expected_remote_bytes
        ):
            raise RuntimeError(
                "Remote GSMaP NOW artifact "
                "changed between discovery "
                "and download."
            )

        with staging_path.open(
            "wb"
        ) as file:
            ftp.retrbinary(
                f"RETR {filename}",
                file.write,
                blocksize=1024 * 1024,
            )

    except Exception:
        if staging_path.exists():
            staging_path.unlink()

        raise

    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()

    if not staging_path.is_file():
        raise RuntimeError(
            "GSMaP NOW staging artifact "
            "was not created."
        )

    local_bytes = (
        staging_path.stat().st_size
    )

    if (
        local_bytes
        != expected_remote_bytes
    ):
        staging_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Downloaded GSMaP NOW byte size "
            "does not match remote size: "
            f"remote={expected_remote_bytes}, "
            f"local={local_bytes}"
        )

    try:
        validation_metrics = (
            validate_gsmap_now_netcdf(
                path=staging_path,
                filename=filename,
                config=config,
            )
        )

    except Exception:
        staging_path.unlink(
            missing_ok=True
        )

        raise

    return {
        "action": "DOWNLOAD_TO_STAGING",
        "filename": filename,
        "event_start_time": latest[
            "event_start_time"
        ],
        "remote_bytes": (
            expected_remote_bytes
        ),
        "local_bytes": local_bytes,
        "remote_mdtm": latest[
            "remote_mdtm"
        ],
        "staging_path": staging_path,
        "physical_ok": (
            validation_metrics[
                "physical_ok"
            ]
        ),
        "semantic_ok": (
            validation_metrics[
                "semantic_ok"
            ]
        ),
        "rain_valid_count": (
            validation_metrics[
                "rain_valid_count"
            ]
        ),
        "gauge_valid_count": (
            validation_metrics[
                "gauge_valid_count"
            ]
        ),
    }

def attribute_text(
    value: Any,
) -> str:
    if isinstance(value, bytes):
        return value.decode(
            "utf-8"
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        if value.size != 1:
            raise ValueError(
                "Expected scalar attribute."
            )

        value = value.reshape(-1)[0]

        if isinstance(
            value,
            bytes,
        ):
            return value.decode(
                "utf-8"
            )

    return str(value)


def attribute_float(
    value: Any,
) -> float:
    array = np.asarray(
        value
    ).reshape(-1)

    if array.size != 1:
        raise ValueError(
            "Expected scalar numeric attribute."
        )

    return float(
        array[0]
    )


def parse_time_units_start_time(
    units: str,
) -> datetime:
    pattern = re.compile(
        r"^1hour since "
        r"(?P<year>\d{4})-"
        r"(?P<month>\d{2})-"
        r"(?P<day>\d{2}) "
        r"(?P<hour>\d{2}):"
        r"(?P<minute>\d{2}):"
        r"(?P<second>\d{1,2}"
        r"(?:\.\d+)?)$"
    )

    match = pattern.fullmatch(
        units.strip()
    )

    if match is None:
        raise ValueError(
            "Unexpected GSMaP NOW "
            f"Time units: {units}"
        )

    second_float = float(
        match.group("second")
    )

    second = int(
        second_float
    )

    microsecond = int(
        round(
            (
                second_float
                - second
            )
            * 1_000_000
        )
    )

    return datetime(
        year=int(
            match.group("year")
        ),
        month=int(
            match.group("month")
        ),
        day=int(
            match.group("day")
        ),
        hour=int(
            match.group("hour")
        ),
        minute=int(
            match.group("minute")
        ),
        second=second,
        microsecond=microsecond,
        tzinfo=timezone.utc,
    )

def validate_gsmap_now_netcdf(
    path: Path,
    filename: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "GSMaP NOW NetCDF artifact "
            f"does not exist: {path}"
        )

    file_bytes = (
        path.stat().st_size
    )

    if file_bytes <= 0:
        raise ValueError(
            "GSMaP NOW NetCDF artifact "
            "is empty."
        )

    with path.open(
        "rb"
    ) as file:
        signature = file.read(8)

    hdf5_signature = (
        b"\x89HDF\r\n\x1a\n"
    )

    if signature != hdf5_signature:
        raise ValueError(
            "Artifact is not a "
            "NetCDF-4/HDF5 container."
        )

    event_time = (
        parse_event_start_time(
            filename=filename,
            config=config,
        )
    )

    validation = config[
        "validation"
    ]

    variables = config[
        "variables"
    ]

    spatial = config[
        "spatial"
    ]

    required_datasets = set(
        validation[
            "required_root_datasets"
        ]
    )

    expected_shapes = validation[
        "expected_shapes"
    ]

    rain_contract = variables[
        "rainfall"
    ]

    gauge_contract = variables[
        "rainfall_gauge_calibrated"
    ]

    latitude_contract = variables[
        "latitude"
    ]

    longitude_contract = variables[
        "longitude"
    ]

    time_contract = variables[
        "time"
    ]

    with h5py.File(
        path,
        "r",
    ) as h5:
        root_datasets = set(
            h5.keys()
        )

        missing_datasets = (
            required_datasets
            - root_datasets
        )

        if missing_datasets:
            raise ValueError(
                "Missing required GSMaP NOW "
                "datasets: "
                f"{sorted(missing_datasets)}"
            )

        for dataset_name, shape_list in (
            expected_shapes.items()
        ):
            actual_shape = tuple(
                h5[
                    dataset_name
                ].shape
            )

            expected_shape = tuple(
                shape_list
            )

            if (
                actual_shape
                != expected_shape
            ):
                raise ValueError(
                    "Unexpected shape for "
                    f"{dataset_name}: "
                    f"expected={expected_shape}, "
                    f"actual={actual_shape}"
                )

        dtype_contracts = {
            latitude_contract[
                "dataset_name"
            ]: latitude_contract[
                "dtype"
            ],
            longitude_contract[
                "dataset_name"
            ]: longitude_contract[
                "dtype"
            ],
            time_contract[
                "dataset_name"
            ]: time_contract[
                "dtype"
            ],
            rain_contract[
                "dataset_name"
            ]: rain_contract[
                "dtype"
            ],
            gauge_contract[
                "dataset_name"
            ]: gauge_contract[
                "dtype"
            ],
        }

        for (
            dataset_name,
            expected_dtype,
        ) in dtype_contracts.items():
            actual_dtype = str(
                h5[
                    dataset_name
                ].dtype
            )

            if (
                actual_dtype
                != expected_dtype
            ):
                raise ValueError(
                    "Unexpected dtype for "
                    f"{dataset_name}: "
                    f"expected={expected_dtype}, "
                    f"actual={actual_dtype}"
                )

        latitude = h5[
            latitude_contract[
                "dataset_name"
            ]
        ][:]

        longitude = h5[
            longitude_contract[
                "dataset_name"
            ]
        ][:]

        if not np.all(
            np.diff(latitude) > 0
        ):
            raise ValueError(
                "Latitude axis is not "
                "strictly ascending."
            )

        if not np.all(
            np.diff(longitude) > 0
        ):
            raise ValueError(
                "Longitude axis is not "
                "strictly ascending."
            )

        if not np.allclose(
            np.diff(latitude),
            spatial[
                "latitude_resolution_degrees"
            ],
            atol=1e-5,
        ):
            raise ValueError(
                "Unexpected latitude "
                "grid resolution."
            )

        if not np.allclose(
            np.diff(longitude),
            spatial[
                "longitude_resolution_degrees"
            ],
            atol=1e-5,
        ):
            raise ValueError(
                "Unexpected longitude "
                "grid resolution."
            )

        if not np.isclose(
            latitude[0],
            spatial[
                "latitude_first"
            ],
            atol=1e-4,
        ):
            raise ValueError(
                "Unexpected first latitude."
            )

        if not np.isclose(
            latitude[-1],
            spatial[
                "latitude_last"
            ],
            atol=1e-4,
        ):
            raise ValueError(
                "Unexpected last latitude."
            )

        if not np.isclose(
            longitude[0],
            spatial[
                "longitude_first"
            ],
            atol=1e-4,
        ):
            raise ValueError(
                "Unexpected first longitude."
            )

        if not np.isclose(
            longitude[-1],
            spatial[
                "longitude_last"
            ],
            atol=1e-4,
        ):
            raise ValueError(
                "Unexpected last longitude."
            )

        latitude_units = (
            attribute_text(
                h5[
                    latitude_contract[
                        "dataset_name"
                    ]
                ].attrs["units"]
            )
        )

        longitude_units = (
            attribute_text(
                h5[
                    longitude_contract[
                        "dataset_name"
                    ]
                ].attrs["units"]
            )
        )

        if (
            latitude_units
            != latitude_contract["units"]
        ):
            raise ValueError(
                "Unexpected Latitude units."
            )

        if (
            longitude_units
            != longitude_contract["units"]
        ):
            raise ValueError(
                "Unexpected Longitude units."
            )

        time_dataset = h5[
            time_contract[
                "dataset_name"
            ]
        ]

        if "units" not in time_dataset.attrs:
            raise ValueError(
                "Time dataset has no units."
            )

        time_units = attribute_text(
            time_dataset.attrs[
                "units"
            ]
        )

        time_start = (
            parse_time_units_start_time(
                time_units
            )
        )

        if (
            validation.get(
                "require_filename_timestamp_match_time_units",
                False,
            )
            and time_start != event_time
        ):
            raise ValueError(
                "Filename event time does "
                "not match NetCDF Time units: "
                f"filename={event_time.isoformat()}, "
                f"time_units={time_start.isoformat()}"
            )

        rain_dataset = h5[
            rain_contract[
                "dataset_name"
            ]
        ]

        gauge_dataset = h5[
            gauge_contract[
                "dataset_name"
            ]
        ]

        rain_units = attribute_text(
            rain_dataset.attrs[
                "units"
            ]
        )

        gauge_units = attribute_text(
            gauge_dataset.attrs[
                "units"
            ]
        )

        if (
            rain_units
            != rain_contract["units"]
        ):
            raise ValueError(
                "Unexpected rainfall units."
            )

        if (
            gauge_units
            != gauge_contract["units"]
        ):
            raise ValueError(
                "Unexpected gauge rainfall units."
            )

        rain_long_name = (
            attribute_text(
                rain_dataset.attrs[
                    "long_name"
                ]
            )
        )

        gauge_long_name = (
            attribute_text(
                gauge_dataset.attrs[
                    "long_name"
                ]
            )
        )

        if (
            rain_long_name
            != rain_contract[
                "long_name"
            ]
        ):
            raise ValueError(
                "Unexpected rainfall long_name."
            )

        if (
            gauge_long_name
            != gauge_contract[
                "long_name"
            ]
        ):
            raise ValueError(
                "Unexpected gauge rainfall "
                "long_name."
            )

        rain_fill = attribute_float(
            rain_dataset.attrs[
                "_FillValue"
            ]
        )

        gauge_fill = attribute_float(
            gauge_dataset.attrs[
                "_FillValue"
            ]
        )

        if not np.isclose(
            rain_fill,
            rain_contract[
                "fill_value"
            ],
            atol=1e-3,
        ):
            raise ValueError(
                "Unexpected rainfall "
                "fill value."
            )

        if not np.isclose(
            gauge_fill,
            gauge_contract[
                "fill_value"
            ],
            atol=1e-3,
        ):
            raise ValueError(
                "Unexpected gauge rainfall "
                "fill value."
            )

        rain = rain_dataset[:]
        gauge = gauge_dataset[:]

        rain_invalid_mask = np.isclose(
            rain,
            rain_fill,
            atol=1e-3,
        )

        for sentinel in rain_contract.get(
                "missing_sentinel_values",
                [],
        ):
            rain_invalid_mask |= np.isclose(
                rain,
                float(sentinel),
                atol=1e-6,
            )

        rain_valid_mask = (
                np.isfinite(rain)
                & ~rain_invalid_mask
        )

        gauge_invalid_mask = np.isclose(
            gauge,
            gauge_fill,
            atol=1e-3,
        )

        for sentinel in gauge_contract.get(
                "missing_sentinel_values",
                [],
        ):
            gauge_invalid_mask |= np.isclose(
                gauge,
                float(sentinel),
                atol=1e-6,
            )

        gauge_valid_mask = (
                np.isfinite(gauge)
                & ~gauge_invalid_mask
        )

        rain_valid = rain[
            rain_valid_mask
        ]

        gauge_valid = gauge[
            gauge_valid_mask
        ]

        if rain_valid.size == 0:
            raise ValueError(
                "No valid rainfall values."
            )

        if gauge_valid.size == 0:
            raise ValueError(
                "No valid gauge rainfall values."
            )

        if validation.get(
            "require_nonnegative_valid_rainfall",
            False,
        ):
            if np.any(
                rain_valid < 0
            ):
                raise ValueError(
                    "Negative valid rainfall "
                    "values detected."
                )

            if np.any(
                gauge_valid < 0
            ):
                raise ValueError(
                    "Negative valid gauge "
                    "rainfall values detected."
                )

        return {
            "file_bytes": file_bytes,
            "event_start_time": event_time,
            "time_units": time_units,
            "root_datasets": sorted(
                root_datasets
            ),
            "rain_valid_count": int(
                rain_valid.size
            ),
            "rain_min": float(
                rain_valid.min()
            ),
            "rain_max": float(
                rain_valid.max()
            ),
            "gauge_valid_count": int(
                gauge_valid.size
            ),
            "gauge_min": float(
                gauge_valid.min()
            ),
            "gauge_max": float(
                gauge_valid.max()
            ),
            "physical_ok": True,
            "semantic_ok": True,
        }

def promote_staging_to_bronze(
    filename: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    event_time = parse_event_start_time(
        filename=filename,
        config=config,
    )

    staging_path = (
        staging_path_for_filename(
            filename=filename,
            config=config,
        )
    )

    bronze_path = (
        bronze_path_for_filename(
            filename=filename,
            config=config,
        )
    )

    if not staging_path.is_file():
        raise FileNotFoundError(
            "GSMaP NOW staging artifact "
            f"does not exist: {staging_path}"
        )

    if bronze_path.exists():
        raise FileExistsError(
            "GSMaP NOW Bronze artifact "
            f"already exists: {bronze_path}"
        )

    staging_metrics = (
        validate_gsmap_now_netcdf(
            path=staging_path,
            filename=filename,
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
        staging_path.resolve()
        .anchor.lower()
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
            "GSMaP NOW byte size changed "
            "during promotion."
        )

    bronze_metrics = (
        validate_gsmap_now_netcdf(
            path=bronze_path,
            filename=filename,
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

    return {
        "filename": filename,
        "event_start_time": event_time,
        "bronze_path": bronze_path,
        "file_bytes": bronze_bytes,
        "physical_ok": (
            bronze_metrics["physical_ok"]
        ),
        "semantic_ok": (
            bronze_metrics["semantic_ok"]
        ),
        "rain_valid_count": (
            bronze_metrics[
                "rain_valid_count"
            ]
        ),
        "gauge_valid_count": (
            bronze_metrics[
                "gauge_valid_count"
            ]
        ),
    }

def get_valid_existing_bronze(
    filename: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    bronze_path = (
        bronze_path_for_filename(
            filename=filename,
            config=config,
        )
    )

    if not bronze_path.is_file():
        return None

    metrics = validate_gsmap_now_netcdf(
        path=bronze_path,
        filename=filename,
        config=config,
    )

    staging_path = (
        staging_path_for_filename(
            filename=filename,
            config=config,
        )
    )

    if staging_path.exists():
        staging_path.unlink()

    return {
        "filename": filename,
        "bronze_path": bronze_path,
        "file_bytes": metrics[
            "file_bytes"
        ],
        "event_start_time": metrics[
            "event_start_time"
        ],
        "physical_ok": metrics[
            "physical_ok"
        ],
        "semantic_ok": metrics[
            "semantic_ok"
        ],
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GSMaP NOW Bronze collector "
            "contract utilities."
        )
    )

    parser.add_argument(
        "--filename",
        default=(
            "gsmap_now_rain."
            "20261013.0830.nc"
        ),
        help=(
            "GSMaP NOW NetCDF filename "
            "used for contract validation."
        ),
    )
    parser.add_argument(
        "--discover-latest",
        action="store_true",
        help=(
            "Discover the latest valid "
            "GSMaP NOW rain NetCDF "
            "artifact from JAXA FTP."
        ),
    )
    parser.add_argument(
        "--download-latest",
        action="store_true",
        help=(
            "Download the latest valid "
            "GSMaP NOW rainfall NetCDF "
            "artifact into staging and "
            "validate it."
        ),
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Atomically promote a validated "
            "GSMaP NOW staging artifact "
            "into Bronze raw."
        ),
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help=(
            "Check whether the requested "
            "GSMaP NOW artifact already "
            "exists as a valid Bronze artifact."
        ),
    )

    args = parser.parse_args()

    config = load_config()

    validate_config(
        config
    )

    if args.discover_latest:
        latest = (
            discover_latest_rain_netcdf(
                config
            )
        )

        print("FTP_DISCOVERY_OK=True")

        print(
            "CANDIDATE_COUNT="
            f"{latest['candidate_count']}"
        )

        print(
            "LATEST_FILENAME="
            f"{latest['filename']}"
        )

        print(
            "LATEST_EVENT_START_TIME="
            f"{latest['event_start_time'].isoformat()}"
        )

        print(
            "REMOTE_BYTES="
            f"{latest['remote_bytes']}"
        )

        print(
            "REMOTE_MDTM="
            f"{latest['remote_mdtm']}"
        )

        return

    if args.download_latest:
        metrics = (
            download_latest_to_staging(
                config
            )
        )
        print(
            "BRONZE_ACTION="
            f"{metrics['action']}"
        )

        if (
                metrics["action"]
                == "SKIP_EXISTING"
        ):
            print(
                "BRONZE_EXISTS=True"
            )

            print(
                "BRONZE_VALID=True"
            )

            print(
                "FILENAME="
                f"{metrics['filename']}"
            )

            print(
                "EVENT_START_TIME="
                f"{metrics['event_start_time'].isoformat()}"
            )

            print(
                "BRONZE_FILE="
                f"{metrics['bronze_path']}"
            )

            print(
                "FILE_BYTES="
                f"{metrics['local_bytes']}"
            )

            print(
                "PHYSICAL_OK="
                f"{metrics['physical_ok']}"
            )

            print(
                "SEMANTIC_OK="
                f"{metrics['semantic_ok']}"
            )

            return

        print(
            "DOWNLOAD_OK=True"
        )

        print(
            "FILENAME="
            f"{metrics['filename']}"
        )

        print(
            "EVENT_START_TIME="
            f"{metrics['event_start_time'].isoformat()}"
        )

        print(
            "REMOTE_BYTES="
            f"{metrics['remote_bytes']}"
        )

        print(
            "LOCAL_BYTES="
            f"{metrics['local_bytes']}"
        )

        print(
            "BYTE_SIZE_MATCH="
            f"{metrics['remote_bytes'] == metrics['local_bytes']}"
        )

        print(
            "STAGING_FILE="
            f"{metrics['staging_path']}"
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
            "RAIN_VALID_COUNT="
            f"{metrics['rain_valid_count']}"
        )

        print(
            "GAUGE_VALID_COUNT="
            f"{metrics['gauge_valid_count']}"
        )

        return

    if args.promote:
        metrics = (
            promote_staging_to_bronze(
                filename=args.filename,
                config=config,
            )
        )

        print(
            "PROMOTE_OK=True"
        )

        print(
            "FILENAME="
            f"{metrics['filename']}"
        )

        print(
            "EVENT_START_TIME="
            f"{metrics['event_start_time'].isoformat()}"
        )

        print(
            "BRONZE_FILE="
            f"{metrics['bronze_path']}"
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
            "RAIN_VALID_COUNT="
            f"{metrics['rain_valid_count']}"
        )

        print(
            "GAUGE_VALID_COUNT="
            f"{metrics['gauge_valid_count']}"
        )

        return

    if args.check_existing:
        existing = (
            get_valid_existing_bronze(
                filename=args.filename,
                config=config,
            )
        )

        if existing is None:
            print(
                "BRONZE_EXISTS=False"
            )

        else:
            print(
                "BRONZE_EXISTS=True"
            )

            print(
                "BRONZE_VALID=True"
            )

            print(
                "BRONZE_ACTION=SKIP_EXISTING"
            )

            print(
                "FILENAME="
                f"{existing['filename']}"
            )

            print(
                "EVENT_START_TIME="
                f"{existing['event_start_time'].isoformat()}"
            )

            print(
                "BRONZE_FILE="
                f"{existing['bronze_path']}"
            )

            print(
                "FILE_BYTES="
                f"{existing['file_bytes']}"
            )

            print(
                "PHYSICAL_OK="
                f"{existing['physical_ok']}"
            )

            print(
                "SEMANTIC_OK="
                f"{existing['semantic_ok']}"
            )

        return

    filename = args.filename

    event_time = parse_event_start_time(
        filename=filename,
        config=config,
    )

    event_date = event_date_for_filename(
        filename=filename,
        config=config,
    )

    staging_path = (
        staging_path_for_filename(
            filename=filename,
            config=config,
        )
    )

    bronze_path = (
        bronze_path_for_filename(
            filename=filename,
            config=config,
        )
    )

    print("CONFIG_VALID=True")
    print(
        "DATASET="
        f"{config['dataset']}"
    )
    print(
        "FILENAME="
        f"{filename}"
    )
    print(
        "EVENT_START_TIME="
        f"{event_time.isoformat()}"
    )
    print(
        "EVENT_DATE="
        f"{event_date}"
    )
    print(
        "STAGING_PATH="
        f"{staging_path}"
    )
    print(
        "BRONZE_PATH="
        f"{bronze_path}"
    )


if __name__ == "__main__":
    main()