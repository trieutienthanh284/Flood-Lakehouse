import argparse
from datetime import date
from pathlib import Path

import yaml
import cdsapi
import os

from eccodes import (
    codes_get,
    codes_grib_new_from_file,
    codes_release,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "era5_land.yaml"
DEFAULT_STAGING_DIR = (
    PROJECT_ROOT / "data" / "bronze" / "era5_land" / "staging"
)

DEFAULT_BRONZE_RAW_DIR = (
    PROJECT_ROOT / "data" / "bronze" / "era5_land" / "raw"
)
EXPECTED_DATASET = "reanalysis-era5-land"

EXPECTED_VARIABLES = {
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
}

EXPECTED_GRIB_VARIABLES = {
    "2t": {
        "units": "K",
    },
    "2d": {
        "units": "K",
    },
    "10u": {
        "units": "m s**-1",
    },
    "10v": {
        "units": "m s**-1",
    },
    "swvl1": {
        "units": "m**3 m**-3",
    },
    "swvl2": {
        "units": "m**3 m**-3",
    },
}
GRIB_MAGIC = b"GRIB"
GRIB_END_MARKER = b"7777"
MIN_GRIB_BYTES = 8

def grib_validity_key(gid) -> str:
    validity_date = int(codes_get(gid, "validityDate"))
    validity_time = int(codes_get(gid, "validityTime"))

    year = validity_date // 10000
    month = (validity_date // 100) % 100
    day = validity_date % 100

    hour = validity_time // 100
    minute = validity_time % 100

    return (
        f"{year:04d}-{month:02d}-{day:02d}"
        f"T{hour:02d}:{minute:02d}"
    )

def validate_grib_semantics(
    path: Path,
    request_date: date,
    expected_times: list[str],
) -> dict:
    if not has_valid_grib_signature(path):
        raise ValueError(
            f"Invalid GRIB physical signature: {path}"
        )

    expected_validity_keys = {
        f"{request_date.isoformat()}T{time_value}"
        for time_value in expected_times
    }

    expected_pairs = {
        (short_name, validity_key)
        for short_name in EXPECTED_GRIB_VARIABLES
        for validity_key in expected_validity_keys
    }

    observed_pairs = set()
    message_count = 0

    with path.open("rb") as f:
        while True:
            gid = codes_grib_new_from_file(f)

            if gid is None:
                break

            try:
                message_count += 1

                short_name = str(
                    codes_get(gid, "shortName")
                )

                if short_name not in EXPECTED_GRIB_VARIABLES:
                    raise ValueError(
                        f"Unexpected GRIB variable: {short_name}"
                    )

                units = str(codes_get(gid, "units"))
                expected_units = (
                    EXPECTED_GRIB_VARIABLES[short_name]["units"]
                )

                if units != expected_units:
                    raise ValueError(
                        f"Unexpected units for {short_name}: "
                        f"{units!r}; expected {expected_units!r}"
                    )

                validity_key = grib_validity_key(gid)

                if validity_key not in expected_validity_keys:
                    raise ValueError(
                        f"Unexpected validity time: "
                        f"{validity_key}"
                    )

                pair = (short_name, validity_key)

                if pair in observed_pairs:
                    raise ValueError(
                        f"Duplicate GRIB message: {pair}"
                    )

                observed_pairs.add(pair)

                grid_type = str(
                    codes_get(gid, "gridType")
                )

                ni = int(codes_get(gid, "Ni"))
                nj = int(codes_get(gid, "Nj"))
                points = int(
                    codes_get(gid, "numberOfDataPoints")
                )

                lat_first = float(
                    codes_get(
                        gid,
                        "latitudeOfFirstGridPointInDegrees",
                    )
                )
                lon_first = float(
                    codes_get(
                        gid,
                        "longitudeOfFirstGridPointInDegrees",
                    )
                )
                lat_last = float(
                    codes_get(
                        gid,
                        "latitudeOfLastGridPointInDegrees",
                    )
                )
                lon_last = float(
                    codes_get(
                        gid,
                        "longitudeOfLastGridPointInDegrees",
                    )
                )

                if grid_type != "regular_ll":
                    raise ValueError(
                        f"Unexpected grid type: {grid_type}"
                    )

                if (ni, nj, points) != (4, 3, 12):
                    raise ValueError(
                        "Unexpected ERA5-Land grid dimensions: "
                        f"Ni={ni}, Nj={nj}, points={points}"
                    )

                tolerance = 1e-6

                if abs(lat_first - 21.1) > tolerance:
                    raise ValueError(
                        f"Unexpected LAT_FIRST={lat_first}"
                    )

                if abs(lon_first - 105.7) > tolerance:
                    raise ValueError(
                        f"Unexpected LON_FIRST={lon_first}"
                    )

                if abs(lat_last - 20.9) > tolerance:
                    raise ValueError(
                        f"Unexpected LAT_LAST={lat_last}"
                    )

                if abs(lon_last - 106.0) > tolerance:
                    raise ValueError(
                        f"Unexpected LON_LAST={lon_last}"
                    )

            finally:
                codes_release(gid)

    missing_pairs = expected_pairs - observed_pairs
    unexpected_pairs = observed_pairs - expected_pairs

    if missing_pairs:
        raise ValueError(
            f"Missing GRIB variable/time pairs: "
            f"{sorted(missing_pairs)}"
        )

    if unexpected_pairs:
        raise ValueError(
            f"Unexpected GRIB variable/time pairs: "
            f"{sorted(unexpected_pairs)}"
        )

    if message_count != len(expected_pairs):
        raise ValueError(
            f"Unexpected message count: "
            f"{message_count}; "
            f"expected {len(expected_pairs)}"
        )

    return {
        "message_count": message_count,
        "variable_count": len(EXPECTED_GRIB_VARIABLES),
        "time_count": len(expected_validity_keys),
        "semantic_ok": True,
    }

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"ERA5-Land config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("ERA5-Land config must be a YAML mapping.")

    return config


def validate_config(config: dict) -> None:
    dataset = config.get("dataset")
    if dataset != EXPECTED_DATASET:
        raise ValueError(
            f"Unexpected dataset: {dataset!r}. "
            f"Expected {EXPECTED_DATASET!r}."
        )

    temporal = config.get("temporal")
    if not isinstance(temporal, dict):
        raise ValueError("Missing or invalid temporal configuration.")

    if temporal.get("timezone") != "UTC":
        raise ValueError("ERA5-Land temporal.timezone must be UTC.")

    if temporal.get("resolution") != "1h":
        raise ValueError("ERA5-Land temporal.resolution must be 1h.")

    if temporal.get("request_partition") != "1_day":
        raise ValueError(
            "ERA5-Land temporal.request_partition must be 1_day."
        )

    area = config.get("area")
    if not isinstance(area, dict):
        raise ValueError("Missing or invalid area configuration.")

    required_area_keys = {"north", "west", "south", "east"}
    missing_area_keys = required_area_keys - area.keys()

    if missing_area_keys:
        raise ValueError(
            f"Missing ERA5-Land area keys: {sorted(missing_area_keys)}"
        )

    north = float(area["north"])
    west = float(area["west"])
    south = float(area["south"])
    east = float(area["east"])

    if north <= south:
        raise ValueError("ERA5-Land area requires north > south.")

    if east <= west:
        raise ValueError("ERA5-Land area requires east > west.")

    variables = config.get("variables")
    if not isinstance(variables, list):
        raise ValueError("ERA5-Land variables must be a YAML list.")

    actual_variables = set(variables)

    if actual_variables != EXPECTED_VARIABLES:
        missing = EXPECTED_VARIABLES - actual_variables
        unexpected = actual_variables - EXPECTED_VARIABLES

        raise ValueError(
            "ERA5-Land variable contract mismatch. "
            f"Missing={sorted(missing)}, "
            f"Unexpected={sorted(unexpected)}"
        )

    if config.get("data_format") != "grib":
        raise ValueError("ERA5-Land data_format must be grib.")

    if config.get("download_format") != "unarchived":
        raise ValueError(
            "ERA5-Land download_format must be unarchived."
        )


def parse_utc_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Date must use YYYY-MM-DD format."
        ) from exc


def hourly_times() -> list[str]:
    return [f"{hour:02d}:00" for hour in range(24)]

def daily_filename(request_date: date) -> str:
    return f"era5_land_{request_date:%Y%m%d}.grib"


def staging_path_for_date(
    request_date: date,
    staging_root: Path = DEFAULT_STAGING_DIR,
) -> Path:
    return staging_root / f"{daily_filename(request_date)}.part"


def bronze_path_for_date(
    request_date: date,
    bronze_root: Path = DEFAULT_BRONZE_RAW_DIR,
) -> Path:
    return (
        bronze_root
        / f"year={request_date:%Y}"
        / f"month={request_date:%m}"
        / f"day={request_date:%d}"
        / daily_filename(request_date)
    )

def has_valid_grib_signature(path: Path) -> bool:
    """
    Kiểm tra physical signature tối thiểu của một GRIB artifact.

    Đây chỉ là physical validation, chưa xác nhận biến,
    thời gian, grid hoặc số lượng GRIB messages.
    """
    if not path.is_file():
        return False

    if path.stat().st_size < MIN_GRIB_BYTES:
        return False

    try:
        with path.open("rb") as f:
            first_four = f.read(4)

            f.seek(-4, 2)
            last_four = f.read(4)
    except OSError:
        return False

    return (
        first_four == GRIB_MAGIC
        and last_four == GRIB_END_MARKER
    )

def build_daily_request(config: dict, request_date: date) -> dict:
    area = config["area"]

    return {
        "variable": list(config["variables"]),
        "year": f"{request_date.year:04d}",
        "month": f"{request_date.month:02d}",
        "day": [f"{request_date.day:02d}"],
        "time": hourly_times(),
        "data_format": config["data_format"],
        "download_format": config["download_format"],
        "area": [
            float(area["north"]),
            float(area["west"]),
            float(area["south"]),
            float(area["east"]),
        ],
    }

def download_to_staging(
    config: dict,
    request_date: date,
    staging_root: Path = DEFAULT_STAGING_DIR,
) -> dict:
    request = build_daily_request(
        config=config,
        request_date=request_date,
    )

    staging_path = staging_path_for_date(
        request_date=request_date,
        staging_root=staging_root,
    )

    staging_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Không tái sử dụng một partial artifact cũ.
    staging_path.unlink(missing_ok=True)

    client = cdsapi.Client()

    try:
        client.retrieve(
            config["dataset"],
            request,
            str(staging_path),
        )
    except Exception:
        staging_path.unlink(missing_ok=True)
        raise

    if not staging_path.exists():
        raise RuntimeError(
            f"CDS download completed but staging file "
            f"does not exist: {staging_path}"
        )

    if not has_valid_grib_signature(staging_path):
        staging_path.unlink(missing_ok=True)

        raise ValueError(
            "Downloaded ERA5-Land artifact does not have "
            "a valid GRIB physical signature."
        )

    semantic_result = validate_grib_semantics(
        path=staging_path,
        request_date=request_date,
        expected_times=request["time"],
    )

    return {
        "staging_path": staging_path,
        "bytes_written": staging_path.stat().st_size,
        "message_count": semantic_result["message_count"],
        "variable_count": semantic_result["variable_count"],
        "time_count": semantic_result["time_count"],
        "physical_ok": True,
        "semantic_ok": semantic_result["semantic_ok"],
    }

def promote_to_bronze(
    config: dict,
    request_date: date,
    staging_root: Path = DEFAULT_STAGING_DIR,
    bronze_root: Path = DEFAULT_BRONZE_RAW_DIR,
) -> dict:
    """
    Validate lại staging artifact rồi atomically promote sang Bronze raw.

    Hàm không được phép âm thầm ghi đè một Bronze artifact
    đã tồn tại.
    """
    staging_path = staging_path_for_date(
        request_date=request_date,
        staging_root=staging_root,
    )

    bronze_path = bronze_path_for_date(
        request_date=request_date,
        bronze_root=bronze_root,
    )

    if not staging_path.is_file():
        raise FileNotFoundError(
            f"ERA5-Land staging artifact not found: "
            f"{staging_path}"
        )

    if bronze_path.exists():
        raise FileExistsError(
            f"ERA5-Land Bronze artifact already exists: "
            f"{bronze_path}"
        )

    if not has_valid_grib_signature(staging_path):
        raise ValueError(
            f"Invalid GRIB physical signature: "
            f"{staging_path}"
        )

    request = build_daily_request(
        config=config,
        request_date=request_date,
    )

    semantic_result = validate_grib_semantics(
        path=staging_path,
        request_date=request_date,
        expected_times=request["time"],
    )

    bytes_before_move = staging_path.stat().st_size

    bronze_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # os.replace phải diễn ra trên cùng filesystem
    # để giữ atomic promotion semantics.
    staging_drive = staging_path.resolve().drive.lower()
    bronze_drive = bronze_path.resolve().drive.lower()

    if staging_drive != bronze_drive:
        raise RuntimeError(
            "Staging and Bronze are not on the same filesystem "
            "drive; atomic promotion cannot be guaranteed."
        )

    os.replace(
        staging_path,
        bronze_path,
    )

    if not bronze_path.is_file():
        raise RuntimeError(
            f"Atomic promotion failed; Bronze artifact "
            f"does not exist: {bronze_path}"
        )

    if staging_path.exists():
        raise RuntimeError(
            f"Atomic promotion failed; staging artifact "
            f"still exists: {staging_path}"
        )

    if bronze_path.stat().st_size != bytes_before_move:
        raise RuntimeError(
            "Bronze artifact size changed during promotion."
        )

    if not has_valid_grib_signature(bronze_path):
        raise RuntimeError(
            "Bronze artifact failed physical validation "
            "after promotion."
        )

    return {
        "bronze_path": bronze_path,
        "bytes_written": bronze_path.stat().st_size,
        "message_count": semantic_result["message_count"],
        "variable_count": semantic_result["variable_count"],
        "time_count": semantic_result["time_count"],
        "promotion_ok": True,
    }

def get_valid_existing_bronze(
    config: dict,
    request_date: date,
    bronze_root: Path = DEFAULT_BRONZE_RAW_DIR,
) -> Path | None:
    """
    Trả về Bronze artifact nếu nó đã tồn tại và hợp lệ.

    Nếu chưa tồn tại:
        return None

    Nếu tồn tại nhưng không hợp lệ:
        raise error
    """
    bronze_path = bronze_path_for_date(
        request_date=request_date,
        bronze_root=bronze_root,
    )

    if not bronze_path.exists():
        return None

    if not bronze_path.is_file():
        raise ValueError(
            f"ERA5-Land Bronze path exists but is not a file: "
            f"{bronze_path}"
        )

    if not has_valid_grib_signature(bronze_path):
        raise ValueError(
            f"ERA5-Land Bronze artifact has invalid "
            f"GRIB signature: {bronze_path}"
        )

    request = build_daily_request(
        config=config,
        request_date=request_date,
    )

    validate_grib_semantics(
        path=bronze_path,
        request_date=request_date,
        expected_times=request["time"],
    )

    return bronze_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ERA5-Land daily ingestion request."
    )

    parser.add_argument(
        "--date",
        required=True,
        type=parse_utc_date,
        help="UTC date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download one complete ERA5-Land UTC day "
            "to staging and validate it."
        ),
    )

    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Validate an existing ERA5-Land staging artifact "
            "and atomically promote it to Bronze raw."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(CONFIG_PATH)
    validate_config(config)

    request = build_daily_request(config, args.date)
    staging_path = staging_path_for_date(args.date)
    bronze_path = bronze_path_for_date(args.date)

    print("ERA5_CONFIG_VALID=True")
    print(f"DATASET={config['dataset']}")
    print(f"REQUEST_DATE={args.date.isoformat()}")
    print(f"VARIABLE_COUNT={len(request['variable'])}")
    print(f"TIME_COUNT={len(request['time'])}")
    print(f"FIRST_TIME={request['time'][0]}")
    print(f"LAST_TIME={request['time'][-1]}")
    print(f"YEAR={request['year']}")
    print(f"MONTH={request['month']}")
    print(f"DAY={request['day']}")
    print(f"AREA={request['area']}")
    print(f"DATA_FORMAT={request['data_format']}")
    print(f"DOWNLOAD_FORMAT={request['download_format']}")
    print(
        "STAGING_PATH="
        f"{staging_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "BRONZE_PATH="
        f"{bronze_path.relative_to(PROJECT_ROOT)}"
    )

    if args.download or args.promote:
        existing_bronze = get_valid_existing_bronze(
            config=config,
            request_date=args.date,
        )

        if existing_bronze is not None:
            stale_staging_path = staging_path_for_date(
                request_date=args.date,
            )

            stale_staging_path.unlink(
                missing_ok=True,
            )

            print("BRONZE_EXISTS=True")
            print("BRONZE_VALID=True")
            print("BRONZE_ACTION=SKIP_EXISTING")
            print(
                "BRONZE_FILE="
                f"{existing_bronze.relative_to(PROJECT_ROOT)}"
            )

            return

    if args.download:
        result = download_to_staging(
            config=config,
            request_date=args.date,
        )

        print("DOWNLOAD_OK=True")
        print(
            "STAGING_FILE="
            f"{result['staging_path'].relative_to(PROJECT_ROOT)}"
        )
        print(f"STAGING_BYTES={result['bytes_written']}")
        print(f"GRIB_PHYSICAL_OK={result['physical_ok']}")
        print(f"GRIB_SEMANTIC_OK={result['semantic_ok']}")
        print(f"MESSAGE_COUNT={result['message_count']}")
        print(f"VARIABLE_COUNT={result['variable_count']}")
        print(f"TIME_COUNT={result['time_count']}")

    if args.promote:
        promotion = promote_to_bronze(
            config=config,
            request_date=args.date,
        )

        print("PROMOTE_OK=True")
        print(
            "BRONZE_FILE="
            f"{promotion['bronze_path'].relative_to(PROJECT_ROOT)}"
        )
        print(f"BRONZE_BYTES={promotion['bytes_written']}")
        print(
            f"MESSAGE_COUNT="
            f"{promotion['message_count']}"
        )
        print(
            f"VARIABLE_COUNT="
            f"{promotion['variable_count']}"
        )
        print(
            f"TIME_COUNT="
            f"{promotion['time_count']}"
        )


if __name__ == "__main__":
    main()