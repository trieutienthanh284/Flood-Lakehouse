from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from eccodes import (
    codes_get,
    codes_grib_new_from_file,
    codes_release,
)

try:
    from .era5_land_collector import (
        DEFAULT_BRONZE_RAW_DIR,
        EXPECTED_GRIB_VARIABLES,
        PROJECT_ROOT,
        bronze_path_for_date,
        grib_validity_key,
        has_valid_grib_signature,
        hourly_times,
        validate_grib_semantics,
    )
except ImportError:
    from era5_land_collector import (
        DEFAULT_BRONZE_RAW_DIR,
        EXPECTED_GRIB_VARIABLES,
        PROJECT_ROOT,
        bronze_path_for_date,
        grib_validity_key,
        has_valid_grib_signature,
        hourly_times,
        validate_grib_semantics,
    )


EXPECTED_VARIABLE_COUNT = len(
    EXPECTED_GRIB_VARIABLES
)

EXPECTED_TIME_COUNT = len(
    hourly_times()
)

EXPECTED_MESSAGE_COUNT = (
    EXPECTED_VARIABLE_COUNT
    * EXPECTED_TIME_COUNT
)


@dataclass(frozen=True)
class DayQualityResult:
    bronze_exists: bool
    physical_ok: bool

    semantic_contract_ok: bool
    semantic_error: str | None

    message_count: int
    unique_pairs: int
    unique_variables: int
    unique_valid_times: int

    duplicate_pairs: int
    missing_pairs: int
    unexpected_pairs: int

    first_valid_time: str | None
    last_valid_time: str | None

    full_day_coverage: bool


def parse_utc_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Date must use YYYY-MM-DD format."
        ) from exc


def expected_validity_keys(
    request_date: date,
) -> set[str]:
    return {
        f"{request_date.isoformat()}T{time_value}"
        for time_value in hourly_times()
    }


def expected_variable_time_pairs(
    request_date: date,
) -> set[tuple[str, str]]:
    valid_times = expected_validity_keys(
        request_date
    )

    return {
        (short_name, valid_time)
        for short_name in EXPECTED_GRIB_VARIABLES
        for valid_time in valid_times
    }


def validate_bronze_day(
    request_date: date,
    bronze_root: Path = DEFAULT_BRONZE_RAW_DIR,
) -> DayQualityResult:
    bronze_path = bronze_path_for_date(
        request_date=request_date,
        bronze_root=bronze_root,
    )

    # -------------------------------------------------
    # 1. Bronze existence check
    # -------------------------------------------------

    if not bronze_path.is_file():
        return DayQualityResult(
            bronze_exists=False,
            physical_ok=False,
            semantic_contract_ok=False,
            semantic_error=(
                "Bronze artifact does not exist."
            ),
            message_count=0,
            unique_pairs=0,
            unique_variables=0,
            unique_valid_times=0,
            duplicate_pairs=0,
            missing_pairs=EXPECTED_MESSAGE_COUNT,
            unexpected_pairs=0,
            first_valid_time=None,
            last_valid_time=None,
            full_day_coverage=False,
        )

    # -------------------------------------------------
    # 2. Physical GRIB validation
    # -------------------------------------------------

    physical_ok = has_valid_grib_signature(
        bronze_path
    )

    if not physical_ok:
        return DayQualityResult(
            bronze_exists=True,
            physical_ok=False,
            semantic_contract_ok=False,
            semantic_error=(
                "Invalid GRIB physical signature."
            ),
            message_count=0,
            unique_pairs=0,
            unique_variables=0,
            unique_valid_times=0,
            duplicate_pairs=0,
            missing_pairs=EXPECTED_MESSAGE_COUNT,
            unexpected_pairs=0,
            first_valid_time=None,
            last_valid_time=None,
            full_day_coverage=False,
        )

    # -------------------------------------------------
    # 3. Build expected daily variable-time contract
    # -------------------------------------------------

    expected_pairs = (
        expected_variable_time_pairs(
            request_date
        )
    )

    observed_pairs: set[
        tuple[str, str]
    ] = set()

    observed_variables: set[str] = set()
    observed_valid_times: set[str] = set()

    duplicate_pairs = 0
    message_count = 0

    # -------------------------------------------------
    # 4. Inspect all GRIB messages
    # -------------------------------------------------

    with bronze_path.open("rb") as f:
        while True:
            gid = codes_grib_new_from_file(f)

            if gid is None:
                break

            try:
                message_count += 1

                short_name = str(
                    codes_get(
                        gid,
                        "shortName",
                    )
                )

                valid_time = (
                    grib_validity_key(gid)
                )

                pair = (
                    short_name,
                    valid_time,
                )

                if pair in observed_pairs:
                    duplicate_pairs += 1

                observed_pairs.add(pair)

                observed_variables.add(
                    short_name
                )

                observed_valid_times.add(
                    valid_time
                )

            finally:
                codes_release(gid)

    # -------------------------------------------------
    # 5. Completeness comparison
    # -------------------------------------------------

    missing = (
        expected_pairs
        - observed_pairs
    )

    unexpected = (
        observed_pairs
        - expected_pairs
    )

    sorted_valid_times = sorted(
        observed_valid_times
    )

    first_valid_time = (
        sorted_valid_times[0]
        if sorted_valid_times
        else None
    )

    last_valid_time = (
        sorted_valid_times[-1]
        if sorted_valid_times
        else None
    )

    # -------------------------------------------------
    # 6. Full semantic contract validation
    #
    # Reuse the same semantic contract used by
    # the collector:
    #
    # - expected variables
    # - expected units
    # - validity times
    # - grid type
    # - grid dimensions
    # - spatial bounds
    # - duplicate / missing / unexpected messages
    #
    # DQ records semantic failure instead of
    # terminating immediately.
    # -------------------------------------------------

    semantic_contract_ok = True
    semantic_error = None

    try:
        validate_grib_semantics(
            path=bronze_path,
            request_date=request_date,
            expected_times=hourly_times(),
        )
    except Exception as exc:
        semantic_contract_ok = False
        semantic_error = (
            f"{type(exc).__name__}: {exc}"
        )

    # -------------------------------------------------
    # 7. Final daily DQ decision
    # -------------------------------------------------

    full_day_coverage = (
        physical_ok
        and semantic_contract_ok
        and message_count
        == EXPECTED_MESSAGE_COUNT
        and len(observed_pairs)
        == EXPECTED_MESSAGE_COUNT
        and len(observed_variables)
        == EXPECTED_VARIABLE_COUNT
        and len(observed_valid_times)
        == EXPECTED_TIME_COUNT
        and duplicate_pairs == 0
        and len(missing) == 0
        and len(unexpected) == 0
    )

    return DayQualityResult(
        bronze_exists=True,
        physical_ok=physical_ok,
        semantic_contract_ok=(
            semantic_contract_ok
        ),
        semantic_error=semantic_error,
        message_count=message_count,
        unique_pairs=len(
            observed_pairs
        ),
        unique_variables=len(
            observed_variables
        ),
        unique_valid_times=len(
            observed_valid_times
        ),
        duplicate_pairs=duplicate_pairs,
        missing_pairs=len(missing),
        unexpected_pairs=len(unexpected),
        first_valid_time=first_valid_time,
        last_valid_time=last_valid_time,
        full_day_coverage=full_day_coverage,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one complete ERA5-Land "
            "Bronze UTC day."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        type=parse_utc_date,
        help="UTC date in YYYY-MM-DD format.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = validate_bronze_day(
        args.date
    )

    bronze_path = bronze_path_for_date(
        args.date
    )

    print(
        f"DATE                = "
        f"{args.date}"
    )

    print(
        "BRONZE_PATH         = "
        f"{bronze_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"BRONZE_EXISTS       = "
        f"{result.bronze_exists}"
    )

    print(
        f"PHYSICAL_OK         = "
        f"{result.physical_ok}"
    )

    print(
        f"SEMANTIC_CONTRACT_OK= "
        f"{result.semantic_contract_ok}"
    )

    print(
        f"SEMANTIC_ERROR      = "
        f"{result.semantic_error}"
    )

    print(
        f"MESSAGE_COUNT       = "
        f"{result.message_count}"
    )

    print(
        f"UNIQUE_PAIRS        = "
        f"{result.unique_pairs}"
    )

    print(
        f"UNIQUE_VARIABLES    = "
        f"{result.unique_variables}"
    )

    print(
        f"UNIQUE_VALID_TIMES  = "
        f"{result.unique_valid_times}"
    )

    print(
        f"DUPLICATE_PAIRS     = "
        f"{result.duplicate_pairs}"
    )

    print(
        f"MISSING_PAIRS       = "
        f"{result.missing_pairs}"
    )

    print(
        f"UNEXPECTED_PAIRS    = "
        f"{result.unexpected_pairs}"
    )

    print(
        f"FIRST_VALID_TIME    = "
        f"{result.first_valid_time}"
    )

    print(
        f"LAST_VALID_TIME     = "
        f"{result.last_valid_time}"
    )

    print(
        f"FULL_DAY_COVERAGE   = "
        f"{result.full_day_coverage}"
    )

    if not result.full_day_coverage:
        raise SystemExit(1)


if __name__ == "__main__":
    main()