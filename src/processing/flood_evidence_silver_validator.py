from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DEFAULT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "flood_evidence_silver.yaml"
)


FORBIDDEN_FIELDS = {
    "h3",
    "h3_index",
    "h3_cell",
    "flood_label",
    "label",
    "gold_label",
}


def load_contract(
    path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Silver contract not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        contract = yaml.safe_load(f)

    if not isinstance(
        contract,
        dict,
    ):
        raise ValueError(
            "Silver contract must be a YAML object."
        )

    return contract


def parse_optional_datetime(
    value: Any,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            f"{field_name} must be a string "
            "or null."
        )

    normalized = (
        value.strip()
    )

    if not normalized:
        raise ValueError(
            f"{field_name} must not be empty."
        )

    if normalized.endswith("Z"):
        normalized = (
            normalized[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            normalized
        )
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must use "
            "ISO-8601 datetime format."
        ) from exc

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must be "
            "timezone-aware."
        )

    return parsed


def validate_nonempty_string(
    record: dict[str, Any],
    field_name: str,
) -> str:
    value = record.get(
        field_name
    )

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            f"{field_name} must be a string."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field_name} must not be empty."
        )

    return value


def validate_optional_number(
    record: dict[str, Any],
    field_name: str,
) -> float | None:
    value = record.get(
        field_name
    )

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{field_name} must be numeric "
            "or null."
        )

    return float(
        value
    )


def validate_silver_record(
    record: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(
        record,
        dict,
    ):
        raise ValueError(
            "Silver record must be "
            "a JSON object."
        )

    forbidden_present = sorted(
        FORBIDDEN_FIELDS
        & set(record)
    )

    if forbidden_present:
        raise ValueError(
            "Forbidden Silver fields present: "
            + ", ".join(
                forbidden_present
            )
        )

    record_id = (
        validate_nonempty_string(
            record,
            "record_id",
        )
    )

    evidence_id = (
        validate_nonempty_string(
            record,
            "evidence_id",
        )
    )

    if not evidence_id.startswith(
        "fe_"
    ):
        raise ValueError(
            "evidence_id must reference "
            "a Bronze flood evidence ID."
        )

    event_type = (
        validate_nonempty_string(
            record,
            "event_type",
        )
    )

    allowed_event_types = (
        contract[
            "event"
        ][
            "event_type"
        ][
            "allowed_values"
        ]
    )

    if (
        event_type
        not in allowed_event_types
    ):
        raise ValueError(
            "Unsupported event_type: "
            f"{event_type}"
        )

    extracted_text = (
        validate_nonempty_string(
            record,
            "extracted_text",
        )
    )

    extraction_method = (
        validate_nonempty_string(
            record,
            "extraction_method",
        )
    )

    allowed_methods = (
        contract[
            "evidence_text"
        ][
            "extraction_method"
        ][
            "allowed_values"
        ]
    )

    if (
        extraction_method
        not in allowed_methods
    ):
        raise ValueError(
            "Unsupported extraction_method: "
            f"{extraction_method}"
        )

    confidence = (
        validate_nonempty_string(
            record,
            "confidence",
        )
    )

    allowed_confidence = (
        contract[
            "quality"
        ][
            "confidence"
        ][
            "allowed_values"
        ]
    )

    if (
        confidence
        not in allowed_confidence
    ):
        raise ValueError(
            "Unsupported confidence: "
            f"{confidence}"
        )

    review_status = (
        validate_nonempty_string(
            record,
            "review_status",
        )
    )

    allowed_review_status = (
        contract[
            "quality"
        ][
            "review_status"
        ][
            "allowed_values"
        ]
    )

    if (
        review_status
        not in allowed_review_status
    ):
        raise ValueError(
            "Unsupported review_status: "
            f"{review_status}"
        )

    observed_start = (
        parse_optional_datetime(
            record.get(
                "observed_time_start"
            ),
            "observed_time_start",
        )
    )

    observed_end = (
        parse_optional_datetime(
            record.get(
                "observed_time_end"
            ),
            "observed_time_end",
        )
    )

    if (
        observed_start is not None
        and observed_end is not None
        and observed_end < observed_start
    ):
        raise ValueError(
            "observed_time_end must not "
            "precede observed_time_start."
        )

    latitude = (
        validate_optional_number(
            record,
            "latitude",
        )
    )

    longitude = (
        validate_optional_number(
            record,
            "longitude",
        )
    )

    if (
        latitude is None
    ) != (
        longitude is None
    ):
        raise ValueError(
            "latitude and longitude must "
            "either both be present or "
            "both be null."
        )

    if (
        latitude is not None
        and not (
            -90.0
            <= latitude
            <= 90.0
        )
    ):
        raise ValueError(
            "latitude must be between "
            "-90 and 90."
        )

    if (
        longitude is not None
        and not (
            -180.0
            <= longitude
            <= 180.0
        )
    ):
        raise ValueError(
            "longitude must be between "
            "-180 and 180."
        )

    flood_depth_m = (
        validate_optional_number(
            record,
            "flood_depth_m",
        )
    )

    if (
        flood_depth_m is not None
        and flood_depth_m < 0
    ):
        raise ValueError(
            "flood_depth_m must not "
            "be negative."
        )

    inundation_duration_hours = (
        validate_optional_number(
            record,
            "inundation_duration_hours",
        )
    )

    if (
        inundation_duration_hours is not None
        and inundation_duration_hours < 0
    ):
        raise ValueError(
            "inundation_duration_hours must "
            "not be negative."
        )

    rainfall_mm = (
        validate_optional_number(
            record,
            "rainfall_mm",
        )
    )

    if (
        rainfall_mm is not None
        and rainfall_mm < 0
    ):
        raise ValueError(
            "rainfall_mm must not be negative."
        )

    rainfall_duration_hours = (
        validate_optional_number(
            record,
            "rainfall_duration_hours",
        )
    )

    if (
        rainfall_duration_hours is not None
        and rainfall_duration_hours < 0
    ):
        raise ValueError(
            "rainfall_duration_hours must "
            "not be negative."
        )

    return {
        "record_id": record_id,
        "evidence_id": evidence_id,
        "event_type": event_type,
        "extracted_text": extracted_text,
        "extraction_method": (
            extraction_method
        ),
        "confidence": confidence,
        "review_status": review_status,
        "temporal_ok": True,
        "spatial_ok": True,
        "measurement_ok": True,
        "forbidden_fields_ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one structured "
            "Silver flood evidence record."
        )
    )

    parser.add_argument(
        "--record",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
    )

    args = parser.parse_args()

    try:
        contract = load_contract(
            args.contract
        )

        record = json.loads(
            args.record.read_text(
                encoding="utf-8"
            )
        )

        metrics = (
            validate_silver_record(
                record,
                contract,
            )
        )

        print(
            "SILVER_RECORD_VALID=True"
        )

        for key, value in (
            metrics.items()
        ):
            print(
                f"{key.upper()}={value}"
            )

    except Exception as exc:
        print(
            "SILVER_RECORD_VALID=False"
        )

        print(
            "ERROR_TYPE="
            f"{type(exc).__name__}"
        )

        print(
            "ERROR="
            f"{exc}"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()