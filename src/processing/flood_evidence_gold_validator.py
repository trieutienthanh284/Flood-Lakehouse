from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import h3
import yaml


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DEFAULT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "flood_evidence_gold.yaml"
)


def load_contract(
    path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    data = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Gold contract must be an object."
        )

    return data


def resolve_project_path(
    value: str,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def validate_nonempty_string(
    record: dict[str, Any],
    field: str,
) -> str:
    value = record.get(field)

    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a string."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field} must not be empty."
        )

    return value


def parse_optional_date(
    record: dict[str, Any],
    field: str,
) -> date | None:
    value = record.get(field)

    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be an ISO date string."
        )

    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{field} must be a valid ISO date."
        ) from exc

    return parsed


def parse_optional_datetime(
    record: dict[str, Any],
    field: str,
) -> datetime | None:
    value = record.get(field)

    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be an ISO datetime string."
        )

    try:
        parsed = datetime.fromisoformat(
            value
        )
    except ValueError as exc:
        raise ValueError(
            f"{field} must be a valid ISO datetime."
        ) from exc

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        raise ValueError(
            f"{field} must be timezone-aware."
        )

    return parsed


def derive_expected_temporal_from_silver(
    silver_record: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    policy = contract[
        "temporal"
    ][
        "derivation_policy"
    ]

    if (
        policy.get("policy")
        != "deterministic_from_silver"
    ):
        raise ValueError(
            "Unsupported temporal derivation policy."
        )

    if (
        policy.get(
            "temporal_text_date_order"
        )
        != "DMY"
    ):
        raise ValueError(
            "Unsupported temporal text date order."
        )

    observed_policy = policy[
        "observed_timestamp"
    ]

    parsed_policy = policy[
        "parsed_temporal_text"
    ]

    no_source_policy = policy[
        "no_temporal_source"
    ]

    observed_start_raw = silver_record.get(
        "observed_time_start"
    )

    observed_end_raw = silver_record.get(
        "observed_time_end"
    )

    temporal_text = silver_record.get(
        "temporal_text"
    )

    if (
        observed_end_raw is not None
        and observed_start_raw is None
        and observed_policy.get(
            "observed_time_end_requires_observed_time_start"
        )
        is True
    ):
        raise ValueError(
            "Silver observed_time_end requires "
            "observed_time_start."
        )

    if observed_start_raw is not None:
        if not isinstance(
            observed_start_raw,
            str,
        ):
            raise ValueError(
                "Silver observed_time_start must "
                "be an ISO datetime string."
            )

        try:
            observed_start = (
                datetime.fromisoformat(
                    observed_start_raw
                )
            )
        except ValueError as exc:
            raise ValueError(
                "Silver observed_time_start must "
                "be a valid ISO datetime."
            ) from exc

        if (
            observed_start.tzinfo is None
            or observed_start.utcoffset()
            is None
        ):
            raise ValueError(
                "Silver observed_time_start must "
                "be timezone-aware."
            )

        observed_end = None

        if observed_end_raw is not None:
            if not isinstance(
                observed_end_raw,
                str,
            ):
                raise ValueError(
                    "Silver observed_time_end must "
                    "be an ISO datetime string."
                )

            try:
                observed_end = (
                    datetime.fromisoformat(
                        observed_end_raw
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    "Silver observed_time_end must "
                    "be a valid ISO datetime."
                ) from exc

            if (
                observed_end.tzinfo is None
                or observed_end.utcoffset()
                is None
            ):
                raise ValueError(
                    "Silver observed_time_end must "
                    "be timezone-aware."
                )

            if (
                observed_end
                < observed_start
            ):
                raise ValueError(
                    "Silver observed_time_end must "
                    "not precede observed_time_start."
                )

        return {
            "derivation_class":
                "observed_timestamp",

            "temporal_resolution_status":
                observed_policy[
                    "required_resolution_status"
                ],

            "event_date_start":
                observed_start.date(),

            "event_date_end":
                (
                    observed_end.date()
                    if observed_end is not None
                    else observed_start.date()
                ),

            "event_time_start":
                observed_start,

            "event_time_end":
                observed_end,
        }

    if temporal_text is None:
        return {
            "derivation_class":
                "no_temporal_source",

            "temporal_resolution_status":
                no_source_policy[
                    "required_resolution_status"
                ],

            "event_date_start":
                None,

            "event_date_end":
                None,

            "event_time_start":
                None,

            "event_time_end":
                None,
        }

    if (
        not isinstance(
            temporal_text,
            str,
        )
        or not temporal_text.strip()
    ):
        raise ValueError(
            "Silver temporal_text must be "
            "a non-empty string or null."
        )

    single_date_pattern = re.compile(
        r"^\s*"
        r"(\d{1,2})/"
        r"(\d{1,2})/"
        r"(\d{4})"
        r"\s*$"
    )

    same_month_range_pattern = re.compile(
        r"^\s*"
        r"(\d{1,2})-"
        r"(\d{1,2})/"
        r"(\d{1,2})/"
        r"(\d{4})"
        r"\s*$"
    )

    cross_month_range_pattern = re.compile(
        r"^\s*"
        r"(\d{1,2})/"
        r"(\d{1,2})-"
        r"(\d{1,2})/"
        r"(\d{1,2})/"
        r"(\d{4})"
        r"\s*$"
    )

    embedded_date_pattern = re.compile(
        r"(?<!\d)"
        r"(\d{1,2})/"
        r"(\d{1,2})/"
        r"(\d{4})"
        r"(?!\d)"
    )

    def make_date(
        year: str,
        month: str,
        day: str,
    ) -> date:
        try:
            return date(
                int(year),
                int(month),
                int(day),
            )
        except ValueError as exc:
            raise ValueError(
                "Silver temporal_text contains "
                "an invalid calendar date."
            ) from exc

    derivation_class = None
    start_date = None
    end_date = None

    match = cross_month_range_pattern.fullmatch(
        temporal_text
    )

    if match:
        (
            start_day,
            start_month,
            end_day,
            end_month,
            year,
        ) = match.groups()

        derivation_class = (
            "cross_month_date_range"
        )

        start_date = make_date(
            year,
            start_month,
            start_day,
        )

        end_date = make_date(
            year,
            end_month,
            end_day,
        )

    if derivation_class is None:
        match = (
            same_month_range_pattern.fullmatch(
                temporal_text
            )
        )

        if match:
            (
                start_day,
                end_day,
                month,
                year,
            ) = match.groups()

            derivation_class = (
                "same_month_date_range"
            )

            start_date = make_date(
                year,
                month,
                start_day,
            )

            end_date = make_date(
                year,
                month,
                end_day,
            )

    if derivation_class is None:
        match = single_date_pattern.fullmatch(
            temporal_text
        )

        if match:
            (
                day,
                month,
                year,
            ) = match.groups()

            derivation_class = (
                "single_date"
            )

            start_date = make_date(
                year,
                month,
                day,
            )

            end_date = start_date

    if derivation_class is None:
        match = embedded_date_pattern.search(
            temporal_text
        )

        if match:
            (
                day,
                month,
                year,
            ) = match.groups()

            derivation_class = (
                "date_embedded_in_text"
            )

            start_date = make_date(
                year,
                month,
                day,
            )

            end_date = start_date

    supported_shapes = set(
        parsed_policy.get(
            "supported_shapes",
            [],
        )
    )

    if (
        derivation_class is None
        or derivation_class
        not in supported_shapes
    ):
        raise ValueError(
            "Unsupported Silver temporal_text "
            "shape for deterministic Gold "
            "temporal derivation."
        )

    if (
        start_date is not None
        and end_date is not None
        and end_date < start_date
    ):
        raise ValueError(
            "Derived temporal date interval "
            "is reversed."
        )

    return {
        "derivation_class":
            derivation_class,

        "temporal_resolution_status":
            parsed_policy[
                "required_resolution_status"
            ],

        "event_date_start":
            start_date,

        "event_date_end":
            end_date,

        "event_time_start":
            None,

        "event_time_end":
            None,
    }


def validate_optional_number(
    record: dict[str, Any],
    field: str,
) -> float | int | None:
    value = record.get(field)

    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise ValueError(
            f"{field} must be numeric."
        )

    return value


def load_h3_reference(
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    scope = contract["scope"]

    reference_path = resolve_project_path(
        scope["h3_reference_path"]
    )

    data = json.loads(
        reference_path.read_text(
            encoding="utf-8"
        )
    )

    if data.get("type") != "FeatureCollection":
        raise ValueError(
            "H3 reference must be a FeatureCollection."
        )

    id_field = scope[
        "h3_reference_id_field"
    ]

    resolution_field = scope[
        "h3_reference_resolution_field"
    ]

    expected_resolution = int(
        scope["h3_resolution"]
    )

    lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    for feature in data.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        h3_id = properties.get(
            id_field
        )

        if (
            not isinstance(h3_id, str)
            or not h3.is_valid_cell(h3_id)
        ):
            raise ValueError(
                "Invalid H3 cell in reference."
            )

        if (
            h3.get_resolution(h3_id)
            != expected_resolution
        ):
            raise ValueError(
                "Unexpected H3 resolution "
                "in reference."
            )

        if (
            properties.get(
                resolution_field
            )
            != expected_resolution
        ):
            raise ValueError(
                "Reference resolution property "
                "does not match contract."
            )

        if h3_id in lookup:
            raise ValueError(
                "Duplicate H3 cell in reference."
            )

        lookup[h3_id] = properties

    if not lookup:
        raise ValueError(
            "H3 reference is empty."
        )

    return lookup


def find_silver_record(
    silver_record_id: str,
) -> Path:
    root = (
        PROJECT_ROOT
        / "data"
        / "silver"
        / "flood_evidence"
        / "records"
    )

    matches = list(
        root.rglob(
            f"{silver_record_id}.json"
        )
    )

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one Silver "
            f"record for {silver_record_id}; "
            f"found {len(matches)}."
        )

    return matches[0]


def validate_gold_record(
    record: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(
            "Gold record must be an object."
        )

    gold_record_id = validate_nonempty_string(
        record,
        "gold_record_id",
    )

    silver_record_id = validate_nonempty_string(
        record,
        "silver_record_id",
    )

    evidence_id = validate_nonempty_string(
        record,
        "evidence_id",
    )

    if not gold_record_id.startswith(
        "feg_"
    ):
        raise ValueError(
            "gold_record_id must start with feg_."
        )

    if not silver_record_id.startswith(
        "fer_"
    ):
        raise ValueError(
            "silver_record_id must start with fer_."
        )

    if not evidence_id.startswith(
        "fe_"
    ):
        raise ValueError(
            "evidence_id must start with fe_."
        )

    silver_path = find_silver_record(
        silver_record_id
    )

    silver_record = json.loads(
        silver_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        silver_record.get("evidence_id")
        != evidence_id
    ):
        raise ValueError(
            "Gold evidence_id does not match "
            "the Silver source record."
        )

    event_type = validate_nonempty_string(
        record,
        "event_type",
    )

    allowed_event_types = contract[
        "event"
    ][
        "event_type"
    ][
        "allowed_values"
    ]

    if event_type not in allowed_event_types:
        raise ValueError(
            "event_type is not allowed "
            "by Gold contract."
        )

    if (
        silver_record.get("event_type")
        != event_type
    ):
        raise ValueError(
            "Gold event_type must match "
            "the Silver source record."
        )

    semantic_policy = contract[
        "source_semantic_preservation"
    ]

    if (
        semantic_policy.get(
            "policy"
        )
        != "exact_match"
    ):
        raise ValueError(
            "Unsupported source semantic "
            "preservation policy."
        )

    required_equal_fields = (
        semantic_policy.get(
            "required_equal_fields",
            [],
        )
    )

    if not isinstance(
        required_equal_fields,
        list,
    ):
        raise ValueError(
            "source_semantic_preservation."
            "required_equal_fields must be a list."
        )

    for field in required_equal_fields:
        gold_value = record.get(
            field
        )

        silver_value = silver_record.get(
            field
        )

        if gold_value != silver_value:
            raise ValueError(
                "Gold source semantic field "
                f"{field} does not match "
                "the Silver source record."
            )

    event_date_start = parse_optional_date(
        record,
        "event_date_start",
    )

    event_date_end = parse_optional_date(
        record,
        "event_date_end",
    )

    if (
        event_date_start is not None
        and event_date_end is not None
        and event_date_end
        < event_date_start
    ):
        raise ValueError(
            "event_date_end must not be "
            "before event_date_start."
        )

    event_time_start = parse_optional_datetime(
        record,
        "event_time_start",
    )

    event_time_end = parse_optional_datetime(
        record,
        "event_time_end",
    )

    if (
        event_time_start is not None
        and event_time_end is not None
        and event_time_end
        < event_time_start
    ):
        raise ValueError(
            "event_time_end must not be "
            "before event_time_start."
        )

    temporal_status = validate_nonempty_string(
        record,
        "temporal_resolution_status",
    )

    allowed_temporal_status = contract[
        "temporal"
    ][
        "temporal_resolution_status"
    ][
        "allowed_values"
    ]

    if (
        temporal_status
        not in allowed_temporal_status
    ):
        raise ValueError(
            "Invalid temporal_resolution_status."
        )

    temporal_semantics = contract[
        "temporal"
    ][
        "temporal_resolution_status"
    ].get(
        "semantics",
        {},
    )

    status_semantics = (
        temporal_semantics.get(
            temporal_status,
            {},
        )
    )

    if (
        status_semantics.get(
            "requires_event_time_start"
        )
        is True
        and event_time_start is None
    ):
        raise ValueError(
            "resolved temporal status requires "
            "event_time_start."
        )

    if (
        status_semantics.get(
            "requires_event_date_start"
        )
        is True
        and event_date_start is None
    ):
        raise ValueError(
            "partial temporal status requires "
            "event_date_start."
        )

    if (
        status_semantics.get(
            "prohibits_event_time"
        )
        is True
        and (
            event_time_start is not None
            or event_time_end is not None
        )
    ):
        raise ValueError(
            "partial temporal status prohibits "
            "event_time_start and event_time_end."
        )

    if (
        status_semantics.get(
            "prohibits_normalized_date_time"
        )
        is True
        and any(
            value is not None
            for value in [
                event_date_start,
                event_date_end,
                event_time_start,
                event_time_end,
            ]
        )
    ):
        raise ValueError(
            "unresolved temporal status prohibits "
            "normalized date/time fields."
        )

    expected_temporal = (
        derive_expected_temporal_from_silver(
            silver_record,
            contract,
        )
    )

    expected_temporal_status = (
        expected_temporal[
            "temporal_resolution_status"
        ]
    )

    if (
        temporal_status
        != expected_temporal_status
    ):
        raise ValueError(
            "Gold temporal_resolution_status "
            "does not match deterministic "
            "Silver temporal derivation."
        )

    derived_temporal_fields = {
        "event_date_start": (
            event_date_start,
            expected_temporal[
                "event_date_start"
            ],
        ),
        "event_date_end": (
            event_date_end,
            expected_temporal[
                "event_date_end"
            ],
        ),
        "event_time_start": (
            event_time_start,
            expected_temporal[
                "event_time_start"
            ],
        ),
        "event_time_end": (
            event_time_end,
            expected_temporal[
                "event_time_end"
            ],
        ),
    }

    for (
        field,
        (
            gold_value,
            expected_value,
        ),
    ) in derived_temporal_fields.items():
        if gold_value != expected_value:
            raise ValueError(
                "Gold derived temporal field "
                f"{field} does not match "
                "deterministic Silver derivation."
            )

    latitude = validate_optional_number(
        record,
        "latitude",
    )

    longitude = validate_optional_number(
        record,
        "longitude",
    )

    if (
        latitude is None
    ) != (
        longitude is None
    ):
        raise ValueError(
            "latitude and longitude must "
            "both be present or both be null."
        )

    if (
        latitude is not None
        and not -90 <= latitude <= 90
    ):
        raise ValueError(
            "latitude must be between -90 and 90."
        )

    if (
        longitude is not None
        and not -180 <= longitude <= 180
    ):
        raise ValueError(
            "longitude must be between "
            "-180 and 180."
        )

    h3_index = validate_nonempty_string(
        record,
        "h3_index",
    )

    if not h3.is_valid_cell(
        h3_index
    ):
        raise ValueError(
            "h3_index is not a valid H3 cell."
        )

    expected_resolution = int(
        contract["scope"][
            "h3_resolution"
        ]
    )

    if (
        h3.get_resolution(h3_index)
        != expected_resolution
    ):
        raise ValueError(
            "h3_index has the wrong resolution."
        )

    h3_resolution = record.get(
        "h3_resolution"
    )

    if (
        isinstance(h3_resolution, bool)
        or not isinstance(
            h3_resolution,
            int,
        )
        or h3_resolution
        != expected_resolution
    ):
        raise ValueError(
            "h3_resolution must equal "
            f"{expected_resolution}."
        )

    reference = load_h3_reference(
        contract
    )

    if h3_index not in reference:
        raise ValueError(
            "h3_index is outside the Gold "
            "H3 reference scope."
        )

    spatial_method = validate_nonempty_string(
        record,
        "spatial_resolution_method",
    )

    allowed_spatial_methods = contract[
        "spatial"
    ][
        "spatial_resolution_method"
    ][
        "allowed_values"
    ]

    if (
        spatial_method
        not in allowed_spatial_methods
    ):
        raise ValueError(
            "Invalid spatial_resolution_method."
        )

    spatial_reference_config = contract[
        "spatial"
    ][
        "spatial_reference"
    ]

    spatial_reference = record.get(
        "spatial_reference"
    )

    if (
        spatial_reference is not None
        and not isinstance(
            spatial_reference,
            dict,
        )
    ):
        raise ValueError(
            "spatial_reference must be "
            "an object or null."
        )

    required_reference_methods = (
        spatial_reference_config.get(
            "required_for_methods",
            [],
        )
    )

    reference_required = (
        spatial_method
        in required_reference_methods
    )

    if (
        reference_required
        and not isinstance(
            spatial_reference,
            dict,
        )
    ):
        raise ValueError(
            "spatial_reference is required "
            "for this spatial_resolution_method."
        )

    if isinstance(
        spatial_reference,
        dict,
    ):
        reference_fields = (
            spatial_reference_config[
                "fields"
            ]
        )

        for (
            reference_field,
            reference_field_config,
        ) in reference_fields.items():
            value = spatial_reference.get(
                reference_field
            )

            if (
                reference_field_config.get(
                    "required"
                )
                is True
            ):
                if (
                    not isinstance(
                        value,
                        str,
                    )
                    or not value.strip()
                ):
                    raise ValueError(
                        "spatial_reference."
                        f"{reference_field} "
                        "must be a non-empty string."
                    )

        snapshot_date = (
            spatial_reference.get(
                "snapshot_date"
            )
        )

        if isinstance(
            snapshot_date,
            str,
        ):
            try:
                __import__(
                    "datetime"
                ).date.fromisoformat(
                    snapshot_date
                )

            except ValueError as exc:
                raise ValueError(
                    "spatial_reference.snapshot_date "
                    "must be an ISO date."
                ) from exc

    scope_status = validate_nonempty_string(
        record,
        "scope_status",
    )

    if scope_status != "in_scope":
        raise ValueError(
            "scope_status must be in_scope."
        )

    measurement_fields = [
        "flood_depth_m",
        "inundation_duration_hours",
        "rainfall_mm",
        "rainfall_duration_hours",
    ]

    for field in measurement_fields:
        value = validate_optional_number(
            record,
            field,
        )

        if (
            value is not None
            and value < 0
        ):
            raise ValueError(
                f"{field} must not be negative."
            )

    flood_depth_text = record.get(
        "flood_depth_text"
    )

    if (
        flood_depth_text is not None
        and not isinstance(
            flood_depth_text,
            str,
        )
    ):
        raise ValueError(
            "flood_depth_text must be "
            "a string or null."
        )

    flood_label = validate_nonempty_string(
        record,
        "flood_label",
    )

    allowed_labels = contract[
        "label"
    ][
        "flood_label"
    ][
        "allowed_values"
    ]

    if flood_label not in allowed_labels:
        raise ValueError(
            "Invalid flood_label."
        )

    label_basis = validate_nonempty_string(
        record,
        "label_basis",
    )

    allowed_basis = contract[
        "label"
    ][
        "label_basis"
    ][
        "allowed_values"
    ]

    if label_basis not in allowed_basis:
        raise ValueError(
            "Invalid label_basis."
        )

    label_confidence = validate_nonempty_string(
        record,
        "label_confidence",
    )

    allowed_confidence = contract[
        "label"
    ][
        "label_confidence"
    ][
        "allowed_values"
    ]

    if (
        label_confidence
        not in allowed_confidence
    ):
        raise ValueError(
            "Invalid label_confidence."
        )

    validate_nonempty_string(
        record,
        "label_policy_version",
    )

    compatibility = contract[
        "label"
    ][
        "compatibility"
    ][
        flood_label
    ]

    if (
        event_type
        not in compatibility[
            "eligible_event_types"
        ]
    ):
        raise ValueError(
            "flood_label is incompatible "
            "with event_type."
        )

    if (
        label_basis
        not in compatibility[
            "allowed_label_basis"
        ]
    ):
        raise ValueError(
            "label_basis is incompatible "
            "with flood_label."
        )

    review_status = validate_nonempty_string(
        record,
        "review_status",
    )

    allowed_review_status = contract[
        "quality"
    ][
        "review_status"
    ][
        "allowed_values"
    ]

    if (
        review_status
        not in allowed_review_status
    ):
        raise ValueError(
            "Invalid review_status."
        )

    required_review_status = (
        compatibility.get(
            "required_review_status"
        )
    )

    if (
        required_review_status is not None
        and review_status
        != required_review_status
    ):
        raise ValueError(
            "review_status is incompatible "
            "with flood_label."
        )

    negative_config = contract[
        "label"
    ][
        "negative_evidence"
    ]

    negative_evidence = record.get(
        "negative_evidence"
    )

    if (
        negative_evidence is not None
        and not isinstance(
            negative_evidence,
            dict,
        )
    ):
        raise ValueError(
            "negative_evidence must be "
            "an object or null."
        )

    negative_required = (
        flood_label
        in negative_config.get(
            "required_for_labels",
            [],
        )
    )

    negative_prohibited = (
        flood_label
        in negative_config.get(
            "prohibited_for_labels",
            [],
        )
    )

    explicit_negative_rule = (
        contract[
            "rules"
        ].get(
            "no_flood_requires_explicit_negative_evidence"
        )
        is True
    )

    if (
        negative_required
        and explicit_negative_rule
    ):
        if not isinstance(
            negative_evidence,
            dict,
        ):
            raise ValueError(
                "NO_FLOOD requires explicit "
                "negative_evidence."
            )

        if (
            negative_evidence.get(
                "verified"
            )
            is not True
        ):
            raise ValueError(
                "NO_FLOOD requires "
                "negative_evidence.verified=true."
            )

        verification_note = (
            negative_evidence.get(
                "verification_note"
            )
        )

        if (
            not isinstance(
                verification_note,
                str,
            )
            or not verification_note.strip()
        ):
            raise ValueError(
                "NO_FLOOD requires a non-empty "
                "negative evidence verification note."
            )

    if negative_prohibited:
        if isinstance(
            negative_evidence,
            dict,
        ):
            verified = (
                negative_evidence.get(
                    "verified"
                )
            )

            verification_note = (
                negative_evidence.get(
                    "verification_note"
                )
            )

            if (
                verified is True
                or (
                    isinstance(
                        verification_note,
                        str,
                    )
                    and verification_note.strip()
                )
            ):
                raise ValueError(
                    "FLOOD must not carry verified "
                    "negative evidence."
                )

    return {
        "gold_record_id":
            gold_record_id,

        "silver_record_id":
            silver_record_id,

        "evidence_id":
            evidence_id,

        "event_type":
            event_type,

        "h3_index":
            h3_index,

        "flood_label":
            flood_label,

        "traceability_ok":
            True,

        "temporal_ok":
            True,

        "spatial_ok":
            True,

        "measurement_ok":
            True,

        "label_ok":
            True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "record_path",
        type=Path,
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
            args.record_path.read_text(
                encoding="utf-8"
            )
        )

        metrics = validate_gold_record(
            record,
            contract,
        )

        print(
            "GOLD_VALIDATION_OK=True"
        )

        print(
            json.dumps(
                metrics,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    except Exception as exc:
        print(
            "GOLD_VALIDATION_OK=False"
        )

        print(
            "ERROR_TYPE="
            + type(exc).__name__
        )

        print(
            "ERROR="
            + str(exc)
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
