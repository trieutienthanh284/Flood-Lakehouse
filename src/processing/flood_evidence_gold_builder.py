from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.processing.flood_evidence_gold_validator import (
    PROJECT_ROOT,
    load_contract,
    resolve_project_path,
    validate_gold_record,
)


def normalize_identity_value(
    value: Any,
) -> Any:
    if isinstance(value, str):
        return value.strip()

    return value


def canonical_identity_payload(
    record: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    fields = contract[
        "record_identity"
    ][
        "gold_record_id"
    ][
        "canonical_fields"
    ]

    payload = {}

    for field in fields:
        payload[field] = (
            normalize_identity_value(
                record.get(field)
            )
        )

    return payload


def build_gold_record_id(
    record: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[str, str]:
    identity_config = contract[
        "record_identity"
    ][
        "gold_record_id"
    ]

    payload = canonical_identity_payload(
        record,
        contract,
    )

    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        canonical_json.encode(
            "utf-8"
        )
    ).hexdigest()

    prefix = identity_config[
        "prefix"
    ]

    prefix_length = int(
        identity_config[
            "sha256_prefix_length"
        ]
    )

    record_id = (
        f"{prefix}_"
        f"{digest[:prefix_length]}"
    )

    return record_id, digest


def validate_path_component(
    value: str,
    field: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a string."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field} must not be empty."
        )

    if (
        "/" in value
        or "\\" in value
        or value in {".", ".."}
        or ".." in value
    ):
        raise ValueError(
            f"{field} contains unsafe "
            "path characters."
        )

    return value


def build_gold_record_path(
    record: dict[str, Any],
    contract: dict[str, Any],
) -> Path:
    h3_index = validate_path_component(
        record.get("h3_index"),
        "h3_index",
    )

    gold_record_id = validate_path_component(
        record.get("gold_record_id"),
        "gold_record_id",
    )

    records_dir = resolve_project_path(
        contract[
            "storage"
        ][
            "records_dir"
        ]
    )

    return (
        records_dir
        / f"h3_index={h3_index}"
        / f"{gold_record_id}.json"
    )


def write_gold_record(
    record: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    candidate = dict(record)

    expected_record_id, identity_sha256 = (
        build_gold_record_id(
            candidate,
            contract,
        )
    )

    supplied_record_id = candidate.get(
        "gold_record_id"
    )

    if (
        supplied_record_id is not None
        and supplied_record_id
        != expected_record_id
    ):
        raise ValueError(
            "Supplied gold_record_id does not "
            "match deterministic identity."
        )

    candidate[
        "gold_record_id"
    ] = expected_record_id

    validation_metrics = (
        validate_gold_record(
            candidate,
            contract,
        )
    )

    target_path = build_gold_record_path(
        candidate,
        contract,
    )

    idempotency = contract[
        "storage"
    ][
        "idempotency"
    ]

    if target_path.exists():
        existing = json.loads(
            target_path.read_text(
                encoding="utf-8"
            )
        )

        validate_gold_record(
            existing,
            contract,
        )

        existing_record_id, existing_digest = (
            build_gold_record_id(
                existing,
                contract,
            )
        )

        if (
            existing_record_id
            != expected_record_id
            or existing_digest
            != identity_sha256
        ):
            raise RuntimeError(
                "Gold record_id collision "
                "or corrupted existing record."
            )

        conflict_fields = (
            idempotency.get(
                "semantic_conflict_fields",
                [],
            )
        )

        semantic_conflicts = {}

        for field in conflict_fields:
            existing_value = existing.get(
                field
            )

            incoming_value = candidate.get(
                field
            )

            if existing_value != incoming_value:
                semantic_conflicts[field] = {
                    "existing":
                        existing_value,

                    "incoming":
                        incoming_value,
                }

        if semantic_conflicts:
            conflict_policy = (
                idempotency.get(
                    "semantic_conflict_policy"
                )
            )

            if conflict_policy == "error":
                raise RuntimeError(
                    "Gold semantic conflict "
                    "for existing record: "
                    + json.dumps(
                        semantic_conflicts,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )

            raise RuntimeError(
                "Unsupported Gold semantic "
                "conflict policy: "
                + repr(conflict_policy)
            )

        return {
            "action":
                "SKIP_EXISTING",

            "gold_record_id":
                expected_record_id,

            "record_path":
                target_path,

            "validation_ok":
                True,
        }

    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = target_path.with_suffix(
        target_path.suffix + ".tmp"
    )

    try:
        temp_path.write_text(
            json.dumps(
                candidate,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        written = json.loads(
            temp_path.read_text(
                encoding="utf-8"
            )
        )

        validate_gold_record(
            written,
            contract,
        )

        written_record_id, written_digest = (
            build_gold_record_id(
                written,
                contract,
            )
        )

        if (
            written_record_id
            != expected_record_id
            or written_digest
            != identity_sha256
        ):
            raise RuntimeError(
                "Written Gold record identity "
                "verification failed."
            )

        os.replace(
            temp_path,
            target_path,
        )

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        raise

    final_record = json.loads(
        target_path.read_text(
            encoding="utf-8"
        )
    )

    validate_gold_record(
        final_record,
        contract,
    )

    final_record_id, final_digest = (
        build_gold_record_id(
            final_record,
            contract,
        )
    )

    if (
        final_record_id
        != expected_record_id
        or final_digest
        != identity_sha256
    ):
        raise RuntimeError(
            "Promoted Gold record identity "
            "verification failed."
        )

    return {
        "action":
            "WRITTEN",

        "gold_record_id":
            expected_record_id,

        "record_path":
            target_path,

        "validation_ok":
            all(
                [
                    validation_metrics[
                        "traceability_ok"
                    ],
                    validation_metrics[
                        "temporal_ok"
                    ],
                    validation_metrics[
                        "spatial_ok"
                    ],
                    validation_metrics[
                        "measurement_ok"
                    ],
                    validation_metrics[
                        "label_ok"
                    ],
                ]
            ),
    }


if __name__ == "__main__":
    contract = load_contract()

    print(
        "GOLD_BUILDER_LOAD_OK=True"
    )
