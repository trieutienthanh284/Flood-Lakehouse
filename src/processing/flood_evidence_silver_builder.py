from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import os

try:
    from .flood_evidence_silver_validator import (
        PROJECT_ROOT,
        load_contract,
        validate_silver_record,
    )
except ImportError:
    from flood_evidence_silver_validator import (
        PROJECT_ROOT,
        load_contract,
        validate_silver_record,
    )


def normalize_identity_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        str,
    ):
        return value.strip()

    return value


def canonical_identity_payload(
    record: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        contract[
            "record_identity"
        ][
            "record_id"
        ][
            "canonical_fields"
        ]
    )

    return {
        field: normalize_identity_value(
            record.get(field)
        )
        for field in fields
    }


def build_record_id(
    record: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[str, str]:
    identity_config = (
        contract[
            "record_identity"
        ][
            "record_id"
        ]
    )

    prefix = (
        identity_config[
            "prefix"
        ]
    )

    prefix_length = int(
        identity_config[
            "sha256_prefix_length"
        ]
    )

    payload = (
        canonical_identity_payload(
            record=record,
            contract=contract,
        )
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

    record_id = (
        f"{prefix}_"
        f"{digest[:prefix_length]}"
    )

    return (
        record_id,
        digest,
    )


def build_record_path(
    record_id: str,
    evidence_id: str,
    contract: dict[str, Any],
) -> Path:
    if not record_id.startswith(
        "fer_"
    ):
        raise ValueError(
            "record_id must start "
            "with 'fer_'."
        )

    if not evidence_id.startswith(
        "fe_"
    ):
        raise ValueError(
            "evidence_id must start "
            "with 'fe_'."
        )

    storage = contract[
        "storage"
    ]

    records_dir = Path(
        storage[
            "records_dir"
        ]
    )

    if not records_dir.is_absolute():
        records_dir = (
            PROJECT_ROOT
            / records_dir
        )

    template = (
        storage[
            "layout"
        ][
            "record_path_template"
        ]
    )

    relative_path = PurePosixPath(
        template.format(
            evidence_id=evidence_id,
            record_id=record_id,
        )
    )

    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise ValueError(
            "Silver record path must "
            "remain inside records_dir."
        )

    return (
        records_dir
        / Path(
            *relative_path.parts
        )
    )

def find_bronze_manifest(
    evidence_id: str,
) -> Path:
    manifest_root = (
        PROJECT_ROOT
        / "data"
        / "bronze"
        / "flood_evidence"
        / "manifest"
    )

    if not manifest_root.is_dir():
        raise FileNotFoundError(
            "Bronze flood evidence manifest "
            "directory does not exist."
        )

    matches = sorted(
        path
        for path in manifest_root.rglob(
            f"{evidence_id}.json"
        )
        if path.is_file()
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            "No Bronze manifest found for "
            f"evidence_id={evidence_id}."
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple Bronze manifests found "
            f"for evidence_id={evidence_id}."
        )

    return matches[0]

def write_silver_record(
    record: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    candidate = dict(
        record
    )

    evidence_id = candidate.get(
        "evidence_id"
    )

    if not isinstance(
        evidence_id,
        str,
    ) or not evidence_id.strip():
        raise ValueError(
            "evidence_id is required."
        )

    evidence_id = evidence_id.strip()

    bronze_manifest = (
        find_bronze_manifest(
            evidence_id
        )
    )

    expected_record_id, identity_sha256 = (
        build_record_id(
            record=candidate,
            contract=contract,
        )
    )

    supplied_record_id = candidate.get(
        "record_id"
    )

    if (
        supplied_record_id is not None
        and supplied_record_id
        != expected_record_id
    ):
        raise ValueError(
            "Supplied record_id does not "
            "match deterministic record_id."
        )

    candidate[
        "record_id"
    ] = expected_record_id

    validate_silver_record(
        record=candidate,
        contract=contract,
    )

    target_path = (
        build_record_path(
            record_id=expected_record_id,
            evidence_id=evidence_id,
            contract=contract,
        )
    )

    if target_path.exists():
        existing = json.loads(
            target_path.read_text(
                encoding="utf-8"
            )
        )

        validate_silver_record(
            record=existing,
            contract=contract,
        )

        existing_record_id, existing_digest = (
            build_record_id(
                record=existing,
                contract=contract,
            )
        )

        if (
            existing_record_id
            != expected_record_id
            or existing_digest
            != identity_sha256
        ):
            raise RuntimeError(
                "Silver record_id collision "
                "or corrupted existing record."
            )

        idempotency_config = (
            contract["storage"]["idempotency"]
        )

        conflict_policy = (
            idempotency_config.get(
                "semantic_conflict_policy"
            )
        )

        conflict_fields = (
            idempotency_config.get(
                "semantic_conflict_fields",
                [],
            )
        )

        semantic_conflicts = {}

        for field in conflict_fields:
            existing_value = existing.get(
                field
            )

            candidate_value = candidate.get(
                field
            )

            if existing_value != candidate_value:
                semantic_conflicts[field] = {
                    "existing":
                        existing_value,
                    "incoming":
                        candidate_value,
                }

        if semantic_conflicts:
            if conflict_policy == "error":
                raise RuntimeError(
                    "Silver semantic conflict "
                    "for existing record: "
                    + json.dumps(
                        semantic_conflicts,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )

            raise RuntimeError(
                "Unsupported semantic conflict "
                "policy: "
                + repr(conflict_policy)
            )

        return {
            "action": "SKIP_EXISTING",
            "record_id": (
                expected_record_id
            ),
            "evidence_id": evidence_id,
            "record_path": target_path,
            "identity_sha256": (
                identity_sha256
            ),
            "bronze_manifest": (
                bronze_manifest
            ),
            "validation_ok": True,
        }

    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        target_path.with_suffix(
            target_path.suffix
            + ".tmp"
        )
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

        validate_silver_record(
            record=written,
            contract=contract,
        )

        written_record_id, written_digest = (
            build_record_id(
                record=written,
                contract=contract,
            )
        )

        if (
            written_record_id
            != expected_record_id
            or written_digest
            != identity_sha256
        ):
            raise ValueError(
                "Written Silver record "
                "identity verification failed."
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

    validate_silver_record(
        record=final_record,
        contract=contract,
    )

    return {
        "action": "WRITTEN",
        "record_id": expected_record_id,
        "evidence_id": evidence_id,
        "record_path": target_path,
        "identity_sha256": identity_sha256,
        "bronze_manifest": bronze_manifest,
        "validation_ok": True,
    }