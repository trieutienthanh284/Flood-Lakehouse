from __future__ import annotations

import argparse
import hashlib
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
    / "flood_evidence_manifest.yaml"
)


def load_contract(
    path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest contract does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        contract = yaml.safe_load(
            file
        )

    if not isinstance(
        contract,
        dict,
    ):
        raise ValueError(
            "Manifest contract must be a YAML mapping."
        )

    return contract


def load_manifest(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(
            file
        )

    if not isinstance(
        manifest,
        dict,
    ):
        raise ValueError(
            "Manifest must be a JSON object."
        )

    return manifest


def calculate_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def validate_timezone_aware(
    value: str,
) -> bool:
    try:
        parsed = datetime.fromisoformat(
            value
        )
    except ValueError:
        return False

    return (
        parsed.tzinfo
        is not None
        and parsed.utcoffset()
        is not None
    )


def validate_artifact_manifest(
    artifact_path: Path,
    manifest_path: Path,
    contract: dict[str, Any],
    expected_raw_artifact_path: Path | None = None,
) -> dict[str, Any]:
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Artifact does not exist: {artifact_path}"
        )

    manifest = load_manifest(
        manifest_path
    )

    required_fields = {
        "evidence_id",
        "evidence_type",
        "source_name",
        "source_uri",
        "ingestion_time",
        "raw_artifact_path",
        "sha256",
        "media_type",
        "file_bytes",
    }

    missing_fields = sorted(
        required_fields
        - set(
            manifest
        )
    )

    if missing_fields:
        raise ValueError(
            "Missing required manifest fields: "
            f"{missing_fields}"
        )

    evidence_id = manifest[
        "evidence_id"
    ]

    if not isinstance(
        evidence_id,
        str,
    ):
        raise ValueError(
            "evidence_id must be a string."
        )

    if artifact_path.stem != evidence_id:
        raise ValueError(
            "Artifact filename does not match evidence_id."
        )

    if manifest_path.stem != evidence_id:
        raise ValueError(
            "Manifest filename does not match evidence_id."
        )

    allowed_types = set(
        contract[
            "identity"
        ][
            "evidence_type"
        ][
            "allowed_values"
        ]
    )

    if (
        manifest["evidence_type"]
        not in allowed_types
    ):
        raise ValueError(
            "Unsupported evidence_type."
        )

    raw_artifact_path = Path(
        manifest[
            "raw_artifact_path"
        ]
    )

    if expected_raw_artifact_path is None:
        expected_path = (
            artifact_path
        )
    else:
        expected_path = (
            expected_raw_artifact_path
        )

    if not raw_artifact_path.is_absolute():
        raw_artifact_path = (
                PROJECT_ROOT
                / raw_artifact_path
        )

    if (
            raw_artifact_path.resolve()
            != expected_path.resolve()
    ):
        raise ValueError(
            "Manifest raw_artifact_path "
            "does not match expected "
            "canonical artifact path."
        )

    actual_file_bytes = (
        artifact_path.stat().st_size
    )

    if (
        manifest["file_bytes"]
        != actual_file_bytes
    ):
        raise ValueError(
            "Artifact file size does not "
            "match manifest file_bytes."
        )

    expected_sha256 = manifest[
        "sha256"
    ]

    if not isinstance(
        expected_sha256,
        str,
    ):
        raise ValueError(
            "sha256 must be a string."
        )

    if len(
        expected_sha256
    ) != 64:
        raise ValueError(
            "sha256 must contain 64 hexadecimal characters."
        )

    try:
        int(
            expected_sha256,
            16,
        )
    except ValueError as exc:
        raise ValueError(
            "sha256 is not valid hexadecimal."
        ) from exc

    actual_sha256 = calculate_sha256(
        artifact_path
    )

    if (
        actual_sha256
        != expected_sha256
    ):
        raise ValueError(
            "Artifact SHA-256 does not match manifest."
        )

    if not validate_timezone_aware(
        manifest[
            "ingestion_time"
        ]
    ):
        raise ValueError(
            "ingestion_time must be timezone-aware."
        )

    forbidden_fields = {
        "h3",
        "h3_index",
        "flood_label",
        "label",
        "geocoding_result",
        "text_extraction_result",
    }

    forbidden_present = sorted(
        forbidden_fields
        & set(
            manifest
        )
    )

    if forbidden_present:
        raise ValueError(
            "Forbidden Bronze manifest fields: "
            f"{forbidden_present}"
        )

    return {
        "evidence_id": evidence_id,
        "evidence_type": manifest[
            "evidence_type"
        ],
        "artifact_path": artifact_path,
        "manifest_path": manifest_path,
        "file_bytes": actual_file_bytes,
        "sha256": actual_sha256,
        "physical_ok": True,
        "semantic_ok": True,
        "integrity_ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one historical flood "
            "evidence artifact and manifest."
        )
    )

    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
    )

    args = parser.parse_args()

    contract = load_contract(
        args.contract
    )

    metrics = (
        validate_artifact_manifest(
            artifact_path=(
                args.artifact
            ),
            manifest_path=(
                args.manifest
            ),
            contract=contract,
        )
    )

    print(
        "VALIDATION_OK=True"
    )

    print(
        "EVIDENCE_ID="
        f"{metrics['evidence_id']}"
    )

    print(
        "EVIDENCE_TYPE="
        f"{metrics['evidence_type']}"
    )

    print(
        "FILE_BYTES="
        f"{metrics['file_bytes']}"
    )

    print(
        "SHA256="
        f"{metrics['sha256']}"
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
        "INTEGRITY_OK="
        f"{metrics['integrity_ok']}"
    )


if __name__ == "__main__":
    main()