from __future__ import annotations

import argparse
import hashlib
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
import shutil
import yaml
import json
import os

try:
    from .flood_evidence_validator import (
        load_contract,
        validate_artifact_manifest,
    )
except ImportError:
    from flood_evidence_validator import (
        load_contract,
        validate_artifact_manifest,
    )

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "flood_evidence.yaml"
)


def load_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Flood evidence config does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "Flood evidence config must be a YAML mapping."
        )

    return config


def calculate_sha256(
    path: Path,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"Artifact does not exist: {path}"
        )

    if path.stat().st_size <= 0:
        raise ValueError(
            "Artifact must not be empty."
        )

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


def validate_evidence_type(
    evidence_type: str,
    config: dict[str, Any],
) -> str:
    normalized = (
        evidence_type
        .strip()
        .lower()
    )

    allowed_types = set(
        config[
            "evidence"
        ][
            "supported_types"
        ]
    )

    if normalized not in allowed_types:
        raise ValueError(
            "Unsupported evidence_type: "
            f"{evidence_type}. "
            f"Allowed: {sorted(allowed_types)}"
        )

    return normalized


def build_evidence_id(
    artifact_path: Path,
    evidence_type: str,
    config: dict[str, Any],
) -> tuple[str, str]:
    normalized_type = (
        validate_evidence_type(
            evidence_type=evidence_type,
            config=config,
        )
    )

    sha256 = calculate_sha256(
        artifact_path
    )

    id_config = (
        config[
            "ingestion"
        ][
            "evidence_id"
        ]
    )

    prefix = id_config[
        "prefix"
    ]

    hash_length = id_config[
        "sha256_prefix_length"
    ]

    if not isinstance(
        hash_length,
        int,
    ):
        raise ValueError(
            "sha256_prefix_length must be an integer."
        )

    if (
        hash_length <= 0
        or hash_length > 64
    ):
        raise ValueError(
            "sha256_prefix_length must be between 1 and 64."
        )

    sha_prefix = sha256[
        :hash_length
    ]

    evidence_id = (
        f"{prefix}_"
        f"{normalized_type}_"
        f"{sha_prefix}"
    )

    return (
        evidence_id,
        sha256,
    )

def validate_ingestion_date(
    ingestion_date: str,
) -> str:
    try:
        parsed = date.fromisoformat(
            ingestion_date
        )
    except ValueError as exc:
        raise ValueError(
            "ingestion_date must use "
            "ISO format YYYY-MM-DD."
        ) from exc

    normalized = parsed.isoformat()

    if normalized != ingestion_date:
        raise ValueError(
            "ingestion_date must be "
            "canonical YYYY-MM-DD."
        )

    return normalized


def artifact_suffix(
    artifact_path: Path,
) -> str:
    suffix = artifact_path.suffix.lower()

    if not suffix:
        raise ValueError(
            "Artifact must have a file suffix."
        )

    return suffix

def detect_artifact_format(
    artifact_path: Path,
) -> tuple[str, str]:
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Artifact not found: {artifact_path}"
        )

    with artifact_path.open("rb") as f:
        header = f.read(4096)

    if not header:
        raise ValueError(
            "Artifact is empty."
        )

    # PDF
    if header.startswith(b"%PDF-"):
        return (
            "application/pdf",
            ".pdf",
        )

    # JPEG
    if header.startswith(
        b"\xff\xd8\xff"
    ):
        return (
            "image/jpeg",
            ".jpg",
        )

    # PNG
    if header.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        return (
            "image/png",
            ".png",
        )

    # WebP
    if (
        len(header) >= 12
        and header[:4] == b"RIFF"
        and header[8:12] == b"WEBP"
    ):
        return (
            "image/webp",
            ".webp",
        )

    # AVIF / AVIS
    if (
        len(header) >= 16
        and header[4:8] == b"ftyp"
        and (
            b"avif" in header[8:32]
            or b"avis" in header[8:32]
        )
    ):
        return (
            "image/avif",
            ".avif",
        )

    # GIF
    if (
        header.startswith(b"GIF87a")
        or header.startswith(b"GIF89a")
    ):
        return (
            "image/gif",
            ".gif",
        )

    # TIFF little endian
    if header.startswith(
        b"II*\x00"
    ):
        return (
            "image/tiff",
            ".tif",
        )

    # TIFF big endian
    if header.startswith(
        b"MM\x00*"
    ):
        return (
            "image/tiff",
            ".tif",
        )

    # HTML / article
    html_probe = (
        header
        .lstrip(b"\xef\xbb\xbf \t\r\n")
        .lower()
    )

    if (
        b"<!doctype html" in html_probe[:1024]
        or b"<html" in html_probe[:1024]
        or b"<head" in html_probe[:1024]
        or b"<meta" in html_probe[:1024]
    ):
        return (
            "text/html",
            ".html",
        )

    raise ValueError(
        "Unsupported or unrecognized "
        "artifact format."
    )


def resolve_artifact_format(
    artifact_path: Path,
    evidence_type: str,
    declared_media_type: str | None,
) -> tuple[str, str]:
    detected_media_type, detected_suffix = (
        detect_artifact_format(
            artifact_path
        )
    )

    if evidence_type == "pdf":
        if (
            detected_media_type
            != "application/pdf"
        ):
            raise ValueError(
                "evidence_type='pdf' does not "
                "match detected artifact format "
                f"'{detected_media_type}'."
            )

    elif evidence_type == "image":
        if not detected_media_type.startswith(
            "image/"
        ):
            raise ValueError(
                "evidence_type='image' does not "
                "match detected artifact format "
                f"'{detected_media_type}'."
            )

    elif evidence_type == "article":
        if detected_media_type != "text/html":
            raise ValueError(
                "evidence_type='article' does not "
                "match detected artifact format "
                f"'{detected_media_type}'."
            )

    else:
        raise ValueError(
            "Unsupported evidence_type: "
            f"{evidence_type}"
        )

    if declared_media_type is not None:
        normalized_declared = (
            declared_media_type
            .strip()
            .lower()
        )

        if (
            normalized_declared
            != detected_media_type
        ):
            raise ValueError(
                "Declared media_type "
                f"'{normalized_declared}' "
                "does not match detected "
                f"media_type "
                f"'{detected_media_type}'."
            )

    return (
        detected_media_type,
        detected_suffix,
    )

def relative_template_to_path(
    value: str,
) -> Path:
    pure_path = PurePosixPath(
        value
    )

    if pure_path.is_absolute():
        raise ValueError(
            "Bronze template must produce "
            "a relative path."
        )

    if ".." in pure_path.parts:
        raise ValueError(
            "Bronze template must not "
            "contain parent traversal."
        )

    return Path(
        *pure_path.parts
    )


def staging_paths(
    evidence_id: str,
    suffix: str,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    staging_root = (
        PROJECT_ROOT
        / config[
            "ingestion"
        ][
            "staging"
        ][
            "directory"
        ]
    )

    evidence_staging_dir = (
        staging_root
        / evidence_id
    )

    artifact_path = (
        evidence_staging_dir
        / f"{evidence_id}{suffix}"
    )

    manifest_path = (
        evidence_staging_dir
        / f"{evidence_id}.json"
    )

    return (
        artifact_path,
        manifest_path,
    )


def bronze_raw_path(
    evidence_id: str,
    evidence_type: str,
    ingestion_date: str,
    suffix: str,
    config: dict[str, Any],
) -> Path:
    normalized_type = (
        validate_evidence_type(
            evidence_type=(
                evidence_type
            ),
            config=config,
        )
    )

    normalized_date = (
        validate_ingestion_date(
            ingestion_date
        )
    )

    template = (
        config[
            "bronze_layout"
        ][
            "raw_path_template"
        ]
    )

    relative_text = template.format(
        evidence_type=normalized_type,
        ingestion_date=normalized_date,
        evidence_id=evidence_id,
        suffix=suffix,
    )

    relative_path = (
        relative_template_to_path(
            relative_text
        )
    )

    return (
        PROJECT_ROOT
        / config[
            "bronze"
        ][
            "raw_dir"
        ]
        / relative_path
    )


def bronze_manifest_path(
    evidence_id: str,
    evidence_type: str,
    ingestion_date: str,
    config: dict[str, Any],
) -> Path:
    normalized_type = (
        validate_evidence_type(
            evidence_type=(
                evidence_type
            ),
            config=config,
        )
    )

    normalized_date = (
        validate_ingestion_date(
            ingestion_date
        )
    )

    template = (
        config[
            "bronze_layout"
        ][
            "manifest_path_template"
        ]
    )

    relative_text = template.format(
        evidence_type=normalized_type,
        ingestion_date=normalized_date,
        evidence_id=evidence_id,
    )

    relative_path = (
        relative_template_to_path(
            relative_text
        )
    )

    return (
        PROJECT_ROOT
        / config[
            "bronze"
        ][
            "manifest_dir"
        ]
        / relative_path
    )


def project_relative_posix(
    path: Path,
) -> str:
    resolved = path.resolve()

    try:
        relative = resolved.relative_to(
            PROJECT_ROOT.resolve()
        )
    except ValueError as exc:
        raise ValueError(
            "Path must be inside "
            "PROJECT_ROOT."
        ) from exc

    return relative.as_posix()

def copy_artifact_to_staging(
    source_path: Path,
    staging_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source artifact does not exist: {source_path}"
        )

    source_bytes = (
        source_path.stat().st_size
    )

    if source_bytes <= 0:
        raise ValueError(
            "Source artifact must not be empty."
        )

    source_sha256 = calculate_sha256(
        source_path
    )

    if source_sha256 != expected_sha256:
        raise ValueError(
            "Source SHA-256 does not match "
            "expected SHA-256."
        )

    staging_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if staging_path.exists():
        staging_path.unlink()

    shutil.copy2(
        source_path,
        staging_path,
    )

    if not staging_path.is_file():
        raise RuntimeError(
            "Staging artifact was not created."
        )

    staging_bytes = (
        staging_path.stat().st_size
    )

    staging_sha256 = calculate_sha256(
        staging_path
    )

    byte_size_match = (
        source_bytes
        == staging_bytes
    )

    sha256_match = (
        source_sha256
        == staging_sha256
    )

    if not byte_size_match:
        raise ValueError(
            "Staging artifact byte size "
            "does not match source."
        )

    if not sha256_match:
        raise ValueError(
            "Staging artifact SHA-256 "
            "does not match source."
        )

    return {
        "source_path": source_path,
        "staging_path": staging_path,
        "source_bytes": source_bytes,
        "staging_bytes": staging_bytes,
        "source_sha256": source_sha256,
        "staging_sha256": staging_sha256,
        "byte_size_match": (
            byte_size_match
        ),
        "sha256_match": (
            sha256_match
        ),
    }

def create_staging_manifest(
    staging_manifest_path: Path,
    bronze_artifact_path: Path,
    evidence_id: str,
    evidence_type: str,
    sha256: str,
    file_bytes: int,
    source_name: str,
    source_uri: str,
    media_type: str,
    ingestion_time: datetime,
    title: str | None = None,
    author: str | None = None,
    published_time: str | None = None,
    observed_time: str | None = None,
    location_text: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if (
        ingestion_time.tzinfo is None
        or ingestion_time.utcoffset() is None
    ):
        raise ValueError(
            "ingestion_time must be timezone-aware."
        )

    if len(sha256) != 64:
        raise ValueError(
            "sha256 must contain 64 characters."
        )

    if file_bytes <= 0:
        raise ValueError(
            "file_bytes must be greater than zero."
        )

    manifest = {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "source_name": source_name,
        "source_uri": source_uri,
        "title": title,
        "author": author,
        "published_time": published_time,
        "observed_time": observed_time,
        "ingestion_time": (
            ingestion_time.isoformat()
        ),
        "location_text": location_text,
        "raw_artifact_path": (
            project_relative_posix(
                bronze_artifact_path
            )
        ),
        "sha256": sha256,
        "media_type": media_type,
        "file_bytes": file_bytes,
        "notes": notes,
    }

    staging_manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if not staging_manifest_path.is_file():
        raise RuntimeError(
            "Staging manifest was not created."
        )

    return manifest

def promote_staging_pair(
    evidence_id: str,
    evidence_type: str,
    ingestion_date: str,
    suffix: str,
    config: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    staging_artifact, staging_manifest = (
        staging_paths(
            evidence_id=evidence_id,
            suffix=suffix,
            config=config,
        )
    )

    bronze_artifact = bronze_raw_path(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        ingestion_date=ingestion_date,
        suffix=suffix,
        config=config,
    )

    bronze_manifest = (
        bronze_manifest_path(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            ingestion_date=ingestion_date,
            config=config,
        )
    )

    if not staging_artifact.is_file():
        raise FileNotFoundError(
            "Staging artifact does not exist: "
            f"{staging_artifact}"
        )

    if not staging_manifest.is_file():
        raise FileNotFoundError(
            "Staging manifest does not exist: "
            f"{staging_manifest}"
        )

    pre_metrics = (
        validate_artifact_manifest(
            artifact_path=(
                staging_artifact
            ),
            manifest_path=(
                staging_manifest
            ),
            contract=contract,
            expected_raw_artifact_path=(
                bronze_artifact
            ),
        )
    )

    if bronze_artifact.exists():
        raise FileExistsError(
            "Bronze artifact already exists: "
            f"{bronze_artifact}"
        )

    if bronze_manifest.exists():
        raise FileExistsError(
            "Bronze manifest already exists: "
            f"{bronze_manifest}"
        )

    bronze_artifact.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bronze_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = (
        staging_artifact,
        staging_manifest,
        bronze_artifact,
        bronze_manifest,
    )

    anchors = {
        path.resolve().anchor.lower()
        for path in paths
    }

    if len(anchors) != 1:
        raise ValueError(
            "Staging and Bronze must be "
            "on the same filesystem volume "
            "for atomic file replacement."
        )

    artifact_moved = False
    manifest_moved = False

    try:
        os.replace(
            staging_artifact,
            bronze_artifact,
        )

        artifact_moved = True

        os.replace(
            staging_manifest,
            bronze_manifest,
        )

        manifest_moved = True

        post_metrics = (
            validate_artifact_manifest(
                artifact_path=(
                    bronze_artifact
                ),
                manifest_path=(
                    bronze_manifest
                ),
                contract=contract,
            )
        )

    except Exception:
        staging_artifact.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            manifest_moved
            and bronze_manifest.exists()
        ):
            os.replace(
                bronze_manifest,
                staging_manifest,
            )

        if (
            artifact_moved
            and bronze_artifact.exists()
        ):
            os.replace(
                bronze_artifact,
                staging_artifact,
            )

        raise

    staging_dir = (
        staging_artifact.parent
    )

    if (
        staging_dir.is_dir()
        and not any(
            staging_dir.iterdir()
        )
    ):
        staging_dir.rmdir()

    return {
        "evidence_id": evidence_id,
        "bronze_artifact": (
            bronze_artifact
        ),
        "bronze_manifest": (
            bronze_manifest
        ),
        "file_bytes": post_metrics[
            "file_bytes"
        ],
        "sha256": post_metrics[
            "sha256"
        ],
        "physical_ok": post_metrics[
            "physical_ok"
        ],
        "semantic_ok": post_metrics[
            "semantic_ok"
        ],
        "integrity_ok": post_metrics[
            "integrity_ok"
        ],
        "pre_validation_ok": (
            pre_metrics[
                "physical_ok"
            ]
            and pre_metrics[
                "semantic_ok"
            ]
            and pre_metrics[
                "integrity_ok"
            ]
        ),
    }

def get_valid_existing_bronze_pair(
    evidence_id: str,
    evidence_type: str,
    suffix: str,
    config: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    normalized_type = (
        validate_evidence_type(
            evidence_type=evidence_type,
            config=config,
        )
    )

    raw_root = (
        PROJECT_ROOT
        / config[
            "bronze"
        ][
            "raw_dir"
        ]
    )

    manifest_root = (
        PROJECT_ROOT
        / config[
            "bronze"
        ][
            "manifest_dir"
        ]
    )

    raw_matches = []

    if raw_root.is_dir():
        raw_matches = sorted(
            path
            for path in raw_root.rglob(
                f"{evidence_id}{suffix}"
            )
            if path.is_file()
        )

    manifest_matches = []

    if manifest_root.is_dir():
        manifest_matches = sorted(
            path
            for path in manifest_root.rglob(
                f"{evidence_id}.json"
            )
            if path.is_file()
        )

    if (
        not raw_matches
        and not manifest_matches
    ):
        return None

    if len(raw_matches) != 1:
        raise RuntimeError(
            "Expected exactly one Bronze "
            "artifact for existing evidence_id, "
            f"found {len(raw_matches)}."
        )

    if len(manifest_matches) != 1:
        raise RuntimeError(
            "Expected exactly one Bronze "
            "manifest for existing evidence_id, "
            f"found {len(manifest_matches)}."
        )

    bronze_artifact = (
        raw_matches[0]
    )

    bronze_manifest = (
        manifest_matches[0]
    )

    metrics = (
        validate_artifact_manifest(
            artifact_path=(
                bronze_artifact
            ),
            manifest_path=(
                bronze_manifest
            ),
            contract=contract,
        )
    )

    if (
        metrics["evidence_id"]
        != evidence_id
    ):
        raise ValueError(
            "Existing Bronze evidence_id "
            "does not match expected evidence_id."
        )

    if (
        metrics["evidence_type"]
        != normalized_type
    ):
        raise ValueError(
            "Existing Bronze evidence_type "
            "does not match expected type."
        )

    staging_artifact, staging_manifest = (
        staging_paths(
            evidence_id=evidence_id,
            suffix=suffix,
            config=config,
        )
    )

    if staging_artifact.exists():
        staging_artifact.unlink()

    if staging_manifest.exists():
        staging_manifest.unlink()

    staging_dir = (
        staging_artifact.parent
    )

    if (
        staging_dir.is_dir()
        and not any(
            staging_dir.iterdir()
        )
    ):
        staging_dir.rmdir()

    return {
        "action": "SKIP_EXISTING",
        "evidence_id": evidence_id,
        "evidence_type": (
            normalized_type
        ),
        "bronze_artifact": (
            bronze_artifact
        ),
        "bronze_manifest": (
            bronze_manifest
        ),
        "file_bytes": metrics[
            "file_bytes"
        ],
        "sha256": metrics[
            "sha256"
        ],
        "physical_ok": metrics[
            "physical_ok"
        ],
        "semantic_ok": metrics[
            "semantic_ok"
        ],
        "integrity_ok": metrics[
            "integrity_ok"
        ],
    }

def ingest_local_artifact(
    source_path: Path,
    evidence_type: str,
    source_name: str,
    source_uri: str,
    media_type: str,
    ingestion_time: datetime,
    config: dict[str, Any],
    contract: dict[str, Any],
    title: str | None = None,
    author: str | None = None,
    published_time: str | None = None,
    observed_time: str | None = None,
    location_text: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if (
        ingestion_time.tzinfo is None
        or ingestion_time.utcoffset() is None
    ):
        raise ValueError(
            "ingestion_time must be timezone-aware."
        )

    normalized_type = (
        validate_evidence_type(
            evidence_type=evidence_type,
            config=config,
        )
    )

    evidence_id, sha256 = (
        build_evidence_id(
            artifact_path=source_path,
            evidence_type=normalized_type,
            config=config,
        )
    )

    resolved_media_type, suffix = (
        resolve_artifact_format(
            artifact_path=source_path,
            evidence_type=normalized_type,
            declared_media_type=media_type,
        )
    )

    ingestion_date = (
        ingestion_time
        .date()
        .isoformat()
    )

    existing = (
        get_valid_existing_bronze_pair(
            evidence_id=evidence_id,
            evidence_type=normalized_type,
            suffix=suffix,
            config=config,
            contract=contract,
        )
    )

    if existing is not None:
        return existing

    staging_artifact, staging_manifest = (
        staging_paths(
            evidence_id=evidence_id,
            suffix=suffix,
            config=config,
        )
    )

    bronze_artifact = bronze_raw_path(
        evidence_id=evidence_id,
        evidence_type=normalized_type,
        ingestion_date=ingestion_date,
        suffix=suffix,
        config=config,
    )

    copy_metrics = (
        copy_artifact_to_staging(
            source_path=source_path,
            staging_path=staging_artifact,
            expected_sha256=sha256,
        )
    )

    try:
        create_staging_manifest(
            staging_manifest_path=(
                staging_manifest
            ),
            bronze_artifact_path=(
                bronze_artifact
            ),
            evidence_id=evidence_id,
            evidence_type=normalized_type,
            sha256=sha256,
            file_bytes=(
                copy_metrics[
                    "staging_bytes"
                ]
            ),
            source_name=source_name,
            source_uri=source_uri,
            media_type=resolved_media_type,
            ingestion_time=ingestion_time,
            title=title,
            author=author,
            published_time=published_time,
            observed_time=observed_time,
            location_text=location_text,
            notes=notes,
        )

        promoted = (
            promote_staging_pair(
                evidence_id=evidence_id,
                evidence_type=normalized_type,
                ingestion_date=ingestion_date,
                suffix=suffix,
                config=config,
                contract=contract,
            )
        )

    except Exception:
        if staging_artifact.exists():
            staging_artifact.unlink()

        if staging_manifest.exists():
            staging_manifest.unlink()

        staging_dir = (
            staging_artifact.parent
        )

        if (
            staging_dir.is_dir()
            and not any(
                staging_dir.iterdir()
            )
        ):
            staging_dir.rmdir()

        raise

    return {
        "action": "INGESTED",
        "evidence_id": evidence_id,
        "evidence_type": normalized_type,
        "bronze_artifact": promoted[
            "bronze_artifact"
        ],
        "bronze_manifest": promoted[
            "bronze_manifest"
        ],
        "file_bytes": promoted[
            "file_bytes"
        ],
        "sha256": promoted[
            "sha256"
        ],
        "physical_ok": promoted[
            "physical_ok"
        ],
        "semantic_ok": promoted[
            "semantic_ok"
        ],
        "integrity_ok": promoted[
            "integrity_ok"
        ],
    }

def parse_cli_ingestion_time(
    value: str | None,
) -> datetime:
    if value is None:
        return datetime.now(
            timezone.utc
        )

    normalized = (
        value.strip()
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
            "ingestion_time must use "
            "ISO-8601 datetime format."
        ) from exc

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        raise ValueError(
            "ingestion_time must be timezone-aware."
        )

    return parsed

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled Bronze ingestion "
            "for historical flood evidence."
        )
    )

    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help=(
            "Local article, PDF, or image "
            "artifact to ingest."
        ),
    )

    parser.add_argument(
        "--evidence-type",
        required=True,
        help=(
            "Evidence type: "
            "article, pdf, or image."
        ),
    )

    parser.add_argument(
        "--source-name",
        required=True,
        help=(
            "Human-readable source name."
        ),
    )

    parser.add_argument(
        "--source-uri",
        required=True,
        help=(
            "Original URL or source URI."
        ),
    )

    parser.add_argument(
        "--media-type",
        default=None,
        help=(
            "Optional declared media type. "
            "If omitted, the collector detects "
            "the actual format from file content. "
            "If provided, it must match the "
            "detected format."
        ),
    )

    parser.add_argument(
        "--ingestion-time",
        default=None,
        help=(
            "Optional timezone-aware "
            "ISO-8601 ingestion datetime. "
            "Defaults to current UTC time."
        ),
    )

    parser.add_argument(
        "--title",
        default=None,
    )

    parser.add_argument(
        "--author",
        default=None,
    )

    parser.add_argument(
        "--published-time",
        default=None,
    )

    parser.add_argument(
        "--observed-time",
        default=None,
    )

    parser.add_argument(
        "--location-text",
        default=None,
    )

    parser.add_argument(
        "--notes",
        default=None,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Optional collector config override."
        ),
    )

    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help=(
            "Optional manifest contract override."
        ),
    )

    args = parser.parse_args()

    try:
        config = load_config(
            args.config
        )

        if args.contract is None:
            contract = load_contract()
        else:
            contract = load_contract(
                args.contract
            )

        ingestion_time = (
            parse_cli_ingestion_time(
                args.ingestion_time
            )
        )

        source_path = (
            args.artifact.resolve()
        )

        metrics = ingest_local_artifact(
            source_path=source_path,
            evidence_type=(
                args.evidence_type
            ),
            source_name=(
                args.source_name
            ),
            source_uri=(
                args.source_uri
            ),
            media_type=(
                args.media_type
            ),
            ingestion_time=(
                ingestion_time
            ),
            config=config,
            contract=contract,
            title=args.title,
            author=args.author,
            published_time=(
                args.published_time
            ),
            observed_time=(
                args.observed_time
            ),
            location_text=(
                args.location_text
            ),
            notes=args.notes,
        )

        print(
            "INGESTION_OK=True"
        )

        print(
            "ACTION="
            f"{metrics['action']}"
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
            "BRONZE_ARTIFACT="
            f"{project_relative_posix(metrics['bronze_artifact'])}"
        )

        print(
            "BRONZE_MANIFEST="
            f"{project_relative_posix(metrics['bronze_manifest'])}"
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

    except Exception as exc:
        print(
            "INGESTION_OK=False"
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