from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyogrio
import yaml


DEFAULT_CONFIG_PATH = Path(
    "configs/osm_spatial_reference_cache.yaml"
)


def load_config(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cache config does not exist: {path}"
        )

    config = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "Cache config must be a YAML object."
        )

    return config


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest().upper()


def validate_lineage_hashes(
    config: dict[str, Any],
    source_sha256: str,
    h3_reference_sha256: str,
) -> None:
    lineage_config = config[
        "lineage"
    ]

    if (
        lineage_config.get(
            "require_live_hash_verification"
        )
        is not True
    ):
        return


    source_config = config[
        "source"
    ]

    scope_config = config[
        "scope"
    ]


    expected_source_sha256 = (
        source_config.get(
            "expected_sha256"
        )
    )

    expected_h3_sha256 = (
        scope_config.get(
            "expected_sha256"
        )
    )


    if not isinstance(
        expected_source_sha256,
        str,
    ):
        raise ValueError(
            "source.expected_sha256 "
            "is required when live hash "
            "verification is enabled."
        )


    if not isinstance(
        expected_h3_sha256,
        str,
    ):
        raise ValueError(
            "scope.expected_sha256 "
            "is required when live hash "
            "verification is enabled."
        )


    if (
        source_sha256.upper()
        != expected_source_sha256.upper()
    ):
        raise ValueError(
            "Live OSM source SHA-256 does not "
            "match source.expected_sha256."
        )


    if (
        h3_reference_sha256.upper()
        != expected_h3_sha256.upper()
    ):
        raise ValueError(
            "Live H3 reference SHA-256 does not "
            "match scope.expected_sha256."
        )


def derive_cache_paths(
    config: dict[str, Any],
) -> tuple[Path, Path]:
    cache_config = config[
        "cache"
    ]

    output_dir = (
        Path(
            cache_config[
                "root"
            ]
        )
        / cache_config[
            "snapshot_partition"
        ]
    )

    cache_path = (
        output_dir
        / cache_config[
            "file_name"
        ]
    )

    manifest_path = (
        output_dir
        / config[
            "lineage"
        ][
            "manifest_file_name"
        ]
    )

    return (
        cache_path,
        manifest_path,
    )


def build_cache(
    config: dict[str, Any],
) -> dict[str, Any]:
    source_config = config[
        "source"
    ]

    scope_config = config[
        "scope"
    ]

    cache_config = config[
        "cache"
    ]

    lineage_config = config[
        "lineage"
    ]


    source_path = Path(
        source_config[
            "path"
        ]
    )

    h3_reference_path = Path(
        scope_config[
            "h3_reference_path"
        ]
    )


    for required_path in [
        source_path,
        h3_reference_path,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required input does not exist: "
                f"{required_path}"
            )


    hash_algorithm = lineage_config[
        "hash_algorithm"
    ]

    if hash_algorithm != "sha256":
        raise ValueError(
            "Only sha256 lineage hashing "
            "is supported."
        )


    source_sha256 = sha256_file(
        source_path
    )

    h3_reference_sha256 = sha256_file(
        h3_reference_path
    )


    validate_lineage_hashes(
        config,
        source_sha256,
        h3_reference_sha256,
    )


    h3_frame = gpd.read_file(
        h3_reference_path
    )

    if h3_frame.crs is None:
        raise ValueError(
            "H3 reference CRS is missing."
        )


    expected_h3_resolution = int(
        scope_config[
            "h3_resolution"
        ]
    )


    h3_resolution_values = set(
        int(value)
        for value in h3_frame[
            "h3_resolution"
        ].dropna()
    )


    if h3_resolution_values != {
        expected_h3_resolution
    }:
        raise ValueError(
            "H3 reference resolution does not "
            "match cache config."
        )


    h3_frame = h3_frame.to_crs(
        source_config[
            "source_crs"
        ]
    )


    study_bbox = tuple(
        float(value)
        for value in h3_frame.total_bounds
    )


    retained_fields = list(
        cache_config[
            "retained_fields"
        ]
    )


    frame = pyogrio.read_dataframe(
        source_path,
        layer=source_config[
            "layer"
        ],
        bbox=study_bbox,
        columns=retained_fields,
    )


    id_field = source_config[
        "id_field"
    ]


    if id_field not in frame.columns:
        raise ValueError(
            f"OSM identifier field is missing: "
            f"{id_field}"
        )


    frame = frame[
        frame[
            id_field
        ].notna()
    ].copy()


    frame[
        id_field
    ] = frame[
        id_field
    ].astype(str)


    require_unique = (
        cache_config.get(
            "require_unique_feature_id"
        )
        is True
    )


    duplicate_count = int(
        frame[
            id_field
        ].duplicated(
            keep=False
        ).sum()
    )


    if (
        require_unique
        and duplicate_count != 0
    ):
        raise ValueError(
            "OSM cache contains duplicate "
            f"{id_field} values."
        )


    expected_crs = cache_config[
        "crs"
    ]


    if frame.crs is None:
        raise ValueError(
            "Extracted OSM cache CRS is missing."
        )


    if (
        frame.crs.to_string()
        != expected_crs
    ):
        frame = frame.to_crs(
            expected_crs
        )


    cache_path, manifest_path = (
        derive_cache_paths(
            config
        )
    )


    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temporary_cache_path = (
        cache_path.with_name(
            cache_path.stem
            + ".tmp"
            + cache_path.suffix
        )
    )


    temporary_manifest_path = (
        manifest_path.with_name(
            manifest_path.name
            + ".tmp"
        )
    )


    for temporary_path in [
        temporary_cache_path,
        temporary_manifest_path,
    ]:
        if temporary_path.exists():
            temporary_path.unlink()


    pyogrio.write_dataframe(
        frame,
        temporary_cache_path,
        layer=cache_config[
            "layer"
        ],
        driver=cache_config[
            "driver"
        ],
    )


    cache_sha256 = sha256_file(
        temporary_cache_path
    )


    manifest = {
        "manifest_version":
            "1.0",

        "cache_name":
            config[
                "cache_name"
            ],

        "purpose":
            config[
                "purpose"
            ],

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "source": {
            "provider":
                source_config[
                    "provider"
                ],

            "format":
                source_config[
                    "format"
                ],

            "snapshot_date":
                source_config[
                    "snapshot_date"
                ],

            "path":
                source_path.as_posix(),

            "layer":
                source_config[
                    "layer"
                ],

            "feature_type":
                source_config[
                    "feature_type"
                ],

            "id_field":
                id_field,

            "source_crs":
                source_config[
                    "source_crs"
                ],

            "size_bytes":
                source_path.stat().st_size,

            "sha256":
                source_sha256,
        },

        "scope": {
            "source":
                scope_config[
                    "source"
                ],

            "h3_reference_path":
                h3_reference_path.as_posix(),

            "h3_resolution":
                expected_h3_resolution,

            "h3_reference_count":
                len(h3_frame),

            "h3_reference_size_bytes":
                h3_reference_path.stat().st_size,

            "h3_reference_sha256":
                h3_reference_sha256,

            "study_bbox":
                list(
                    study_bbox
                ),
        },

        "cache": {
            "path":
                cache_path.as_posix(),

            "layer":
                cache_config[
                    "layer"
                ],

            "driver":
                cache_config[
                    "driver"
                ],

            "crs":
                expected_crs,

            "feature_count":
                len(frame),

            "duplicate_feature_id_row_count":
                duplicate_count,

            "size_bytes":
                temporary_cache_path.stat().st_size,

            "sha256":
                cache_sha256,
        },
    }


    temporary_manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    os.replace(
        temporary_cache_path,
        cache_path,
    )

    os.replace(
        temporary_manifest_path,
        manifest_path,
    )


    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )

    args = parser.parse_args()


    config = load_config(
        args.config
    )

    manifest = build_cache(
        config
    )


    print(
        "OSM_SPATIAL_REFERENCE_CACHE_BUILD_OK=True"
    )

    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
