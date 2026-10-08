from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime, timezone, timedelta

import time
import h5py
import requests
from dotenv import load_dotenv

MAX_DOWNLOAD_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2

RETRYABLE_STATUS_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

IMERG_SHORT_NAME = "GPM_3IMERGHH"
IMERG_VERSION = "07"

DATA_REL = "http://esipfed.org/ns/fedsearch/1.1/data#"

REQUEST_TIMEOUT_SECONDS = 30
DOWNLOAD_TIMEOUT_SECONDS = 120

DEFAULT_STAGING_DIR = (
    PROJECT_ROOT / "data" / "bronze" / "imerg" / "staging"
)

DEFAULT_BRONZE_RAW_DIR = (
    PROJECT_ROOT / "data" / "bronze" / "imerg" / "raw"
)

HDF5_MAGIC_SIGNATURE = b"\x89HDF\r\n\x1a\n"

@dataclass(frozen=True)
class GranuleMetadata:
    title: str
    time_start: str
    time_end: str
    data_url: str

@dataclass(frozen=True)
class DownloadResult:
    staging_path: Path
    http_status: int
    redirects: int
    bytes_written: int
    hdf5_signature_valid: bool
    semantic_valid: bool
    attempts: int

@dataclass(frozen=True)
class SemanticValidationResult:
    grid_exists: bool
    precipitation_exists: bool
    precipitation_shape: tuple[int, ...]
    precipitation_units: str
    window_start: datetime
    window_end: datetime
    duration_seconds: int

@dataclass(frozen=True)
class PromotionResult:
    bronze_path: Path
    promoted: bool
    bytes_written: int

def parse_cmr_utc(value: str) -> datetime:
    """
    Parse ISO-8601 UTC timestamp do NASA CMR trả về.
    """
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).astimezone(timezone.utc)

def bronze_path_for_granule(
    granule: GranuleMetadata,
    bronze_root: Path = DEFAULT_BRONZE_RAW_DIR,
) -> Path:
    """
    Tạo Bronze path phân vùng theo ngày UTC của granule.
    """
    start_time = parse_cmr_utc(granule.time_start)

    partition_dir = (
        bronze_root
        / f"year={start_time.year:04d}"
        / f"month={start_time.month:02d}"
        / f"day={start_time.day:02d}"
    )

    return partition_dir / granule_filename(granule)

def get_earthdata_token() -> str:
    """
    Đọc EARTHDATA_TOKEN từ file .env tại project root.

    Token chỉ được giữ trong memory và tuyệt đối không được log ra console.
    """
    load_dotenv(dotenv_path=ENV_FILE)

    token = os.getenv("EARTHDATA_TOKEN")

    if token is None or not token.strip():
        raise RuntimeError(
            f"Không tìm thấy EARTHDATA_TOKEN hợp lệ trong {ENV_FILE}"
        )

    return token.strip()


def find_data_url(links: list[dict[str, Any]]) -> str:
    """
    Tìm direct HDF5 data URL trong danh sách link do NASA CMR trả về.

    Không sử dụng OPeNDAP ở bước này.
    """
    for link in links:
        rel = link.get("rel")
        href = link.get("href", "")

        if rel == DATA_REL and href.lower().endswith(".hdf5"):
            return href

    raise ValueError(
        "Không tìm thấy direct HDF5 data URL trong CMR metadata."
    )


def discover_granules(
    start_time: str,
    end_time: str,
    page_size: int = 100,
) -> list[GranuleMetadata]:
    """
    Truy vấn NASA CMR để tìm các granule IMERG V07
    giao với khoảng thời gian yêu cầu.
    """
    params = {
        "short_name": IMERG_SHORT_NAME,
        "version": IMERG_VERSION,
        "temporal": f"{start_time},{end_time}",
        "page_size": page_size,
    }

    response = requests.get(
        CMR_GRANULES_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    payload = response.json()
    entries = payload.get("feed", {}).get("entry", [])

    granules: list[GranuleMetadata] = []

    for entry in entries:
        granules.append(
            GranuleMetadata(
                title=entry["title"],
                time_start=entry["time_start"],
                time_end=entry["time_end"],
                data_url=find_data_url(entry.get("links", [])),
            )
        )

    granules.sort(key=lambda item: item.time_start)

    return granules


def granule_filename(granule: GranuleMetadata) -> str:
    """
    Chuyển CMR title thành tên file HDF5 gốc.

    Ví dụ:
    GPM_3IMERGHH.07:3B-HHR....HDF5
    ->
    3B-HHR....HDF5
    """
    prefix = f"{IMERG_SHORT_NAME}.{IMERG_VERSION}:"

    if not granule.title.startswith(prefix):
        raise ValueError(
            f"Granule title không có prefix mong đợi: {granule.title}"
        )

    return granule.title.removeprefix(prefix)

def has_valid_hdf5_signature(path: Path) -> bool:
    """
    Kiểm tra 8-byte magic signature của HDF5.

    Validation này chỉ xác nhận định dạng container HDF5,
    chưa xác nhận schema IMERG hay temporal metadata.
    """
    with path.open("rb") as file:
        signature = file.read(8)

    return signature == HDF5_MAGIC_SIGNATURE

def validate_imerg_semantics(
    path: Path,
    granule: GranuleMetadata,
) -> SemanticValidationResult:
    """
    Xác nhận file staging là đúng IMERG V07 half-hourly granule.

    Validation bao gồm:
    - /Grid
    - /Grid/precipitation
    - precipitation shape
    - precipitation units
    - time_bnds
    - duration 30 phút
    - window_start khớp CMR metadata
    """
    epoch = datetime(
        1980,
        1,
        6,
        tzinfo=timezone.utc,
    )

    with h5py.File(path, "r") as file:
        grid_exists = "Grid" in file

        if not grid_exists:
            raise ValueError(
                f"Thiếu /Grid trong granule: {granule.title}"
            )

        precipitation_exists = "Grid/precipitation" in file

        if not precipitation_exists:
            raise ValueError(
                f"Thiếu /Grid/precipitation: {granule.title}"
            )

        precipitation = file["Grid/precipitation"]

        precipitation_shape = tuple(precipitation.shape)

        if precipitation_shape != (1, 3600, 1800):
            raise ValueError(
                "Unexpected precipitation shape "
                f"{precipitation_shape}: {granule.title}"
            )

        raw_units = precipitation.attrs.get("units")

        if isinstance(raw_units, bytes):
            precipitation_units = raw_units.decode("utf-8")
        else:
            precipitation_units = str(raw_units)

        if precipitation_units != "mm/hr":
            raise ValueError(
                "Unexpected precipitation units "
                f"{precipitation_units!r}: {granule.title}"
            )

        if "Grid/time_bnds" not in file:
            raise ValueError(
                f"Thiếu /Grid/time_bnds: {granule.title}"
            )

        bounds = file["Grid/time_bnds"][0]

        start_seconds = int(bounds[0])
        end_seconds = int(bounds[1])

        duration_seconds = end_seconds - start_seconds

        window_start = epoch + timedelta(
            seconds=start_seconds
        )
        window_end = epoch + timedelta(
            seconds=end_seconds
        )

    if duration_seconds != 1800:
        raise ValueError(
            "IMERG granule không có duration 1800 giây: "
            f"{granule.title}"
        )

    expected_start = parse_cmr_utc(granule.time_start)

    if window_start != expected_start:
        raise ValueError(
            "time_bnds start không khớp CMR metadata: "
            f"{granule.title}"
        )

    return SemanticValidationResult(
        grid_exists=grid_exists,
        precipitation_exists=precipitation_exists,
        precipitation_shape=precipitation_shape,
        precipitation_units=precipitation_units,
        window_start=window_start,
        window_end=window_end,
        duration_seconds=duration_seconds,
    )

def promote_to_bronze(
    granule: GranuleMetadata,
    staging_path: Path,
    bronze_root: Path = DEFAULT_BRONZE_RAW_DIR,
) -> PromotionResult:
    """
    Atomically promote một file .part đã validate sang Bronze chính thức.

    Hàm này chỉ được gọi sau:
    - HTTP validation
    - HDF5 signature validation
    - semantic validation
    """
    if not staging_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy staging file: {staging_path}"
        )

    bronze_path = bronze_path_for_granule(
        granule=granule,
        bronze_root=bronze_root,
    )

    bronze_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if staging_path.drive.lower() != bronze_path.drive.lower():
        raise RuntimeError(
            "Staging và Bronze không cùng filesystem; "
            "không thể đảm bảo atomic promotion."
        )

    os.replace(
        staging_path,
        bronze_path,
    )

    if not bronze_path.exists():
        raise RuntimeError(
            f"Promotion thất bại: {bronze_path}"
        )

    if staging_path.exists():
        raise RuntimeError(
            f"Staging file vẫn còn sau promotion: {staging_path}"
        )

    return PromotionResult(
        bronze_path=bronze_path,
        promoted=True,
        bytes_written=bronze_path.stat().st_size,
    )

def get_valid_existing_bronze(
    granule: GranuleMetadata,
    bronze_root: Path = DEFAULT_BRONZE_RAW_DIR,
) -> Path | None:
    """
    Kiểm tra granule đã tồn tại trong Bronze hay chưa.

    Nếu chưa tồn tại:
        return None.

    Nếu đã tồn tại:
        - kiểm tra HDF5 signature;
        - kiểm tra IMERG semantic contract;
        - chỉ khi hợp lệ mới trả về path.

    Bronze tồn tại nhưng không hợp lệ sẽ raise error,
    không tự động ghi đè.
    """
    bronze_path = bronze_path_for_granule(
        granule=granule,
        bronze_root=bronze_root,
    )

    if not bronze_path.exists():
        return None

    if not has_valid_hdf5_signature(bronze_path):
        raise ValueError(
            f"Bronze file tồn tại nhưng HDF5 signature không hợp lệ: "
            f"{bronze_path}"
        )

    validate_imerg_semantics(
        path=bronze_path,
        granule=granule,
    )

    return bronze_path

def download_to_staging(
    granule: GranuleMetadata,
    earthdata_token: str,
    staging_dir: Path = DEFAULT_STAGING_DIR,
) -> DownloadResult:
    """
    Tải một granule vào file .part với bounded retry.

    Pipeline:
    - authenticated HTTP download
    - retry transient failures
    - HDF5 signature validation
    - IMERG semantic validation

    File chỉ nằm ở staging; hàm chưa thực hiện atomic promotion.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    filename = granule_filename(granule)
    staging_path = staging_dir / f"{filename}.part"

    headers = {
        "Authorization": f"Bearer {earthdata_token}",
    }

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        staging_path.unlink(missing_ok=True)

        try:
            with requests.get(
                granule.data_url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                response.raise_for_status()

                with staging_path.open("wb") as output:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            output.write(chunk)

                bytes_written = staging_path.stat().st_size

                if bytes_written == 0:
                    raise ValueError(
                        f"Download tạo file rỗng: {granule.title}"
                    )

                hdf5_signature_valid = (
                    has_valid_hdf5_signature(staging_path)
                )

                if not hdf5_signature_valid:
                    raise ValueError(
                        "File tải về không có HDF5 signature hợp lệ: "
                        f"{granule.title}"
                    )

                validate_imerg_semantics(
                    path=staging_path,
                    granule=granule,
                )

                return DownloadResult(
                    staging_path=staging_path,
                    http_status=response.status_code,
                    redirects=len(response.history),
                    bytes_written=bytes_written,
                    hdf5_signature_valid=True,
                    semantic_valid=True,
                    attempts=attempt,
                )

        except Exception as exc:
            staging_path.unlink(missing_ok=True)

            should_retry = is_retryable_download_error(exc)

            if not should_retry:
                raise

            if attempt >= MAX_DOWNLOAD_ATTEMPTS:
                raise

            wait_seconds = (
                RETRY_BACKOFF_SECONDS
                * (2 ** (attempt - 1))
            )

            print(
                f"DOWNLOAD_RETRY = attempt {attempt} failed; "
                f"waiting {wait_seconds}s"
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Download thất bại ngoài dự kiến: {granule.title}"
    )

def is_retryable_download_error(exc: Exception) -> bool:
    """
    Xác định lỗi download có nên retry hay không.

    Chỉ retry lỗi mạng và HTTP status có tính chất tạm thời.
    Không retry lỗi authentication, not-found hoặc validation.
    """
    if isinstance(
        exc,
        (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True

    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response

        if response is None:
            return False

        return response.status_code in RETRYABLE_STATUS_CODES

    return False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and stage NASA GPM IMERG V07 "
            "half-hourly granules."
        )
    )

    parser.add_argument(
        "--start",
        required=True,
        help="UTC start time, ví dụ: 2025-09-30T00:00:00Z",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="UTC end time, ví dụ: 2025-09-30T00:29:59Z",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Tải các granule được discover vào file staging .part.",
    )

    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Sau khi download và validation thành công, "
            "atomic promote file từ staging sang Bronze raw."
        ),
    )

    args = parser.parse_args()

    if args.promote and not args.download:
        parser.error("--promote yêu cầu phải có --download")

    return args


def main() -> None:
    args = parse_args()

    earthdata_token = get_earthdata_token()

    granules = discover_granules(
        start_time=args.start,
        end_time=args.end,
    )

    print(
        "EARTHDATA_TOKEN_LOADED =",
        bool(earthdata_token),
    )
    print(f"COLLECTION     = {IMERG_SHORT_NAME}")
    print(f"VERSION        = {IMERG_VERSION}")
    print(f"QUERY_START    = {args.start}")
    print(f"QUERY_END      = {args.end}")
    print(f"GRANULE_COUNT  = {len(granules)}")

    for index, granule in enumerate(granules, start=1):
        print(f"GRANULE_{index}_TITLE      = {granule.title}")
        print(f"GRANULE_{index}_START      = {granule.time_start}")
        print(f"GRANULE_{index}_END        = {granule.time_end}")
        print(
            f"GRANULE_{index}_DATA_FOUND = "
            f"{bool(granule.data_url)}"
        )

    if not args.download:
        return

    for index, granule in enumerate(granules, start=1):
        if args.promote:
            existing_bronze = get_valid_existing_bronze(
                granule=granule,
            )

            if existing_bronze is not None:
                stale_staging_path = (
                        DEFAULT_STAGING_DIR
                        / f"{granule_filename(granule)}.part"
                )

                stale_staging_path.unlink(
                    missing_ok=True,
                )

                print(
                    f"BRONZE_{index}_EXISTS = True"
                )
                print(
                    f"BRONZE_{index}_VALID  = True"
                )
                print(
                    f"BRONZE_{index}_ACTION = SKIP_EXISTING"
                )
                print(
                    f"BRONZE_{index}_PATH   = "
                    f"{existing_bronze.relative_to(PROJECT_ROOT)}"
                )

                continue

        result = download_to_staging(
            granule=granule,
            earthdata_token=earthdata_token,
        )

        result = download_to_staging(
            granule=granule,
            earthdata_token=earthdata_token,
        )

        print(f"DOWNLOAD_{index}_STATUS    = {result.http_status}")
        print(f"DOWNLOAD_{index}_REDIRECTS = {result.redirects}")
        print(f"DOWNLOAD_{index}_BYTES     = {result.bytes_written}")
        print(
            f"DOWNLOAD_{index}_PART      = "
            f"{result.staging_path.name}"
        )
        print(
            f"DOWNLOAD_{index}_HDF5_OK   = "
            f"{result.hdf5_signature_valid}"
        )
        print(
            f"DOWNLOAD_{index}_SEMANTIC_OK = "
            f"{result.semantic_valid}"
        )
        print(
            f"DOWNLOAD_{index}_ATTEMPTS  = "
            f"{result.attempts}"
        )

        if args.promote:
            promotion = promote_to_bronze(
                granule=granule,
                staging_path=result.staging_path,
            )

            print(
                f"PROMOTE_{index}_OK        = "
                f"{promotion.promoted}"
            )
            print(
                f"PROMOTE_{index}_BYTES     = "
                f"{promotion.bytes_written}"
            )
            print(
                f"PROMOTE_{index}_PATH      = "
                f"{promotion.bronze_path.relative_to(PROJECT_ROOT)}"
            )
            print(
                f"PROMOTE_{index}_PART_LEFT = "
                f"{result.staging_path.exists()}"
            )


if __name__ == "__main__":
    main()