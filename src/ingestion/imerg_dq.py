from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BRONZE_ROOT = (
    PROJECT_ROOT / "data" / "bronze" / "imerg" / "raw"
)

IMERG_GRANULE_DURATION_SECONDS = 1800
IMERG_GRANULES_PER_DAY = 48


@dataclass(frozen=True)
class DayQualityResult:
    date: str
    file_count: int
    unique_windows: int
    first_start: datetime
    last_end: datetime
    all_duration_1800: bool
    gap_count: int
    overlap_count: int
    all_gaps_zero: bool
    full_day_coverage: bool


def parse_utc_date(value: str) -> datetime:
    """
    Parse YYYY-MM-DD thành thời điểm 00:00:00 UTC.
    """
    return datetime.strptime(
        value,
        "%Y-%m-%d",
    ).replace(tzinfo=timezone.utc)


def bronze_day_directory(
    day: datetime,
    bronze_root: Path = DEFAULT_BRONZE_ROOT,
) -> Path:
    """
    Trả về partition Bronze tương ứng với một ngày UTC.
    """
    return (
        bronze_root
        / f"year={day.year:04d}"
        / f"month={day.month:02d}"
        / f"day={day.day:02d}"
    )


def read_time_window(path: Path) -> tuple[datetime, datetime]:
    """
    Đọc time_bnds của một granule IMERG.
    """
    epoch = datetime(
        1980,
        1,
        6,
        tzinfo=timezone.utc,
    )

    with h5py.File(path, "r") as file:
        if "Grid/time_bnds" not in file:
            raise ValueError(
                f"Thiếu Grid/time_bnds: {path}"
            )

        bounds = file["Grid/time_bnds"][0]

        start = epoch + timedelta(
            seconds=int(bounds[0])
        )

        end = epoch + timedelta(
            seconds=int(bounds[1])
        )

    return start, end


def validate_bronze_day(
    date: str,
    bronze_root: Path = DEFAULT_BRONZE_ROOT,
) -> DayQualityResult:
    """
    Kiểm tra tính đầy đủ và continuity của một ngày IMERG Bronze.
    """
    expected_start = parse_utc_date(date)
    expected_end = expected_start + timedelta(days=1)

    day_dir = bronze_day_directory(
        expected_start,
        bronze_root=bronze_root,
    )

    if not day_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy Bronze partition: {day_dir}"
        )

    files = sorted(day_dir.glob("*.HDF5"))

    if not files:
        raise ValueError(
            f"Không có HDF5 granule trong: {day_dir}"
        )

    windows: list[tuple[str, datetime, datetime]] = []

    for path in files:
        start, end = read_time_window(path)

        windows.append(
            (
                path.name,
                start,
                end,
            )
        )

    windows.sort(key=lambda item: item[1])

    durations = [
        int((end - start).total_seconds())
        for _, start, end in windows
    ]

    gaps = [
        int(
            (
                windows[index + 1][1]
                - windows[index][2]
            ).total_seconds()
        )
        for index in range(len(windows) - 1)
    ]

    unique_windows = {
        (start, end)
        for _, start, end in windows
    }

    all_duration_1800 = all(
        value == IMERG_GRANULE_DURATION_SECONDS
        for value in durations
    )

    gap_count = sum(
        value > 0
        for value in gaps
    )

    overlap_count = sum(
        value < 0
        for value in gaps
    )

    all_gaps_zero = all(
        value == 0
        for value in gaps
    )

    full_day_coverage = (
        len(windows) == IMERG_GRANULES_PER_DAY
        and len(unique_windows) == IMERG_GRANULES_PER_DAY
        and windows[0][1] == expected_start
        and windows[-1][2] == expected_end
        and all_duration_1800
        and all_gaps_zero
    )

    return DayQualityResult(
        date=date,
        file_count=len(windows),
        unique_windows=len(unique_windows),
        first_start=windows[0][1],
        last_end=windows[-1][2],
        all_duration_1800=all_duration_1800,
        gap_count=gap_count,
        overlap_count=overlap_count,
        all_gaps_zero=all_gaps_zero,
        full_day_coverage=full_day_coverage,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate completeness and temporal continuity "
            "of one IMERG Bronze day."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        help="UTC date ở dạng YYYY-MM-DD",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = validate_bronze_day(
        date=args.date,
    )

    print(f"DATE              = {result.date}")
    print(f"FILE_COUNT        = {result.file_count}")
    print(f"UNIQUE_WINDOWS    = {result.unique_windows}")
    print(
        f"FIRST_START       = "
        f"{result.first_start.isoformat()}"
    )
    print(
        f"LAST_END          = "
        f"{result.last_end.isoformat()}"
    )
    print(
        f"ALL_DURATION_1800 = "
        f"{result.all_duration_1800}"
    )
    print(f"GAP_COUNT         = {result.gap_count}")
    print(f"OVERLAP_COUNT     = {result.overlap_count}")
    print(
        f"ALL_GAPS_ZERO     = "
        f"{result.all_gaps_zero}"
    )
    print(
        f"FULL_DAY_COVERAGE = "
        f"{result.full_day_coverage}"
    )

    if not result.full_day_coverage:
        raise SystemExit(1)


if __name__ == "__main__":
    main()