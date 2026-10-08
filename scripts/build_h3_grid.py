from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import h3
from shapely.geometry import Point, Polygon


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "hanoi_core_9_legacy_districts.geojson"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "hanoi_core_h3_res9.geojson"
)

H3_RESOLUTION = 9
PARENT_RESOLUTION = 8

# Kết quả đã được kiểm chứng ở bước thực nghiệm trước.
EXPECTED_CELL_COUNT = 1481

# Tên trong source geoBoundaries -> tên chuẩn tiếng Việt dùng trong project.
DISTRICT_NAME_MAP = {
    "Ba Dinh": "Ba Đình",
    "Cau Giay": "Cầu Giấy",
    "Dong Da": "Đống Đa",
    "Ha Dong": "Hà Đông",
    "Hai Ba Trung": "Hai Bà Trưng",
    "Hoan Kiem": "Hoàn Kiếm",
    "Hoang Mai": "Hoàng Mai",
    "Tay Ho": "Tây Hồ",
    "Thanh Xuan": "Thanh Xuân",
}


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Loading and validation
# ============================================================

def load_district_boundaries(path: Path) -> gpd.GeoDataFrame:
    """
    Đọc boundary của 9 quận cũ và kiểm tra tính hợp lệ trước khi tạo H3.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy input boundary: {path}"
        )

    gdf = gpd.read_file(path)

    if len(gdf) != 9:
        raise ValueError(
            f"Expected 9 district polygons, nhưng nhận được {len(gdf)}."
        )

    if "shapeName" not in gdf.columns:
        raise ValueError(
            "Input boundary không có cột 'shapeName'."
        )

    if gdf.crs is None:
        raise ValueError(
            "Input boundary không có CRS."
        )

    # H3 làm việc với latitude/longitude WGS84.
    if gdf.crs.to_epsg() != 4326:
        logger.info(
            "Chuyển CRS từ %s sang EPSG:4326.",
            gdf.crs,
        )
        gdf = gdf.to_crs(epsg=4326)

    invalid_count = int((~gdf.geometry.is_valid).sum())

    if invalid_count > 0:
        raise ValueError(
            f"Phát hiện {invalid_count} geometry không hợp lệ."
        )

    actual_names = set(gdf["shapeName"])
    expected_names = set(DISTRICT_NAME_MAP)

    missing_names = expected_names - actual_names
    unexpected_names = actual_names - expected_names

    if missing_names:
        raise ValueError(
            f"Thiếu district trong source: {sorted(missing_names)}"
        )

    if unexpected_names:
        raise ValueError(
            f"Có district ngoài phạm vi project: "
            f"{sorted(unexpected_names)}"
        )

    logger.info(
        "Đã đọc thành công %d district polygons.",
        len(gdf),
    )

    return gdf


# ============================================================
# Spatial helper functions
# ============================================================

def find_legacy_district(
    point: Point,
    districts: gpd.GeoDataFrame,
) -> str:
    """
    Xác định quận cũ chứa tâm của một H3 cell.

    Dùng covers() thay vì contains() để không loại bỏ trường hợp
    điểm nằm chính xác trên boundary.
    """

    matches: list[str] = []

    for row in districts.itertuples(index=False):
        if row.geometry.covers(point):
            matches.append(row.shapeName)

    if len(matches) != 1:
        raise ValueError(
            "Không xác định duy nhất legacy district cho point "
            f"{point.wkt}. Matches={matches}"
        )

    source_name = matches[0]

    return DISTRICT_NAME_MAP[source_name]


def h3_cell_to_polygon(cell: str) -> Polygon:
    """
    Chuyển H3 cell thành Shapely Polygon.

    H3 trả boundary theo thứ tự:
        (latitude, longitude)

    Shapely cần:
        (x, y) = (longitude, latitude)
    """

    boundary = h3.cell_to_boundary(cell)

    lon_lat_coordinates = [
        (longitude, latitude)
        for latitude, longitude in boundary
    ]

    return Polygon(lon_lat_coordinates)


# ============================================================
# H3 grid builder
# ============================================================

def build_h3_grid(
    districts: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Tạo H3 resolution 9 cho toàn bộ AOI.

    AOI được tạo bằng union của đúng 9 district polygons.
    """

    aoi_geometry = districts.geometry.union_all()

    cells = sorted(
        h3.geo_to_cells(
            aoi_geometry,
            H3_RESOLUTION,
        )
    )

    logger.info(
        "AOI tạo ra %d H3 resolution %d cells.",
        len(cells),
        H3_RESOLUTION,
    )

    records: list[dict] = []

    for cell in cells:
        centroid_lat, centroid_lon = h3.cell_to_latlng(cell)

        centroid_point = Point(
            centroid_lon,
            centroid_lat,
        )

        legacy_district = find_legacy_district(
            centroid_point,
            districts,
        )

        parent_res8 = h3.cell_to_parent(
            cell,
            PARENT_RESOLUTION,
        )

        geometry = h3_cell_to_polygon(cell)

        records.append(
            {
                "h3_id": cell,
                "h3_resolution": H3_RESOLUTION,
                "h3_parent_res8": parent_res8,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "legacy_district": legacy_district,
                "geometry": geometry,
            }
        )

    grid = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326",
    )

    return grid


# ============================================================
# Data quality checks
# ============================================================

def validate_h3_grid(
    grid: gpd.GeoDataFrame,
) -> None:
    """
    Kiểm tra các invariant quan trọng của spatial grid.
    """

    if len(grid) != EXPECTED_CELL_COUNT:
        raise ValueError(
            "H3 cell count không đúng. "
            f"Expected={EXPECTED_CELL_COUNT}, actual={len(grid)}"
        )

    if grid["h3_id"].isna().any():
        raise ValueError(
            "Phát hiện h3_id null."
        )

    if not grid["h3_id"].is_unique:
        duplicates = grid.loc[
            grid["h3_id"].duplicated(keep=False),
            "h3_id",
        ].tolist()

        raise ValueError(
            f"Phát hiện duplicate H3 IDs: {duplicates}"
        )

    resolutions = set(grid["h3_resolution"])

    if resolutions != {H3_RESOLUTION}:
        raise ValueError(
            f"H3 resolution không nhất quán: {resolutions}"
        )

    if grid["h3_parent_res8"].isna().any():
        raise ValueError(
            "Phát hiện h3_parent_res8 null."
        )

    parent_resolutions = {
        h3.get_resolution(parent)
        for parent in grid["h3_parent_res8"]
    }

    if parent_resolutions != {PARENT_RESOLUTION}:
        raise ValueError(
            "Parent H3 resolution không đúng: "
            f"{parent_resolutions}"
        )

    if grid["legacy_district"].isna().any():
        raise ValueError(
            "Phát hiện legacy_district null."
        )

    expected_districts = set(DISTRICT_NAME_MAP.values())
    actual_districts = set(grid["legacy_district"])

    if actual_districts != expected_districts:
        raise ValueError(
            "Danh sách legacy districts không đúng. "
            f"Expected={sorted(expected_districts)}, "
            f"Actual={sorted(actual_districts)}"
        )

    invalid_geometry_count = int(
        (~grid.geometry.is_valid).sum()
    )

    if invalid_geometry_count > 0:
        raise ValueError(
            f"Có {invalid_geometry_count} H3 geometry không hợp lệ."
        )

    if grid.crs is None or grid.crs.to_epsg() != 4326:
        raise ValueError(
            f"Output CRS không đúng: {grid.crs}"
        )

    logger.info("Data quality checks: PASSED.")


# ============================================================
# Output
# ============================================================

def save_grid(
    grid: gpd.GeoDataFrame,
    path: Path,
) -> None:
    """
    Ghi spatial reference grid ra GeoJSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid.to_file(
        path,
        driver="GeoJSON",
    )

    logger.info(
        "Đã ghi H3 grid: %s",
        path,
    )


def log_summary(
    grid: gpd.GeoDataFrame,
) -> None:
    """
    In summary sau khi build thành công.
    """

    logger.info(
        "Tổng số H3 cells: %d",
        len(grid),
    )

    logger.info(
        "H3 resolution: %d",
        H3_RESOLUTION,
    )

    logger.info(
        "Số parent res8 duy nhất: %d",
        grid["h3_parent_res8"].nunique(),
    )

    logger.info(
        "Phân bố cell theo legacy district:"
    )

    district_counts = (
        grid.groupby("legacy_district")
        .size()
        .sort_index()
    )

    for district, count in district_counts.items():
        logger.info(
            "  %-15s %d cells",
            district,
            count,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    logger.info(
        "Bắt đầu build official H3 spatial grid."
    )

    districts = load_district_boundaries(
        INPUT_PATH
    )

    grid = build_h3_grid(
        districts
    )

    validate_h3_grid(
        grid
    )

    save_grid(
        grid,
        OUTPUT_PATH,
    )

    log_summary(
        grid
    )

    logger.info(
        "Hoàn thành build H3 spatial grid."
    )


if __name__ == "__main__":
    main()