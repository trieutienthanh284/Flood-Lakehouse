# Hanoi Flood Intelligence Lakehouse
## Source & License Matrix

**Phiên bản:** 1.0<br>
**Trạng thái:** Active<br>
**AOI:** 9 quận đô thị cũ của Hà Nội<br>
**Prediction horizons:** 30 / 60 phút

---

## 1. Nguyên tắc lựa chọn nguồn

Mỗi nguồn dữ liệu phải được đánh giá theo:

- vai trò trong hệ thống;
- historical hay realtime;
- spatial resolution;
- temporal resolution;
- latency;
- license / terms of use;
- yêu cầu tài khoản;
- khả năng tự động hóa;
- fallback;
- rủi ro phụ thuộc nguồn.

Không viết collector cho một nguồn trước khi các yếu tố trên được xác minh.

---

## 2. Source Matrix

| Source | Vai trò | Temporal | Spatial | Access / License | Trạng thái |
|---|---|---|---|---|---|
| NASA GPM IMERG | Historical rainfall, target 30m/60m, backfill | 30 phút | ~0.1° / ~10 km | Free; Earthdata account có thể được yêu cầu | PRIMARY - HISTORICAL |
| ERA5-Land | Historical weather, antecedent rain, soil/wind/temp context | 1 giờ | ~0.1° / ~9 km | CC BY 4.0; CDS account/API access | PRIMARY - HISTORICAL |
| NASA SRTMGL1 V003 | Elevation / DEM | Static | ~30 m | Free / public reuse | PRIMARY - STATIC |
| ESA WorldCover | Land cover / built-up proxy | Static | 10 m | CC BY 4.0 | PRIMARY - STATIC |
| OpenStreetMap | Roads, waterways, urban vector features | Near-static | Vector | ODbL; attribution required | PRIMARY - STATIC |
| GSMaP NOW | Latest rainfall context for realtime serving | cập nhật ~30 phút | ~0.1° / ~10 km | Free registration; JAXA terms + attribution | CANDIDATE - REALTIME |
| geoBoundaries gbOpen ADM2 | Legacy district AOI reference | Static | Polygon | CC BY 4.0; attribution required; source metadata records CC BY 3.0 IGO | REFERENCE |
| Copernicus DEM GLO-30 | Optional DEM alternative | Static | 30 m | Access restricted to authorized categories | OPTIONAL |

---

## 3. Vai trò theo pipeline

### Historical Training / Backfill

Nguồn chính:

- IMERG
- ERA5-Land
- SRTM
- WorldCover
- OpenStreetMap

Mục tiêu:

- tạo historical rainfall features;
- tạo target `target_rain_exceed_30m`;
- tạo target `target_rain_exceed_60m`;
- tính antecedent rainfall;
- tạo susceptibility features.

---

### Realtime / Near-Realtime

Candidate ban đầu:

- GSMaP NOW;
- forecast source sẽ được xác minh riêng.

GSMaP NOW không được dùng thay thế trực tiếp IMERG historical mà không có kiểm tra phân bố và bias.

---

## 4. Rainfall Source Separation

Historical target source:

`IMERG`

Realtime rainfall candidate:

`GSMaP NOW`

Do hai nguồn khác nhau, project phải kiểm tra:

- bias;
- scale mismatch;
- distribution shift;
- missing-data behavior;
- training-serving skew.

Nếu sự khác biệt quá lớn, phải có calibration hoặc common feature representation.

---

## 5. Hazard Resolution

Rainfall hazard phải giữ đúng spatial resolution thực của nguồn.

Không downscale IMERG hoặc GSMaP xuống H3 resolution 9 rồi tuyên bố đó là rainfall observation ở độ phân giải H3.

Một rainfall pixel có thể được ánh xạ tới nhiều H3 cells.

Các H3 cells có thể nhận cùng hazard nhưng có susceptibility khác nhau.

---

## 6. Static Susceptibility Sources

### Elevation

Primary:

`NASA SRTMGL1 V003`

Candidate derived features:

- elevation;
- slope;
- flow accumulation;
- HAND;
- TWI.

Giới hạn:

DEM khoảng 30 m không thể mô tả đầy đủ micro-topography, curb, cống hoặc cao độ mặt đường đô thị.

---

### Land Cover

Primary:

`ESA WorldCover`

Candidate features:

- built-up ratio;
- land-cover class;
- impervious-surface proxy.

Built-up ratio chỉ được xem là proxy cho impervious surface nếu chưa có nguồn impervious trực tiếp.

---

### Road / Water Network

Primary:

`OpenStreetMap`

Candidate features:

- road density;
- road class;
- intersection density;
- distance to waterway;
- distance to river;
- urban network characteristics.

Phải giữ attribution theo ODbL khi dữ liệu được công bố hoặc hiển thị.

---

## 7. Realtime Source Requirement

Một realtime source chỉ được đưa vào production khi xác minh được:

- access ổn định;
- latency;
- timestamp semantics;
- sampling interval;
- license;
- rate limit;
- failure behavior;
- freshness rule;
- fallback.

Không phụ thuộc sống còn vào một nguồn duy nhất.

---

## 8. Fallback Principle

Mỗi nguồn quan trọng phải có phương án fallback hoặc degradation mode.

Ví dụ:

- DEM: SRTM -> optional alternative DEM;
- rainfall realtime: GSMaP NOW -> forecast / secondary source;
- roads: cached OSM snapshot;
- land cover: cached WorldCover snapshot.

Nếu realtime rainfall source lỗi, hệ thống phải đánh dấu dữ liệu stale thay vì tiếp tục đưa ra prediction như bình thường.

---

## 9. Data Lineage

Mọi dataset tải về phải lưu tối thiểu metadata:

- `source`;
- `dataset`;
- `source_version`;
- `retrieval_time`;
- `license_reference`;
- `spatial_resolution`;
- `temporal_resolution`;
- `coverage`;
- `checksum` khi phù hợp.

---

## 10. Current Decisions

Primary historical rainfall:

`NASA GPM IMERG`

Primary historical weather:

`ERA5-Land`

Primary DEM:

`NASA SRTMGL1 V003`

Primary land cover:

`ESA WorldCover`

Primary road / water vector:

`OpenStreetMap`

Realtime rainfall candidate:

`GSMaP NOW`

Primary spatial identity:

`H3 resolution 9`

Prediction horizons:

`30 / 60 minutes`

### 10.1. Provenance của AOI và H3 grid
Nguồn boundary dùng để xây AOI 9 quận legacy:

- Provider: geoBoundaries
- Product: gbOpen
- Country: Viet Nam (`VNM`)
- Administrative level: `ADM2`
- Canonical level: `District`
- Boundary year represented: `2020`
- Boundary source: `OCHA ROAP, Government of Viet Nam`
- geoBoundaries distribution license: `CC BY 4.0`
- Original source license recorded by geoBoundaries: `CC BY 3.0 IGO`

File:

`data/reference/hanoi_core_9_legacy_districts.geojson`

là tập con gồm 9 quận legacy được trích từ boundary ADM2 nêu trên.

File:

`data/reference/hanoi_core_h3_res9.geojson`

là dữ liệu dẫn xuất do project tạo ra bằng cách polyfill AOI 9 quận
với H3 resolution 9. File này không phải boundary gốc của geoBoundaries.

Spatial identity chính của hệ thống là `h3_id`.
Thông tin `legacy_district` chỉ là administrative metadata được gán theo
tâm H3 cell và không thay đổi định danh H3.

Attribution:

- Runfola, D. et al. (2020), *geoBoundaries: A global database of political administrative boundaries*, PLoS ONE 15(4): e0231866.
- DOI: `10.1371/journal.pone.0231866`
