# Hanoi Flood Intelligence Lakehouse
## Problem Contract

**Phiên bản contract:** 2.2  
**Trạng thái:** Active  
**Phạm vi:** Hà Nội, Việt Nam

---

## 1. Mục tiêu hệ thống

Hệ thống thực hiện:

- dự báo mưa ngắn hạn;
- đánh giá rủi ro ngập đô thị tại Hà Nội.

Ở giai đoạn hiện tại, hệ thống không tuyên bố tạo ra xác suất ngập đã được kiểm chứng.

---

## 2. Bài toán Machine Learning chính

Với mỗi đơn vị không gian và thời điểm dự báo `t`, mô hình ước lượng xác suất lượng mưa vượt một ngưỡng cấu hình trong:

- 30 phút tiếp theo;
- 60 phút tiếp theo.

Horizon 15 phút đã được loại khỏi phạm vi hiện tại vì các nguồn dữ liệu mưa lịch sử miễn phí được lựa chọn chưa cung cấp ground truth 15 phút đủ phù hợp để huấn luyện và đánh giá nghiêm túc.

---

## 3. Target của mô hình

Hai target chính thức là:

- `target_rain_exceed_30m`
- `target_rain_exceed_60m`

Quy tắc tổng quát:

`target_rain_exceed_{h}m = 1`

nếu lượng mưa trong khoảng thời gian mục tiêu vượt ngưỡng mưa được cấu hình.

Ngược lại:

`target_rain_exceed_{h}m = 0`

Trong đó:

`h ∈ {30, 60}`

Ngưỡng mưa phải nằm trong configuration, không được hard-code trong feature pipeline hoặc model code.

---

## 4. Hazard

`hazard_score` biểu diễn mức độ nghiêm trọng của tình huống mưa.

Các input tiềm năng gồm:

- xác suất mưa vượt ngưỡng;
- lượng mưa tích lũy gần đây;
- antecedent rainfall;
- forecast rainfall;
- radar features khi có dữ liệu phù hợp.

Output:

`hazard_score ∈ [0, 1]`

Hazard thay đổi theo thời gian.

---

## 5. Flood Susceptibility

`susceptibility_score` biểu diễn mức độ dễ bị ngập vốn có của một khu vực khi xuất hiện mưa lớn.

Các feature tiềm năng gồm:

- elevation;
- HAND;
- TWI;
- flow accumulation;
- built-up / impervious-surface proxy;
- distance to river hoặc water body;
- historical known flood locations;
- đặc trưng đường và đô thị.

Output:

`susceptibility_score ∈ [0, 1]`

Các trọng số và tham số susceptibility phải nằm trong configuration và có tài liệu giải thích.

---

## 6. Flood Risk Index

Công thức ban đầu:

`risk_score = hazard_score * susceptibility_score`

Trong đó:

`risk_score ∈ [0, 1]`

Risk score được ánh xạ thành:

- `LOW`
- `MEDIUM`
- `HIGH`

Các ngưỡng phân loại phải nằm trong configuration.

Flood Risk Index là chỉ số rủi ro phục vụ nghiên cứu.

Không được mô tả `risk_score` là xác suất ngập đã được kiểm chứng cho tới khi project có đủ flood labels đáng tin cậy.

---

## 7. Prediction Horizon

Hai horizon chính thức:

- 30 phút;
- 60 phút.

Horizon 15 phút không còn thuộc hệ thống hiện tại.

Không tạo production schema, feature pipeline, model, test, API hoặc dashboard phụ thuộc vào horizon 15 phút.

---

## 8. Ngữ nghĩa thời gian

Các dataset realtime và model phải phân biệt rõ:

- `event_time`
- `ingestion_time`
- `forecast_issue_time`
- `valid_time`
- `feature_time`
- `prediction_time`
- `target_time`
- `prediction_horizon`

Với prediction tại thời điểm `t`, feature chỉ được sử dụng thông tin thực sự khả dụng tại hoặc trước `t`.

Không được sử dụng future data làm feature.

---

## 9. Spatial Unit

Primary spatial unit của hệ thống là:

`H3 resolution 9`

H3 resolution 9 được sử dụng cho:

- spatial identity chính của hệ thống;
- susceptibility features;
- Flood Risk Index;
- prediction output;
- dashboard mapping.

Trong AOI hiện tại gồm 9 quận đô thị cũ của Hà Nội, H3 resolution 9 tạo khoảng:

`1,481 cells`

Diện tích trung bình của một cell H3 resolution 9:

`~0.1053 km²`

Độ dài cạnh trung bình:

`~0.2008 km`

H3 resolution 8 được giữ làm parent / aggregation level.

Trong cùng AOI, H3 resolution 8 tạo khoảng:

`210 cells`

H3 resolution 8 có thể được sử dụng cho:

- aggregation;
- summary;
- parent-level analysis;
- spatial comparison ở mức thô hơn.

Việc sử dụng H3 resolution 9 không có nghĩa dữ liệu rainfall đạt độ phân giải tương ứng.

Rainfall hazard phải giữ đúng độ phân giải thực của nguồn dữ liệu.

Ví dụ, một rainfall pixel có thể phủ nhiều H3 resolution 9 cells.

Các cell đó có thể nhận cùng hazard nhưng có susceptibility khác nhau do các đặc trưng như:

- elevation;
- HAND / TWI;
- built-up ratio;
- road density;
- distance to river.

Do đó:

`hazard resolution != H3 risk resolution`

Project AOI được giới hạn trong footprint của 9 quận đô thị cũ:

- Ba Đình;
- Cầu Giấy;
- Đống Đa;
- Hà Đông;
- Hai Bà Trưng;
- Hoàn Kiếm;
- Hoàng Mai;
- Tây Hồ;
- Thanh Xuân.

H3 index là spatial identity chính thức của hệ thống.

Thông tin hành chính được xem là metadata và có thể bao gồm:

- `legacy_district`;
- `current_ward`;
- `admin_version`.


---

## 10. Output Contract

Mỗi prediction record tối thiểu phải hỗ trợ các field logic sau:

- `grid_id`
- `prediction_time`
- `target_time`
- `horizon_min`
- `p_rain_exceed`
- `rain_threshold_mm_h`
- `hazard_score`
- `susceptibility_score`
- `risk_score`
- `risk_level`
- `explanation`
- `model_version`

Lineage metadata và data-quality metadata sẽ được bổ sung khi thiết kế schema chi tiết.

---

## 11. Những tuyên bố được phép

Project có thể được mô tả là:

**Hệ thống dự báo mưa ngắn hạn và đánh giá rủi ro ngập đô thị tại Hà Nội.**

Ở giai đoạn hiện tại, project không được tuyên bố:

- cung cấp xác suất ngập đã được kiểm chứng;
- dự đoán `FLOOD / NO FLOOD` đã được kiểm chứng.

---

## 12. Hướng phát triển sau này

Các báo cáo ngập từ tin tức, HSDC/iHanoi manual validation và các nguồn quan sát đáng tin cậy khác sẽ được tích lũy theo thời gian.

Khi flood labels đủ về số lượng và chất lượng, project có thể bổ sung supervised flood classification model.

Lớp này là phần mở rộng tương lai và không được làm cản trở pipeline rainfall nowcasting + Flood Risk Index hiện tại.