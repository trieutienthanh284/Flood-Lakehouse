# Hanoi Flood Intelligence Lakehouse
## Temporal & Bitemporal Contract

**Phiên bản:** 1.0<br>
**Trạng thái:** Active<br>
**Prediction horizons:** 30 phút và 60 phút

---

## 1. Mục đích

Contract này quy định ngữ nghĩa thời gian cho toàn bộ pipeline.

Mục tiêu chính là bảo đảm:

- không sử dụng future data;
- không xảy ra point-in-time leakage;
- historical replay phản ánh đúng dữ liệu mà hệ thống thực sự biết tại từng thời điểm;
- training và realtime serving sử dụng cùng nguyên tắc thời gian.

---

## 2. Các timestamp chính

### event_time

Thời điểm hiện tượng xảy ra ngoài đời.

Ví dụ:

`event_time = 2026-10-06 09:30`

nghĩa là observation mô tả trạng thái tại 09:30.

---

### ingestion_time

Thời điểm hệ thống Flood Lakehouse thực sự nhận được record.

Ví dụ:

`event_time = 09:30`

`ingestion_time = 09:36`

nghĩa là dữ liệu có độ trễ 6 phút.

Một model chạy trước 09:36 không được sử dụng record này.

---

### forecast_issue_time

Thời điểm nhà cung cấp phát hành một bản forecast.

Ví dụ:

`forecast_issue_time = 08:00`

nghĩa là forecast được phát hành lúc 08:00.

---

### valid_time

Thời điểm mà forecast mô tả hoặc có hiệu lực.

Ví dụ:

`forecast_issue_time = 08:00`

`valid_time = 10:00`

nghĩa là lúc 08:00 đã tồn tại một forecast cho thời điểm 10:00.

`valid_time` không phải thời điểm hệ thống biết forecast.

---

### feature_time

Thời điểm mà feature vector đại diện.

Mọi dữ liệu dùng để tạo feature tại thời điểm `t` phải thỏa mãn quy tắc point-in-time của contract này.

---

### prediction_time

Thời điểm model thực sự sinh prediction.

Ví dụ:

`prediction_time = 10:00`

---

### target_time

Thời điểm mục tiêu mà prediction hướng tới.

Với hệ thống hiện tại:

- horizon 30 phút: `target_time = prediction_time + 30 phút`
- horizon 60 phút: `target_time = prediction_time + 60 phút`

---

### prediction_horizon

Khoảng cách giữa prediction time và target time.

Giá trị hợp lệ:

- `30`
- `60`

Đơn vị chuẩn: phút.

---

## 3. Hai trục thời gian cốt lõi

Hệ thống phải phân biệt:

1. thời gian của thế giới thực;
2. thời gian hệ thống biết thông tin.

Ví dụ:

`event_time = 09:30`

`ingestion_time = 10:05`

Dù sự kiện đã xảy ra lúc 09:30, hệ thống chỉ biết record đó từ 10:05.

Do đó prediction chạy lúc 10:00 không được sử dụng record này.

---

## 4. Quy tắc Point-in-Time Correctness

Với feature được tạo tại thời điểm `t`:

### Observation data

Record chỉ được sử dụng nếu tối thiểu:

`event_time <= t`

và:

`ingestion_time <= t`

### Forecast data

Forecast chỉ được sử dụng nếu:

`forecast_issue_time <= t`

và:

`ingestion_time <= t`

`valid_time` có thể lớn hơn `t` vì đó chính là thời điểm forecast hướng tới.

Không được lựa chọn một forecast chỉ vì `valid_time` phù hợp nếu forecast đó chưa được phát hành hoặc chưa được hệ thống ingest tại thời điểm `t`.

---

## 5. Availability Time

Khái niệm logic `available_time` biểu diễn thời điểm sớm nhất mà hệ thống thực sự có thể sử dụng một record.

Với observation thông thường:

`available_time = ingestion_time`

Với forecast:

`available_time = max(forecast_issue_time, ingestion_time)`

Quy tắc tổng quát:

`available_time <= feature_time`

Record có:

`available_time > feature_time`

phải bị loại khỏi feature calculation.

---

## 6. Ví dụ hợp lệ

Prediction chạy lúc:

`10:00`

Record A:

`event_time = 09:30`

`ingestion_time = 09:35`

Kết luận:

`09:35 <= 10:00`

Record A được phép sử dụng.

---

## 7. Ví dụ leakage

Prediction chạy lúc:

`10:00`

Record B:

`event_time = 09:30`

`ingestion_time = 10:05`

Mặc dù:

`event_time < prediction_time`

nhưng:

`ingestion_time > prediction_time`

Do đó Record B không được sử dụng.

Nếu sử dụng Record B trong training, hệ thống đã tạo point-in-time leakage.

---

## 8. Forecast Example

Prediction chạy lúc:

`10:00`

Forecast A:

`forecast_issue_time = 09:45`

`ingestion_time = 09:47`

`valid_time = 10:30`

Forecast A được phép sử dụng.

Forecast B:

`forecast_issue_time = 09:55`

`ingestion_time = 10:04`

`valid_time = 10:30`

Forecast B không được sử dụng cho prediction lúc 10:00 vì hệ thống chỉ nhận được nó lúc 10:04.

---

## 9. Training và Historical Replay

Historical training không được sử dụng dữ liệu chỉ dựa vào event timestamp.

Khi replay prediction tại thời điểm `t`, pipeline phải tái dựng:

"Những dữ liệu nào thực sự available tại thời điểm t?"

Không được sử dụng revision, reanalysis hoặc forecast được phát hành sau `t` nếu chúng không tồn tại tại thời điểm prediction giả lập.

---

## 10. As-Of Join

Các temporal join phục vụ feature engineering phải tuân theo nguyên tắc as-of.

Với mỗi feature row tại thời điểm `t`, chỉ được lấy record phù hợp gần nhất mà:

`available_time <= t`

Không được join với record tương lai chỉ vì record đó gần `t` hơn.

---

## 11. Time Zone

Timestamp lưu trữ trong các tầng dữ liệu chuẩn hóa phải có timezone rõ ràng.

Chuẩn lưu trữ ưu tiên:

`UTC`

Khi hiển thị cho người dùng tại Hà Nội:

`Asia/Ho_Chi_Minh`

Không được lưu timestamp không có timezone nếu timestamp đó tham gia vào logic realtime, feature engineering hoặc model evaluation.

---

## 12. Leakage Tests bắt buộc

Pipeline phải có automated tests kiểm tra tối thiểu:

- `event_time <= feature_time` đối với observation được sử dụng;
- `ingestion_time <= feature_time`;
- `forecast_issue_time <= feature_time` đối với forecast;
- `available_time <= feature_time`;
- `target_time > prediction_time`;
- `prediction_horizon in {30, 60}`;
- không có future observation trong feature set.

Các test này phải được đưa vào CI khi CI pipeline được triển khai.

---

## 13. Nguyên tắc bất biến

Không được hy sinh point-in-time correctness chỉ để tăng metric offline.

Một model có metric thấp hơn nhưng sử dụng đúng dữ liệu khả dụng tại thời điểm dự đoán có giá trị hơn một model có metric cao nhờ future information.

Training và serving phải tuân theo cùng một temporal contract.
