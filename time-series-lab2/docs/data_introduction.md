# TÀI LIỆU GIỚI THIỆU DỮ LIỆU THỰC NGHIỆM
## BỘ DỮ LIỆU AIRPASSENGERS (1949 - 1960)

---

## 1. NGUỒN GỐC VÀ BỐI CẢNH LỊCH SỬ

Bộ dữ liệu **AirPassengers** (thường được biết đến với tên gọi kinh điển *Box & Jenkins Airline Data*) là một trong những bộ dữ liệu chuẩn mực (benchmark dataset) nổi tiếng và có ảnh hưởng sâu rộng nhất trong lĩnh vực Thống kê ứng dụng và Khai phá Chuỗi Thời gian (Time Series Analysis & Forecasting).

- **Nguồn xuất bản gốc:** Được công bố và phân tích chuyên sâu lần đầu trong công trình nghiên cứu kinh điển của George E. P. Box và Gwilym M. Jenkins:
  > **Box, G. E. P., & Jenkins, G. M. (1976).** *Time Series Analysis: Forecasting and Control.* Holden-Day, San Francisco.
- **Bối cảnh thu thập:** Ghi nhận tổng số lượng hành khách trên các chuyến bay quốc tế của các hãng hàng không thương mại trong giai đoạn phục hồi và bùng nổ sau Chiến tranh thế giới thứ hai (thập niên 1950).
- **Vị trí lưu trữ trong dự án:**
  - Dữ liệu gốc (Raw): [`../data/raw/AirPassengers.csv`](../data/raw/AirPassengers.csv)
  - Dữ liệu chuẩn hóa (Processed): [`../data/processed/AirPassengers_resampled.csv`](../data/processed/AirPassengers_resampled.csv)

---

## 2. ĐẶC TẢ THUỘC TÍNH VÀ SCHEMA DỮ LIỆU

Bộ dữ liệu có cấu trúc đơn giản, tinh gọn nhưng phản ánh đầy đủ mọi đặc trưng phức tạp của chuỗi thời gian thực tế:

| Tên trường (Column) | Kiểu dữ liệu gốc | Kiểu dữ liệu chuẩn hóa | Đơn vị đo lường | Mô tả chi tiết |
| :--- | :---: | :---: | :---: | :--- |
| `Month` | `string` (`YYYY-MM`) | `pd.DatetimeIndex` (`freq='MS'`) | Tháng - Năm | Thời điểm quan sát bắt đầu từ ngày mùng 1 hàng tháng (`1949-01-01` đến `1960-12-01`). |
| `Passengers` | `int64` | `float64` | Nghìn lượt người ($10^3$) | Tổng lượng hành khách quốc tế di chuyển bằng đường hàng không trong tháng tương ứng (ví dụ: `112` tương đương $112,000$ hành khách). |

---

## 3. THỐNG KÊ MÔ TẢ ĐỊNH LƯỢNG

Trích xuất từ kết quả chạy thực nghiệm tại [`../outputs/tables/summary_statistics.csv`](../outputs/tables/summary_statistics.csv):

| Chỉ số thống kê | Ký hiệu toán học | Giá trị tính toán | Diễn giải ý nghĩa |
| :--- | :---: | :---: | :--- |
| **Số lượng quan sát (Count)** | $n$ | 144 | 144 tháng liên tục trong suốt 12 năm (1949 - 1960). |
| **Giá trị trung bình (Mean)** | $\bar{y}$ | 280.30 | Bình quân mỗi tháng có 280,300 lượt khách bay. |
| **Độ lệch chuẩn mẫu (Std)** | $s$ | 119.97 | Biên độ phân tán lớn quanh giá trị trung bình. |
| **Giá trị nhỏ nhất (Min)** | $y_{\min}$ | 104.00 | Đáy thấp kỷ lục vào Tháng 11/1949. |
| **Phân vị 25% (Q1)** | $Q_1$ | 180.00 | 25% số tháng có lượng khách dưới 180,000 người. |
| **Trung vị (Median / Q2)** | $\text{Med}$ | 265.50 | Điểm cân bằng của phân phối mẫu. |
| **Phân vị 75% (Q3)** | $Q_3$ | 360.50 | 75% số tháng có lượng khách dưới 360,500 người. |
| **Giá trị lớn nhất (Max)** | $y_{\max}$ | 622.00 | Đỉnh cao kỷ lục vào Tháng 07/1960. |
| **Hệ số bất đối xứng (Skewness)** | $\gamma_1$ | +0.5831 | Phân phối lệch dương nhẹ do xu hướng tăng trưởng liên tục. |
| **Độ nhọn (Kurtosis)** | $\gamma_2$ | -0.3650 | Đỉnh phân phối phẳng hơn so với phân phối chuẩn Gauss. |

### Tính toàn vẹn của chuỗi dữ liệu gốc:
- Số bản ghi bị thiếu (Missing/NaN): **0 / 144 (0.0%)** $\rightarrow$ Dữ liệu nguyên bản hoàn hảo 100%.
- Tính trùng lặp mốc thời gian (Duplicate timestamps): **0 bản ghi** $\rightarrow$ Chỉ mục thời gian đơn điệu tăng nghiêm ngặt.
- Tần suất quan sát: Đều đặn hàng tháng (`MS` - Month Start).

---

## 4. QUYỀN SỞ HỮU TRÍ TUỆ VÀ ĐIỀU KHOẢN SỬ DỤNG

- **Bản quyền dữ liệu (License):** Bộ dữ liệu thuộc phạm vi **Miền công cộng (Public Domain / CC0: Public Domain Dedication)**.
- **Quyền sử dụng:**
  - Hoàn toàn tự do sử dụng cho mục đích giảng dạy, nghiên cứu khoa học, tái hiện thực nghiệm và phát triển mô hình.
  - Được phép tích hợp vào các gói thư viện mã nguồn mở (hiện được tích hợp sẵn trong `R` package `datasets`, Python `statsmodels`, `seaborn` và `sktime`).
- **Trích dẫn học thuật chuẩn mực (BibTeX):**
```bibtex
@book{boxjenkins1976,
  author    = {George E. P. Box and Gwilym M. Jenkins},
  title     = {Time Series Analysis: Forecasting and Control},
  publisher = {Holden-Day},
  address   = {San Francisco},
  year      = {1976},
  isbn      = {978-0816211043}
}
```

---

## 5. PHẠM VI ỨNG DỤNG TRONG LAB 2

Dữ liệu `AirPassengers` là hình mẫu lý tưởng để thực nghiệm các kỹ thuật:
1. **Kiểm tra tính dừng (Stationarity):** Chuỗi có xu hướng tăng rõ rệt và phương sai phình rộng dần theo thời gian (Heteroskedasticity), vi phạm tính dừng nghiêm trọng.
2. **Phân tích ACF/PACF:** Thể hiện đồng thời tính tự tương quan dương cực mạnh suy giảm chậm và các xung nhịp mùa vụ 12 tháng.
3. **Mô hình hóa phân rã:** Thử nghiệm đối chứng xuất sắc giữa mô hình Cộng và mô hình Nhân.
4. **Kiểm toán chất lượng dữ liệu:** Là môi trường đối chứng chuẩn (Ground Truth) để đánh giá độ chính xác của các thuật toán bù khuyết (Interpolation) và thuật toán lọc ngoại lai (Hampel Filter).
