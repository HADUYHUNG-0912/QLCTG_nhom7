# BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM LAB 2 (QLCTG - NHÓM 7)

Dưới đây là các bảng số liệu trích xuất từ chuỗi thực nghiệm trên bộ dữ liệu `AirPassengers.csv` (144 quan sát, từ 01/1949 đến 12/1960).

---

## 1. Thống kê mô tả tổng quan dữ liệu gốc

| Chỉ số thống kê | Giá trị | Đơn vị tính | Diễn giải |
| :--- | :---: | :---: | :--- |
| **Số lượng quan sát (Count)** | 144 | Tháng | 12 năm liên tục (1949 - 1960) |
| **Giá trị trung bình (Mean)** | 280.30 | Nghìn lượt khách | Mức hành khách bình quân/tháng |
| **Độ lệch chuẩn (Std Dev)** | 119.97 | Nghìn lượt khách | Biến động mạnh quanh trung bình |
| **Giá trị nhỏ nhất (Min)** | 104.00 | Nghìn lượt khách | Tháng 11/1949 |
| **Phân vị 25% (Q1)** | 180.00 | Nghìn lượt khách | 25% số tháng có lượng khách < 180 |
| **Trung vị 50% (Median)** | 265.50 | Nghìn lượt khách | Điểm cân bằng của phân phối |
| **Phân vị 75% (Q3)** | 360.50 | Nghìn lượt khách | 75% số tháng có lượng khách < 360.5 |
| **Giá trị lớn nhất (Max)** | 622.00 | Nghìn lượt khách | Tháng 07/1960 |
| **Hệ số bất đối xứng (Skewness)** | +0.5831 | Không thứ nguyên | Lệch phải nhẹ (xu hướng tăng trưởng) |
| **Độ nhọn (Kurtosis)** | -0.3650 | Không thứ nguyên | Phẳng hơn phân phối chuẩn |

---

## 2. Kiểm định Ljung-Box Test (White Noise vs AirPassengers)

Kiểm định giả thuyết:
- $H_0$: Chuỗi dữ liệu là nhiễu trắng ngẫu nhiên (không có tự tương quan).
- $H_1$: Chuỗi dữ liệu không phải nhiễu trắng (chứa cấu trúc tự tương quan / thông tin quy luật).

| Chuỗi dữ liệu | Độ trễ ($h$) | Thống kê $Q$ | Trị số p-value | Mức ý nghĩa $\alpha$ | Kết luận thống kê |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gaussian White Noise** | 10 | 7.71 | 0.6571 | 0.05 | Chấp nhận $H_0$ (Là nhiễu trắng) |
| **Gaussian White Noise** | 20 | 15.16 | 0.7675 | 0.05 | Chấp nhận $H_0$ (Là nhiễu trắng) |
| **AirPassengers** | 10 | 857.07 | $1.10 \times 10^{-177}$ | 0.05 | Bác bỏ $H_0$ (Chứa tín hiệu quy luật mạnh) |
| **AirPassengers** | 20 | 1434.15 | $5.30 \times 10^{-292}$ | 0.05 | Bác bỏ $H_0$ (Chứa tín hiệu quy luật mạnh) |

---

## 3. So sánh các mô hình phân rã chuỗi thời gian

| Mô hình phân rã | Công thức toán học | Tăng trưởng xu hướng | Đỉnh mùa vụ (Peak) | Đáy mùa vụ (Trough) | Bản chất phần dư (Residual) | Đánh giá mức độ phù hợp |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Additive (Cộng)** | $Y_t = T_t + S_t + R_t$ | +274.7% | Tháng 7 (+53.5 khách) | Tháng 11 (-56.2 khách) | Phương sai thay đổi theo thời gian | Kém phù hợp (Biên độ mùa mở rộng) |
| **Multiplicative (Nhân)** | $Y_t = T_t \times S_t \times R_t$ | +274.7% | Tháng 7 ($\times 1.27$) | Tháng 11 ($\times 0.82$) | Phương sai ổn định quanh 1.0 | **TỐI ƯU VƯỢT TRỘI** |
| **STL Decomposition** | $Y_t = T_t + S_t + R_t$ (Loess) | +274.7% | Tháng 7 (+51.2 khách) | Tháng 11 (-54.8 khách) | Linh hoạt, thích ứng phi tuyến | Rất tốt cho phân tích chuyên sâu |

---

## 4. Đánh giá chất lượng và phục hồi dữ liệu (Data Quality)

### a. Bù khuyết dữ liệu thiếu (10% Missing Values)
| Thuật toán | MAE (nghìn khách) | RMSE (nghìn khách) | Đánh giá kỹ thuật |
| :--- | :---: | :---: | :--- |
| **Forward-fill (ffill)** | 23.31 | 28.59 | Bị chậm pha, trễ nhịp so với xu hướng dốc lên |
| **Time-based Interpolation** | **16.43** | **18.83** | **Tốt nhất:** Bám sát độ dốc tuyến tính giữa 2 mốc thời gian |

### b. Phát hiện ngoại lai bất thường (Spikes 3.5x)
| Phương pháp | Vị trí phát hiện | Số lượng phát hiện | Đánh giá |
| :--- | :---: | :---: | :--- |
| **Rolling Z-Score** | `[25, 75, 110]` | 3/3 spikes | Phát hiện chính xác, nhưng nhạy cảm với giả định chuẩn |
| **Rolling IQR** | `[25, 43, 75, 110]` | 4 phát hiện (báo nhầm 1) | Báo nhầm đỉnh cao điểm du lịch hè tháng 43 |
| **Hampel Filter (Median & MAD)** | `[25, 75, 110]` | **3/3 spikes (0 nhầm)** | **Kháng nhiễu tuyệt đối**, chuẩn xác 100% |

### c. Hiệu quả khôi phục toàn diện End-to-End Pipeline
| Trạng thái chuỗi dữ liệu | MAE (nghìn khách) | RMSE (nghìn khách) | Mức độ cải thiện sai số |
| :--- | :---: | :---: | :--- |
| **Sau nội suy, trước khi lọc Hampel** | 15.04 | 99.18 | Lỗi RMSE rất lớn do các đột biến 3.5x |
| **Sau nội suy và qua bộ lọc Hampel Filter** | **1.64** | **5.90** | **RMSE giảm ~17 lần**, dữ liệu phục hồi gần như tuyệt đối |

---

## 5. Phân tích mùa vụ trung bình 12 tháng

| Tháng | Tên tháng | Lượng khách TB (nghìn) | Độ lệch chuẩn (nghìn) | Thấp nhất (nghìn) | Cao nhất (nghìn) | Chỉ số mùa vụ (Multiplicative) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Tháng 1 | 241.75 | 76.54 | 112 | 417 | 0.8625 |
| 2 | Tháng 2 | 235.00 | 69.86 | 118 | 391 | 0.8384 |
| 3 | Tháng 3 | 270.17 | 81.34 | 132 | 419 | 0.9638 |
| 4 | Tháng 4 | 267.08 | 74.34 | 129 | 461 | 0.9528 |
| 5 | Tháng 5 | 271.83 | 77.29 | 121 | 472 | 0.9698 |
| 6 | Tháng 6 | 311.67 | 92.03 | 135 | 535 | 1.1119 |
| **7** | **Tháng 7 (Peak)** | **351.33** | **109.18** | **148** | **622** | **1.2534 (Đỉnh hè)** |
| 8 | Tháng 8 | 351.08 | 107.56 | 148 | 606 | 1.2525 |
| 9 | Tháng 9 | 314.42 | 89.63 | 136 | 508 | 1.1217 |
| 10 | Tháng 10 | 274.00 | 79.41 | 119 | 461 | 0.9775 |
| **11** | **Tháng 11 (Trough)**| **232.83** | **64.06** | **104** | **390** | **0.8306 (Đáy năm)** |
| 12 | Tháng 12 | 261.83 | 74.54 | 118 | 432 | 0.9341 |
