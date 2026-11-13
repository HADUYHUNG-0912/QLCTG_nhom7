# BÀI TẬP THỰC HÀNH 2: KIỂM ĐỊNH ADF & CHUẨN BỊ MÔ HÌNH ARIMA / SARIMA
## MÔN HỌC: PHÂN TÍCH DỮ LIỆU CHUỖI THỜI GIAN VÀ DỰ BÁO

> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Nhóm thực hiện:** Nhóm 7  
> **Tài liệu gốc:** `excercise_stats_models.pdf`  

---

## 🎯 MỤC TIÊU CỦA BÀI THỰC HÀNH (OBJECTIVES)

1. **Nhận diện chuỗi dừng / không dừng:** Quan sát và nhận diện trực quan các thành phần xu hướng (trend) và mùa vụ (seasonality).
2. **Thực hiện và diễn giải kiểm định ADF:** Nắm vững giả thuyết $H_0$, thống kê kiểm định và giá trị $p$-value.
3. **Xác định tham số $d$ trong ARIMA:** Lấy sai phân để đưa chuỗi về trạng thái dừng (Stationarity).
4. **Kết hợp ADF với ACF/PACF:** Đọc biểu đồ tự tương quan và tự tương quan riêng phần để ước lượng bậc $(p, d, q)$.
5. **Chuẩn bị dữ liệu đúng cách cho ARIMA / SARIMA:** Phân chia tập huấn luyện và kiểm thử theo trật tự thời gian (không shuffle).
6. **Đánh giá mô hình:** Chẩn đoán phần dư (Residual Diagnostics) và đo lường sai số dự báo (MAE, RMSE, MAPE).

---

## 📊 BÀI TẬP THỰC HÀNH CƠ BẢN (SỬ DỤNG DATASET 1)

> **Bộ dữ liệu áp dụng:** **Dataset 1 — Monthly Retail Sales (a10.csv)**  
> *Doanh số bán lẻ theo tháng tại Mỹ (Đặc điểm: Có cả Trend + Seasonality).*  
> **Nguồn tải:** [https://raw.githubusercontent.com/selva86/datasets/master/a10.csv](https://raw.githubusercontent.com/selva86/datasets/master/a10.csv)

---

### Bài 1: Khám phá dữ liệu (EDA)

* **Nhiệm vụ thực hiện:**
  - Nạp dữ liệu vào DataFrame.
  - Chuyển đổi cột `date` thành `DatetimeIndex` với tần suất phù hợp.
  - Vẽ đồ thị chuỗi thời gian hoàn chỉnh theo trục thời gian.
* **Câu hỏi cần trả lời:**
  1. Chuỗi thời gian có xu hướng (Trend) rõ ràng không?
  2. Chuỗi có dấu hiệu mùa vụ (Seasonality) lặp lại theo chu kỳ không?
  3. Bằng mắt thường, chuỗi có khả năng là chuỗi dừng (Stationary) không? Vì sao?

---

### Bài 2: Kiểm định ADF trên chuỗi gốc

* **Nhiệm vụ thực hiện:**
  - Thực hiện kiểm định **ADF (Augmented Dickey-Fuller Test)** cho chuỗi gốc ban đầu.
  - Trích xuất các chỉ số:
    * Giá trị thống kê ADF (ADF statistic).
    * Giá trị $p$-value.
    * Các giá trị ngưỡng tới hạn (Critical Values tại $1\%, 5\%, 10\%$).
  - Đưa ra kết luận: Chuỗi gốc là **Dừng** hay **Không dừng**?
* **Câu hỏi cần trả lời:**
  1. Giả thuyết vô hiệu ($H_0$) và giả thuyết đối ($H_1$) của kiểm định ADF là gì?
  2. Giá trị $p$-value phản ánh điều gì trong việc bác bỏ hoặc chấp nhận $H_0$?
  3. Căn cứ vào số liệu tính toán, chuỗi gốc có dừng không?

---

### Bài 3: Sai phân & Kiểm định lại

* **Nhiệm vụ thực hiện:**
  - Thực hiện lấy sai phân bậc 1: $\Delta Y_t = Y_t - Y_{t-1}$.
  - Vẽ lại đồ thị đường của chuỗi sau khi lấy sai phân bậc 1.
  - Thực hiện kiểm định ADF lần 2 trên chuỗi sai phân.
* **Câu hỏi cần trả lời:**
  1. Sau khi lấy sai phân bậc 1, chuỗi đã đạt trạng thái dừng chưa ($p\text{-value} < 0.05$)?
  2. Giá trị tham số tích hợp $d$ trong mô hình $\text{ARIMA}(p, d, q)$ được xác định là bao nhiêu?

---

### Bài 4: Phân tích biểu đồ ACF & PACF

* **Nhiệm vụ thực hiện:**
  - Vẽ biểu đồ hàm tự tương quan (**ACF — Autocorrelation Function**) của chuỗi đã dừng.
  - Vẽ biểu đồ hàm tự tương quan riêng phần (**PACF — Partial Autocorrelation Function**) của chuỗi đã dừng.
  - Phân tích hiện tượng cắt cụt (cutoff) hoặc suy giảm dần (tails off).
  - Đề xuất các bộ tham số $(p, d, q)$ khả dĩ.
* **Câu hỏi cần trả lời:**
  1. Biểu đồ ACF cắt cụt (vào khoảng tin cậy 95%) ở bậc trễ (lag) nào?
  2. Biểu đồ PACF cắt cụt ở bậc trễ (lag) nào?
  3. Gợi ý ít nhất 2 cấu hình mô hình $\text{ARIMA}(p, d, q)$ tiềm năng cho chuỗi dữ liệu?

---

### Bài 5: Thiết lập ARIMA / SARIMA – Dự đoán – Đánh giá

* **Nhiệm vụ thực hiện:**
  1. **Phân chia dữ liệu (Temporal Split):**
     * Tập Huấn luyện (Train): $80\%$ dữ liệu thời gian đầu.
     * Tập Kiểm thử (Test): $20\%$ dữ liệu thời gian cuối.
     * ⚠️ *Quy tắc bắt buộc:* Giữ nguyên thứ tự thời gian tuần tự, tuyệt đối **không shuffle**.
  2. **Thiết lập & Huấn luyện mô hình ARIMA:**
     * Dựa trên kết quả ACF/PACF từ Bài 4, đề xuất 1–2 cấu hình mô hình (Ví dụ: $\text{ARIMA}(1, 1, 1)$, $\text{ARIMA}(2, 1, 1)$).
     * Huấn luyện mô hình trên tập Train.
     * Ghi nhận và so sánh các tiêu chuẩn thông tin **AIC** và **BIC**.
  3. **Chẩn đoán phần dư (Residual Diagnostics):**
     * Vẽ đồ thị phần dư (Residuals) theo trục thời gian.
     * Vẽ đồ thị ACF của phần dư.
     * Kiểm tra tính chất của phần dư:
       - Phần dư có kỳ vọng trung bình xấp xỉ 0 ($\text{mean} \approx 0$)?
       - Phần dư có còn hiện tượng tự tương quan (Autocorrelation) không?
  4. **Dự báo (Forecasting):**
     * Thực hiện dự báo trên khoảng thời gian của tập Test.
     * Vẽ đồ thị trực quan hóa đa đường: Train, Test (thực tế), và Forecast (dự báo).
  5. **Đánh giá định lượng mô hình ARIMA:**
     * Tính toán bắt buộc 3 chỉ số đo lường sai số trên tập Test:
       - **MAE (Mean Absolute Error):**
         $$\text{MAE} = \frac{1}{N} \sum_{t=1}^N |y_t - \hat{y}_t|$$
       - **RMSE (Root Mean Squared Error):**
         $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N (y_t - \hat{y}_t)^2}$$
       - **MAPE (Mean Absolute Percentage Error):**
         $$\text{MAPE} = \frac{1}{N} \sum_{t=1}^N \left| \frac{y_t - \hat{y}_t}{y_t} \right| \times 100\%$$
* **Bộ 9 câu hỏi thảo luận lý luận:**
  1. Vì sao **không được shuffle** dữ liệu chuỗi thời gian? Tập Test set trong bài toán chuỗi thời gian dùng để làm gì?
  2. Ý nghĩa thống kê cụ thể của các tham số $p, d, q$ đã chọn?
  3. Quá trình tối ưu của mô hình có hội tụ không?
  4. Phần dư (Residuals) có thỏa mãn điều kiện của nhiễu trắng (**White Noise**) không?
  5. Nếu phần dư vẫn còn hiện tượng tự tương quan thì cần điều chỉnh gì ở mô hình?
  6. Đường dự báo có bám sát được xu hướng và chu kỳ dao động thực tế không?
  7. Sai lệch lớn nhất giữa dự báo và thực tế xảy ra ở đâu (thời điểm nào, biên độ bao nhiêu)?
  8. Trong các chỉ số MAE, RMSE, MAPE, chỉ số nào **nhạy cảm nhất với các điểm dị biệt (outliers)**? Vì sao?
  9. Giá trị RMSE lớn hay nhỏ phản ánh điều gì về chất lượng dự báo của mô hình?

---

## 🚀 BÀI TẬP NÂNG CAO VỚI MÔ HÌNH SARIMA (SỬ DỤNG DATASET 2)

> **Bộ dữ liệu áp dụng:** **Dataset 2 — AirPassengers (airline-passengers.csv)**  
> *Số lượng hành khách hàng không quốc tế theo tháng từ năm 1949 đến 1960 (144 quan sát).*  
> **Nguồn tải:** [https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv)

### Các bước thực hiện:
1. **Trực quan hóa chuỗi thời gian:** Vẽ biểu đồ đường nhận diện chu kỳ mùa vụ hàng năm ($s = 12$ tháng).
2. **Kiểm định ADF chuỗi gốc:** Kiểm chứng tính không dừng ($p\text{-value} > 0.05$).
3. **Thực hiện sai phân kép:**
   * Sai phân thường bậc 1: $d = 1$ để loại bỏ xu hướng tăng trưởng (Trend).
   * Sai phân mùa vụ bậc 1: $D = 1$ với chu kỳ $s = 12$ để triệt tiêu tính mùa vụ.
4. **Phân tích ACF/PACF đa chu kỳ:** Xác định các bậc trễ ngắn hạn $(p, q)$ và các bậc trễ mùa vụ $(P, Q)$ tại các lag bội số của 12 (lag 12, 24, 36).
5. **Huấn luyện mô hình SARIMA:**
   $$\text{SARIMA}(p, d, q) \times (P, D, Q)_{s=12}$$
6. **Dự báo & Đánh giá sai số:** Dự báo tập Test, tính toán định lượng các thước đo MAE, RMSE, MAPE.
7. **Đối chuẩn so sánh:** Lập bảng so sánh đối đầu giữa mô hình $\text{ARIMA}$ phi mùa vụ và mô hình $\text{SARIMA}$ có xét mùa vụ.

* **Gợi ý thiết lập tham số mùa vụ:**
  - Chu kỳ mùa vụ: $s = 12$ tháng.
  - Bậc sai phân: $d = 1$ (loại bỏ Trend), $D = 1$ (loại bỏ Seasonality).

---

## 📁 DANH MỤC DATASET THAM CHIẾU

| Tên bộ dữ liệu | Nội dung & Phạm vi | Đặc tính chuỗi | Đường dẫn tải trực tiếp (Raw CSV) |
| :--- | :--- | :--- | :--- |
| **Dataset 1: Monthly Retail Sales** | Doanh số bán lẻ theo tháng tại Mỹ | Xu hướng tăng + Chu kỳ mùa vụ rõ nét | `https://raw.githubusercontent.com/selva86/datasets/master/a10.csv` |
| **Dataset 2: AirPassengers** | Số hành khách hàng không theo tháng (1949–1960, 144 mẫu) | Xu hướng tăng + Biên độ dao động mùa vụ tăng dần ($s = 12$) | `https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv` |
