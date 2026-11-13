# BÀI TẬP THỰC HÀNH 2: KIỂM ĐỊNH ADF & CHUẨN BỊ MÔ HÌNH ARIMA/SARIMA

> **Học phần:** Phân tích dữ liệu chuỗi thời gian và dự báo  
> **Chủ đề:** Kiểm định tính dừng (ADF Test), Phân tích tương quan (ACF/PACF) và Xây dựng mô hình ARIMA / SARIMA  

---

## 🎯 MỤC TIÊU (OBJECTIVES)

1. Nhận diện chuỗi dừng / không dừng (Stationary vs Non-Stationary).
2. Thực hiện và diễn giải kiểm định nghiệm đơn vị ADF (Augmented Dickey-Fuller Test).
3. Xác định bậc sai phân $d$ thích hợp trong mô hình ARIMA.
4. Kết hợp kiểm định ADF với đồ thị tự tương quan (ACF) và tự tương quan riêng phần (PACF).
5. Chuẩn bị dữ liệu đúng cách theo trình tự thời gian cho mô hình ARIMA / SARIMA.
6. Đánh giá chất lượng và độ chính xác dự báo của mô hình.

---

## 📝 BÀI TẬP THỰC HÀNH (PRACTICE EXERCISES)

> **Dữ liệu thực nghiệm:** Sử dụng **Dataset 1** (Monthly Retail Sales) trong phần [DATASET](#-dataset) bên dưới.

---

### Bài 1: Khám phá dữ liệu (EDA)

* **Các bước thực hiện:**
  * Load dữ liệu từ nguồn Dataset 1.
  * Chuyển cột `date` thành `DatetimeIndex` với tần suất phù hợp.
  * Trực quan hóa chuỗi thời gian (Vẽ đồ thị đường).
* **Câu hỏi cần trả lời:**
  1. Chuỗi có xu hướng (Trend) rõ rệt không?
  2. Chuỗi có dấu hiệu mùa vụ (Seasonality) lặp lại chu kỳ không?
  3. Bằng quan sát trực quan ban đầu, chuỗi có khả năng dừng không?

---

### Bài 2: Kiểm định ADF trên chuỗi gốc

* **Các bước thực hiện:**
  * Thực hiện kiểm định ADF (Augmented Dickey-Fuller) cho chuỗi dữ liệu gốc.
  * Ghi nhận các giá trị:
    * Giá trị thống kê kiểm định: **ADF statistic**.
    * Giá trị xác suất: **$p$-value**.
    * Các giá trị ngưỡng tới hạn (Critical Values tại $1\%, 5\%, 10\%$).
  * Đưa ra kết luận: Chuỗi **Dừng** hay **Không dừng**?
* **Câu hỏi cần trả lời:**
  1. Giả thuyết vô hiệu ($H_0$) và giả thuyết đối ($H_1$) của kiểm định ADF là gì?
  2. Giá trị $p$-value phản ánh điều gì? Với mức ý nghĩa $\alpha = 0.05$, quy tắc bác bỏ $H_0$ như thế nào?
  3. Chuỗi dữ liệu gốc có thỏa mãn tính dừng không?

---

### Bài 3: Sai phân & Kiểm định lại

* **Các bước thực hiện:**
  * Lấy sai phân bậc 1 ($\Delta y_t = y_t - y_{t-1}$).
  * Vẽ lại đồ thị đường của chuỗi sau khi lấy sai phân bậc 1.
  * Thực hiện kiểm định ADF lần 2 trên chuỗi sau sai phân.
* **Câu hỏi cần trả lời:**
  1. Sau khi lấy sai phân bậc 1, chuỗi dữ liệu đã đạt tính dừng chưa ($p$-value $< 0.05$)?
  2. Nếu chuỗi đã dừng, giá trị tham số sai phân $d$ trong mô hình ARIMA là bao nhiêu?

---

### Bài 4: Phân tích ACF & PACF

* **Các bước thực hiện:**
  * Vẽ đồ thị hàm tự tương quan (**ACF**) và tự tương quan riêng phần (**PACF**) cho chuỗi đã đạt tính dừng.
  * Phân tích các bước trễ (lags) vượt ra ngoài khoảng tin cậy 95%.
  * Đề xuất các bộ tham số $(p, d, q)$ tiềm năng cho mô hình ARIMA.
* **Câu hỏi cần trả lời:**
  1. Đồ thị ACF cắt cụt (cut off) hay giảm dần (tail off) ở lag nào?
  2. Đồ thị PACF cắt cụt hay giảm dần ở lag nào?
  3. Gợi ý bộ tham số khởi tạo cho mô hình $\text{ARIMA}(p, d, q)$ là gì?

---

### Bài 5: Thiết lập ARIMA / SARIMA – Dự đoán – Đánh giá

1. **Phân chia tập dữ liệu:**
   * Chia dữ liệu thành: **Train set (80%)** và **Test set (20%)**.
   * **Bắt buộc:** Giữ nguyên thứ tự thời gian, tuyệt đối **không shuffle**.
2. **Thiết lập mô hình ARIMA:**
   * Dựa trên phân tích ACF/PACF từ Bài 4, đề xuất 1–2 cấu hình mô hình ARIMA ứng viên (Ví dụ: $\text{ARIMA}(1, 1, 1)$, $\text{ARIMA}(2, 1, 1)$, v.v.).
   * Huấn luyện (fit) mô hình trên tập Train.
   * Ghi nhận và so sánh tiêu chuẩn thông tin **AIC** (Akaike Information Criterion) và **BIC** (Bayesian Information Criterion) giữa các mô hình.
3. **Chẩn đoán phần dư (Residual Diagnostics):**
   * Vẽ biểu đồ phân phối phần dư (Residuals) theo thời gian.
   * Vẽ đồ thị ACF của phần dư.
   * Kiểm tra điều kiện:
     * Phần dư có kỳ vọng trung bình $\text{mean} \approx 0$ không?
     * Phần dư có còn tự tương quan không?
4. **Dự đoán (Forecasting):**
   * Thực hiện dự báo ngoài mẫu (Out-of-sample forecast) trên khoảng thời gian của tập Test.
   * Trực quan hóa đối chứng: Đồ thị kết hợp **Train (quá khứ)**, **Test (thực tế)** và **Forecast (dự báo)**.
5. **Đánh giá hiệu năng:**
   * Đo lường sai số dự báo bằng 3 chỉ số bắt buộc:
     * **MAE** (Mean Absolute Error)
     * **RMSE** (Root Mean Squared Error)
     * **MAPE** (Mean Absolute Percentage Error)

* **Câu hỏi cần trả lời:**
  1. Vì sao không được phép shuffle dữ liệu chuỗi thời gian khi chia tập huấn luyện/kiểm thử? Tập Test set đóng vai trò gì?
  2. Ý nghĩa thực tiễn của từng tham số $p, d, q$ đã chọn là gì?
  3. Quá trình huấn luyện mô hình có hội tụ không?
  4. Phần dư của mô hình có thỏa mãn là một chuỗi nhiễu trắng (White Noise) không?
  5. Nếu phần dư vẫn còn tự tương quan thì cần điều chỉnh gì ở cấu trúc mô hình?
  6. Đường dự báo có bám sát được xu hướng và chu kỳ dao động thực tế không?
  7. Sai lệch lớn nhất giữa giá trị dự báo và thực tế xảy ra tại thời điểm nào?
  8. Trong các chỉ số MAE, RMSE, MAPE, chỉ số nào nhạy cảm nhất với điểm dị biệt (outliers)? Vì sao?
  9. Độ lớn của giá trị RMSE nói lên điều gì về chất lượng dự báo của mô hình?

---

## 🚀 BÀI TẬP NÂNG CAO: MÔ HÌNH HÓA VỚI SARIMA

> **Dữ liệu thực nghiệm:** Sử dụng **Dataset 2** (AirPassengers) trong phần [DATASET](#-dataset) bên dưới.

* **Quy trình thực hiện:**
  1. Vẽ đồ thị chuỗi thời gian $\longrightarrow$ Nhận diện đặc tính xu hướng và tính mùa vụ (Seasonality).
  2. Kiểm định ADF trên chuỗi gốc $\longrightarrow$ Xác nhận chuỗi không dừng.
  3. Thực hiện kết hợp **sai phân thường** ($d$) và **sai phân mùa vụ** ($D$).
  4. Quan sát đồ thị ACF/PACF tại các lag mùa vụ $\longrightarrow$ Xác định bộ tham số $(p, d, q) \times (P, D, Q)_s$.
  5. Huấn luyện mô hình $\text{SARIMA}(p, d, q)(P, D, Q)_s$.
  6. Dự báo trên tập Test và đánh giá định lượng bằng MAE, RMSE, MAPE.
  7. So sánh hiệu năng đối chuẩn giữa mô hình **SARIMA** và mô hình **ARIMA tiêu chuẩn**.

* **Gợi ý thiết lập tham số mùa vụ:**
  * Chu kỳ mùa vụ: $s = 12$ (dữ liệu tháng).
  * Sai phân thường: $d = 1$ (triệt tiêu xu hướng - Trend).
  * Sai phân mùa vụ: $D = 1$ (triệt tiêu yếu tố mùa vụ lặp lại - Seasonality).

---

## 📊 DATASET

### Dataset 1: Monthly Retail Sales (Bán lẻ – Dành cho ARIMA / SARIMA)
* **Mô tả:** Doanh số bán lẻ theo tháng của thị trường Mỹ.
* **Đặc điểm:** Tồn tại đồng thời xu hướng tăng trưởng (Trend) và dao động mùa vụ (Seasonality).
* **Đường dẫn tải dữ liệu:**  
  `https://raw.githubusercontent.com/selva86/datasets/master/a10.csv`

### Dataset 2: AirPassengers (Monthly Airline Passenger Numbers)
* **Mô tả:** Số lượng hành khách hàng không quốc tế theo từng tháng từ năm 1949 đến năm 1960.
* **Quy mô:** 144 quan sát liên tục theo tháng.
* **Đặc tính chuỗi:**
  * Tần suất lấy mẫu: Theo tháng (Monthly, `freq='MS'`).
  * Chu kỳ mùa vụ: $s = 12$ tháng.
* **Đường dẫn tải dữ liệu:**  
  `https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv`

