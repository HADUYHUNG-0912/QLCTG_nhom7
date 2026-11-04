# PRACTICE 4 – MULTIVARIATE TIME SERIES FORECASTING
## DỰ BÁO CHUỖI THỜI GIAN ĐA BIẾN: ĐIỆN NĂNG TIÊU THỤ & THỜI TIẾT (ELECTRICITY LOAD + WEATHER)

> **Môn học:** Quản lý & Khai phá Chuỗi Thời Gian (QLCTG)  
> **Chủ đề:** Multivariate Time Series Forecasting (Thống kê vs Machine Learning vs Deep Learning)  
> **Bộ dữ liệu chính:** Electricity Load Diagrams Dataset (UCI) kết hợp dữ liệu thời tiết (Weather Data)  
> **Bộ dữ liệu nâng cao:** METR-LA / PEMS-BAY (California Highway Sensors)  

---

## 🎯 1. MỤC TIÊU CỦA BÀI THỰC HÀNH (OBJECTIVES)

Trong bài thực hành này, sinh viên sẽ xây dựng, thực nghiệm và so sánh các mô hình dự báo chuỗi thời gian đa biến (Multivariate Time Series) dựa trên dữ liệu phụ tải điện năng và các yếu tố thời tiết ngoại sinh:
1. **Hiểu bản chất mô hình hóa chuỗi thời gian đa biến:** Cách tích hợp thông tin lịch sử của nhiều biến số phụ thuộc lẫn nhau vào quá trình dự báo.
2. **Triển khai và đối chuẩn 3 trường phái mô hình tiêu biểu:**
   * **VAR (Vector Autoregression):** Đại diện cho trường phái **Mô hình Thống kê cổ điển (Statistical Baseline)**.
   * **XGBoost Regressor:** Đại diện cho trường phái **Học máy (Machine Learning / Gradient Boosting)**.
   * **LSTM (Long Short-Term Memory):** Đại diện cho trường phái **Học sâu (Deep Learning / Recurrent Neural Networks)**.
3. **Đánh giá và so sánh định lượng chất lượng dự báo qua 3 chỉ số chuẩn:**
   * $\text{RMSE}$ (Root Mean Squared Error)
   * $\text{MAE}$ (Mean Absolute Error)
   * $\text{MAPE}$ (Mean Absolute Percentage Error)

---

## 📊 2. DATASET VÀ BÀI TOÁN DỰ BÁO (PROBLEM FORMULATION)

### 2.1. Các biến số quan sát
Sử dụng bộ dữ liệu **Electricity Load Diagrams Dataset (UCI)** tích hợp thông tin khí tượng, gồm 4 biến số:
* $\text{load}$: Lượng điện năng tiêu thụ (Biến mục tiêu chính cần dự báo).
* $\text{temperature}$: Nhiệt độ không khí.
* $\text{humidity}$: Độ ẩm không khí.
* $\text{wind\_speed}$: Tốc độ gió.

Tại mỗi bước thời gian $t$, vector quan sát đa biến được định nghĩa là:
$$X_t = [\text{load}_t, \text{temp}_t, \text{humidity}_t, \text{wind}_t]^T$$

### 2.2. Mục tiêu dự báo (Forecasting Horizon)
Sử dụng lịch sử của cả 4 biến số trong **24 giờ gần nhất** (cửa sổ quá khứ $L = 24$) để dự đoán lượng điện năng tiêu thụ ở bước tiếp theo ($t+1$):
$$X_{t-24:t} \Longrightarrow \text{load}_{t+1}$$

---

## 🔍 3. KHÁM PHÁ VÀ THẤU HIỂU DỮ LIỆU (DATA UNDERSTANDING)

Trước khi xây dựng mô hình, cần phân tích khám phá (EDA) để hiểu sâu sắc hành vi của dữ liệu, đặc biệt là mối tương quan và tác động qua lại giữa thời tiết và phụ tải điện:

* **Kiểm tra chất lượng dữ liệu:**
  * Giá trị khuyết thiếu (Missing values / NaNs).
  * Điểm dị biệt (Outliers / Anomalies).
  * Tính mùa vụ (Seasonality): Chu kỳ ngày (Daily 24h), chu kỳ tuần (Weekly 168h).
* **Trực quan hóa:**
  * Biểu đồ đường của `load` theo trục thời gian.
  * Biểu đồ phân tán (Scatter plot): `load` vs `temperature` (quan sát hình thái phi tuyến chữ U hoặc V).
  * Ma trận hệ số tương quan (Correlation Matrix Heatmap) giữa 4 biến: `load`, `temperature`, `humidity`, `wind_speed`.

---

## ⚙️ 4. TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)

### 4.1. Chuẩn hóa dữ liệu (Feature Scaling)
Tất cả 4 biến số phải được chuẩn hóa bằng **`StandardScaler`** (đưa về phân phối chuẩn có $\mu = 0, \sigma = 1$):
* **Lý do:**
  * Giúp mạng **LSTM** học ổn định, tránh hiện tượng bùng nổ hoặc tiêu biến đạo hàm (Exploding/Vanishing Gradients).
  * Loại bỏ thiên lệch (bias) do chênh lệch độ lớn và đơn vị đo (Megawatt vs $^\circ\text{C}$ vs $\%$ vs $\text{m/s}$).
  * Cải thiện tốc độ hội tụ và độ chính xác của **VAR** và **XGBoost**.

### 4.2. Phân chia tập dữ liệu theo trình tự thời gian (Temporal Split)
Dữ liệu chuỗi thời gian phải được cắt tuần tự đơn điệu theo thời gian, **tuyệt đối không shuffle**:
* **Tập Huấn luyện (Train set):** $70\%$ dữ liệu thời gian đầu.
* **Tập Xác thực (Validation set):** $15\%$ dữ liệu thời gian tiếp theo (dùng điều chỉnh siêu tham số và early stopping).
* **Tập Kiểm thử (Test set):** $15\%$ dữ liệu thời gian cuối cùng (dùng đánh giá và đối chuẩn cuối cùng).

---

## 🧬 5. TRÍCH XUẤT ĐẶC TRƯNG CHO ML VÀ DL (FEATURE ENGINEERING)

### 5.1. Cửa sổ trễ (Lag Window)
* Độ dài cửa sổ lịch sử: $L = 24$ (24 giờ trước).
* Mỗi mẫu quan sát tại thời điểm $t$ có ma trận/vector lịch sử gồm 96 đặc trưng trễ:
  $$[\text{load}_{t-1}, \dots, \text{load}_{t-24}, \; \text{temp}_{t-1}, \dots, \text{temp}_{t-24}, \; \text{humidity}_{t-1}, \dots, \text{humidity}_{t-24}, \; \text{wind}_{t-1}, \dots, \text{wind}_{t-24}]$$

### 5.2. Rolling Features (Đặc thù bổ trợ riêng cho XGBoost)
Cây quyết định trong XGBoost không tự duy trì trạng thái bộ nhớ như RNN, do đó cần trích xuất thêm các đặc trưng thống kê cuộn (lưu ý tính trên dữ liệu trễ để tránh Lookahead Bias):
* **Rolling Mean 6 giờ:** Trung bình trượt của phụ tải trong 6 giờ qua.
* **Rolling Standard Deviation 24 giờ:** Độ lệch chuẩn trượt đo lường mức độ biến động trong ngày.
* **Load Trend:** Độ biến thiên xu hướng ngày:
  $$\text{Trend}_t = \text{load}_t - \text{load}_{t-24}$$

---

## 🤖 6. MÔ HÌNH 1 – VECTOR AUTOREGRESSION (VAR - STATISTICAL BASELINE)

Mô hình thống kê đồng thời mô hình hóa mối quan hệ tuyến tính động giữa tất cả các chuỗi thời gian trong hệ thống:

1. **Kiểm tra tính dừng (Stationarity Test):**
   * Thực hiện kiểm định **ADF (Augmented Dickey-Fuller Test)** cho cả 4 chuỗi.
   * Nếu có chuỗi không dừng ($p\text{-value} > 0.05$) $\Longrightarrow$ thực hiện lấy sai phân bậc 1 (first differencing $\Delta X_t = X_t - X_{t-1}$).
2. **Lựa chọn bậc trễ tối ưu $p$:**
   * Thử nghiệm $p \in [1, 24]$, chọn $p$ tối ưu dựa trên tiêu chuẩn thông tin **AIC** (Akaike Information Criterion) hoặc **BIC** (Bayesian Information Criterion).
3. **Huấn luyện mô hình VAR:**
   $$X_t = \sum_{i=1}^p A_i X_{t-i} + \varepsilon_t$$
   * Mô hình VAR dự báo đồng thời toàn bộ vector $[\text{load}_{t+1}, \text{temp}_{t+1}, \text{humidity}_{t+1}, \text{wind}_{t+1}]$.
   * Khi đánh giá sai số, chỉ trích xuất cột $\text{load}$ (và đảo chuẩn hóa/tích lũy sai phân về lại thang đo gốc).

---

## 🌲 7. MÔ HÌNH 2 – XGBOOST REGRESSOR (MACHINE LEARNING)

Ứng dụng thuật toán tăng cường độ dốc (Gradient Boosted Decision Trees):

1. **Định dạng Input / Output:**
   * **Input ($X_t$):** Vector phẳng 1 chiều (Flat Vector) gồm 96 lag features ($24 \times 4$) kết hợp các rolling features.
   * **Output ($y_t$):** Giá trị vô hướng $\text{load}_{t+1}$.
2. **Chiến lược Huấn luyện & Tinh chỉnh:**
   * Điều chỉnh các siêu tham số chính: `max_depth`, `n_estimators`, `learning_rate`.
   * Sử dụng cơ chế **Early Stopping** trên tập Validation để chống overfitting.
   * Trích xuất **Feature Importance** để phân tích mức độ đóng góp của từng biến khí tượng so với các biến lag nội sinh của điện năng.

---

## 🧠 8. MÔ HÌNH 3 – LSTM (DEEP LEARNING)

Ứng dụng mạng nơ-ron hồi quy Long Short-Term Memory chuyên biệt cho dữ liệu dạng chuỗi:

1. **Định dạng dữ liệu đầu vào (3D Tensor):**
   $$\text{Shape} = (\text{batch\_samples}, \; \text{timesteps}=24, \; \text{features}=4)$$
2. **Kiến trúc mạng (Network Architecture):**
   * **Layer 1:** $\text{LSTM}(64 \text{ units}, \text{return\_sequences}=\text{True})$
   * **Layer 2:** $\text{LSTM}(32 \text{ units}, \text{return\_sequences}=\text{False})$
   * **Output Layer:** $\text{Dense}(1)$ (dự báo giá trị $\text{load}_{t+1}$)
3. **Cấu hình Huấn luyện:**
   * **Hàm mất mát (Loss):** $\text{MSE}$ (Mean Squared Error).
   * **Thuật toán tối ưu (Optimizer):** $\text{Adam}$.
   * Kết hợp kỹ thuật Early Stopping / Model Checkpoint theo dõi `val_loss`.

---

## 🔁 9. KIỂM THỬ THỰC TẾ: BACKTESTING (WALK-FORWARD VALIDATION)

Áp dụng quy trình kiểm thử cuộn (Walk-forward Validation / Rolling-origin Backtest) cho cả 3 mô hình trên tập Test:
$$\text{Train / Update} \longrightarrow \text{Predict } t+1 \longrightarrow \text{Thêm thực tế } t+1 \text{ vào chuỗi} \longrightarrow \text{Predict } t+2 \longrightarrow \dots$$
* Quy trình này mô phỏng chân thực cách hệ thống điều độ lưới điện vận hành trong thực tế: liên tục cập nhật dữ liệu mới nhất để đưa ra dự báo tiếp theo.

---

## 📈 10. ĐÁNH GIÁ VÀ ĐỐI CHUẨN CÁC MÔ HÌNH (MODEL EVALUATION)

Sử dụng 3 thước đo định lượng trên tập kiểm thử (đưa về đơn vị gốc trước khi tính):
1. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N (y_t - \hat{y}_t)^2}$$
2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{t=1}^N |y_t - \hat{y}_t|$$
3. **Mean Absolute Percentage Error (MAPE):**
   $$\text{MAPE} = \frac{1}{N} \sum_{t=1}^N \left| \frac{y_t - \hat{y}_t}{y_t} \right| \times 100\%$$

* **Bảng so sánh tổng hợp:** Lập bảng đối sánh trực diện hiệu năng (RMSE, MAE, MAPE, thời gian huấn luyện, thời gian dự báo) giữa:
  $$\text{VAR (Statistical)} \quad \text{vs} \quad \text{XGBoost (ML)} \quad \text{vs} \quad \text{LSTM (DL)}$$

---

## 💡 11. CÂU HỎI PHÂN TÍCH LÝ LUẬN (ANALYTICAL QUESTIONS)

Sau khi hoàn thành thực nghiệm, sinh viên phải trả lời 6 câu hỏi thảo luận chuyên sâu:
1. **Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng?** (Dựa trên ma trận tương quan, trọng số hồi quy VAR, và Feature Importance của XGBoost).
2. **Khi nào mô hình VAR thất bại?** (Xét về giả định tuyến tính, vấn đề bùng nổ tham số khi số chiều $K$ hoặc bậc trễ $p$ lớn, và chuỗi phi tuyến/phi chuẩn).
3. **XGBoost học được thông tin gì từ các biến thời tiết?** (Các ngưỡng phân chia nhiệt độ cực đoan mùa hè/mùa đông, tương tác phi tuyến giữa nhiệt độ và độ ẩm).
4. **Mô hình LSTM học được những chu kỳ nào?** (Khả năng ghi nhớ chu kỳ ngày 24h và chu kỳ tuần qua cổng quên/cổng nhớ so với các biến lag cố định).
5. **Mô hình LSTM có bị Overfitting không?** (Đánh giá qua khoảng cách giữa Train Loss và Validation Loss trên đồ thị Learning Curve, các biện pháp kiểm soát: Dropout, Early Stopping).
6. **Sự khác biệt cốt lõi giữa Machine Learning (XGBoost) và Deep Learning (LSTM) trong bài toán này?** (Yêu cầu Feature Engineering thủ công vs Biểu diễn không gian ẩn tuần tự End-to-End, chi phí tính toán, khả năng ngoại suy và tính diễn giải).

---

## 🚀 12. BÀI TẬP NÂNG CAO – DỰ BÁO LƯU LƯỢNG GIAO THÔNG (TRAFFIC FORECASTING)

Dành cho phần mở rộng nghiên cứu:
* **Bộ dữ liệu chuẩn:** **METR-LA** hoặc **PEMS-BAY** (Cảm biến đường cao tốc California).
  * Gồm tốc độ lưu thông phương tiện tại **207 trạm đo**.
  * Tần suất lấy mẫu: **5 phút / lần** trong nhiều tháng.
  * Vector trạng thái tại mỗi thời điểm: $X_t = [\text{speed}_1, \text{speed}_2, \dots, \text{speed}_{207}]^T$.
* **Bài toán dự báo đa bước không gian - thời gian (Spatio-Temporal):**
  $$[X_{t-12}, \dots, X_t] \Longrightarrow X_{t+3}, \; X_{t+6}, \; X_{t+12}$$
  *(Dự báo trước tương lai tại các mốc 15 phút, 30 phút, và 60 phút).*
* **Tài liệu tham khảo chuyên sâu:**
  * Bài báo gốc: *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (DCRNN)*.
  * Đóng vai trò là bài toán chuẩn benchmark quốc tế cho Deep Learning trên đồ thị chuỗi thời gian (Spatio-Temporal Graph Neural Networks).
