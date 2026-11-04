# BÀI TẬP THỰC HÀNH: DỰ ĐOÁN CHUỖI THỜI GIAN SỬ DỤNG MACHINE LEARNING MODELS
## (TIME SERIES FORECASTING USING MACHINE LEARNING MODELS)

> **Môn học:** Quản lý & Khai phá Chuỗi Thời Gian (QLCTG)  
> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin - Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên phụ trách:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Bộ dữ liệu thực nghiệm:** [Hourly Energy Consumption (PJM) - AEP_hourly.csv](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv)

---

## 🎯 MỤC TIÊU (OBJECTIVES)

- **Chuyển bài toán chuỗi thời gian $\rightarrow$ học có giám sát** (Transforming Time Series into Supervised Learning via Feature Engineering).
- **Huấn luyện mô hình XGBoost Regressor** (`xgboost.XGBRegressor`) chuyên sâu cho bài toán hồi quy chuỗi thời gian.
- **Đánh giá chất lượng dự báo** qua các chỉ số định lượng chuẩn mực (MAE, RMSE, MAPE, $R^2$) và trực quan hóa Actual vs Predicted.

---

## 📝 BÀI TẬP THỰC HÀNH CỐT LÕI (CORE WORKFLOW)

Cho một chuỗi thời gian đơn biến $y_t$, hãy dự đoán giá trị tại thời điểm tiếp theo dựa trên các giá trị trong quá khứ ($y_t$ hoặc $y_{t+1}$ từ $y_{t-1}, y_{t-2}, \dots, y_{t-p}$).  
Thực hiện tuần tự theo 7 bước dưới đây:

### Bước 1. Chuẩn bị dữ liệu (Data Ingestion & Preprocessing)
- Tải tệp dữ liệu `AEP_hourly.csv` từ bộ dữ liệu PJM Hourly Energy Consumption.
- Chuyển đổi cột mốc thời gian (`Datetime`) thành kiểu `pd.DatetimeIndex`.
- Sắp xếp tăng dần theo trục thời gian (`df.sort_index(inplace=True)`).
- Kiểm tra tính liên tục, kiểm tra giá trị khuyết thiếu (`isna().sum()`) và trùng lặp chỉ mục (`index.has_duplicates`).

### Bước 2. Tạo lag features (Supervised Learning Transformation)
- Biến đổi bài toán chuỗi thời gian thành bài toán học máy có giám sát bằng kỹ thuật trượt độ trễ:
  $$X_t = [y_{t-1}, y_{t-2}, \dots, y_{t-p}]$$
- Sử dụng hàm `.shift(1)`, `.shift(2)`, $\dots$, `.shift(p)` trong Pandas để tạo các cột độ trễ tương ứng.
- Xử lý các giá trị NaN phát sinh ở đầu chuỗi sau khi tạo lag features (`dropna()`).

### Bước 3. Chia ma trận đặc trưng và biến mục tiêu ($X - y$)
- **Ma trận đặc trưng ($X$):** Gồm các cột lag features $y_{t-1}, \dots, y_{t-p}$ (và các đặc trưng bổ sung nếu có).
- **Vector mục tiêu ($y$):** Giá trị thực tế cần dự đoán tại thời điểm hiện tại $y_t$ (hoặc bước tiếp theo $y_{t+1}$).

### Bước 4. Chia tập Train / Test theo thời gian (Time-Based Train/Test Split)
- **Quy tắc bắt buộc:** Chia theo thứ tự thời gian tuần tự (ví dụ: $80\%$ quan sát đầu dùng làm tập Huấn luyện - Train, $20\%$ quan sát sau dùng làm tập Kiểm thử - Test).
- **Cấm kỵ:** Tuyệt đối **KHÔNG dùng hàm ngẫu nhiên `train_test_split(shuffle=True)`** để tránh hiện tượng rò rỉ dữ liệu tương lai vào quá khứ (Data Leakage / Lookahead Bias).

### Bước 5. Huấn luyện mô hình XGBoost (Model Training)
- Khởi tạo mô hình `XGBRegressor` từ thư viện `xgboost`.
- Thiết lập các siêu tham số cơ bản ban đầu:
  ```python
  from xgboost import XGBRegressor

  model = XGBRegressor(
      n_estimators=1000,
      learning_rate=0.01,
      max_depth=5,
      early_stopping_rounds=50,
      objective='reg:squarederror',
      random_state=42
  )
  model.fit(
      X_train, y_train,
      eval_set=[(X_train, y_train), (X_test, y_test)],
      verbose=100
  )
  ```

### Bước 6. Dự đoán và đánh giá chất lượng (Forecasting & Evaluation)
- Dự báo trên tập kiểm thử: $\hat{y}_{\text{test}} = \operatorname{model.predict}(X_{\text{test}})$.
- Tính toán các chỉ số sai số đánh giá:
  - **MAE (Mean Absolute Error):**
    $$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|$$
  - **RMSE (Root Mean Squared Error):**
    $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$
  - **MAPE (Mean Absolute Percentage Error):**
    $$\text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right| \times 100\%$$
  - **Hệ số xác định ($R^2$ Score):** Đánh giá tỷ lệ phương sai mà mô hình giải thích được.
- Vẽ biểu đồ đối chứng trực quan giữa chuỗi thực tế ($y_{\text{test}}$) và chuỗi dự báo ($\hat{y}_{\text{test}}$) trên cùng một đồ thị theo thời gian.

### Bước 7. Phân tích Feature Importance (Độ quan trọng của đặc trưng)
- Trích xuất độ quan trọng của các độ trễ: `model.feature_importances_`.
- Vẽ biểu đồ cột trực quan hóa thứ hạng quan trọng của các lag (`plot_importance`).
- Phân tích và giải thích: **Lag nào có ảnh hưởng mạnh nhất đến giá trị dự báo?** (Ví dụ: Lag 1h trước, Lag 24h trước - chu kỳ 1 ngày, hoặc Lag 168h trước - chu kỳ 1 tuần).

---

## 🚀 BÀI TẬP NÂNG CAO (ADVANCED TASKS)

Sau khi hoàn thành 7 bước cơ bản, triển khai tiếp 4 nội dung nâng cao sau:

### 1. Thêm Rolling Mean & Rolling Std (Rolling Window Features)
- Tạo thêm các đặc trưng thống kê trượt để cung cấp mức nền xu hướng và mức độ biến động cục bộ:
  - `rolling_mean_24`: Trung bình trượt 24 giờ.
  - `rolling_std_24`: Độ lệch chuẩn trượt 24 giờ.
  - `rolling_mean_168`: Trung bình trượt 1 tuần (168 giờ).
- *Lưu ý kỹ thuật:* Luôn tính rolling trên các giá trị đã trễ (dùng `.shift(1).rolling(w)`) để tránh rò rỉ dữ liệu tại thời điểm dự báo $t$.

### 2. Dự báo Đa bước (Multi-Step Ahead Forecasting)
- Mở rộng bài toán từ dự báo 1 bước ($t+1$) sang dự báo nhiều bước trong tương lai ($t+1, t+2, \dots, t+24$):
  - **Chiến lược Đệ quy (Recursive Strategy):** Lấy giá trị dự báo $\hat{y}_{t+1}$ làm đầu vào để dự báo tiếp $\hat{y}_{t+2}$.
  - **Chiến lược Trực tiếp (Direct Strategy):** Huấn luyện $H$ mô hình độc lập, mỗi mô hình chuyên dự báo cho một bước trễ tương ứng $t+h$.
  - **Chiến lược Đa đầu ra (Multi-Output Regressor):** Sử dụng mô hình hỗ trợ xuất vector nhiều giá trị tương lai cùng lúc.

### 3. Tinh chỉnh Siêu tham số bằng `TimeSeriesSplit` (Hyperparameter Tuning)
- Sử dụng phương pháp Cross-Validation đặc thù cho chuỗi thời gian: `TimeSeriesSplit` từ `sklearn.model_selection` (mở rộng cửa sổ huấn luyện Rolling / Expanding Window, không xáo trộn).
- Kết hợp với `GridSearchCV` hoặc `RandomizedSearchCV` để tìm bộ siêu tham số tối ưu:
  - `max_depth` (3, 5, 7, 9)
  - `learning_rate` (0.01, 0.05, 0.1)
  - `n_estimators` (300, 500, 1000)
  - `subsample` và `colsample_bytree` (0.7, 0.8, 1.0)

### 4. So sánh Thực nghiệm với Random Forest (Benchmark Comparison)
- Huấn luyện song song mô hình `RandomForestRegressor` trên cùng tập đặc trưng $X$ và nhãn $y$.
- Lập bảng so sánh đối chứng giữa **XGBoost** và **Random Forest**:
  - Thời gian huấn luyện (Training Time).
  - Tốc độ suy luận (Inference Speed).
  - Độ chính xác sai số (MAE, RMSE, MAPE).
  - Khả năng tổng quát hóa trên tập kiểm thử (Overfitting Assessment).

---

## ❓ CÂU HỎI LÝ LUẬN CẦN TRẢ LỜI (THEORETICAL QUESTIONS)

Trình bày câu trả lời chi tiết và lập luận khoa học cho 3 câu hỏi dưới đây:

### Câu 1: So sánh XGBoost với Linear Regression trong bài toán dự báo chuỗi thời gian?
- *Gợi ý trả lời:*
  - Khả năng nắm bắt mối quan hệ phi tuyến tính phức tạp và tương tác bậc cao giữa các đặc trưng (XGBoost vượt trội nhờ cấu trúc cây quyết định phân nhánh).
  - Độ nhạy với hiện tượng đa cộng tuyến (Multicollinearity) giữa các lag kề cận ($y_{t-1}, y_{t-2}$ tương quan rất cao, Linear Regression dễ bị phình to phương sai hệ số, trong khi XGBoost chọn đặc trưng tốt nhất tại mỗi nút chia).
  - Khả năng ngoại suy xu hướng dài hạn (Linear Regression ngoại suy tuyến tính tốt ra ngoài miền giá trị, còn Decision Tree/XGBoost bị giới hạn trong khoảng $[y_{\min}, y_{\max}]$ của tập huấn luyện nếu không khử trend trước).

### Câu 2: Điều gì xảy ra khi tăng số lượng độ trễ (lags $p$)?
- *Gợi ý trả lời:*
  - **Lợi ích:** Mô hình tiếp nhận thêm thông tin quá khứ ở các chu kỳ xa hơn (ví dụ lag 24h, 168h), giúp nhận diện đầy đủ các nhịp tuần hoàn ngày và tuần.
  - **Rủi ro:** 
    - Hiện tượng quá khớp (Overfitting): Thêm quá nhiều lag không cần thiết tạo ra nhiễu thông tin.
    - Lời nguyền số chiều (Curse of Dimensionality) và tăng thời gian tính toán.
    - Mất mát dữ liệu đầu chuỗi: Cần bỏ $p$ dòng đầu tiên do chứa NaN khi tạo `.shift(p)`.

### Câu 3: Khi nào mô hình XGBoost phù hợp hơn mô hình truyền thống ARIMA/SARIMA?
- *Gợi ý trả lời:*
  - **Nên dùng XGBoost khi:**
    - Chuỗi thời gian có mối quan hệ phi tuyến tính mạnh, cấu trúc mùa vụ phức tạp nhiều chu kỳ đan xen (Intra-day, Weekly, Annual Seasonality).
    - Có nhiều biến ngoại sinh (Exogenous variables): Nhiệt độ, độ ẩm, ngày lễ, giá điện, biến lịch (Hour, DayOfWeek, Month).
    - Tập dữ liệu có kích thước lớn (hàng chục/hàng trăm nghìn quan sát như dữ liệu đo từng giờ của PJM).
  - **Nên dùng ARIMA khi:**
    - Chuỗi thời gian đơn biến tuyến tính, kích thước mẫu nhỏ hoặc vừa.
    - Cần mô hình có cơ sở thống kê suy diễn chặt chẽ, dễ diễn giải khoảng tin cậy giải tích, hoặc khi cần ngoại suy xu hướng tuyến tính thuần túy.

---

## 📊 THÔNG TIN BỘ DỮ LIỆU (DATASET SPECIFICATION)

- **Tên bộ dữ liệu:** **Hourly Energy Consumption** (PJM Interconnection LLC)
- **Mô tả:** Bộ dữ liệu ghi nhận mức tiêu thụ điện năng hàng giờ trong hơn 10 năm liên tục từ PJM Interconnection (tổ chức truyền tải điện khu vực phục vụ nhiều bang tại Hoa Kỳ), đơn vị đo lường tính bằng **Megawatt (MW)**.
- **Tệp dữ liệu thực nghiệm chỉ định:** `AEP_hourly.csv` (American Electric Power).
- **Đường dẫn tải trực tiếp:** [https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv)
- **Vị trí lưu trữ trong dự án:**
  ```text
  time-series-lab2/data/raw/AEP_hourly.csv
  ```

---

## 📋 CẤU TRÚC BÀI NỘP VÀ TIÊU CHÍ NGHIỆM THU

1. **Thư mục dự án:** [`time-series-lab2/`](../)
2. **Notebook thực thi:** `notebooks/bt_xgboost_energy.ipynb` (hoặc các notebook chia nhỏ theo từng bước).
3. **Mã nguồn script CLI:** `src/xgboost_forecasting.py`.
4. **Hình ảnh trực quan:** Lưu trong `outputs/figures/` (Actual vs Predicted, Feature Importance, Multi-step Forecast, Rolling Stats).
5. **Bảng kết quả:** Lưu trong `outputs/tables/` (Bảng so sánh MAE, RMSE, MAPE giữa XGBoost, Random Forest, Linear Regression).
6. **Báo cáo phân tích:** Trả lời đầy đủ 3 câu hỏi lý luận và tổng kết bài học kinh nghiệm.
