# BÁO CÁO KỸ THUẬT — LAB 2: MACHINE LEARNING & XGBOOST
## Môn học: Quản lý & Khai phá Chuỗi Thời gian (QLCTG)
### Nhóm 7 — Trường Đại học Kinh tế TP.HCM (UEH)

---

## THÔNG TIN NHÓM

| Công việc | Thành viên phụ trách | Trạng thái |
|:----------|:---------------------|:----------:|
| CV01 — Tiền xử lý & Lag Features | Hà Duy Hưng | ✅ Hoàn thành |
| CV02 — Huấn luyện XGBoost Baseline | Khải | ✅ Hoàn thành |
| CV03 — Đánh giá & Feature Importance | Hoàng | ✅ Hoàn thành |
| CV04 — Rolling Features & Dự báo 24h | Trâm | ✅ Hoàn thành |
| CV05 — TimeSeriesSplit & Random Forest | Hiệp | ✅ Hoàn thành |
| CV06 — Tổng hợp & Báo cáo | Bùi Trần Ngọc Khải | ✅ Hoàn thành |

---

## 1. GIỚI THIỆU BÀI TOÁN & MỤC TIÊU

Bài lab 2 ứng dụng kỹ thuật **Machine Learning — XGBoost Regressor** vào bài toán **dự báo mức tiêu thụ điện năng theo giờ** cho lưới điện vùng AEP (American Electric Power, Mỹ) giai đoạn 2004–2018.

**Tập dữ liệu:** `AEP_hourly.csv` — 121.269 quan sát theo tần suất giờ.

**Mục tiêu chính:**
1. Chuyển bài toán chuỗi thời gian thành học có giám sát bằng lag features.
2. Huấn luyện và đánh giá XGBoost Regressor cơ sở.
3. Phân tích tầm quan trọng đặc trưng (Feature Importance).
4. Mở rộng bằng Rolling Features và dự báo đa bước (Multi-step).
5. Tinh chỉnh siêu tham số và so sánh với Random Forest.

---

## 2. MÔ TẢ DỮ LIỆU & TIỀN XỬ LÝ (CV01 — Hà Duy Hưng)

| Chỉ số | Giá trị |
|:-------|:-------:|
| Mốc thời gian bắt đầu | 2004-10-01 01:00:00 |
| Mốc thời gian kết thúc | 2018-08-03 00:00:00 |
| Tổng số quan sát (sau làm sạch) | 121.269 |
| Giá trị khuyết thiếu (NaN) | 0 |
| Mức tiêu thụ trung bình | 15.499,65 MW |
| Độ lệch chuẩn | 2.591,32 MW |
| Min / Max | 9.581 MW / 25.695 MW |

**Chuỗi Lag Features được tạo:** `lag_1`, `lag_2`, `lag_3`, `lag_24`, `lag_48`, `lag_168`

**Phân chia Train/Test (No Shuffle — Không rò rỉ dữ liệu):**
- Train (80%): 2004-10-08 → 2015-10-28 (96.880 quan sát)
- Test (20%): 2015-10-28 → 2018-08-03 (24.221 quan sát)

![Train/Test Split & Lag Correlation](../outputs/figures/cv01_raw_and_lags.png)

---

## 3. HUẤN LUYỆN MÔ HÌNH XGBOOST CƠ SỞ (CV02 — Khải)

**Siêu tham số cơ sở:** `n_estimators=1000`, `learning_rate=0.01`, `max_depth=5`, `early_stopping_rounds=50`, `objective='reg:squarederror'`

Mô hình hội tụ sớm nhờ early stopping, tránh overfitting.

![Learning Curve](../outputs/figures/cv02_learning_curve.png)

---

## 4. ĐÁNH GIÁ SAI SỐ & FEATURE IMPORTANCE (CV03 — Hoàng)

### 4.1. Kết quả đánh giá trên tập Test

| Chỉ số | Giá trị |
|:-------|:-------:|
| MAE | 161.62 MW |
| RMSE | 211.92 MW |
| MAPE | 1.1135 % |
| R² | 0.9925 |

### 4.2. Top 5 Feature Importance (XGBoost Gain)

| Hạng | Đặc trưng | Tầm quan trọng | Loại |
|:----:|:----------|:--------------:|:----:|
| 1 | lag_1 | 0.9187 | Lag |
| 2 | lag_3 | 0.0152 | Lag |
| 3 | hour | 0.0147 | Calendar |
| 4 | lag_24 | 0.0139 | Lag |
| 5 | lag_2 | 0.0113 | Lag |

> **Phát hiện chính:** `lag_1` chiếm tới **91.87%** tổng tầm quan trọng, thể hiện quán tính ngắn hạn cực kỳ mạnh của chuỗi điện năng.

![Actual vs Predicted](../outputs/figures/cv03_actual_vs_predicted.png)
![Feature Importance](../outputs/figures/cv03_feature_importance.png)

---

## 5. ROLLING FEATURES & DỰ BÁO ĐA BƯỚC 24H (CV04 — Trâm)

### 5.1. Tác động của Rolling Features

Thêm các đặc trưng cuộn (`rolling_mean_24`, `rolling_std_24`, `rolling_mean_168`, `rolling_std_168`) giúp mô hình nắm bắt xu hướng và biến động.

![Rolling Features Impact](../outputs/figures/cv04_rolling_features_impact.png)

### 5.2. Kết quả Dự báo Đa bước 24h

| Chiến lược | MAE (MW) | RMSE (MW) |
|:-----------|:--------:|:---------:|
| Direct | 568.33 | 772.73 |
| Recursive | 613.82 | 847.29 |
| SeasonalNaive 24h | 922.17 | 1.222,53 |
| SeasonalNaive 168h | 1.435,36 | 1.909,36 |

> **Nhận xét:** Cả Direct và Recursive đều vượt trội so với Naive baseline. Sai số tăng tuyến tính theo bước dự báo (error propagation).

![Multistep Forecast](../outputs/figures/cv04_multistep_forecast.png)

---

## 6. TINH CHỈNH TIMESERIESSPLIT & ĐỐI CHUẨN RANDOM FOREST (CV05 — Hiệp)

### 6.1. Siêu tham số tối ưu (GridSearchCV + TimeSeriesSplit 5-folds)

| Siêu tham số | Giá trị tối ưu |
|:-------------|:--------------:|
| learning_rate | 0.1 |
| max_depth | 7 |
| n_estimators | 500 |
| subsample | 0.8 |
| CV MAE tốt nhất | 120.49 MW |

### 6.2. Bảng so sánh XGBoost Tuned vs Random Forest

| Chỉ số | XGBoost (Tuned) | Random Forest |
|:-------|:--------------:|:-------------:|
| MAE (MW) | 106.98 | 129.47 |
| RMSE (MW) | 144.75 | 178.29 |
| MAPE (%) | 0.7362 | 0.8904 |
| R² | 0.9965 | 0.9947 |
| Thời gian Train (s) | 6.68 | 39.88 |
| Tốc độ dự báo (ms) | 88.25 | 694.64 |

> **Kết luận:** XGBoost Tuned **vượt trội Random Forest** trên mọi tiêu chí: sai số thấp hơn, train nhanh hơn **~6×**, dự báo nhanh hơn **~8×**.

![Benchmark Comparison](../outputs/figures/cv05_benchmark_comparison.png)

---

## 7. BẢNG SO SÁNH TỔNG HỢP TOÀN BỘ MÔ HÌNH

| Mô hình | MAE (MW) | RMSE (MW) | MAPE (%) | R² | Ghi chú |
|:--------|:--------:|:---------:|:--------:|:--:|:--------|
| XGBoost Baseline (CV02/CV03) | 161.62 | 211.92 | 1.1135 | 0.9925 | Default hyperparams, 6 lags |
| XGBoost (tuned) | 106.98 | 144.75 | 0.7362 | 0.9965 | TimeSeriesSplit 5-folds GridSearchCV |
| Random Forest | 129.47 | 178.29 | 0.8904 | 0.9947 | TimeSeriesSplit 5-folds GridSearchCV |
| XGBoost Recursive 24h (CV04) | 613.82 | 847.29 | 4.1603 | 0.8801 | Recursive multi-step 24h, rolling features |

---

## 8. TRẢ LỜI 3 CÂU HỎI LÝ LUẬN CỦA ĐỀ BÀI

### Câu 1: So sánh XGBoost và Linear Regression trong bài toán chuỗi thời gian

**XGBoost vượt trội Linear Regression vì 3 lý do cốt lõi:**

**1. Xử lý quan hệ phi tuyến và tương tác bậc cao:**
Linear Regression giả định quan hệ tuyến tính đơn thuần: $\hat{y} = \beta_0 + \beta_1 \cdot lag_1 + \cdots$. Trong thực tế, mức tiêu thụ điện phụ thuộc **phi tuyến** vào nhiệt độ, giờ trong ngày và ngày trong tuần đồng thời. Khi `lag_1` cao *và* `hour = 14` (2 giờ chiều hè), nhu cầu tăng vọt do điều hòa — đây là **tương tác bậc hai** mà Linear Regression không thể nắm bắt được. XGBoost xây dựng cây quyết định khai thác các phân chia dữ liệu phức tạp, tự động học các tương tác này.

**2. Hiện tượng đa cộng tuyến (Multicollinearity):**
Các lag liên tiếp (`lag_1`, `lag_2`, `lag_3`) có tương quan cực cao với nhau ($r > 0.82$). Với Linear Regression, điều này làm cho các hệ số $\beta$ bất ổn định và khó giải thích. XGBoost không bị ảnh hưởng bởi đa cộng tuyến vì tại mỗi nút phân chia, cây chỉ chọn 1 đặc trưng tốt nhất — các đặc trưng dư thừa tự nhiên bị bỏ qua.

**3. Khả năng ngoại suy xu hướng:**
Linear Regression có thể ngoại suy tuyến tính ra ngoài miền dữ liệu, nhưng khi xu hướng phi tuyến (ví dụ: tiêu thụ điện tăng đột biến do heatwave), kết quả bị sai lệch nghiêm trọng. XGBoost với cấu trúc cây không ngoại suy vượt ngoài phạm vi huấn luyện nhưng xử lý tốt hơn trong phạm vi đã học.

---

### Câu 2: Tác động khi tăng số lượng bậc trễ (lags p)

**Lợi ích khi tăng p:**
- Tiếp nhận tín hiệu dài hạn: `lag_24` ($r = 0.874$) nắm bắt chu kỳ ngày 24h; `lag_168` ($r = 0.760$) nắm bắt chu kỳ tuần 7 ngày. Nếu chỉ dùng `lag_1, lag_2, lag_3`, mô hình mù quáng trước thông tin mùa vụ.
- Cung cấp ngữ cảnh thời gian phong phú hơn cho mô hình.

**Rủi ro khi tăng p quá lớn:**
- **Curse of Dimensionality**: Tăng số chiều đặc trưng, cần nhiều dữ liệu hơn để học tốt.
- **Overfitting**: Với p rất lớn, mô hình có thể ghi nhớ các pattern nhiễu trong tập train.
- **Mất quan sát đầu chuỗi (NaN)**: Khi tạo `lag_p`, p hàng đầu tiên trở thành NaN và phải loại bỏ — giảm kích thước tập train hiệu dụng.
- **Đa cộng tuyến**: Các lag liên tiếp có tương quan cao, thông tin biên tế mà lag mới thêm vào ngày càng giảm.

**Thực nghiệm của nhóm:** Bộ lags `[1, 2, 3, 24, 48, 168]` được chọn vì cân bằng giữa hiệu suất dự báo ($R^2 = 0.9965$) và kích thước chiều không gian đặc trưng (13 features).

---

### Câu 3: Khi nào XGBoost phù hợp hơn ARIMA/SARIMA?

XGBoost phù hợp hơn trong các điều kiện:

| Tiêu chí | ARIMA/SARIMA | XGBoost |
|:---------|:------------:|:-------:|
| Chuỗi tuyến tính, ổn định | ✅ Tốt nhất | ⚠️ Overhead |
| **Chuỗi phi tuyến phức tạp** | ❌ Kém | ✅ **Vượt trội** |
| **Mùa vụ đa chu kỳ** (ngày + tuần + năm) | ⚠️ SARIMA chỉ 1 chu kỳ | ✅ Tự nhiên qua lags |
| **Nhiều biến ngoại sinh** (nhiệt độ, lịch) | ❌ Phức tạp (ARIMAX) | ✅ Feature-based |
| **Dữ liệu kích thước lớn** (>100k điểm) | ❌ Chậm, khó hội tụ | ✅ Hiệu quả |
| Khoảng tin cậy thống kê | ✅ Tự nhiên | ⚠️ Cần bootstrap |
| Diễn giải toán học | ✅ Rõ ràng | ⚠️ Black-box hơn |

**Kết luận cho bài toán AEP:** XGBoost là lựa chọn đúng đắn vì dữ liệu có **3 chu kỳ mùa vụ lồng nhau** (ngày 24h, tuần 168h, năm 8760h), **phi tuyến** (peak đôi mùa hè/đông), và kích thước **121.269 điểm** — tất cả đều là thế mạnh của XGBoost so với SARIMA.

---

## 9. KẾT LUẬN VÀ ĐÁNH GIÁ

### Tóm tắt kết quả

Nhóm 7 đã hoàn thành toàn bộ pipeline với kết quả nổi bật:

- **XGBoost Baseline**: MAE = 161.62 MW, $R^2$ = 0.9925 — vượt kỳ vọng ban đầu ngay ở bước baseline.
- **XGBoost Tuned** (CV05): MAE = 106.98 MW, $R^2$ = 0.9965 — cải thiện đáng kể so với baseline nhờ TimeSeriesSplit GridSearch.
- **XGBoost vượt trội Random Forest** về cả độ chính xác lẫn tốc độ (nhanh hơn 6× khi train, 8× khi dự báo).
- **Dự báo đa bước 24h** (CV04): MAE tích lũy = 613.82 MW — sai số tăng tuyến tính theo horizon, phù hợp lý thuyết error propagation.

### Tiềm năng cải thiện trong tương lai

1. **LSTM / Transformer**: Học pattern phức tạp dài hạn tốt hơn lag features tuyến tính.
2. **Biến ngoại sinh**: Tích hợp dữ liệu thời tiết (nhiệt độ, độ ẩm) có thể giảm thêm 20–30% sai số.
3. **Conformal Prediction**: Bổ sung khoảng tin cậy cho dự báo điểm.
4. **Ensemble**: Kết hợp XGBoost + SARIMA theo horizon (ngắn hạn vs dài hạn).

---

*Báo cáo được tạo tự động bởi `src/cv06_report_synthesis.py` — Nhóm 7, môn QLCTG, UEH.*
