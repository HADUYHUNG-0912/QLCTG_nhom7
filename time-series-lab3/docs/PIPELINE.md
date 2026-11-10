# PIPELINE QUY TRÌNH KỸ THUẬT: DỰ BÁO CHUỖI THỜI GIAN VỚI MACHINE LEARNING (XGBOOST)
## MÔN HỌC: PHÂN TÍCH DỮ LIỆU CHUỖI THỜI GIAN VÀ DỰ BÁO — LAB 3

> **Kho lưu trữ:** `time-series-lab3`  
> **Bộ dữ liệu thực nghiệm:** `AEP_hourly.csv` (American Electric Power - PJM)  
> **Mô hình trọng tâm:** XGBoost Regressor (`xgboost.XGBRegressor`) & Random Forest Benchmark  

---

## 1. SƠ ĐỒ PIPELINE TỔNG THỂ (ASCII ARCHITECTURE)

```text
========================================================================================================================
                          PIPELINE ARCHITECTURE: TIME SERIES FORECASTING WITH XGBOOST
========================================================================================================================

 [ NGUỒN DỮ LIỆU THÔ ]
 ┌────────────────────────────────────────────────────────┐
 │ File: data/raw/AEP_hourly.csv                          │
 │ 121,269 quan sát theo giờ (2004-2018), công suất MW   │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 GIAI ĐOẠN 1: TIỀN XỬ LÝ & KỸ NGHỆ ĐẶC TRƯNG (DATA PREPROCESSING & FEATURE ENGINEERING)
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MÃ CÔNG VIỆC : CV01                                                                                                │
 │ THÀNH VIÊN   : Hà Duy Hưng (Lead)                                                                                  │
 │ TỆP THỰC THI : src/cv01_data_and_lags.py                                                                            │
 │ ĐẦU VÀO      : data/raw/AEP_hourly.csv                                                                             │
 │ NHIỆM VỤ     : • Chuyển DatetimeIndex, sắp xếp tăng dần, kiểm tra trùng lặp & khuyết thiếu                         │
 │                • Tạo Lag Features: lag_1, lag_2, lag_3, lag_24, lag_48, lag_168                                    │
 │                • Tạo Calendar Features: hour, dayofweek, month                                                     │
 │                • Chia Train/Test (80% / 20%) theo trục thời gian (TUYỆT ĐỐI KHÔNG SHUFFLE)                         │
 └─────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                           │
                               ┌───────────────────────────┴───────────────────────────┐
                               │ ĐẦU RA (OUTPUT CONTRACTS):                            │
                               │ • data/processed/train.csv (96,880 dòng - 80%)        │
                               │ • data/processed/test.csv  (24,221 dòng - 20%)        │
                               │ • data/processed/AEP_features.csv                     │
                               │ • outputs/tables/cv01_data_summary.csv                │
                               │ • outputs/tables/train_test_split_info.csv            │
                               │ • outputs/figures/cv01_raw_and_lags.png               │
                               │ • outputs/figures/eda_aep_hourly.png                  │
                               └───────────────────────────┬───────────────────────────┘
                                                           │
                                                           ▼
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH CƠ SỞ (MODEL TRAINING PHASE - BASELINE)
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MÃ CÔNG VIỆC : CV02                                                                                                │
 │ THÀNH VIÊN   : Khải                                                                                                │
 │ TỆP THỰC THI : src/cv02_xgboost_train.py                                                                           │
 │ ĐẦU VÀO      : data/processed/train.csv, test.csv                                                                  │
 │ NHIỆM VỤ     : • Khởi tạo XGBRegressor (n_estimators=1000, lr=0.01, max_depth=5)                                   │
 │                • Cấu hình early_stopping_rounds=50, eval_metric="rmse"                                             │
 │                • Theo dõi tiến trình hội tụ loss train/val                                                         │
 │                • Đóng băng trọng số và lưu mô hình                                                                 │
 └─────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                           │
                               ┌───────────────────────────┴───────────────────────────┐
                               │ ĐẦU RA (OUTPUT CONTRACTS):                            │
                               │ • outputs/models/xgboost_baseline.json (Model freeze)  │
                               │ • outputs/tables/cv02_train_history.csv               │
                               │ • outputs/figures/cv02_learning_curve.png             │
                               └───────────────────────────┬───────────────────────────┘
                                                           │
                               ┌───────────────────────────┴───────────────────────────┐
                               │                                                       │
                               ▼                                                       ▼
 ═══════════════════════════════════════════════════   ════════════════════════════════════════════════════════════════
 GIAI ĐOẠN 3: SUY LUẬN & ĐÁNH GIÁ (INFERENCE 1-STEP)   GIAI ĐOẠN 4: ĐẶC TRƯNG TRƯỢT & DỰ BÁO ĐA BƯỚC (ROLLING & MULTI)
 ═══════════════════════════════════════════════════   ════════════════════════════════════════════════════════════════
 ┌─────────────────────────────────────────────────┐   ┌──────────────────────────────────────────────────────────────┐
 │ MÃ CÔNG VIỆC : CV03                             │   │ MÃ CÔNG VIỆC : CV04                                          │
 │ THÀNH VIÊN   : Hoàng                            │   │ THÀNH VIÊN   : Trâm                                          │
 │ TỆP THỰC THI : src/cv03_evaluation_importance.py│   │ TỆP THỰC THI : src/cv04_rolling_and_multistep.py             │
 │ ĐẦU VÀO      : data/processed/test.csv          │   │ ĐẦU VÀO      : data/processed/train.csv, test.csv            │
 │                outputs/models/xgboost_baseline  │   │ NHIỆM VỤ     : • Tính rolling_mean_24/168 & rolling_std      │
 │ NHIỆM VỤ     : • Load model cố định             │   │                  (trên chuỗi .shift(1) chống leakage)        │
 │                • Chạy predict(X_test)           │   │                • Thực nghiệm dự báo đệ quy đa bước 24 giờ    │
 │                • Tính MAE, RMSE, MAPE, R2       │   │                  (Recursive Multi-step: t+1 -> t+24)         │
 │                • Trích xuất Feature Importance  │   │                • Đánh giá sai số lan truyền theo từng bước   │
 └────────────────────────┬────────────────────────┘   └──────────────────────────────┬───────────────────────────────┘
                          │                                                           │
 ┌────────────────────────┴────────────────────────┐   ┌──────────────────────────────┴───────────────────────────────┐
 │ ĐẦU RA (OUTPUT CONTRACTS):                      │   │ ĐẦU RA (OUTPUT CONTRACTS):                                   │
 │ • outputs/tables/cv03_evaluation_metrics.csv    │   │ • data/processed/AEP_features_rolling.csv                    │
 │ • outputs/tables/feature_importance_ranking.csv │   │ • outputs/tables/cv04_multistep_metrics.csv                  │
 │ • outputs/figures/cv03_actual_vs_predicted.png  │   │ • outputs/tables/cv04_rolling_features_impact.csv            │
 │ • outputs/figures/cv03_feature_importance.png   │   │ • outputs/figures/cv04_multistep_forecast.png                │
 └────────────────────────┬────────────────────────┘   │ • outputs/figures/cv04_rolling_features_impact.png           │
                          │                            └──────────────────────────────┬───────────────────────────────┘
                          │                                                           │
                          └────────────────────────────┬──────────────────────────────┘
                                                       │
                                                       ▼
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 GIAI ĐOẠN 5: TỐI ƯU SIÊU THAM SỐ & ĐỐI CHUẨN RANDOM FOREST (TUNING & BENCHMARKING)
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MÃ CÔNG VIỆC : CV05                                                                                                │
 │ THÀNH VIÊN   : Hiệp                                                                                                │
 │ TỆP THỰC THI : src/cv05_tuning_random_forest.py                                                                    │
 │ ĐẦU VÀO      : data/processed/train.csv, test.csv                                                                  │
 │ NHIỆM VỤ     : • Cross-Validation chuỗi thời gian: TimeSeriesSplit(n_splits=5) (không shuffle)                     │
 │                • Tinh chỉnh siêu tham số XGBoost (max_depth, learning_rate, subsample, colsample_bytree)           │
 │                • Huấn luyện mô hình đối chuẩn RandomForestRegressor                                                │
 │                • So sánh thời gian huấn luyện, tốc độ suy luận, MAE/RMSE/MAPE giữa XGBoost vs Random Forest         │
 └─────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                           │
                               ┌───────────────────────────┴───────────────────────────┐
                               │ ĐẦU RA (OUTPUT CONTRACTS):                            │
                               │ • outputs/models/rf_benchmark.joblib                  │
                               │ • outputs/tables/cv05_best_hyperparameters.csv        │
                               │ • outputs/tables/cv05_benchmark_summary.csv           │
                               │ • outputs/figures/cv05_benchmark_comparison.png       │
                               └───────────────────────────┬───────────────────────────┘
                                                           │
                                                           ▼
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 GIAI ĐOẠN 6: TỔNG HỢP PIPELINE, KIỂM TOÁN & XUẤT BÁO CÁO (AUDIT, SYNTHESIS & REPORTING)
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MÃ CÔNG VIỆC : CV06                                                                                                │
 │ THÀNH VIÊN   : Phương                                                                                              │
 │ TỆP THỰC THI : src/cv06_report_synthesis.py                                                                        │
 │ ĐẦU VÀO      : Tất cả tệp trong outputs/tables/ và outputs/figures/ từ CV01 đến CV05                              │
 │ NHIỆM VỤ     : • Kiểm toán toàn vẹn dữ liệu End-to-End từ Raw Data -> Features -> Model -> Outputs                 │
 │                • Lập bảng tổng hợp so sánh đa mô hình (Baseline XGB vs Tuned XGB vs Random Forest)                  │
 │                • Giải đáp 3 câu hỏi lý luận khoa học (XGB vs Linear Reg, Lags Count Impact, XGB vs ARIMA)          │
 │                • Biên soạn Báo cáo nghiệm thu hoàn chỉnh                                                           │
 └─────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                           │
                               ┌───────────────────────────┴───────────────────────────┐
                               │ ĐẦU RA (OUTPUT CONTRACTS):                            │
                               │ • outputs/tables/final_model_comparison.csv           │
                               │ • Báo cáo kỹ thuật tổng kết nghiệm thu dự án          │
                               └───────────────────────────────────────────────────────┘
========================================================================================================================
```

---

## 2. MA TRẬN PHÂN CÔNG & QUY TRÌNH THỰC THI (WORKLOAD & I/O CONTRACT MATRIX)

| Giai đoạn | Mã CV | Thành viên | Script thực thi | Dữ liệu đầu vào (Input Contract) | Tệp kết quả đầu ra (Output Contract) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **1. Tiền xử lý & Kỹ nghệ Đặc trưng** | **CV01** | **Hà Duy Hưng (Lead)** | [`src/cv01_data_and_lags.py`](src/cv01_data_and_lags.py) | • `data/raw/AEP_hourly.csv` | • `data/processed/train.csv`<br>• `data/processed/test.csv`<br>• `data/processed/AEP_features.csv`<br>• `outputs/tables/cv01_data_summary.csv`<br>• `outputs/tables/train_test_split_info.csv`<br>• `outputs/figures/cv01_raw_and_lags.png`<br>• `outputs/figures/eda_aep_hourly.png` |
| **2. Huấn luyện Mô hình Cơ sở** | **CV02** | **Khải** | [`src/cv02_xgboost_train.py`](src/cv02_xgboost_train.py) | • `data/processed/train.csv`<br>• `data/processed/test.csv` | • `outputs/models/xgboost_baseline.json`<br>• `outputs/tables/cv02_train_history.csv`<br>• `outputs/figures/cv02_learning_curve.png` |
| **3. Đánh giá & Phân tích Đặc trưng** | **CV03** | **Hoàng** | [`src/cv03_evaluation_importance.py`](src/cv03_evaluation_importance.py) | • `data/processed/test.csv`<br>• `outputs/models/xgboost_baseline.json` | • `outputs/tables/cv03_evaluation_metrics.csv`<br>• `outputs/tables/feature_importance_ranking.csv`<br>• `outputs/figures/cv03_actual_vs_predicted.png`<br>• `outputs/figures/cv03_feature_importance.png` |
| **4. Đặc trưng Trượt & Dự báo Đa bước** | **CV04** | **Trâm** | [`src/cv04_rolling_and_multistep.py`](src/cv04_rolling_and_multistep.py) | • `data/processed/train.csv`<br>• `data/processed/test.csv` | • `data/processed/AEP_features_rolling.csv`<br>• `outputs/tables/cv04_multistep_metrics.csv`<br>• `outputs/tables/cv04_rolling_features_impact.csv`<br>• `outputs/figures/cv04_multistep_forecast.png`<br>• `outputs/figures/cv04_rolling_features_impact.png` |
| **5. Tinh chỉnh Siêu tham số & Đối chuẩn** | **CV05** | **Hiệp** | [`src/cv05_tuning_random_forest.py`](src/cv05_tuning_random_forest.py) | • `data/processed/train.csv`<br>• `data/processed/test.csv` | • `outputs/models/rf_benchmark.joblib`<br>• `outputs/tables/cv05_best_hyperparameters.csv`<br>• `outputs/tables/cv05_benchmark_summary.csv`<br>• `outputs/figures/cv05_benchmark_comparison.png` |
| **6. Tổng hợp & Báo cáo Kỹ thuật** | **CV06** | **Phương** | [`src/cv06_report_synthesis.py`](src/cv06_report_synthesis.py) | • Toàn bộ `outputs/tables/`<br>• Toàn bộ `outputs/figures/` | • `outputs/tables/final_model_comparison.csv`<br>• Báo cáo kỹ thuật tổng kết nghiệm thu hoàn chỉnh |

---

## 3. THỨ TỰ THỰC THI ĐỒNG BỘ TRÊN CLI (EXECUTION SEQUENCE)

Mỗi giai đoạn sau đọc dữ liệu trực tiếp từ kết quả của giai đoạn trước theo đúng thứ tự:

```bash
# Giai đoạn 1: Tiền xử lý, tạo Lag features và chia tập dữ liệu
python src/cv01_data_and_lags.py

# Giai đoạn 2: Khớp mô hình XGBoost baseline và đóng băng tham số
python src/cv02_xgboost_train.py

# Giai đoạn 3: Nạp mô hình đã lưu, đánh giá test và trích xuất độ quan trọng
python src/cv03_evaluation_importance.py

# Giai đoạn 4: Thử nghiệm Rolling Features và chạy suy luận đệ quy đa bước 24h
python src/cv04_rolling_and_multistep.py

# Giai đoạn 5: Tối ưu siêu tham số với TimeSeriesSplit và đối chuẩn Random Forest
python src/cv05_tuning_random_forest.py

# Giai đoạn 6: Kiểm toán hệ thống, tổng hợp bảng đối sánh và kết xuất báo cáo
python src/cv06_report_synthesis.py
```

---

## 4. BỐN NGUYÊN TẮC KỸ THUẬT BẮT BUỘC (PIPELINE GOLDEN RULES)

1. **Phân chia không xáo trộn (Strict Time-Order Split):** Cắt tập dữ liệu theo thứ tự $80\%$ quá khứ đầu để Train, $20\%$ tương lai sau để Test (`shuffle=False`), tuyệt đối không xáo trộn.
2. **Kỹ nghệ đặc trưng an toàn (Anti-Lookahead Bias):** Các biến thống kê trượt (Rolling Mean, Rolling Std) luôn được tính trên chuỗi đã dịch trễ (`.shift(1).rolling(w)`) để không rò rỉ thông tin của điểm dự báo $t$.
3. **Cross-Validation tuần tự (Temporal CV):** Trong pha tinh chỉnh (CV05), chỉ sử dụng `TimeSeriesSplit(n_splits=5)`, không dùng K-Fold ngẫu nhiên.
4. **Mô hình đóng băng độc lập (Decoupled Model Inference):** Mô hình sau khi huấn luyện được xuất ra file JSON/Joblib cố định; pha suy luận (Inference) nạp lại mô hình tĩnh và chạy dự báo mà không huấn luyện lại từ đầu.

