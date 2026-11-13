# PIPELINE QUY TRÌNH KỸ THUẬT: DỰ BÁO CHUỖI THỜI GIAN ĐA BIẾN (MULTIVARIATE FORECASTING)
## MÔN HỌC: PHÂN TÍCH DỮ LIỆU CHUỖI THỜI GIAN VÀ DỰ BÁO — LAB 4

> **Kho lưu trữ:** `time-series-lab4` (GitHub: `https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git`)  
> **Nhóm thực hiện:** Nhóm 7 (6 thành viên: Hà Duy Hưng (Lead), Hiệp, Phương, Châm, Hoàng, Khải)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Bảng tính tiến độ chính thức:** [Google Sheets - Nhóm 7](https://docs.google.com/spreadsheets/d/11OVm-W5E1aIS2YUACTVxofQplb5Q6KJHyQS_mGwKQRM/edit?gid=123070797#gid=123070797)  
> **Bộ dữ liệu thực nghiệm:** `energy_weather.csv` (26,304 quan sát giờ liên tục 2016–2018: Phụ tải điện `load` + `temp`, `humidity`, `wind_speed`)  
> **Mục tiêu cốt lõi:** Thiết kế hệ thống hai pha (**Training Pipeline** & **Inference Pipeline**) đối chuẩn 3 trường phái mô hình: **VAR** (Statistical Baseline), **XGBoost** (Machine Learning), và **LSTM** (Deep Learning) với kiểm thử **Walk-forward Backtesting**.

---

## 1. SƠ ĐỒ PIPELINE GIAI ĐOẠN HUẤN LUYỆN (TRAINING PIPELINE)

```text
====================================================================================================================
                      GIAI ĐOẠN 1: SƠ ĐỒ PIPELINE HUẤN LUYỆN (TRAINING PIPELINE ARCHITECTURE)
====================================================================================================================

 [ NGUỒN DỮ LIỆU THÔ ]
                      ┌──────────────────────────────────────────────────────────────────────┐
                      │ File: data/raw/energy_weather.csv (26,304 giờ | 2016 - 2018)         │
                      │ 4 Kênh biến: load (MW), temp (°C), humidity (%), wind_speed (m/s)    │
                      └───────────────────────────────────┬──────────────────────────────────┘
                                                          │
                                                          ▼
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 BƯỚC T1 & T2: KHỞI TẠO, AUDIT DỮ LIỆU, PHÂN CHIA 70/15/15 & FEATURE ENGINEERING
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MÃ CÔNG VIỆC : CV01 (ĐÃ HOÀN THÀNH 🟢)                                                                           │
│ THÀNH VIÊN   : Hà Duy Hưng (Lead)                                                                                │
│ TỆP THỰC THI : src/cv01_eda_and_preprocessing.py | notebooks/cv01_hung_eda_preprocessing.ipynb                   │
│ ĐẦU VÀO      : data/raw/energy_weather.csv                                                                       │
│ NHIỆM VỤ     : • Audit chất lượng: 0% Missing, 0 Duplicate, kiểm tra tính liên tục DatetimeIndex (freq='h')      │
│                • Thực hiện EDA đa biến: Mối quan hệ phi tuyến U-Curve (Load vs Temp), Seasonal 24h & 168h        │
│                • Phân chia TUẦN TỰ (No Shuffle): Train (70% - 18,412) | Val (15% - 3,945) | Test (15% - 3,947)   │
│                • Fit StandardScaler DUY NHẤT trên tập Train -> Transform độc lập (Chống Lookahead Bias)          │
│                • Khởi tạo ma trận 96 Lag Features (24 bước quá khứ x 4 kênh biến)                                │
└─────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┴────────────────────────────────────┐
                    │ ĐẦU RA (OUTPUT CONTRACTS - CV01):                                        │
                    │ • data/processed/train.csv, val.csv, test.csv (đã scale)                 │
                    │ • data/processed/train_lag.csv, val_lag.csv, test_lag.csv (96 Lags)      │
                    │ • data/processed/scaler.pkl (StandardScaler fit trên Train)              │
                    │ • outputs/figures/ (5 biểu đồ EDA) & outputs/tables/ (4 bảng audit)      │
                    └─────────────────────────────────────┬────────────────────────────────────┘
                                                          │
                  ┌───────────────────────────────────────┼───────────────────────────────────────┐
                  │                                       │                                       │
                  ▼                                       ▼                                       ▼
════════════════════════════════════    ════════════════════════════════════    ════════════════════════════════════
 BƯỚC T3A: STATISTICAL (VAR)             BƯỚC T3B: ML (XGBOOST)                  BƯỚC T3C: DEEP LEARNING (LSTM)     
════════════════════════════════════    ════════════════════════════════════    ════════════════════════════════════
┌──────────────────────────────────┐    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│ MÃ CÔNG VIỆC : CV02 (HOÀN THÀNH 🟢)│  │ MÃ CÔNG VIỆC : CV03 (HOÀN THÀNH 🟢)│  │ MÃ CÔNG VIỆC : CV04 (ĐANG LÀM 🟡) │
│ THÀNH VIÊN   : Hiệp              │    │ THÀNH VIÊN   : Phương            │    │ THÀNH VIÊN   : Châm              │
│ NHÁNH GIT    : main              │    │ NHÁNH GIT    : main              │    │ NHÁNH GIT    : cham-lab04        │
│ SCRIPT:                          │    │ SCRIPT:                          │    │ SCRIPT:                          │
│   src/cv02_var_model.py          │    │   src/cv03_xgboost_model.py      │    │   src/cv04_lstm_model.py         │
│ NOTEBOOK:                        │    │ NOTEBOOK:                        │    │ NOTEBOOK:                        │
│   cv02_hiep_var_model.ipynb      │    │   cv03_phuong_xgboost_model.ipynb│    │   cv04_cham_lstm_model.ipynb     │
│ INPUT  : train.csv, val.csv      │    │ INPUT  : train.csv, val.csv      │    │ INPUT  : train.csv, val.csv      │
│          test.csv, scaler.pkl    │    │          test.csv, scaler.pkl    │    │          test.csv, scaler.pkl    │
│ NHIỆM VỤ:                        │    │ NHIỆM VỤ:                        │    │ NHIỆM VỤ:                        │
│ • Kiểm định dừng ADF Test        │    │ • Flatten 96 Lags + Rolling:     │    │ • Tạo Tensor 3D (N, 24, 4)       │
│ • Sai phân bậc 1 nếu p > 0.05    │    │   Mean 6h, Std 24h, Trend 24h    │    │ • Kiến trúc LSTM(64)->Drop->     │
│ • Chọn lag p in [1..24] theo     │    │ • Fit XGBRegressor(seed=42)      │    │   LSTM(32)->Dense(1)             │
│   tiêu chuẩn tối thiểu AIC/BIC   │    │ • Early Stopping trên Val set    │    │ • Adam Optimizer & MSE Loss      │
│ • Huấn luyện VAR(p) đa biến      │    │ • Trích xuất Feature Import.     │    │ • ModelCheckpoint (val_loss)     │
└─────────────────┬────────────────┘    └─────────────────┬────────────────┘    └─────────────────┬────────────────┘
                  │                                       │                                       │
                  ▼                                       ▼                                       ▼
┌─────────────────┴────────────────┐    ┌─────────────────┴────────────────┐    ┌─────────────────┴────────────────┐
│ ĐẦU RA (OUTPUT - CV02):          │    │ ĐẦU RA (OUTPUT - CV03):          │    │ ĐẦU RA (OUTPUT - CV04):          │
│ • outputs/models/var_model.pkl   │    │ • outputs/models/                │    │ • outputs/models/                │
│ • outputs/tables/                │    │   xgboost_model.json             │    │   lstm_model.keras               │
│   cv02_var_metrics.csv           │    │ • outputs/tables/                │    │   lstm_model_best.h5             │
│   cv02_var_lag_selection.csv     │    │   cv03_xgb_metrics.csv           │    │ • outputs/tables/                │
│   cv02_adf_test_results.csv      │    │   cv03_xgb_feature_import.csv    │    │   cv04_lstm_metrics.csv          │
│ • outputs/figures/               │    │ • outputs/figures/               │    │ • outputs/figures/               │
│   cv02_aic_bic_lag_selection.png │    │   cv03_xgb_learning_curve.png    │    │   cv04_lstm_learning_curve.png   │
│   cv02_var_actual_vs_predicted   │    │   cv03_xgb_feature_import.png    │    │   cv04_lstm_actual_vs_pred.png   │
│   cv02_adf_results.png           │    │   cv03_xgb_forecast_vs_actual.png│    │                                  │
└─────────────────┬────────────────┘    └─────────────────┬────────────────┘    └─────────────────┬────────────────┘
                  │                                       │                                       │
                  └───────────────────────────────────────┴───────────────────────────────────────┘
                                                          │
                                                          ▼
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 BƯỚC T4: KHO ĐĂNG KÝ MÔ HÌNH SẴN SÀNG SUY LUẬN (MODEL ARTIFACT REGISTRY)
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┐
│ • Bộ chuẩn hóa tham chiếu : data/processed/scaler.pkl (mean, std của 4 biến từ Train set)                        │
│ • Bộ 3 Mô hình đã fit     : var_model.pkl (Thống kê) | xgboost_model.json (ML) | lstm_model_best.h5 (DL)         │
│ • Đánh giá Validation     : Đã kiểm soát Overfitting qua Early Stopping & Learning Curves                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
====================================================================================================================
```

---

## 2. SƠ ĐỒ PIPELINE GIAI ĐOẠN DỰ BÁO & BACKTESTING (INFERENCE PIPELINE)

```text
====================================================================================================================
                 GIAI ĐOẠN 2: SƠ ĐỒ PIPELINE DỰ BÁO & BACKTESTING (INFERENCE PIPELINE ARCHITECTURE)
====================================================================================================================

 [ DỮ LIỆU ĐẦU VÀO THỜI GIAN THỰC / WALK-FORWARD TEST STREAM ]
                    ┌──────────────────────────────────────────────────────────────────────────┐
                    │ Trạm đo cảm biến cập nhật mỗi giờ tại thời điểm t:                       │
                    │ Vector quan sát mới: X_t = [load_t, temp_t, humidity_t, wind_speed_t]    │
                    └─────────────────────────────────────┬────────────────────────────────────┘
                                                          │
                                                          ▼
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 BƯỚC I1 & I2: CỬA SỔ TRƯỢT FIFO (L = 24 GIỜ) & CHUẨN HÓA TRỰC TUYẾN (ONLINE SCALING)
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ • Quản lý hàng đợi FIFO   : Đẩy X_t vào cuối, loại bỏ X_{t-24} -> Duy trì ma trận 24 giờ [X_{t-23} ... X_t]      │
│ • Tiền xử lý trực tuyến   : Nạp data/processed/scaler.pkl từ Training Phase                                      │
│ • Áp dụng Transform       : X_scaled = scaler.transform(Buffer_24h) [TUYỆT ĐỐI KHÔNG GỌI fit()]                  │
└─────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┘
                                                          │
                  ┌───────────────────────────────────────┼───────────────────────────────────────┐
                  │                                       │                                       │
                  ▼                                       ▼                                       ▼
════════════════════════════════════    ════════════════════════════════════    ════════════════════════════════════
 BƯỚC I3A: SUY LUẬN VAR                  BƯỚC I3B: SUY LUẬN XGBOOST              BƯỚC I3C: SUY LUẬN LSTM            
════════════════════════════════════    ════════════════════════════════════    ════════════════════════════════════
┌──────────────────────────────────┐    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│ ĐỊNH DẠNG ĐẦU VÀO:               │    │ ĐỊNH DẠNG ĐẦU VÀO:               │    │ ĐỊNH DẠNG ĐẦU VÀO:               │
│ • Ma trận chuỗi (24 x 4)         │    │ • Làm phẳng 96 Lag Features      │    │ • Reshape thành Tensor 3D        │
│ • Sai phân bậc 1 (nếu cần)       │    │ • Tính nóng Rolling Mean 6h,     │    │   Shape = (1, 24, 4)             │
│                                  │    │   Std 24h & Trend 24h            │    │                                  │
│ MÔ HÌNH NẠP:                     │    │ MÔ HÌNH NẠP:                     │    │ MÔ HÌNH NẠP:                     │
│ • var_model.pkl                  │    │ • xgboost_model.json             │    │ • lstm_model_best.h5             │
│ • Bước dự báo 1-step-ahead       │    │ • Cây tăng cường độ dốc          │    │ • Truyền thẳng mạng hồi quy      │
│                                  │    │                                  │    │                                  │
│ KẾT QUẢ CHUẨN HÓA:               │    │ KẾT QUẢ CHUẨN HÓA:               │    │ KẾT QUẢ CHUẨN HÓA:               │
│ -> y_hat_{t+1, VAR}^scaled       │    │ -> y_hat_{t+1, XGB}^scaled       │    │ -> y_hat_{t+1, LSTM}^scaled      │
└─────────────────┬────────────────┘    └─────────────────┬────────────────┘    └─────────────────┬────────────────┘
                  │                                       │                                       │
                  └───────────────────────────────────────┴───────────────────────────────────────┘
                                                          │
                                                          ▼
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 BƯỚC I4: ĐẢO CHUẨN HÓA VỀ ĐƠN VỊ VẬT LÝ (INVERSE SCALING / DENORMALIZATION)
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┐
│ • Sử dụng tham số (mean_load, std_load) trích xuất từ data/processed/scaler.pkl                                  │
│ • Công thức khôi phục: y_hat_{t+1} (MW) = y_hat_{t+1}^scaled * std_load + mean_load                              │
│ • Giá trị xuất xưởng : Dự báo phụ tải điện năng (Megawatt) cho giờ tiếp theo (t + 1)                             │
└─────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 BƯỚC I5: KIỂM ĐỊNH WALK-FORWARD BACKTESTING & ĐỐI CHUẨN MÔ HÌNH
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MÃ CÔNG VIỆC : CV05 (SẴN SÀNG THỰC HIỆN ⏳)                                                                      │
│ THÀNH VIÊN   : Hoàng                                                                                             │
│ NHÁNH GIT    : hoang-lab04                                                                                       │
│ TỆP THỰC THI : src/cv05_backtesting_comparison.py | notebooks/cv05_hoang_backtesting.ipynb                       │
│ ĐẦU VÀO      : data/processed/test.csv (3,947 giờ), scaler.pkl, 3 mô hình đã huấn luyện                          │
│ NHIỆM VỤ     : • Vòng lặp Walk-forward: Dự báo t+1 -> Nhận thực tế y_{t+1} -> Cập nhật Buffer -> Lặp lại         │
│                • Tính định lượng 3 thước đo chuẩn trên thang đo gốc (MW): RMSE, MAE, MAPE (%)                    │
│                • Đo lường thời gian huấn luyện (TrainTime) và độ trễ suy luận (Inference Latency)                │
│                • Phân tích bản đồ nhiệt sai số theo từng khung giờ (0h -> 23h) phát hiện điểm mù mô hình         │
└─────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┘
                                                          │
              ┌───────────────────────────────────────────┴──────────────────────────────────────────┐
              │ ĐẦU RA (OUTPUT CONTRACTS - CV05):                                                    │
              │ • outputs/tables/cv05_final_comparison.csv (Bảng xếp hạng hiệu năng 3 mô hình)       │
              │ • outputs/figures/cv05_actual_vs_all_models.png (Đồ thị 4 đường: Actual vs Models)   │
              │ • outputs/figures/cv05_error_heatmap_by_hour.png (Phân tích lỗi theo giờ trong ngày)  │
              │ • outputs/figures/cv05_model_comparison_bar.png (Biểu đồ cột so sánh RMSE/MAE/MAPE)  │
              └───────────────────────────────────────────┬──────────────────────────────────────────┘
                                                          │
                                                          ▼
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 BƯỚC I6: TỔNG HỢP BÁO CÁO NGHIỆM THU & PHÂN TÍCH LÝ LUẬN CHUYÊN SÂU
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MÃ CÔNG VIỆC : CV06 (KẾ HOẠCH ⏳)                                                                                │
│ THÀNH VIÊN   : Khải                                                                                              │
│ NHÁNH GIT    : khai-lab04                                                                                        │
│ TỆP THỰC THI : src/cv06_report_synthesis.py | notebooks/cv06_khai_report_synthesis.ipynb                          │
│ ĐẦU VÀO      : Toàn bộ tệp dữ liệu trong outputs/tables/*.csv và biểu đồ trong outputs/figures/*.png             │
│ NHIỆM VỤ     : • Rà soát và đối chiếu số liệu RMSE, MAE, MAPE giữa 3 trường phái mô hình                         │
│                • Trả lời đầy đủ 6 câu hỏi phân tích lý luận chuyên sâu (Thời tiết, Thất bại VAR, ML vs DL)       │
│                • Đề xuất khuyến nghị triển khai thực tế và hướng phát triển Spatio-Temporal (METR-LA)            │
└─────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┴────────────────────────────────────┐
                    │ ĐẦU RA CUỐI CÙNG (FINAL DELIVERABLES - CV06):                            │
                    │ • docs/BAO_CAO_LAB4_NHOM7.md (Báo cáo kỹ thuật hoàn chỉnh)               │
                    │ • outputs/tables/cv06_final_model_comparison.csv                         │
                    │ • outputs/figures/cv06_summary_dashboard.png                             │
                    └──────────────────────────────────────────────────────────────────────────┘
====================================================================================================================
```

---

## 3. MA TRẬN PHÂN CÔNG & HỢP ĐỒNG ĐẦU VÀO / ĐẦU RA (I/O CONTRACT MATRIX)

> **Căn cứ phân công:** Khớp 100% với danh mục phân công trên [Google Sheets](https://docs.google.com/spreadsheets/d/11OVm-W5E1aIS2YUACTVxofQplb5Q6KJHyQS_mGwKQRM/edit?gid=123070797#gid=123070797).

| Pha | Mã CV | Thành viên | Nhánh Git | Vai trò phụ trách | Tệp thực thi / Notebook | Hợp đồng Đầu vào (Input) | Hợp đồng Đầu ra (Deliverables) | Trạng thái |
| :--- | :---: | :--- | :---: | :--- | :--- | :--- | :--- | :---: |
| **Training (T1-T2)** | **CV01** | **Hà Duy Hưng (Lead)** | `main` | Quản trị Pipeline, EDA, Scaling & 96 Lags | • [`src/cv01_eda_and_preprocessing.py`](../src/cv01_eda_and_preprocessing.py)<br>• [`notebooks/cv01_hung_eda_preprocessing.ipynb`](../notebooks/cv01_hung_eda_preprocessing.ipynb) | • `data/raw/energy_weather.csv` | • `data/processed/train.csv`, `val.csv`, `test.csv`<br>• `data/processed/train_lag.csv`, `val_lag.csv`, `test_lag.csv`<br>• `data/processed/scaler.pkl`<br>• 5 biểu đồ EDA & 4 bảng thống kê | 🟢 **Hoàn thành** |
| **Training (T3A)** | **CV02** | **Hiệp** | `main` | Statistical Baseline (**VAR Model**) | • [`src/cv02_var_model.py`](../src/cv02_var_model.py)<br>• [`notebooks/cv02_hiep_var_model.ipynb`](../notebooks/cv02_hiep_var_model.ipynb) | • `train.csv`, `val.csv`, `test.csv`<br>• `scaler.pkl` | • `outputs/models/var_model.pkl`<br>• `outputs/tables/cv02_var_metrics.csv`<br>• `outputs/tables/cv02_var_lag_selection.csv`<br>• `outputs/tables/cv02_adf_test_results.csv`<br>• `outputs/figures/cv02_aic_bic_lag_selection.png`<br>• `outputs/figures/cv02_var_actual_vs_predicted.png` | 🟢 **Hoàn thành** |
| **Training (T3B)** | **CV03** | **Phương** | `main` | Machine Learning (**XGBoost Regressor**) | • [`src/cv03_xgboost_model.py`](../src/cv03_xgboost_model.py)<br>• [`notebooks/cv03_phuong_xgboost_model.ipynb`](../notebooks/cv03_phuong_xgboost_model.ipynb) | • `train.csv`, `val.csv`, `test.csv`<br>• `scaler.pkl` | • `outputs/models/xgboost_model.json`<br>• `outputs/tables/cv03_xgb_metrics.csv`<br>• `outputs/tables/cv03_xgb_feature_importance.csv`<br>• `cv03_xgb_learning_curve.png`<br>• `cv03_xgb_feature_importance.png`<br>• `cv03_xgb_forecast_vs_actual.png` | 🟢 **Hoàn thành** |
| **Training (T3C)** | **CV04** | **Châm** | `cham-lab04` | Deep Learning (**LSTM Neural Net**) | • [`src/cv04_lstm_model.py`](../src/cv04_lstm_model.py)<br>• [`notebooks/cv04_cham_lstm_model.ipynb`](../notebooks/cv04_cham_lstm_model.ipynb) | • `train.csv`, `val.csv`, `test.csv`<br>• `scaler.pkl` | • `outputs/models/lstm_model.keras`<br>• `outputs/tables/cv04_lstm_metrics.csv`<br>• `cv04_lstm_learning_curve.png`<br>• `cv04_lstm_actual_vs_predicted.png` | 🟡 **Đang thực hiện** |
| **Inference (I1-I5)** | **CV05** | **Hoàng** | `hoang-lab04` | **Walk-forward Backtesting** & Đối chuẩn | • [`src/cv05_backtesting_comparison.py`](../src/cv05_backtesting_comparison.py)<br>• [`notebooks/cv05_hoang_backtesting.ipynb`](../notebooks/cv05_hoang_backtesting.ipynb) | • `test.csv`, `scaler.pkl`<br>• 3 Model Artifacts<br>• 3 bảng Metrics | • `outputs/tables/cv05_final_comparison.csv`<br>• `cv05_actual_vs_all_models.png`<br>• `cv05_error_heatmap_by_hour.png`<br>• `cv05_model_comparison_bar.png` | ⏳ **Sẵn sàng thực hiện** |
| **Inference (I6)** | **CV06** | **Khải** | `khai-lab04` | **Báo cáo tổng hợp** & 6 câu hỏi lý luận | • [`src/cv06_report_synthesis.py`](../src/cv06_report_synthesis.py)<br>• [`notebooks/cv06_khai_report_synthesis.ipynb`](../notebooks/cv06_khai_report_synthesis.ipynb) | • Tất cả `outputs/tables/*.csv`<br>• Tất cả `outputs/figures/*.png` | • `docs/BAO_CAO_LAB4_NHOM7.md`<br>• `outputs/tables/cv06_final_model_comparison.csv`<br>• `outputs/figures/cv06_summary_dashboard.png`<br>• Lời giải 6 câu hỏi chuyên sâu | ⏳ **Kế hoạch** |

---

## 4. CHI TIẾT CÁC BƯỚC TRONG HAI GIAI ĐOẠN (PHASE BREAKDOWN)

### 4.1. Giai đoạn 1: Training Phase (Huấn luyện & Lưu trữ)
1. **T1 — Data Ingestion & Quality Audit (CV01 - Hưng Lead):** Nạp dữ liệu 26,304 giờ liên tục, kiểm tra 0% missing, kiểm toán điểm dị biệt bằng IQR và phân tích bản chất phụ tải đỉnh hè.
2. **T2 — Temporal Split, Scaling & Feature Engineering (CV01 - Hưng Lead):**
   - Cắt tuần tự 70% Train (18,412 giờ), 15% Validation (3,945 giờ), 15% Test (3,947 giờ).
   - Fit `StandardScaler` duy nhất trên Train, đóng băng tham số $(\mu, \sigma)$ vào `scaler.pkl`, sau đó biến đổi Val và Test.
   - Tạo 96 Lag Features ($L=24 \times 4$ biến).
3. **T3 — Huấn luyện song song 3 trường phái mô hình:**
   - **T3A (CV02 - Hiệp):** Kiểm định tính dừng ADF, lấy sai phân bậc 1 vì `temp` không dừng ($p = 0.0663 > 0.05$), chọn bậc trễ tối ưu $p = 24$ theo AIC/BIC, huấn luyện mô hình VAR(24), lưu `var_model.pkl`.
   - **T3B (CV03 - Phương):** Kết hợp 96 Lag Features với các biến thống kê trượt (Rolling Mean 6h, Rolling Std 24h, Trend 24h) dùng `.shift(1)`, huấn luyện XGBoost Regressor có Early Stopping trên Validation, lưu `xgboost_model.json`.
   - **T3C (CV04 - Châm):** Tái cấu trúc thành Tensor 3D $(\text{Batch}, 24, 4)$, huấn luyện kiến trúc $\text{LSTM}(64) \rightarrow \text{Dropout}(0.2) \rightarrow \text{LSTM}(32) \rightarrow \text{Dense}(1)$ với hàm mất mát MSE, lưu trọng số tốt nhất tại `lstm_model.keras`.
4. **T4 — Model Registry:** Lưu trữ tập trung toàn bộ mô hình tại `outputs/models/` cùng lịch sử hội tụ (`learning_curve.png`).

### 4.2. Giai đoạn 2: Inference & Backtesting Phase (Suy luận & Đánh giá)
1. **I1 — FIFO Sliding Window Buffer:** Khi nhận mẫu quan sát mới $X_t$ tại giờ thứ $t$, đẩy vào hàng đợi 24 giờ gần nhất $[X_{t-23}, \dots, X_t]$ và loại bỏ mẫu cũ nhất $X_{t-24}$.
2. **I2 — Online Transformation:** Nạp `scaler.pkl`, biến đổi toàn bộ cửa sổ 24 giờ về không gian chuẩn hóa. Không bao giờ `fit()` lại trên tập dữ liệu mới.
3. **I3 — Dynamic Format Dispatching & Scoring:** Phân luồng dữ liệu cho 3 bộ máy suy luận:
   - Ma trận $(24 \times 4)$ $\rightarrow$ **VAR** $\rightarrow$ $\hat{y}_{t+1, \text{VAR}}^{\text{scaled}}$
   - Vector phẳng 99 Lags + Rolling $\rightarrow$ **XGBoost** $\rightarrow$ $\hat{y}_{t+1, \text{XGB}}^{\text{scaled}}$
   - Tensor 3D $(1, 24, 4)$ $\rightarrow$ **LSTM** $\rightarrow$ $\hat{y}_{t+1, \text{LSTM}}^{\text{scaled}}$
4. **I4 — Denormalization (Inverse Scaling):** Chuyển ngược dự báo về đơn vị Megawatt (MW): $\hat{y}_{t+1} = \hat{y}_{t+1}^{\text{scaled}} \times \sigma_{\text{load}} + \mu_{\text{load}}$.
5. **I5 — Walk-Forward Backtesting & Error Diagnostics (CV05 - Hoàng):** Đối chiếu giá trị dự báo với giá trị thực tế $y_{t+1}$, tính sai số định lượng (RMSE, MAE, MAPE), benchmark độ trễ suy luận (Inference Latency) và trực quan hóa Error Heatmap theo 24 giờ trong ngày.
6. **I6 — Knowledge & Report Synthesis (CV06 - Khải):** Xuất bản báo cáo hoàn chỉnh `docs/BAO_CAO_LAB4_NHOM7.md` cùng lời giải đáp cho 6 câu hỏi lý luận chuyên sâu.

---

## 5. THỨ TỰ THỰC THI LIÊN TỤC QUA CLI (EXECUTION COMMANDS)

Khởi chạy terminal tại thư mục `time-series-lab4/`:

```bash
# ── GIAI ĐOẠN 1: TRAINING PHASE ───────────────────────────────────────────────────
# 1. CV01 (Hưng Lead): EDA, chuẩn hóa StandardScaler, chia 70/15/15 và tạo 96 Lag Features [ĐÃ XONG 🟢]
python src/cv01_eda_and_preprocessing.py

# 2. CV02 (Hiệp): Kiểm định tính dừng ADF, chọn p theo AIC/BIC và huấn luyện VAR [ĐÃ XONG 🟢]
python src/cv02_var_model.py

# 3. CV03 (Phương): Tạo Rolling Features an toàn và huấn luyện XGBoost Regressor [ĐÃ XONG 🟢]
python src/cv03_xgboost_model.py

# 4. CV04 (Châm): Xây dựng Tensor 3D (N, 24, 4) và huấn luyện mạng LSTM [ĐANG THỰC HIỆN 🟡]
python src/cv04_lstm_model.py

# ── GIAI ĐOẠN 2: INFERENCE & EVALUATION PHASE ─────────────────────────────────────
# 5. CV05 (Hoàng): Chạy Walk-forward Backtesting trên Test set và so sánh 3 mô hình [SẴN SÀNG ⏳]
python src/cv05_backtesting_comparison.py

# 6. CV06 (Khải): Tổng hợp toàn bộ kết quả, biểu đồ và viết báo cáo nghiệm thu [KẾ HOẠCH ⏳]
python src/cv06_report_synthesis.py
```

---

## 6. NĂM NGUYÊN TẮC KỸ THUẬT CỐT LÕI (PIPELINE GOLDEN RULES)

1. **Bảo tồn trình tự thời gian tuyệt đối (No Shuffle):** Tuyệt đối không xáo trộn dữ liệu khi phân chia Train (70%), Validation (15%), Test (15%) hoặc khi tạo sequence cho LSTM.
2. **Chống rò rỉ dữ liệu chuẩn hóa (Strict Scaler Fit on Train):** `StandardScaler` chỉ được phép `fit()` trên tập Train. Tập Validation, Test và dữ liệu Streaming thực tế chỉ sử dụng `transform()` từ `scaler.pkl`.
3. **Ngăn chặn Lookahead Bias ở đặc trưng trượt:** Mọi đặc trưng thống kê trượt cho XGBoost (Rolling Mean 6h, Rolling Std 24h) bắt buộc phải áp dụng `.shift(1).rolling(w)` để đảm bảo chỉ tính trên dữ liệu quá khứ đã biết.
4. **Đảo chuẩn hóa trước khi đánh giá (Evaluate in Physical Unit):** Mọi thước đo RMSE, MAE, MAPE dùng để so sánh VAR, XGBoost và LSTM bắt buộc phải tính sau khi đã `inverse_transform` về đơn vị thực Megawatt (MW).
5. **Tôn trọng đặc tính vật lý của ngoại lai (Physical Outlier Awareness):** 2.89% điểm vượt ngưỡng IQR ở cột `load` phản ánh phụ tải đỉnh nắng nóng mùa hè (U-Curve với nhiệt độ), không phải lỗi cảm biến; không cắt bỏ thô bạo làm mất khả năng dự báo đỉnh phụ tải của mô hình.
