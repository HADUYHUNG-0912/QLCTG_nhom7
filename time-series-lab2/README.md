# TIME SERIES LAB 2 - QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG)
## BÀI THỰC HÀNH SỐ 3: DỰ ĐOÁN CHUỖI THỜI GIAN SỬ DỤNG MACHINE LEARNING MODELS (XGBOOST REGRESSOR)

> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin - Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Nhóm thực hiện:** Nhóm 7 (6 thành viên)  
> **Bộ dữ liệu thực nghiệm:** [Hourly Energy Consumption (PJM) - AEP_hourly.csv](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv)  
> **GitHub Repository:** [https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git)

---

## 1. PHÂN CÔNG KHỐI LƯỢNG CÔNG VIỆC CHO 6 THÀNH VIÊN (WORKLOAD MATRIX)

Dự án được module hóa thành 6 script độc lập tương ứng với 6 thành viên (CV01 đến CV06), đảm bảo luồng dữ liệu thông suốt và tương thích tuyệt đối khi các thành viên sử dụng AI coding agents.

| Mã CV | Thành viên phụ trách | Nhánh Git gợi ý | Tệp thực thi | Đầu vào (Input Contract) | Đầu ra (Output Contract) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CV01** | **Hà Duy Hưng (Lead)** | `main` | [`src/cv01_data_and_lags.py`](src/cv01_data_and_lags.py) | `data/raw/AEP_hourly.csv` | `data/processed/train.csv`, `test.csv`<br>`outputs/figures/cv01_raw_and_lags.png`<br>`outputs/tables/cv01_data_summary.csv` |
| **CV02** | **Khải** | `feat/cv02-khai` | [`src/cv02_xgboost_train.py`](src/cv02_xgboost_train.py) | `data/processed/train.csv`, `test.csv` | `outputs/models/xgboost_baseline.json`<br>`outputs/tables/cv02_train_history.csv`<br>`outputs/figures/cv02_learning_curve.png` |
| **CV03** | **Hoàng** | `feat/cv03-hoang` | [`src/cv03_evaluation_importance.py`](src/cv03_evaluation_importance.py) | `data/processed/test.csv`<br>`outputs/models/xgboost_baseline.json` | `outputs/tables/cv03_evaluation_metrics.csv`<br>`outputs/figures/cv03_actual_vs_predicted.png`<br>`outputs/figures/cv03_feature_importance.png` |
| **CV04** | **Trâm** | `feat/cv04-tram` | [`src/cv04_rolling_and_multistep.py`](src/cv04_rolling_and_multistep.py) | `data/processed/train.csv`, `test.csv` | `outputs/figures/cv04_rolling_features_impact.png`<br>`outputs/figures/cv04_multistep_forecast.png`<br>`outputs/tables/cv04_multistep_metrics.csv` |
| **CV05** | **Hiệp** | `feat/cv05-hiep` | [`src/cv05_tuning_random_forest.py`](src/cv05_tuning_random_forest.py) | `data/processed/train.csv`, `test.csv` | `outputs/models/rf_benchmark.joblib`<br>`outputs/tables/cv05_best_hyperparameters.csv`<br>`outputs/tables/cv05_benchmark_summary.csv`<br>`outputs/figures/cv05_benchmark_comparison.png` |
| **CV06** | **Phương** | `feat/cv06-phuong` | [`src/cv06_report_synthesis.py`](src/cv06_report_synthesis.py) | Tất cả bảng biểu trong `outputs/tables/`<br>Tất cả hình ảnh trong `outputs/figures/` | `docs/BAO_CAO_LAB3_NHOM7.md`<br>`outputs/tables/final_model_comparison.csv` |

---

## 2. NGUYÊN TẮC ĐỒNG BỘ DÀNH CHO THÀNH VIÊN VÀ AI AGENTS

Mọi AI coding agent (Antigravity, Cursor, GitHub Copilot, ChatGPT, Claude...) khi tham gia hỗ trợ các thành viên viết code phải tuân thủ nghiêm ngặt các quy chuẩn sau:

1. **Chuẩn hóa đường dẫn tương đối (Path Management):**
   ```python
   from pathlib import Path
   BASE_DIR = Path(__file__).resolve().parent
   PROJECT_ROOT = BASE_DIR.parent
   DATA_RAW = PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"
   DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
   MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
   FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
   TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
   ```
2. **Quy tắc Chống rò rỉ dữ liệu (Anti-Lookahead Bias / Anti-Data Leakage):**
   - **Tuyệt đối KHÔNG dùng `shuffle=True`** khi chia tập dữ liệu huấn luyện/kiểm thử. Luôn cắt theo trình tự thời gian đơn điệu ($80\%$ train đầu, $20\%$ test sau).
   - Khi tạo đặc trưng thống kê trượt (Rolling Mean, Rolling Std), luôn tính trên chuỗi đã trễ `.shift(1).rolling(w)`.
   - Khi Cross-Validation, **bắt buộc dùng `TimeSeriesSplit(n_splits=5)`**, không dùng K-Fold ngẫu nhiên.
3. **Môi trường Console Windows & Tính tái lập (Reproducibility):**
   - Đặt `sys.stdout.reconfigure(encoding="utf-8")` ở đầu mỗi file.
   - Luôn cố định `random_state=42` cho tất cả các mô hình học máy.
4. **Hợp đồng Đầu vào/Đầu ra (I/O Contracts):**
   - Mỗi thành viên xuất dữ liệu đúng định dạng đã thỏa thuận trong bảng trên để thành viên tiếp theo có thể nạp vào ngay lập tức mà không phát sinh lỗi lệch cột hay thiếu tệp.

---

## 3. CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT DIRECTORY TREE)

```text
time-series-lab2/
├── DE_BAI_LAB.md               # Bản đặc tả đề bài và hướng dẫn đầy đủ
├── README.md                   # Tài liệu phân công và điều phối dự án
├── requirements.txt            # Thư viện phụ thuộc (xgboost, scikit-learn, pandas,...)
│
├── data/
│   ├── raw/
│   │   └── AEP_hourly.csv      # Dữ liệu tiêu thụ điện năng gốc (American Electric Power)
│   └── processed/
│       ├── AEP_features.csv    # Dữ liệu trích xuất Lag & Calendar Features (CV01)
│       └── AEP_features_rolling.csv # Dữ liệu bổ sung Rolling Features (CV04)
│
├── docs/
│   ├── data_introduction.md    # Giới thiệu nguồn gốc & 4 đặc tính trực quan của dữ liệu AEP
│   ├── BAO_CAO_LAB3_NHOM7.md   # Báo cáo kỹ thuật tổng hợp & trả lời 3 câu hỏi lý luận (CV06)
│   ├── KE_HOACH_PHAN_CONG_NHOM7.md # Bảng phân công chi tiết 6 thành viên
│   └── DE_BAI_LAB.md           # Bản đặc tả đề bài và hướng dẫn đầy đủ
│
├── notebooks/                  # Notebooks Jupyter tương tác thực nghiệm
│
├── outputs/
│   ├── figures/                # Đồ thị trực quan hóa xuất ra (.png)
│   │   ├── eda_aep_hourly.png
│   │   ├── training_loss_curve.png
│   │   ├── actual_vs_predicted_xgboost.png
│   │   ├── feature_importance_xgboost.png
│   │   ├── rolling_features_plot.png
│   │   ├── multistep_forecast_24h.png
│   │   └── model_comparison_bar.png
│   ├── models/                 # Trọng số mô hình đã lưu
│   │   ├── xgboost_baseline.json
│   │   └── rf_benchmark.joblib
│   └── tables/                 # Bảng số liệu và metrics xuất ra (.csv)
│       ├── data_summary.csv
│       ├── train_test_split_info.csv
│       ├── baseline_metrics.csv
│       ├── feature_importance_ranking.csv
│       ├── multistep_metrics.csv
│       ├── tuning_best_params.csv
│       ├── xgboost_vs_rf_comparison.csv
│       └── summary_all_experiments.csv
│
└── src/                        # Mã nguồn 6 script thực thi của 6 thành viên
    ├── cv01_data_prep_and_lags.py
    ├── cv02_xgboost_train.py
    ├── cv03_forecast_eval_importance.py
    ├── cv04_rolling_and_multistep.py
    ├── cv05_tuning_random_forest.py
    └── cv06_report_synthesis.py
```

---

## 4. HƯỚNG DẪN CÀI ĐẶT & THỨ TỰ THỰC THI (EXECUTION WORKFLOW)

### 4.1. Cài đặt môi trường
Khởi chạy terminal tại thư mục `time-series-lab2/`:

```bash
cd time-series-lab2

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 4.2. Luồng thực thi liên tục qua CLI
Thực thi các script theo thứ tự logic từ CV01 đến CV06:

```bash
# 1. Thành viên 1: Làm sạch dữ liệu và tạo Lag Features
python src/cv01_data_prep_and_lags.py

# 2. Thành viên 2: Chia Train/Test theo thời gian và huấn luyện XGBoost baseline
python src/cv02_xgboost_train.py

# 3. Thành viên 3: Dự báo tập test, tính MAE/RMSE/MAPE/R2 và phân tích Feature Importance
python src/cv03_forecast_eval_importance.py

# 4. Thành viên 4: Triển khai Rolling Mean/Std và thực nghiệm Dự báo Đa bước (Multi-step 24h)
python src/cv04_rolling_and_multistep.py

# 5. Thành viên 5: Tinh chỉnh TimeSeriesSplit và huấn luyện Random Forest đối sánh
python src/cv05_tuning_random_forest.py

# 6. Thành viên 6: Kiểm toán toàn diện, tổng hợp bảng số liệu và sinh báo cáo Markdown
python src/cv06_report_synthesis.py
```
