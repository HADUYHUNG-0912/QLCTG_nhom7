# KẾ HOẠCH THỰC HIỆN & PHÂN CÔNG NHIỆM VỤ — NHÓM 7
## BÀI THỰC HÀNH 4: MULTIVARIATE TIME SERIES FORECASTING
### Môn học: Quản lý & Khai phá Chuỗi Thời gian (QLCTG)

> **Nhóm trưởng:** Hà Duy Hưng (CV01 — Lead)  
> **Tổng số thành viên:** 6 người  
> **Thời gian thực hiện:** 05/11/2026 – 14/11/2026  
> **Kho mã nguồn:** `e:\Chuỗi_TG\time-series-lab3\`  

---

## 📋 1. TỔNG QUAN KẾ HOẠCH (TIMELINE OVERVIEW)

```
Tuần 1 (05/11 – 08/11)
│
├── [CV01] Hưng (Lead)     ───────────────┤ Deadline: 05/11 23:59
│   EDA + Preprocessing + Lag Features   │
│   → Xuất: train.csv, val.csv, test.csv │
│             scaler.pkl                 │
│
├── [CV02] Khải            ────────────────────────────────────┤ Deadline: 08/11 23:59
│   Mô hình VAR (Statistical Baseline)                        │
│   Phụ thuộc: Output của CV01                               │
│
└── [CV03] Hoàng           ────────────────────────────────────┤ Deadline: 08/11 23:59
    Mô hình XGBoost (Machine Learning)                        │
    Phụ thuộc: Output của CV01                               │

Tuần 2 (09/11 – 12/11)
│
├── [CV04] Trâm            ────────────────────────────────────┤ Deadline: 12/11 23:59
│   Mô hình LSTM (Deep Learning)                              │
│   Phụ thuộc: Output của CV01                               │
│
└── [CV05] Hiệp            ────────────────────────────────────┤ Deadline: 12/11 23:59
    Walk-forward Backtesting + So sánh 3 mô hình              │
    Phụ thuộc: Output của CV02, CV03, CV04                   │

Tuần 2 (13/11 – 14/11)
│
└── [CV06] Phương          ────────────────────────────────────┤ Deadline: 14/11 23:59
    Tổng hợp báo cáo + Trả lời 6 câu hỏi lý luận             │
    Phụ thuộc: Output của TẤT CẢ CV01–CV05                  │
```

---

## 👥 2. MA TRẬN PHÂN CÔNG CHI TIẾT (6 THÀNH VIÊN)

| Mã CV | Thành viên | Vai trò | File thực thi | Deadline |
| :---: | :--- | :--- | :--- | :---: |
| **CV01** | **Hà Duy Hưng (Lead)** | EDA, Preprocessing, Lag Features | [`src/cv01_eda_and_preprocessing.py`](src/cv01_eda_and_preprocessing.py) | **05/11** | 🟢 **Hoàn thành** |
| **CV02** | **Khải** | Mô hình VAR (Statistical Baseline) | [`src/cv02_var_model.py`](src/cv02_var_model.py) | **08/11** |
| **CV03** | **Hoàng** | Mô hình XGBoost (Machine Learning) | [`src/cv03_xgboost_model.py`](src/cv03_xgboost_model.py) | **08/11** |
| **CV04** | **Trâm** | Mô hình LSTM (Deep Learning) | [`src/cv04_lstm_model.py`](src/cv04_lstm_model.py) | **12/11** |
| **CV05** | **Hiệp** | Walk-forward Backtesting + So sánh | [`src/cv05_backtesting_comparison.py`](src/cv05_backtesting_comparison.py) | **12/11** |
| **CV06** | **Phương** | Tổng hợp báo cáo + Câu hỏi lý luận | [`src/cv06_report_synthesis.py`](src/cv06_report_synthesis.py) | **14/11** |

---

## 📦 3. HỢP ĐỒNG ĐẦU VÀO / ĐẦU RA (INPUT / OUTPUT CONTRACTS)

### CV01 — Hà Duy Hưng (Deadline: 05/11)

**Nhiệm vụ:** EDA (Exploratory Data Analysis), Tiền xử lý đa biến, Tạo Lag Features

| Hạng mục | Chi tiết |
| :--- | :--- |
| **Input** | `data/raw/` — Dữ liệu thô: file CSV chứa 4 cột `[datetime, load, temperature, humidity, wind_speed]` theo tần suất giờ |
| **Output Dữ liệu** | `data/processed/train.csv` — 70% đầu chuỗi (đã scale + lag features)<br>`data/processed/val.csv` — 15% tiếp theo<br>`data/processed/test.csv` — 15% cuối cùng<br>`data/processed/scaler.pkl` — StandardScaler đã fit trên train |
| **Output Biểu đồ** | `outputs/figures/cv01_load_timeseries.png`<br>`outputs/figures/cv01_scatter_load_vs_temp.png`<br>`outputs/figures/cv01_correlation_heatmap.png`<br>`outputs/figures/cv01_seasonality_daily_weekly.png` |
| **Output Bảng** | `outputs/tables/cv01_data_quality_report.csv`<br>`outputs/tables/cv01_split_info.csv` |
| **Yêu cầu kỹ thuật** | ✅ StandardScaler fit ONLY trên train, transform val & test<br>✅ Split 70/15/15 TUẦN TỰ theo thời gian, KHÔNG shuffle<br>✅ Lag features: 24 lags × 4 biến = 96 features tổng cộng<br>✅ dropna() sau khi tạo lags |

---

### CV02 — Khải (Deadline: 08/11)

**Nhiệm vụ:** Xây dựng mô hình VAR (Vector Autoregression — Statistical Baseline)

| Hạng mục | Chi tiết |
| :--- | :--- |
| **Input** | `data/processed/train.csv`, `val.csv`, `test.csv`, `scaler.pkl` |
| **Output Mô hình** | `outputs/models/var_model.pkl` — Mô hình VAR đã fit |
| **Output Biểu đồ** | `outputs/figures/cv02_adf_results.png`<br>`outputs/figures/cv02_aic_bic_lag_selection.png`<br>`outputs/figures/cv02_var_actual_vs_predicted.png` |
| **Output Bảng** | `outputs/tables/cv02_adf_test_results.csv`<br>`outputs/tables/cv02_var_metrics.csv` — RMSE, MAE, MAPE |
| **Yêu cầu kỹ thuật** | ✅ Kiểm định ADF cho cả 4 chuỗi<br>✅ Lấy sai phân bậc 1 nếu p-value > 0.05<br>✅ Chọn p tối ưu qua AIC/BIC (thử p = 1..24)<br>✅ Inverse transform kết quả về đơn vị gốc trước khi tính metric |

---

### CV03 — Hoàng (Deadline: 08/11)

**Nhiệm vụ:** Xây dựng mô hình XGBoost Regressor (Machine Learning)

| Hạng mục | Chi tiết |
| :--- | :--- |
| **Input** | `data/processed/train.csv`, `val.csv`, `test.csv`, `scaler.pkl` |
| **Output Mô hình** | `outputs/models/xgboost_model.json` — Trọng số XGBoost |
| **Output Biểu đồ** | `outputs/figures/cv03_xgb_learning_curve.png`<br>`outputs/figures/cv03_xgb_actual_vs_predicted.png`<br>`outputs/figures/cv03_xgb_feature_importance.png` |
| **Output Bảng** | `outputs/tables/cv03_xgb_metrics.csv` — RMSE, MAE, MAPE<br>`outputs/tables/cv03_xgb_feature_importance.csv` |
| **Yêu cầu kỹ thuật** | ✅ Input: vector phẳng 96 lag features + rolling_mean_6h + rolling_std_24h + load_trend<br>✅ Rolling features PHẢI dùng `.shift(1)` (tránh lookahead bias)<br>✅ Early stopping trên val set<br>✅ Trích xuất và visualize Feature Importance (Gain) |

---

### CV04 — Trâm (Deadline: 12/11)

**Nhiệm vụ:** Xây dựng mô hình LSTM (Deep Learning)

| Hạng mục | Chi tiết |
| :--- | :--- |
| **Input** | `data/processed/train.csv`, `val.csv`, `test.csv`, `scaler.pkl` |
| **Output Mô hình** | `outputs/models/lstm_model.keras` — Mô hình LSTM đã train |
| **Output Biểu đồ** | `outputs/figures/cv04_lstm_learning_curve.png`<br>`outputs/figures/cv04_lstm_actual_vs_predicted.png` |
| **Output Bảng** | `outputs/tables/cv04_lstm_metrics.csv` — RMSE, MAE, MAPE |
| **Yêu cầu kỹ thuật** | ✅ Input tensor 3D: shape `(samples, 24, 4)` (timesteps=24, features=4)<br>✅ Kiến trúc: LSTM(64, return_seq=True) → LSTM(32) → Dense(1)<br>✅ EarlyStopping(patience=10, monitor='val_loss') + ModelCheckpoint<br>✅ Optimizer: Adam, Loss: MSE<br>✅ Inverse transform chỉ cột load trước khi tính metric |

---

### CV05 — Hiệp (Deadline: 12/11)

**Nhiệm vụ:** Walk-forward Backtesting & So sánh toàn diện 3 mô hình

| Hạng mục | Chi tiết |
| :--- | :--- |
| **Input** | Tất cả models: `var_model.pkl`, `xgboost_model.json`, `lstm_model.keras`<br>Dữ liệu: `test.csv`, `scaler.pkl` |
| **Output Biểu đồ** | `outputs/figures/cv05_actual_vs_all_models.png` — Overlay 4 đường (Actual + 3 models)<br>`outputs/figures/cv05_error_heatmap_by_hour.png` — Heatmap MAE theo giờ trong ngày<br>`outputs/figures/cv05_model_comparison_bar.png` — Bar chart RMSE/MAE/MAPE |
| **Output Bảng** | `outputs/tables/cv05_final_comparison.csv` — Bảng tổng hợp đầy đủ (RMSE, MAE, MAPE, Train time, Inference time) |
| **Yêu cầu kỹ thuật** | ✅ Walk-forward: cuộn từng bước t+1, cập nhật window thực tế vào lịch sử<br>✅ Đo thời gian train và inference cho cả 3 mô hình |

---

### CV06 — Phương (Deadline: 14/11)

**Nhiệm vụ:** Tổng hợp báo cáo kỹ thuật & Trả lời 6 câu hỏi lý luận

| Hạng mục | Chi tiết |
| :--- | :--- |
| **Input** | Tất cả CSV trong `outputs/tables/`, tất cả PNG trong `outputs/figures/` |
| **Output** | `docs/BAO_CAO_LAB4_NHOM7.md` — Báo cáo kỹ thuật hoàn chỉnh<br>`outputs/tables/cv06_final_model_comparison.csv` |
| **6 Câu hỏi lý luận cần trả lời:** | |
| Q1 | Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng? |
| Q2 | Khi nào mô hình VAR thất bại? |
| Q3 | XGBoost học được thông tin gì từ các biến thời tiết? |
| Q4 | LSTM học được những chu kỳ nào (ngày, tuần)? |
| Q5 | LSTM có bị Overfitting không? Biện pháp kiểm soát? |
| Q6 | Sự khác biệt cốt lõi giữa XGBoost (ML) và LSTM (DL)? |

---

## 🔄 4. SƠ ĐỒ LUỒNG DỮ LIỆU (PIPELINE DAG)

```
data/raw/
  └── [CSV: load, temp, humidity, wind_speed]
                          │
                          ▼
              ┌─────────────────────┐
              │  CV01 — cv01_eda_   │  Deadline: 05/11
              │  and_preprocessing  │
              └─────────┬───────────┘
                        │ data/processed/train.csv
                        │ data/processed/val.csv
                        │ data/processed/test.csv
                        │ data/processed/scaler.pkl
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
  ┌──────────────┐ ┌──────────┐ ┌──────────┐
  │ CV02 — VAR   │ │ CV03 —   │ │ CV04 —   │
  │  Statistical │ │ XGBoost  │ │  LSTM    │
  │  Baseline    │ │    ML    │ │    DL    │
  │  Deadline:   │ │Deadline: │ │Deadline: │
  │  08/11       │ │ 08/11    │ │ 12/11    │
  └──────┬───────┘ └────┬─────┘ └────┬─────┘
         │              │             │
         └──────────────┼─────────────┘
                        ▼
              ┌─────────────────────┐
              │  CV05 — Backtesting │  Deadline: 12/11
              │   & Comparison      │
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  CV06 — Synthesis   │  Deadline: 14/11
              │  Report + Q&A       │
              └─────────────────────┘
                        │
                        ▼
              docs/BAO_CAO_LAB4_NHOM7.md
```

---

## 🧱 5. NGUYÊN TẮC KỸ THUẬT BẮT BUỘC (TECHNICAL GUARDRAILS)

> [!IMPORTANT]
> Tất cả thành viên và AI Agent hỗ trợ PHẢI tuân thủ các quy tắc sau. Vi phạm sẽ gây lỗi cascading cho module sau.

### 5.1. Đường dẫn chuẩn (Path Management)
```python
from pathlib import Path
BASE_DIR     = Path(__file__).resolve().parent        # .../src/
PROJECT_ROOT = BASE_DIR.parent                         # .../time-series-lab3/
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR   = PROJECT_ROOT / "outputs" / "tables"
DOCS_DIR     = PROJECT_ROOT / "docs"
```

### 5.2. Chống rò rỉ dữ liệu (Anti-Lookahead Bias)
- ❌ **TUYỆT ĐỐI KHÔNG** dùng `shuffle=True` khi chia dữ liệu
- ❌ **TUYỆT ĐỐI KHÔNG** fit scaler trên toàn bộ dữ liệu — chỉ fit trên train, transform val & test
- ✅ Rolling features PHẢI dùng: `.shift(1).rolling(window=w).mean()`
- ✅ Cross-validation PHẢI dùng `TimeSeriesSplit` nếu cần

### 5.3. Tái lập kết quả (Reproducibility)
```python
import random, numpy as np, tensorflow as tf
random.seed(42); np.random.seed(42); tf.random.set_seed(42)
# XGBoost: random_state=42
# sklearn: random_state=42
```

### 5.4. Encoding Console (Windows)
```python
import sys
sys.stdout.reconfigure(encoding="utf-8")
```

### 5.5. Chuẩn hóa đầu ra
- Biểu đồ: DPI=150, bbox_inches='tight', font hỗ trợ Unicode (`DejaVu Sans` hoặc `Segoe UI`)
- Bảng: UTF-8 encoding, `index=False` khi ghi CSV
- Metrics: Inverse transform về đơn vị gốc TRƯỚC khi tính RMSE/MAE/MAPE

---

## 📅 6. CHI TIẾT DEADLINE VÀ MILESTONE REVIEW

| Milestone | Ngày | Mô tả | Người review |
| :--- | :---: | :--- | :--- |
| **M0 — Kickoff** | 01/11 | Toàn nhóm đọc đề bài, cài môi trường, tải dataset | Hưng (Lead) |
| **M1 — Data Ready** | 05/11 | CV01 hoàn thành: data/processed/ sẵn sàng đầy đủ | Hưng review |
| **M2 — Models Done** | 08/11 | CV02 (VAR) + CV03 (XGBoost) hoàn thành, metrics CSV có mặt | Hưng review |
| **M3 — DL + Backtest** | 12/11 | CV04 (LSTM) + CV05 (Backtesting) hoàn thành | Hưng review |
| **M4 — Final Report** | 14/11 | CV06 hoàn thành báo cáo, nộp GV | Cả nhóm review |

---

## 🤖 7. HƯỚNG DẪN DÙNG AI AGENT CHO TỪNG THÀNH VIÊN

Copy prompt sau vào Antigravity IDE / Cursor / Claude Code và **đưa kèm file code tương ứng**:

### Prompt mẫu cho mỗi thành viên:
```
Tôi là [TÊN], phụ trách [CV0X] trong dự án time-series-lab3 (QLCTG Nhóm 7).

Mở file src/[cvXX_tên_file.py]. Đọc kỹ block AI_AGENT_INSTRUCTIONS ở đầu file.
Giúp tôi hoàn thiện các hàm trong file đó theo đúng yêu cầu đã mô tả.

Quy tắc KHÔNG được vi phạm:
1. Dùng đường dẫn qua PROJECT_ROOT (không hard-code path)
2. StandardScaler chỉ fit trên train, KHÔNG fit trên val/test
3. Chia dữ liệu KHÔNG shuffle, cắt tuần tự theo thời gian
4. Rolling features phải dùng .shift(1) trước .rolling()
5. Inverse transform về đơn vị gốc trước khi tính RMSE/MAE/MAPE
6. random_state=42 cho mọi mô hình
7. Script phải chạy được bằng: python src/[tên_file.py]
```

---

## 🗂️ 8. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
time-series-lab3/
│
├── DE_BAI_LAB.md                    # Đặc tả đề bài đầy đủ (Practice 4)
├── README.md                        # Tổng quan dự án (sẽ cập nhật)
├── requirements.txt                 # Thư viện cần cài: statsmodels, xgboost, tensorflow...
│
├── data/
│   ├── raw/                         # ← Đặt file CSV dữ liệu gốc vào đây
│   │   └── .gitkeep
│   └── processed/                   # ← CV01 xuất ra đây
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── scaler.pkl
│
├── docs/
│   ├── DE_BAI_LAB.md
│   └── BAO_CAO_LAB4_NHOM7.md       # ← CV06 xuất ra đây
│
├── notebooks/                       # Notebooks Jupyter thực nghiệm
│   └── .gitkeep
│
├── outputs/
│   ├── figures/                     # ← Tất cả biểu đồ PNG
│   ├── models/                      # ← Trọng số mô hình
│   └── tables/                      # ← Tất cả metrics CSV
│
├── src/                             # ← MÃ NGUỒN CHÍNH
│   ├── cv01_eda_and_preprocessing.py   (CV01 — Hưng)
│   ├── cv02_var_model.py               (CV02 — Khải)
│   ├── cv03_xgboost_model.py           (CV03 — Hoàng)
│   ├── cv04_lstm_model.py              (CV04 — Trâm)
│   ├── cv05_backtesting_comparison.py  (CV05 — Hiệp)
│   └── cv06_report_synthesis.py        (CV06 — Phương)
│
└── tests/
    └── .gitkeep
```

---

## ⚡ 9. THỨ TỰ THỰC THI (EXECUTION ORDER)

```bash
# Cài môi trường
cd time-series-lab3
pip install -r requirements.txt

# Chạy tuần tự (bắt buộc theo thứ tự)
python src/cv01_eda_and_preprocessing.py   # CV01 — phải chạy trước
python src/cv02_var_model.py               # CV02 — sau CV01
python src/cv03_xgboost_model.py           # CV03 — sau CV01
python src/cv04_lstm_model.py              # CV04 — sau CV01
python src/cv05_backtesting_comparison.py  # CV05 — sau CV02, CV03, CV04
python src/cv06_report_synthesis.py        # CV06 — sau tất cả
```

---

*Kế hoạch này được tạo tự động — Nhóm 7, môn QLCTG, thực hành Practice 4.*

