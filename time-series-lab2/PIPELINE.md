# PIPELINE QUY TRÌNH KỸ THUẬT: BÀI THỰC HÀNH LAB 2
## KIỂM ĐỊNH ADF & XÂY DỰNG MÔ HÌNH ARIMA / SARIMA
### BỘ DỮ LIỆU: DATASET 1 – MONTHLY RETAIL SALES (a10.csv)

> **Môn học:** Phân tích dữ liệu chuỗi thời gian và dự báo  
> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Nhóm thực hiện:** Nhóm 7 (6 thành viên: Hà Duy Hưng (Lead), Thành viên 2, Thành viên 3, Thành viên 4, Thành viên 5, Thành viên 6)  
> **Tập dữ liệu thực nghiệm:** Monthly Retail Sales (`a10.csv` — 204 quan sát tháng liên tục từ 07/1991 đến 06/2008)  

---

## 1. SƠ ĐỒ LUỒNG DỮ LIỆU PIPELINE (DATAFLOW DAG)

```text
========================================================================================================================
                              SƠ ĐỒ PIPELINE QUY TRÌNH THỰC THI (LAB 2 - 6 THÀNH VIÊN)
========================================================================================================================

 [ NGUỒN DỮ LIỆU GỐC ]
   URL: https://raw.githubusercontent.com/selva86/datasets/master/a10.csv (204 tháng | 1991-07 -> 2008-06)
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CV01: QUẢN TRỊ PIPELINE, EDA & DATA PREPARATION                                                                     │
│ Phụ trách : Hà Duy Hưng (Lead)                                                                                       │
│ Script    : src/cv01_eda_and_data_prep.py | Notebook: notebooks/cv01_hung_eda_data_prep.ipynb                          │
│ Nhiệm vụ  : • Tải a10.csv -> data/raw/monthly_retail_sales.csv                                                       │
│             • Chuyển date thành DatetimeIndex (freq='MS'), trực quan hóa chuỗi gốc (Trend + Seasonality)             │
│             • Phân chia TUẦN TỰ (Không shuffle): Train 80% (163 tháng) | Test 20% (41 tháng)                         │
│             • Trả lời 3 câu hỏi Bài 1                                                                                │
│ Output    : data/processed/train.csv, test.csv | figures/cv01_*.png | tables/cv01_*.csv                              │
└─────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┘
                                              │ data/processed/train.csv (163 tháng)
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CV02: KIỂM ĐỊNH TÍNH DỪNG ADF & SAI PHÂN BẬC 1 (BÀI 2 & BÀI 3)                                                       │
│ Phụ trách : Thành viên 2                                                                                             │
│ Script    : src/cv02_adf_and_differencing.py | Notebook: notebooks/cv02_tv2_adf_differencing.ipynb                      │
│ Nhiệm vụ  : • Kiểm định ADF chuỗi gốc (Bài 2): ADF stat, p-value, critical values -> Kết luận KHÔNG DỪNG            │
│             • Lấy sai phân bậc 1: d = 1 (Bài 3), vẽ biểu đồ chuỗi sau sai phân                                       │
│             • Kiểm định ADF lần 2 -> Khẳng định chuỗi dừng (p < 0.05), xác định tham số d = 1                        │
│             • Trả lời 3 câu hỏi Bài 2 và 2 câu hỏi Bài 3                                                             │
│ Output    : data/processed/train_diff1.csv | tables/cv02_adf_*.csv | figures/cv02_*.png                              │
└─────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┘
                                              │ data/processed/train_diff1.csv
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CV03: PHÂN TÍCH ACF / PACF & ĐỀ XUẤT BỘ THAM SỐ (p, d, q) (BÀI 4)                                                    │
│ Phụ trách : Thành viên 3                                                                                             │
│ Script    : src/cv03_acf_pacf_order_selection.py | Notebook: notebooks/cv03_tv3_acf_pacf_selection.ipynb               │
│ Nhiệm vụ  : • Vẽ đồ thị ACF và PACF trên chuỗi đã dừng (40 lags)                                                     │
│             • Xác định điểm cắt cụt (Cut-off) và giảm dần (Tail-off)                                                 │
│             • Grid Search sơ bộ trên Train: quét p in [0..3], q in [0..3] với d=1 theo tiêu chuẩn AIC/BIC            │
│             • Đề xuất 2-3 mô hình ARIMA ứng viên: ví dụ ARIMA(1,1,1), ARIMA(2,1,1), ARIMA(0,1,1)                     │
│             • Trả lời 3 câu hỏi Bài 4                                                                                │
│ Output    : figures/cv03_acf_pacf_plots.png | figures/cv03_aic_bic_heatmap.png | tables/cv03_suggested_orders.csv   │
└─────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┘
                                              │ Các cấu hình (p, d, q) ứng viên
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CV04: HUẤN LUYỆN ARIMA, CHẨN ĐOÁN PHẦN DƯ & DỰ BÁO TẬP TEST (BÀI 5)                                                  │
│ Phụ trách : Thành viên 4                                                                                             │
│ Script    : src/cv04_arima_model_evaluation.py | Notebook: notebooks/cv04_tv4_arima_evaluation.ipynb                     │
│ Nhiệm vụ  : • Fit các mô hình ARIMA ứng viên trên Train (80%), so sánh AIC/BIC để chọn Best ARIMA                   │
│             • Chẩn đoán phần dư: Residual plot (mean ≈ 0), ACF phần dư, Q-Q plot, kiểm định Ljung-Box (White noise) │
│             • Dự báo Out-of-sample trên Test (20% - 41 tháng) kèm khoảng tin cậy 95%                                 │
│             • Tính các chỉ số bắt buộc: MAE, RMSE, MAPE trên đơn vị gốc                                              │
│             • Trả lời chi tiết 9 câu hỏi chuyên sâu Bài 5                                                            │
│ Output    : models/best_arima_model.pkl | figures/cv04_*.png | tables/cv04_arima_metrics.csv                        │
└─────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┘
                                              │ Baseline ARIMA & Residuals
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CV05: MỞ RỘNG MÔ HÌNH SARIMA XỬ LÝ MÙA VỤ CHU KỲ s = 12 (BÀI TẬP NÂNG CAO)                                          │
│ Phụ trách : Thành viên 5                                                                                             │
│ Script    : src/cv05_sarima_seasonal_modeling.py | Notebook: notebooks/cv05_tv5_sarima_modeling.ipynb                 │
│ Nhiệm vụ  : • Phân tích tính mùa vụ chu kỳ s = 12 tháng của dữ liệu bán lẻ                                           │
│             • Thực hiện kết hợp sai phân thường (d=1) và sai phân mùa vụ (D=1)                                       │
│             • Xác định (p, d, q) x (P, D, Q)_12 qua ACF/PACF tại các mốc lag bội số của 12                          │
│             • Huấn luyện SARIMA, dự báo trên Test set, tính MAE, RMSE, MAPE                                          │
│             • Đối chuẩn trực diện: Vẽ biểu đồ so sánh 3 đường (Actual vs ARIMA vs SARIMA)                            │
│ Output    : models/best_sarima_model.pkl | figures/cv05_*.png | tables/cv05_sarima_metrics.csv                      │
└─────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┘
                                              │ Toàn bộ artifacts, metrics, figures
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CV06: TỔNG HỢP TOÀN DIỆN BÁO CÁO KỸ THUẬT & GIẢI ĐÁP LÝ LUẬN CHUYÊN SÂU                                              │
│ Phụ trách : Thành viên 6                                                                                             │
│ Script    : src/cv06_report_synthesis.py | Notebook: notebooks/cv06_tv6_report_synthesis.ipynb                       │
│ Nhiệm vụ  : • Hợp nhất toàn bộ bảng số liệu: so sánh AIC/BIC, MAE, RMSE, MAPE giữa ARIMA và SARIMA                  │
│             • Kiểm định bổ trợ Walk-forward Rolling Forecast kiểm tra độ trôi tham số                               │
│             • Trả lời trọn vẹn, khoa học tất cả câu hỏi lý luận từ Bài 1 đến Bài 5 (kèm bằng chứng số liệu)          │
│             • Xuất bản báo cáo kỹ thuật nghiệm thu hoàn chỉnh: docs/BAO_CAO_LAB2_NHOM7.md                            │
│ Output    : docs/BAO_CAO_LAB2_NHOM7.md | tables/final_model_benchmark.csv | figures/cv06_final_dashboard.png     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
========================================================================================================================
```

---

## 2. MA TRẬN PHÂN CÔNG CHI TIẾT & HỢP ĐỒNG I/O (6 THÀNH VIÊN)

| Mã CV | Thành viên phụ trách | Vai trò trọng tâm | Tệp thực thi / Notebook | Hợp đồng Đầu vào (Input) | Hợp đồng Đầu ra (Deliverables) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **CV01** | **Hà Duy Hưng (Lead)** | Quản trị Pipeline, EDA & Data Split (Bài 1) | • `src/cv01_eda_and_data_prep.py`<br>• `notebooks/cv01_hung_eda_data_prep.ipynb` | • `https://raw.githubusercontent.com/selva86/datasets/master/a10.csv` | • `data/raw/monthly_retail_sales.csv`<br>• `data/processed/train.csv` (163 tháng)<br>• `data/processed/test.csv` (41 tháng)<br>• `outputs/figures/cv01_timeseries_plot.png`<br>• `outputs/figures/cv01_seasonality_boxplot.png`<br>• `outputs/tables/cv01_data_summary.csv`<br>• `outputs/tables/cv01_split_info.csv` |
| **CV02** | **Thành viên 2** | Kiểm định ADF chuỗi gốc & Sai phân bậc 1 (Bài 2 & 3) | • `src/cv02_adf_and_differencing.py`<br>• `notebooks/cv02_tv2_adf_differencing.ipynb` | • `data/processed/train.csv` | • `data/processed/train_diff1.csv`<br>• `outputs/tables/cv02_adf_original_results.csv`<br>• `outputs/tables/cv02_adf_diff1_results.csv`<br>• `outputs/figures/cv02_original_vs_diff1.png`<br>• `outputs/figures/cv02_adf_pvalues_bar.png` |
| **CV03** | **Thành viên 3** | Phân tích ACF/PACF & Đề xuất bậc $(p, d, q)$ (Bài 4) | • `src/cv03_acf_pacf_order_selection.py`<br>• `notebooks/cv03_tv3_acf_pacf_selection.ipynb` | • `data/processed/train_diff1.csv` | • `outputs/figures/cv03_acf_pacf_plots.png`<br>• `outputs/figures/cv03_aic_bic_heatmap.png`<br>• `outputs/tables/cv03_suggested_orders.csv` |
| **CV04** | **Thành viên 4** | Huấn luyện ARIMA, Chẩn đoán phần dư & Dự báo Test (Bài 5) | • `src/cv04_arima_model_evaluation.py`<br>• `notebooks/cv04_tv4_arima_evaluation.ipynb` | • `data/processed/train.csv`<br>• `data/processed/test.csv`<br>• Bộ $(p, d, q)$ từ CV03 | • `outputs/models/best_arima_model.pkl`<br>• `outputs/figures/cv04_residual_diagnostics.png`<br>• `outputs/figures/cv04_arima_forecast_vs_actual.png`<br>• `outputs/tables/cv04_arima_metrics.csv`<br>• `outputs/tables/cv04_residual_ljung_box.csv` |
| **CV05** | **Thành viên 5** | Mô hình hóa SARIMA với chu kỳ mùa vụ $s = 12$ (Nâng cao) | • `src/cv05_sarima_seasonal_modeling.py`<br>• `notebooks/cv05_tv5_sarima_modeling.ipynb` | • `data/processed/train.csv`<br>• `data/processed/test.csv`<br>• Kết quả ARIMA từ CV04 | • `outputs/models/best_sarima_model.pkl`<br>• `outputs/figures/cv05_seasonal_decomposition.png`<br>• `outputs/figures/cv05_sarima_forecast_vs_actual.png`<br>• `outputs/figures/cv05_arima_vs_sarima_overlay.png`<br>• `outputs/tables/cv05_sarima_metrics.csv`<br>• `outputs/tables/cv05_arima_vs_sarima_comparison.csv` |
| **CV06** | **Thành viên 6** | Đối chuẩn tổng thể, Walk-forward & Báo cáo kỹ thuật | • `src/cv06_report_synthesis.py`<br>• `notebooks/cv06_tv6_report_synthesis.ipynb` | • Tất cả CSV trong `outputs/tables/`<br>• Tất cả PNG trong `outputs/figures/` | • `docs/BAO_CAO_LAB2_NHOM7.md`<br>• `outputs/tables/final_model_benchmark.csv`<br>• `outputs/figures/cv06_final_dashboard.png` |

---

## 3. NĂM NGUYÊN TẮC KỸ THUẬT BẮT BUỘC (PIPELINE GUARDRAILS)

1. **Bảo tồn tính tuần tự thời gian tuyệt đối (No Shuffle):**
   * Phân chia tập dữ liệu theo thứ tự đơn điệu thời gian: 80% Train đầu (163 tháng: 07/1991 đến 01/2005) và 20% Test cuối (41 tháng: 02/2005 đến 06/2008). Tuyệt đối không xáo trộn (shuffle = False).
2. **Quy chuẩn kiểm định tính dừng (Stationarity Strictness):**
   * Chỉ khi kiểm định ADF đạt giá trị $p\text{-value} < 0.05$ thì chuỗi mới được coi là dừng. Bậc sai phân $d$ phải được xác định tối thiểu (thường $d=1$ để tránh over-differencing).
3. **Chẩn đoán phần dư độc lập (Residual Independence):**
   * Mô hình đạt chuẩn bắt buộc phần dư phải thỏa mãn: $\text{mean} \approx 0$, không còn tương quan trong đồ thị ACF phần dư, và kiểm định Ljung-Box có $p\text{-value} > 0.05$ (không bác bỏ giả thuyết phần dư là White Noise).
4. **Đánh giá trên không gian dữ liệu gốc:**
   * Mọi chỉ số sai số (MAE, RMSE, MAPE) dùng để đối chuẩn mô hình phải được tính toán trên giá trị doanh số thực tế của tập Test, bảo đảm tính so sánh công bằng.
5. **Tiêu chuẩn biểu đồ xuất bản:**
   * Tất cả đồ thị xuất ra thư mục `outputs/figures/` phải có độ phân giải tối thiểu DPI = 150, định dạng PNG, nhãn trục và chú thích rõ ràng.

---

## 4. THỨ TỰ THỰC THI TOÀN HỆ THỐNG QUA CLI

```bash
cd time-series-lab2

# Bước 1 (Hưng Lead): Tải a10.csv, phân tích EDA, chia tập Train/Test 80/20
python src/cv01_eda_and_data_prep.py

# Bước 2 (Thành viên 2): Kiểm định ADF chuỗi gốc, sai phân bậc 1 và kiểm định ADF lần 2
python src/cv02_adf_and_differencing.py

# Bước 3 (Thành viên 3): Vẽ ACF/PACF (40 lags), quét lưới AIC/BIC, đề xuất (p, d, q)
python src/cv03_acf_pacf_order_selection.py

# Bước 4 (Thành viên 4): Huấn luyện ARIMA, chẩn đoán phần dư, dự báo tập Test và tính MAE/RMSE/MAPE
python src/cv04_arima_model_evaluation.py

# Bước 5 (Thành viên 5): Huấn luyện mô hình SARIMA(p,d,q)(P,D,Q)_12, đối chuẩn trực tiếp với ARIMA
python src/cv05_sarima_seasonal_modeling.py

# Bước 6 (Thành viên 6): Hợp nhất toàn bộ kết quả, chạy Walk-forward và xuất bản báo cáo kỹ thuật hoàn chỉnh
python src/cv06_report_synthesis.py
```
