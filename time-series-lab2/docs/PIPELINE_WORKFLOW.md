# PIPELINE QUY TRÌNH KỸ THUẬT: MÔ HÌNH HÓA VÀ DỰ BÁO ARIMA / SARIMA
## BÀI THỰC HÀNH 2: KIỂM ĐỊNH ADF & XÂY DỰNG MÔ HÌNH ARIMA / SARIMA
### MÔN HỌC: PHÂN TÍCH DỮ LIỆU CHUỖI THỜI GIAN VÀ DỰ BÁO

> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Kho mã nguồn:** `time-series-lab2`  
> **Bộ dữ liệu thực nghiệm:** **Dataset 1: Monthly Retail Sales** (`a10.csv` — 204 quan sát tháng 07/1991 – 06/2008)  
> **Cơ cấu nhân sự (6 thành viên):**  
> * **Thành viên 1** — EDA, Tiền xử lý, Kiểm toán chuỗi thời gian & Phân chia dữ liệu 80/20 (CV01).  
> * **Thành viên 2** — Kiểm định tính dừng ADF & Sai phân bậc $d$ (CV02).  
> * **Thành viên 3** — Phân tích tương quan động ACF / PACF & Lựa chọn bậc trễ $(p, q)$ (CV03).  
> * **Thành viên 4** — Xây dựng mô hình ARIMA, Chẩn đoán phần dư & Đánh giá (CV04).  
> * **Thành viên 5** — Xây dựng mô hình mùa vụ SARIMA & Đối chuẩn Benchmark với ARIMA (CV05).  
> * **Hà Duy Hưng (Lead)** — Quản trị Pipeline tổng thể, Tổng hợp Báo cáo Kỹ thuật & Trả lời toàn bộ 12 câu hỏi lý luận chuyên sâu (CV06).  

---

## 1. SƠ ĐỒ KIẾN TRÚC PIPELINE TỔNG THỂ (PIPELINE DAG)

```text
========================================================================================================================
                     SƠ ĐỒ QUY TRÌNH KỸ THUẬT MÔ HÌNH HÓA ARIMA / SARIMA (LAB 2 PIPELINE DAG)
========================================================================================================================

 [ NGUỒN DỮ LIỆU THÔ ]
                      ┌────────────────────────────────────────────────────────────────────────┐
                      │ File: data/raw/a10.csv (204 tháng | 07/1991 - 06/2008)                 │
                      │ Cấu trúc: 2 cột [date, value] - Doanh số bán lẻ theo tháng             │
                      └───────────────────────────────────┬────────────────────────────────────┘
                                                          │
                                                          ▼
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 BƯỚC 1: KHỞI TẠO, AUDIT DỮ LIỆU, PHÂN CHIA 80/20 & PHÂN TÍCH EDA
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MÃ CÔNG VIỆC : CV01                                                                                                  │
│ THÀNH VIÊN   : Thành viên 1                                                                                          │
│ TỆP THỰC THI : src/cv01_eda_and_preprocessing.py | notebooks/cv01_member1_eda_preprocessing.ipynb                   │
│ ĐẦU VÀO      : data/raw/a10.csv                                                                                      │
│ NHIỆM VỤ     : • Audit dữ liệu: 0% missing, 0 duplicate, thiết lập DatetimeIndex tần suất tháng (freq='MS')          │
│                • Phân tích EDA: Biểu đồ đường (Line plot), phân rã Xu hướng (Trend) & Mùa vụ 12 tháng (Seasonality)  │
│                • Phân chia TUẦN TỰ (No Shuffle): Train 80% (163 tháng đầu) | Test 20% (41 tháng cuối)                │
└─────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┴────────────────────────────────────┐
                    │ ĐẦU RA (DELIVERABLES - CV01):                                            │
                    │ • data/processed/train.csv (163 tháng | 1991-07 đến 2004-12)             │
                    │ • data/processed/test.csv  (41 tháng  | 2005-01 đến 2008-06)             │
                    │ • outputs/figures/cv01_eda_timeseries.png                                │
                    │ • outputs/figures/cv01_monthly_seasonality_boxplot.png                   │
                    │ • outputs/tables/cv01_data_summary.csv, cv01_split_info.csv              │
                    └─────────────────────────────────────┬────────────────────────────────────┘
                                                          │
                                                          ▼
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 BƯỚC 2: KIỂM ĐỊNH TÍNH DỪNG (ADF TEST) & XÁC ĐỊNH BẬC SAI PHÂN d
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MÃ CÔNG VIỆC : CV02                                                                                                  │
│ THÀNH VIÊN   : Thành viên 2                                                                                          │
│ TỆP THỰC THI : src/cv02_stationarity_and_differencing.py | notebooks/cv02_member2_stationarity_differencing.ipynb    │
│ ĐẦU VÀO      : data/processed/train.csv                                                                              │
│ NHIỆM VỤ     : • Kiểm định ADF trên chuỗi Train gốc: ADF statistic, p-value so với ngưỡng alpha = 0.05               │
│                • Nếu chuỗi không dừng: Thực hiện lấy sai phân bậc 1 (d = 1)                                          │
│                • Kiểm định ADF lần 2 trên chuỗi sau sai phân; vẽ biểu đồ so sánh trước/sau sai phân                  │
│                • Kết luận bậc sai phân d tối ưu để đạt chuỗi dừng                                                    │
└─────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┴────────────────────────────────────┐
                    │ ĐẦU RA (DELIVERABLES - CV02):                                            │
                    │ • data/processed/train_differenced.csv (chuỗi đã dừng sau sai phân)      │
                    │ • outputs/tables/cv02_adf_test_summary.csv                               │
                    │ • outputs/figures/cv02_adf_before_after_differencing.png                 │
                    └─────────────────────────────────────┬────────────────────────────────────┘
                                                          │
                                                          ▼
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 BƯỚC 3: PHÂN TÍCH TƯƠNG QUAN ĐỘNG ACF / PACF & ĐỀ XUẤT BẬC (p, q)
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MÃ CÔNG VIỆC : CV03                                                                                                  │
│ THÀNH VIÊN   : Thành viên 3                                                                                          │
│ TỆP THỰC THI : src/cv03_acf_pacf_analysis.py | notebooks/cv03_member3_acf_pacf_analysis.ipynb                        │
│ ĐẦU VÀO      : data/processed/train_differenced.csv                                                                  │
│ NHIỆM VỤ     : • Tính và vẽ hàm tự tương quan (ACF) và hàm tự tương quan riêng phần (PACF) đến lag 40                │
│                • Xác định hiện tượng cắt cụt (Cut-off) hoặc suy giảm dần (Tail-off) để gợi ý p và q                  │
│                • Nhận diện các gai nhọn (spikes) tại lag bội số của 12 (chu kỳ mùa vụ s = 12)                        │
│                • Lập danh sách các bộ tham số ứng viên (p, d, q) theo quy tắc Box-Jenkins                            │
└─────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┴────────────────────────────────────┐
                    │ ĐẦU RA (DELIVERABLES - CV03):                                            │
                    │ • outputs/figures/cv03_acf_pacf_plots.png                                │
                    │ • outputs/figures/cv03_lag_correlation_analysis.png                      │
                    │ • outputs/tables/cv03_candidate_orders.csv                               │
                    └─────────────────────────────────────┬────────────────────────────────────┘
                                                          │
                                ┌─────────────────────────┴─────────────────────────┐
                                │                                                   │
                                ▼                                                   ▼
═══════════════════════════════════════════════════   ═══════════════════════════════════════════════════
 BƯỚC 4: XÂY DỰNG MÔ HÌNH TIÊU CHUẨN ARIMA(p,d,q)      BƯỚC 5: XÂY DỰNG MÔ HÌNH MÙA VỤ SARIMA(p,d,q)(P,D,Q)s
═══════════════════════════════════════════════════   ═══════════════════════════════════════════════════
┌─────────────────────────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│ MÃ CÔNG VIỆC : CV04                             │   │ MÃ CÔNG VIỆC : CV05                             │
│ THÀNH VIÊN   : Thành viên 4                     │   │ THÀNH VIÊN   : Thành viên 5                     │
│ TỆP THỰC THI :                                  │   │ TỆP THỰC THI :                                  │
│   src/cv04_arima_model.py                       │   │   src/cv05_sarima_model.py                      │
│   notebooks/cv04_member4_arima_model.ipynb      │   │   notebooks/cv05_member5_sarima_model.ipynb     │
│ ĐẦU VÀO      : train.csv, test.csv,             │   │ ĐẦU VÀO      : train.csv, test.csv,             │
│                cv03_candidate_orders.csv        │   │                cv03_candidate_orders.csv        │
│ NHIỆM VỤ     :                                  │   │ NHIỆM VỤ     :                                  │
│ • Huấn luyện 1-2 mô hình ARIMA ứng viên         │   │ • Thiết lập tham số mùa vụ s = 12, D = 1        │
│ • Ghi lại AIC, BIC, kiểm tra tính hội tụ        │   │ • Huấn luyện SARIMA(p,d,q)(P,D,Q)[12] tối ưu    │
│ • Chẩn đoán phần dư (Residual Diagnostics):     │   │ • Chẩn đoán phần dư: Ljung-Box test White Noise │
│   Mean ≈ 0?, ACF residuals có tự tương quan?    │   │ • Dự báo 41 tháng trên tập Test (Out-of-sample) │
│ • Dự báo trên tập Test (41 tháng)               │   │ • Đánh giá: MAE, RMSE, MAPE                     │
│ • Tính chỉ số: MAE, RMSE, MAPE                  │   │ • Đối chuẩn Benchmark: ARIMA vs SARIMA          │
└────────────────────────┬────────────────────────┘   └────────────────────────┬────────────────────────┘
                         │                                                     │
                         ▼                                                     ▼
┌─────────────────────────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│ ĐẦU RA (DELIVERABLES - CV04):                   │   │ ĐẦU RA (DELIVERABLES - CV05):                   │
│ • outputs/models/arima_model.pkl                │   │ • outputs/models/sarima_model.pkl               │
│ • outputs/tables/cv04_arima_metrics.csv         │   │ • outputs/tables/cv05_sarima_metrics.csv        │
│ • outputs/tables/cv04_residual_stats.csv        │   │ • outputs/tables/cv05_model_benchmark_table.csv │
│ • outputs/figures/cv04_arima_residual_diag.png  │   │ • outputs/figures/cv05_sarima_forecast_vs_actual│
│ • outputs/figures/cv04_arima_forecast_vs_actual │   │ • outputs/figures/cv05_arima_vs_sarima_comp.png │
└────────────────────────┬────────────────────────┘   └────────────────────────┬────────────────────────┘
                         │                                                     │
                         └──────────────────────────┬──────────────────────────┘
                                                    │
                                                    ▼
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 BƯỚC 6: QUẢN TRỊ PIPELINE, TỔNG HỢP BÁO CÁO NGHIỆM THU & GIẢI ĐÁP TOÀN BỘ CÂU HỎI LÝ LUẬN
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MÃ CÔNG VIỆC : CV06                                                                                                  │
│ THÀNH VIÊN   : Hà Duy Hưng (Lead)                                                                                    │
│ TỆP THỰC THI : src/cv06_report_synthesis.py | notebooks/cv06_hung_lead_report_synthesis.ipynb                       │
│ ĐẦU VÀO      : Toàn bộ bảng dữ liệu trong outputs/tables/*.csv và biểu đồ trong outputs/figures/*.png               │
│ NHIỆM VỤ     : • Tổng hợp và đối chiếu bảng so sánh hoàn chỉnh (AIC, BIC, MAE, RMSE, MAPE giữa ARIMA và SARIMA)     │
│                • Trả lời chi tiết và khoa học toàn bộ 12 câu hỏi lý luận chuyên sâu trong đề bài                     │
│                • Xuất bản Báo cáo Kỹ thuật nghiệm thu hoàn chỉnh: docs/BAO_CAO_LAB02_NHOM7.md                        │
│                • Vẽ Dashboard trực quan hóa tổng kết toàn diện bài thực hành                                         │
└─────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┴────────────────────────────────────┐
                    │ ĐẦU RA CUỐI CÙNG (FINAL DELIVERABLES - CV06):                            │
                    │ • docs/BAO_CAO_LAB02_NHOM7.md (Báo cáo tổng kết nghiệm thu Markdown)     │
                    │ • outputs/tables/final_summary_report.csv                                │
                    │ • outputs/figures/cv06_summary_dashboard.png                             │
                    └──────────────────────────────────────────────────────────────────────────┘
========================================================================================================================
```

---

## 1.1. CHIẾN LƯỢC THỰC THI SONG SONG (PARALLEL EXECUTION STRATEGY)

Để tối ưu hóa thời gian hoàn thành của 6 thành viên, Pipeline được thiết kế theo cơ chế **Phân tầng hướng hợp đồng dữ liệu (Data-Contract Phasing)**, cho phép **làm song song tới 85% khối lượng công việc**:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PHA 1: KHỞI TẠO HỢP ĐỒNG DỮ LIỆU (2 PHÚT)                               │
│                         Lead Hưng / Thành viên 1 sinh train.csv (80%) & test.csv (20%)                  │
└────────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                     │ (Giải phóng điểm nghẽn - Unblock 100%)
                     ┌───────────────────────────────┼───────────────────────────────┐
                     ▼                               ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐   ┌───────────────────────────────┐
│ LUỒNG THỐNG KÊ & EDA (SONG²)  │   │ LUỒNG MÔ HÌNH HÓA (SONG SONG) │   │ LUỒNG LÝ LUẬN & BÁO CÁO (SONG)│
├───────────────────────────────┤   ├───────────────────────────────┤   ├───────────────────────────────┤
│ • CV01: EDA & Boxplot 12 tháng│   │ • CV04: Fit ARIMA(p,d,q)      │   │ • CV06 (Lead Hưng):           │
│ • CV02: Kiểm định ADF & d=1   │   │   Residuals White Noise       │   │   Khung Báo cáo Markdown &    │
│ • CV03: Vẽ ACF/PACF 40 lags,  │   │   Dự báo 41 tháng tập Test    │   │   Giải đáp phần lớn lý luận   │
│   nhận diện spikes mùa vụ     │   │ • CV05: Fit SARIMA(p,d,q)[12] │   │   toán học của 12 câu hỏi     │
│                               │   │   Dự báo 41 tháng tập Test    │   │                               │
└───────────────┬───────────────┘   └───────────────┬───────────────┘   └───────────────┬───────────────┘
                │                                   │                                   │
                └───────────────────────────────────┼───────────────────────────────────┘
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                PHA 3: CỔNG HỘI TỤ NGHIỆM THU (SYNC GATE - 10 PHÚT)                      │
│ • CV05: Ghép cv04_arima_metrics.csv vào cv05_model_benchmark_table.csv & vẽ biểu đồ đối chuẩn 2 mô hình │
│ • CV06: Điền số liệu thực nghiệm chính xác vào BAO_CAO_LAB02_NHOM7.md & xuất bản Dashboard tổng kết     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Bảng phân định tính chất Tuần tự vs Song song chi tiết:

| Mã CV | Thành viên phụ trách | Tính chất thực thi | Phụ thuộc đầu vào (Prerequisites) | Khả năng chạy song song |
| :---: | :--- | :---: | :--- | :--- |
| **CV01 (Split)** | Thành viên 1 (hoặc Lead) | **Tuần tự (Cổng 1)** | `data/raw/a10.csv` | Chạy đầu tiên trong 2 phút để tạo `train.csv` và `test.csv`. |
| **CV01 (EDA)** | Thành viên 1 | **Song song 100%** | `train.csv` | Chạy song song độc lập với CV02, CV03, CV04, CV05, CV06. |
| **CV02** | Thành viên 2 | **Song song 100%** | `train.csv` | Lấy sai phân `diff(1)` và kiểm định ADF; chạy song song với tất cả các bước khác. |
| **CV03** | Thành viên 3 | **Song song 100%** | `train.csv` | Đọc `train.csv`, tính `diff(1)` và vẽ ACF/PACF 40 lags; không cần đợi CV02. |
| **CV04** | Thành viên 4 | **Song song 100%** | `train.csv`, `test.csv` | Huấn luyện các mô hình $\text{ARIMA}(1,1,1), (2,1,1)$; chạy hoàn toàn song song với SARIMA. |
| **CV05** | Thành viên 5 | **Song song 100%** | `train.csv`, `test.csv` | Huấn luyện $\text{SARIMA}(1,1,1)(0,1,1)_{12}$; chỉ cần đợi CV04 ở bước cuối để ghép bảng Benchmark. |
| **CV06** | **Hà Duy Hưng (Lead)** | **Song song từ đầu** | Đề bài `DE_BAI_LAB02.md` | Soạn khung báo cáo và giải quyết 12 câu hỏi lý thuyết; chỉ đợi bảng metrics ở bước nghiệm thu cuối. |

---

## 2. MA TRẬN PHÂN CÔNG NHIỆM VỤ CHI TIẾT (WORKLOAD MATRIX)

| Mã CV | Thành viên | Vai trò phụ trách | Tệp mã nguồn & Notebook | Hợp đồng Đầu vào (Input) | Hợp đồng Đầu ra (Deliverables) | Câu hỏi lý luận chịu trách nhiệm |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **CV01** | **Thành viên 1** | EDA, Tiền xử lý & Split 80/20 | • `src/cv01_eda_and_preprocessing.py`<br>• `notebooks/cv01_member1_eda_preprocessing.ipynb` | • `data/raw/a10.csv` | • `data/processed/train.csv`<br>• `data/processed/test.csv`<br>• `cv01_eda_timeseries.png`<br>• `cv01_monthly_seasonality_boxplot.png`<br>• `cv01_data_summary.csv`, `cv01_split_info.csv` | • Câu hỏi 1, 2, 3 (Bài 1): Xu hướng, Mùa vụ, Tính dừng trực quan. |
| **CV02** | **Thành viên 2** | Kiểm định ADF & Sai phân xác định $d$ | • `src/cv02_stationarity_and_differencing.py`<br>• `notebooks/cv02_member2_stationarity_differencing.ipynb` | • `data/processed/train.csv` | • `data/processed/train_differenced.csv`<br>• `cv02_adf_test_summary.csv`<br>• `cv02_adf_before_after_differencing.png` | • Câu hỏi 1, 2, 3 (Bài 2): Bản chất $H_0$, ý nghĩa $p$-value.<br>• Câu hỏi 1, 2 (Bài 3): Tính dừng sau sai phân, giá trị tham số $d$. |
| **CV03** | **Thành viên 3** | Phân tích tương quan ACF/PACF & Bậc trễ | • `src/cv03_acf_pacf_analysis.py`<br>• `notebooks/cv03_member3_acf_pacf_analysis.ipynb` | • `data/processed/train_differenced.csv` | • `cv03_acf_pacf_plots.png`<br>• `cv03_lag_correlation_analysis.png`<br>• `cv03_candidate_orders.csv` | • Câu hỏi 1, 2, 3 (Bài 4): Điểm cắt cụt ACF/PACF, đề xuất bộ tham số $(p, d, q)$ khởi tạo. |
| **CV04** | **Thành viên 4** | Xây dựng mô hình tiêu chuẩn ARIMA(p,d,q) | • `src/cv04_arima_model.py`<br>• `notebooks/cv04_member4_arima_model.ipynb` | • `train.csv`, `test.csv`<br>• `cv03_candidate_orders.csv` | • `outputs/models/arima_model.pkl`<br>• `cv04_arima_metrics.csv`<br>• `cv04_residual_stats.csv`<br>• `cv04_arima_residual_diag.png`<br>• `cv04_arima_forecast_vs_actual.png` | • Câu hỏi 1–5 (Bài 5): Tại sao không shuffle, ý nghĩa $p, d, q$, hội tụ, kiểm định phần dư White Noise. |
| **CV05** | **Thành viên 5** | Xây dựng mô hình mùa vụ SARIMA & Benchmark | • `src/cv05_sarima_model.py`<br>• `notebooks/cv05_member5_sarima_model.ipynb` | • `train.csv`, `test.csv`<br>• `arima_model.pkl`<br>• `cv04_arima_metrics.csv` | • `outputs/models/sarima_model.pkl`<br>• `cv05_sarima_metrics.csv`<br>• `cv05_model_benchmark_table.csv`<br>• `cv05_sarima_forecast_vs_actual.png`<br>• `cv05_arima_vs_sarima_comparison.png` | • Câu hỏi 6–9 (Bài 5): Dự báo bám xu hướng, vị trí sai lệch lớn nhất, độ nhạy outlier, ý nghĩa RMSE. |
| **CV06** | **Hà Duy Hưng (Lead)** | Quản trị Pipeline, Tổng hợp Báo cáo & Lập luận toàn diện | • `src/cv06_report_synthesis.py`<br>• `notebooks/cv06_hung_lead_report_synthesis.ipynb` | • Toàn bộ CSV trong `outputs/tables/`<br>• Toàn bộ PNG trong `outputs/figures/` | • `docs/BAO_CAO_LAB02_NHOM7.md`<br>• `outputs/tables/final_summary_report.csv`<br>• `outputs/figures/cv06_summary_dashboard.png` | • Hợp nhất và phản biện khoa học toàn bộ 12 câu hỏi lý luận chuyên sâu của bài Lab. |

---

## 3. CHI TIẾT KỸ THUẬT TỪNG BƯỚC THỰC HIỆN

### 3.1. Bước 1 — EDA, Tiền xử lý & Split Dữ liệu (CV01 - Thành viên 1)
* **Tiền xử lý chuỗi thời gian:**
  * Đọc `data/raw/a10.csv`, kiểm tra kiểu dữ liệu của `date` và ép kiểu sang `pd.to_datetime()`.
  * Đặt `date` làm `DatetimeIndex`, gán tần suất chu kỳ tháng đầu tháng: `df.index.freq = 'MS'`.
  * Kiểm toán chất lượng dữ liệu: Đảm bảo $0\%$ missing values, $0$ bản ghi trùng lặp, chuỗi liên tục từ tháng 07/1991 đến tháng 06/2008 ($204$ tháng).
* **Phân tích khám phá (EDA):**
  * Vẽ đồ thị đường theo thời gian quan sát biên độ dao động mở rộng dần theo thời gian (dấu hiệu của mô hình nhân tính Multiplicative).
  * Vẽ biểu đồ hộp (Boxplot) 12 tháng trong năm để phân tích tính mùa vụ: Đỉnh điểm doanh số rơi vào tháng 1 và đáy rơi vào tháng 2/3.
* **Phân chia tập dữ liệu theo thời gian (Temporal Split):**
  * **Train set:** $80\%$ đầu chuỗi = $163$ tháng (từ tháng 07/1991 đến tháng 01/2005).
  * **Test set:** $20\%$ cuối chuỗi = $41$ tháng (từ tháng 02/2005 đến tháng 06/2008).
  * **Quy tắc vàng:** Tuyệt đối không dùng `shuffle=True`, bảo tồn tính thứ tự nhân quả thời gian.

### 3.2. Bước 2 — Kiểm định tính dừng & Sai phân (CV02 - Thành viên 2)
* **Kiểm định ADF trên chuỗi Train gốc:**
  * Giả thuyết $H_0$: Chuỗi tồn tại nghiệm đơn vị (Unit Root) $\implies$ Chuỗi không dừng.
  * Giả thuyết $H_1$: Chuỗi không có nghiệm đơn vị $\implies$ Chuỗi dừng.
  * Nếu $p$-value $\ge 0.05$: Không đủ cơ sở bác bỏ $H_0$, chuỗi gốc không dừng.
* **Thực hiện sai phân bậc 1 ($d = 1$):**
  * Áp dụng công thức sai phân: $\Delta y_t = y_t - y_{t-1}$.
  * Kiểm định ADF lần 2 trên $\Delta y_t$.
  * Nếu chuỗi đạt $p$-value $< 0.05 \implies$ Bác bỏ $H_0$, chuỗi đạt tính dừng với bậc sai phân $d = 1$.

### 3.3. Bước 3 — Phân tích ACF / PACF & Lựa chọn bậc trễ (CV03 - Thành viên 3)
* **Phân tích hàm tự tương quan (ACF & PACF):**
  * Vẽ ACF và PACF cho chuỗi dừng $\Delta y_t$ với độ dài $40$ lags (bao quát hơn 3 chu kỳ mùa vụ 12 tháng).
  * Xác định bậc tự hồi quy $p$ thông qua độ dài lag PACF cắt cụt (vượt dải tin cậy $\pm 1.96/\sqrt{N}$).
  * Xác định bậc trung bình trượt $q$ thông qua độ dài lag ACF cắt cụt.
  * Quan sát sự xuất hiện của các gai nhọn (spikes) nổi bật tại các lag mùa vụ $12, 24, 36$ để chuẩn bị tham số cho SARIMA.
  * Đề xuất bảng các bộ tham số $(p, d, q)$ tiềm năng kèm AIC/BIC ban đầu.

### 3.4. Bước 4 — Huấn luyện Mô hình ARIMA & Chẩn đoán phần dư (CV04 - Thành viên 4)
* **Xây dựng mô hình ARIMA:**
  * Fit các mô hình ứng viên trên tập Train (Ví dụ: $\text{ARIMA}(1, 1, 1)$, $\text{ARIMA}(2, 1, 1)$).
  * Chọn mô hình có AIC/BIC tối ưu nhất.
* **Chẩn đoán phần dư (Residual Diagnostics):**
  * Biểu đồ đường phần dư: Dao động ổn định quanh mức $0$, không có biến động bất thường.
  * Kiểm tra tự tương quan phần dư qua biểu đồ ACF residuals.
  * Kiểm định Ljung-Box test: Kiểm tra xem các giá trị phần dư có độc lập hoàn toàn (White Noise) hay không ($p$-value $> 0.05$).
* **Dự báo ngoài mẫu (Out-of-sample Forecast):**
  * Dự báo 41 bước thời gian tiếp theo tương ứng với tập Test.
  * Tính toán các chỉ số sai số chuẩn:
    $$\text{MAE} = \frac{1}{N} \sum_{t=1}^N |y_t - \hat{y}_t|$$
    $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N (y_t - \hat{y}_t)^2}$$
    $$\text{MAPE} = \frac{1}{N} \sum_{t=1}^N \left| \frac{y_t - \hat{y}_t}{y_t} \right| \times 100\%$$

### 3.5. Bước 5 — Xây dựng mô hình mùa vụ SARIMA & Benchmark (CV05 - Thành viên 5)
* **Tích hợp yếu tố mùa vụ:**
  * Chu kỳ mùa vụ: $s = 12$ (12 tháng trong năm).
  * Áp dụng sai phân mùa vụ: $D = 1$ ($\Delta_{12} y_t = y_t - y_{t-12}$).
  * Thiết lập cấu trúc mô hình: $\text{SARIMA}(p, d, q)(P, D, Q)_{12}$.
* **Huấn luyện và đánh giá SARIMA:**
  * Huấn luyện mô hình tối ưu trên tập Train.
  * Chẩn đoán phần dư SARIMA (đảm bảo sạch tự tương quan tại các lag mùa vụ $12, 24$).
  * Dự báo 41 tháng trên tập Test.
  * Lập bảng đối chuẩn trực diện giữa **ARIMA tiêu chuẩn** và **SARIMA mùa vụ**:
    - So sánh AIC, BIC, Train Time, Test MAE, Test RMSE, Test MAPE.
    - Phân tích sự vượt trội của SARIMA trong việc nắm bắt đỉnh nhọn doanh số tháng 1 hàng năm.

### 3.6. Bước 6 — Quản trị Pipeline, Tổng hợp Báo cáo & Lập luận Chuyên sâu (CV06 - Hà Duy Hưng Lead)
* **Quản trị Pipeline & Tích hợp liên tục:**
  * Điều phối tiến độ các thành viên 1 đến 5, đảm bảo hợp đồng giao tiếp dữ liệu (I/O contracts) chuẩn xác giữa các bước.
  * Kiểm duyệt mã nguồn, loại bỏ hoàn toàn hiện tượng rò rỉ dữ liệu (Lookahead bias), chuẩn hóa định dạng kết quả.
* **Tổng kết nghiệm thu:**
  * Nạp toàn bộ kết quả thực nghiệm từ `outputs/tables/` và `outputs/figures/`.
  * Soạn thảo báo cáo kỹ thuật hoàn chỉnh: `docs/BAO_CAO_LAB02_NHOM7.md`.
  * Thiết kế Dashboard trực quan hóa tổng kết toàn diện (`outputs/figures/cv06_summary_dashboard.png`).
* **Giải đáp chi tiết toàn bộ câu hỏi lý luận chuyên sâu:**
  1. *EDA:* Chuỗi có trend tăng mạnh, chu kỳ mùa vụ 12 tháng rõ rệt, phương sai mở rộng dần, chuỗi gốc không thể dừng.
  2. *ADF Test:* $H_0$ là chuỗi không dừng (có nghiệm đơn vị); $p$-value là xác suất quan sát được mẫu dữ liệu nếu $H_0$ đúng; với mức ý nghĩa $5\%$, $p \ge 0.05$ không thể bác bỏ $H_0$.
  3. *Sai phân:* Sai phân bậc 1 ($d = 1$) triệt tiêu xu hướng tuyến tính; kiểm tra lại đạt tính dừng.
  4. *ACF/PACF:* Lag cắt cụt thể hiện bậc của quá trình AR và MA; các gai nhọn tại lag 12 phản ánh mùa vụ mạnh mẽ.
  5. *Đánh giá mô hình:* Tuyệt đối không shuffle để tránh rò rỉ thông tin tương lai; phần dư lý tưởng là White Noise; RMSE phạt nặng các sai số lớn do lấy bình phương nên nhạy cảm nhất với outlier; SARIMA khắc phục hoàn toàn nhược điểm bám đỉnh mùa vụ của ARIMA.

---

## 4. DANH MỤC TỆP ĐẦU RA TOÀN DỰ ÁN (DELIVERABLES REGISTRY)

```text
time-series-lab2/
├── README.md                                          # Tài liệu giới thiệu & hướng dẫn thực thi chung
├── requirements.txt                                   # Danh sách thư viện (numpy, pandas, statsmodels,...)
│
├── data/
│   ├── raw/
│   │   └── a10.csv                                   # File dữ liệu gốc 204 tháng (tải từ GitHub source)
│   └── processed/
│       ├── train.csv                                 # 80% dữ liệu huấn luyện (163 tháng)
│       ├── test.csv                                  # 20% dữ liệu kiểm thử (41 tháng)
│       └── train_differenced.csv                     # Chuỗi đã lấy sai phân bậc 1
│
├── docs/
│   ├── DE_BAI_LAB02.md                               # Đề bài và đặc tả yêu cầu kỹ thuật chuẩn của Lab 2
│   ├── PIPELINE_WORKFLOW.md                          # Kiến trúc kỹ thuật & phân công nhiệm vụ (Tệp này)
│   └── BAO_CAO_LAB02_NHOM7.md                        # Báo cáo kỹ thuật tổng kết nghiệm thu (CV06 xuất bản)
│
├── notebooks/                                        # 6 Jupyter Notebooks dành cho 6 thành viên thực nghiệm
│   ├── cv01_member1_eda_preprocessing.ipynb          # Notebook của Thành viên 1
│   ├── cv02_member2_stationarity_differencing.ipynb  # Notebook của Thành viên 2
│   ├── cv03_member3_acf_pacf_analysis.ipynb          # Notebook của Thành viên 3
│   ├── cv04_member4_arima_model.ipynb                # Notebook của Thành viên 4
│   ├── cv05_member5_sarima_model.ipynb               # Notebook của Thành viên 5
│   └── cv06_hung_lead_report_synthesis.ipynb         # Notebook của Hà Duy Hưng (Lead)
│
├── outputs/
│   ├── figures/                                      # Biểu đồ trực quan hóa độ phân giải cao (DPI 150)
│   │   ├── cv01_eda_timeseries.png                   # Biểu đồ chuỗi thời gian gốc
│   │   ├── cv01_monthly_seasonality_boxplot.png      # Phân tích mùa vụ 12 tháng qua Boxplot
│   │   ├── cv02_adf_before_after_differencing.png    # So sánh chuỗi trước và sau sai phân
│   │   ├── cv03_acf_pacf_plots.png                   # Đồ thị ACF & PACF 40 lags
│   │   ├── cv03_lag_correlation_analysis.png         # Phân tích tương quan theo bước trễ
│   │   ├── cv04_arima_residual_diag.png              # Chẩn đoán phần dư mô hình ARIMA
│   │   ├── cv04_arima_forecast_vs_actual.png         # Dự báo ARIMA vs Thực tế trên Test set
│   │   ├── cv05_sarima_forecast_vs_actual.png        # Dự báo SARIMA vs Thực tế trên Test set
│   │   ├── cv05_arima_vs_sarima_comparison.png       # Đối chuẩn trực diện 2 mô hình ARIMA và SARIMA
│   │   └── cv06_summary_dashboard.png                # Dashboard tổng kết nghiệm thu toàn bộ bài Lab
│   │
│   ├── models/                                       # Trọng số và cấu hình mô hình đã huấn luyện
│   │   ├── arima_model.pkl                           # Mô hình ARIMA(p,d,q) đã fit
│   │   └── sarima_model.pkl                          # Mô hình SARIMA(p,d,q)(P,D,Q)[12] đã fit
│   │
│   └── tables/                                       # Bảng dữ liệu và kết quả đánh giá số liệu
│       ├── cv01_data_summary.csv                     # Thống kê mô tả dữ liệu ban đầu
│       ├── cv01_split_info.csv                       # Chi tiết phân chia tập Train (80%) / Test (20%)
│       ├── cv02_adf_test_summary.csv                 # Bảng số liệu kiểm định ADF trước và sau sai phân
│       ├── cv03_candidate_orders.csv                 # Danh sách các bộ tham số (p, d, q) ứng viên
│       ├── cv04_arima_metrics.csv                    # Kết quả đánh giá ARIMA (AIC, BIC, MAE, RMSE, MAPE)
│       ├── cv04_residual_stats.csv                   # Thống kê kiểm định White Noise phần dư ARIMA
│       ├── cv05_sarima_metrics.csv                   # Kết quả đánh giá SARIMA (AIC, BIC, MAE, RMSE, MAPE)
│       ├── cv05_model_benchmark_table.csv            # Bảng so sánh trực diện ARIMA vs SARIMA
│       └── final_summary_report.csv                  # Bảng số liệu tổng hợp nghiệm thu cuối cùng
│
├── src/                                              # 6 Python scripts module hóa độc lập
│   ├── cv01_eda_and_preprocessing.py                 # Script CV01 (Thành viên 1)
│   ├── cv02_stationarity_and_differencing.py         # Script CV02 (Thành viên 2)
│   ├── cv03_acf_pacf_analysis.py                     # Script CV03 (Thành viên 3)
│   ├── cv04_arima_model.py                           # Script CV04 (Thành viên 4)
│   ├── cv05_sarima_model.py                          # Script CV05 (Thành viên 5)
│   └── cv06_report_synthesis.py                      # Script CV06 (Hà Duy Hưng Lead)
│
└── tests/
    └── .gitkeep                                      # Kiểm thử tự động tính toàn vẹn của pipeline
```

---

## 5. THỨ TỰ THỰC THI LIÊN TỤC QUA DÒNG LỆNH (CLI EXECUTION)

### Cách 1: Thực thi tuần tự từng bước (Sequential Mode)
Dành cho việc kiểm tra từng bước hoặc chạy trên một terminal duy nhất:

```bash
# Điều hướng vào thư mục bài thực hành
cd time-series-lab2

# Bước 1: Nạp a10.csv, kiểm toán chất lượng, phân chia 80/20 và vẽ biểu đồ EDA
python src/cv01_eda_and_preprocessing.py

# Bước 2: Kiểm định ADF chuỗi gốc, thực hiện sai phân bậc 1 và kiểm định lại tính dừng
python src/cv02_stationarity_and_differencing.py

# Bước 3: Vẽ đồ thị ACF/PACF 40 lags và đề xuất các bộ tham số (p, d, q) ứng viên
python src/cv03_acf_pacf_analysis.py

# Bước 4: Huấn luyện mô hình ARIMA, kiểm định phần dư White Noise và dự báo tập Test
python src/cv04_arima_model.py

# Bước 5: Huấn luyện mô hình mùa vụ SARIMA và đối chuẩn trực tiếp hiệu năng với ARIMA
python src/cv05_sarima_model.py

# Bước 6: Tổng hợp toàn bộ bảng biểu, biểu đồ và xuất bản Báo cáo nghiệm thu Markdown
python src/cv06_report_synthesis.py
```

### Cách 2: Thực thi song song tốc độ cao (Fast-track Parallel Mode)
Tận dụng kiến trúc không đồng bộ (Asynchronous / Parallel Jobs) để chạy cùng lúc các bước độc lập sau khi Bước 1 tạo xong dữ liệu:

```powershell
# Pha 1: Mở cổng dữ liệu (Bắt buộc chạy đầu tiên - 2 phút)
python src/cv01_eda_and_preprocessing.py

# Pha 2: Chạy song song 4 luồng độc lập (PowerShell Background Jobs)
$j2 = Start-Job { Set-Location e:\Chuỗi_TG\time-series-lab2; python src/cv02_stationarity_and_differencing.py }
$j3 = Start-Job { Set-Location e:\Chuỗi_TG\time-series-lab2; python src/cv03_acf_pacf_analysis.py }
$j4 = Start-Job { Set-Location e:\Chuỗi_TG\time-series-lab2; python src/cv04_arima_model.py }
$j5 = Start-Job { Set-Location e:\Chuỗi_TG\time-series-lab2; python src/cv05_sarima_model.py }

# Đợi các luồng huấn luyện và phân tích hoàn tất
Wait-Job $j2, $j3, $j4, $j5
Receive-Job $j2, $j3, $j4, $j5

# Pha 3: Nghiệm thu và xuất bản Báo cáo Kỹ thuật (Lead Hưng)
python src/cv06_report_synthesis.py
```

---

## 6. NĂM NGUYÊN TẮC KỸ THUẬT BẮT BUỘC (PIPELINE GOLDEN RULES)

1. **Bảo tồn trật tự thời gian tuyệt đối (Strict Temporal Order):**
   * Tuyệt đối không dùng `shuffle=True` khi chia tập Train ($80\%$) và Test ($20\%$).
   * Giữ nguyên cấu trúc `DatetimeIndex` với tần suất `freq='MS'`.
2. **Ngăn chặn rò rỉ dữ liệu (Anti-Data Leakage):**
   * Mọi phân tích tính dừng (ADF), ước lượng tham số tự hồi quy (AR), trung bình trượt (MA) và huấn luyện mô hình chỉ được phép thực hiện trên **tập Train** ($163$ tháng đầu).
   * Tập Test ($41$ tháng cuối) chỉ dùng cho mục đích dự báo ngoài mẫu (Out-of-sample Forecast) và đánh giá độc lập.
3. **Quy chuẩn chẩn đoán phần dư (Residual Diagnostics Standard):**
   * Mô hình chỉ được coi là đạt yêu cầu khi phần dư thỏa mãn tính chất **White Noise** (nhiễu trắng): Kỳ vọng trung bình xấp xỉ $0$, không còn tự tương quan đáng kể trong biểu đồ ACF residuals và vượt qua kiểm định Ljung-Box test ($p$-value $> 0.05$).
4. **Đánh giá đa chỉ số khách quan:**
   * Bắt buộc đánh giá đầy đủ 3 thước đo: **MAE**, **RMSE**, và **MAPE**.
   * So sánh song song tiêu chuẩn thông tin (**AIC**, **BIC**) trong giai đoạn huấn luyện và sai số dự báo trên tập Test.
5. **Đồng bộ hóa tên gọi và tính tái lập (Reproducibility):**
   * Cố định `random_state = 42` ở mọi bước cần sinh số ngẫu nhiên.
   * Toàn bộ biểu đồ xuất bản phải lưu ở độ phân giải cao ($\text{DPI} = 150$), định dạng PNG, phông chữ hỗ trợ Unicode (`DejaVu Sans` hoặc `Arial`).

