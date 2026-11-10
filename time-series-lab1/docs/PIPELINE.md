# PIPELINE QUY TRÌNH KỸ THUẬT: KHÁM PHÁ ĐẶC TÍNH & LÀM SẠCH CHẤT LƯỢNG TIME SERIES
## MÔN HỌC: PHÂN TÍCH DỮ LIỆU CHUỖI THỜI GIAN VÀ DỰ BÁO — LAB 1 (CHƯƠNG 2)

> **Kho lưu trữ:** `time-series-lab1`  
> **Bộ dữ liệu thực nghiệm:** `AirPassengers.csv` (Box & Jenkins 1976 - 144 tháng liên tục 1949–1960)  
> **Mục tiêu cốt lõi:** Phân tích tương quan (ACF/PACF), kiểm định White Noise, phân rã Additive/Multiplicative/STL, làm sạch Missing & Hampel Outlier, trực quan hóa chuyên sâu Rolling/Seasonal/Heatmap.

---

## 1. SƠ ĐỒ PIPELINE TỔNG THỂ (ASCII ARCHITECTURE)

```text
========================================================================================================================
                      PIPELINE ARCHITECTURE: TIME SERIES EXPLORATORY DATA ANALYSIS & CLEANING
========================================================================================================================

 [ NGUỒN DỮ LIỆU THÔ ]
 ┌────────────────────────────────────────────────────────┐
 │ File: data/raw/AirPassengers.csv                       │
 │ 144 quan sát tháng (1949-1960), số lượng hành khách    │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 GIAI ĐOẠN 1: KHỞI TẠO, AUDIT DỮ LIỆU & QUẢN TRỊ PIPELINE
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MÃ CÔNG VIỆC : CV01                                                                                                │
 │ PHỤ TRÁCH    : Hà Duy Hưng (Lead)                                                                                  │
 │ TỆP THỰC THI : Điều phối toàn bộ src/ và tổng hợp tài liệu                                                         │
 │ ĐẦU VÀO      : data/raw/AirPassengers.csv                                                                          │
 │ NHIỆM VỤ     : • Thiết lập kiến trúc dự án chuẩn module hóa 5 bài tập                                              │
 │                • Kiểm tra tính toàn vẹn (0 missing, 0 duplicate, DatetimeIndex freq='MS')                          │
 │                • Xuất bảng thống kê mô tả tổng quan và hình ảnh tổng hợp 4 góc nhìn                                │
 └─────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                           │
                               ┌───────────────────────────┴───────────────────────────┐
                               │ ĐẦU RA (OUTPUT CONTRACTS):                            │
                               │ • outputs/tables/summary_statistics.csv               │
                               │ • outputs/figures/data_characteristics_overview.png   │
                               │ • data/processed/AirPassengers_resampled.csv          │
                               └───────────────────────────┬───────────────────────────┘
                                                           │
                               ┌───────────────────────────┴───────────────────────────┐
                               │                                                       │
                               ▼                                                       ▼
 ═══════════════════════════════════════════════════   ════════════════════════════════════════════════════════════════
 GIAI ĐOẠN 2A: PHÂN TÍCH TƯƠNG QUAN (ACF & PACF)       GIAI ĐOẠN 2B: KIỂM ĐỊNH TÍN HIỆU (WHITE NOISE TESTING)
 ═══════════════════════════════════════════════════   ════════════════════════════════════════════════════════════════
 ┌─────────────────────────────────────────────────┐   ┌──────────────────────────────────────────────────────────────┐
 │ MÃ CÔNG VIỆC : CV02 (Bài tập 1)                 │   │ MÃ CÔNG VIỆC : CV03 (Bài tập 2)                              │
 │ THÀNH VIÊN   : Ngọc Trâm                        │   │ THÀNH VIÊN   : Thanh Phương                                  │
 │ TỆP THỰC THI : src/bt1_acf_pacf.py              │   │ TỆP THỰC THI : src/bt2_white_noise.py                         │
 │ NOTEBOOK     : notebooks/bt1_acf_pacf.ipynb     │   │ NOTEBOOK     : notebooks/bt2_white_noise.ipynb                 │
 │ ĐẦU VÀO      : data/raw/AirPassengers.csv       │   │ ĐẦU VÀO      : data/raw/AirPassengers.csv                   │
 │ NHIỆM VỤ     : • Tính toán ACF & PACF 40 lags   │   │ NHIỆM VỤ     : • Sinh Gaussian White Noise (N=1000, seed=42) │
 │                • Dải tin cậy 95% Bartlett       │   │                • So sánh Waveform & ACF White Noise vs Dữ liệu│
 │                • Nhận diện Trend & Mùa vụ 12T   │   │                • Chạy kiểm định Ljung-Box Test (lag 10, 20)  │
 └────────────────────────┬────────────────────────┘   └──────────────────────────────┬───────────────────────────────┘
                          │                                                           │
 ┌────────────────────────┴────────────────────────┐   ┌──────────────────────────────┴───────────────────────────────┐
 │ ĐẦU RA (OUTPUT CONTRACTS):                      │   │ ĐẦU RA (OUTPUT CONTRACTS):                                   │
 │ • outputs/figures/bai1_acf_pacf.png             │   │ • outputs/figures/bai2_white_noise_comparison.png            │
 │   (ACF tail-off chậm, PACF cut-off sau lag 1)   │   │ • outputs/tables/ljung_box_test_results.csv                  │
 └────────────────────────┬────────────────────────┘   │   (p-value ~ 0 => Bác bỏ H0 => Có thể dự báo)                │
                          │                            └──────────────────────────────┬───────────────────────────────┘
                          │                                                           │
                          └────────────────────────────┬──────────────────────────────┘
                                                       │
                                                       ▼
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 GIAI ĐOẠN 3: PHÂN RÃ ĐỘNG HỌC CHUỖI THỜI GIAN (TIME-SERIES DECOMPOSITION)
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MÃ CÔNG VIỆC : CV04 (Bài tập 3)                                                                                    │
 │ THÀNH VIÊN   : Hiệp                                                                                               │
 │ TỆP THỰC THI : src/bt3_decomposition.py                                                                            │
 │ NOTEBOOK     : notebooks/bt3_decomposition.ipynb                                                                   │
 │ ĐẦU VÀO      : data/raw/AirPassengers.csv                                                                          │
 │ NHIỆM VỤ     : • Tách 4 thành phần: Quan sát (Y_t), Xu hướng (T_t), Mùa vụ (S_t), Phần dư (R_t)                    │
 │                • Thực hiện mô hình Cộng cổ điển: Y_t = T_t + S_t + R_t                                             │
 │                • Thực hiện mô hình Nhân cổ điển: Y_t = T_t * S_t * R_t                                             │
 │                • Thực hiện phân rã tiên tiến STL (Seasonal-Trend decomposition using Loess, robust=True)            │
 │                • Chứng minh mô hình Nhân và STL vượt trội do biên độ mùa vụ tăng tỷ lệ thuận với xu hướng          │
 └─────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                           │
                               ┌───────────────────────────┴───────────────────────────┐
                               │ ĐẦU RA (OUTPUT CONTRACTS):                            │
                               │ • outputs/figures/bai3_1_additive_decomposition.png   │
                               │ • outputs/figures/bai3_2_multiplicative_decomposition.png │
                               │ • outputs/figures/bai3_3_stl_decomposition.png        │
                               │ • outputs/tables/decomposition_summary.csv            │
                               └───────────────────────────┬───────────────────────────┘
                                                           │
                                                           ▼
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 GIAI ĐOẠN 4: KỸ THUẬT CHẤT LƯỢNG DỮ LIỆU & LÀM SẠCH (DATA QUALITY CLEANING PIPELINE)
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MÃ CÔNG VIỆC : CV05 (Bài tập 4)                                                                                    │
 │ THÀNH VIÊN   : Hoàng                                                                                              │
 │ TỆP THỰC THI : src/bt4_data_quality.py                                                                             │
 │ NOTEBOOK     : notebooks/bt4_data_quality.ipynb                                                                    │
 │ ĐẦU VÀO      : data/raw/AirPassengers.csv                                                                          │
 │ NHIỆM VỤ     : • Giả lập lỗi: Tạo ngẫu nhiên 10% NaN + tiêm 3 Outlier spikes (gấp 3.5 lần tại idx 25, 75, 110)      │
 │                • Đánh giá xử lý Missing: Forward-fill (ffill) vs Nội suy theo thời gian (Time Interpolation)       │
 │                • Phát hiện ngoại lai bằng bộ lọc kháng nhiễu Hampel Filter (Rolling Median +/- 3 * MAD)            │
 │                • Xây dựng luồng khôi phục dữ liệu sạch hoàn chỉnh và đo sai số MAE / RMSE                          │
 └─────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                           │
                               ┌───────────────────────────┴───────────────────────────┐
                               │ ĐẦU RA (OUTPUT CONTRACTS):                            │
                               │ • data/processed/AirPassengers_cleaned.csv            │
                               │ • outputs/tables/data_cleaning_summary.csv            │
                               │ • outputs/figures/bai4_data_cleaning_before_after.png │
                               └───────────────────────────┬───────────────────────────┘
                                                           │
                                                           ▼
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 GIAI ĐOẠN 5: TRỰC QUAN HÓA CHUYÊN SÂU ĐA CHIỀU (ADVANCED TIME SERIES EDA)
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MÃ CÔNG VIỆC : CV06 (Bài tập 5)                                                                                    │
 │ THÀNH VIÊN   : Khải                                                                                               │
 │ TỆP THỰC THI : src/bt5_visualization.py                                                                            │
 │ NOTEBOOK     : notebooks/bt5_visualization.ipynb                                                                   │
 │ ĐẦU VÀO      : data/raw/AirPassengers.csv                                                                          │
 │ NHIỆM VỤ     : • Tính toán Rolling Statistics (Cửa sổ 12 tháng: Rolling Mean & dải +/- 2*Std)                      │
 │                • Chứng minh hiện tượng phương sai thay đổi (Heteroskedasticity) qua các năm                        │
 │                • Vẽ Monthly Seasonal Plot: 12 đường cong tương ứng 12 năm, phân tích tính đồng dạng                │
 │                • Xây dựng Seasonality Heatmap (Ma trận Năm x Tháng) nhận diện vùng nhiệt cao điểm (T7-T8)          │
 └─────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                           │
                               ┌───────────────────────────┴───────────────────────────┐
                               │ ĐẦU RA (OUTPUT CONTRACTS):                            │
                               │ • outputs/figures/bai5_1_rolling_statistics.png       │
                               │ • outputs/figures/bai5_2_seasonal_plot.png            │
                               │ • outputs/figures/bai5_3_seasonality_heatmap.png      │
                               │ • outputs/tables/monthly_seasonal_summary.csv         │
                               └───────────────────────────┬───────────────────────────┘
                                                           │
                                                           ▼
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 GIAI ĐOẠN 6: TỔNG HỢP PIPELINE, ĐÁNH GIÁ TIỀM NĂNG DỰ BÁO & NGHIỆM THU
 ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MÃ CÔNG VIỆC : CV07 / CV01 (Chủ trì)                                                                               │
 │ PHỤ TRÁCH    : Hưng (Lead) & Cả nhóm (6 thành viên)                                                                │
 │ NHIỆM VỤ     : • Rà soát đồng bộ số liệu giữa tất cả các bài tập (Bài 1 -> Bài 5)                                  │
 │                • Tổng kết đặc trưng chuỗi: Non-stationary, Upward trend, Multiplicative 12M Seasonality             │
 │                • Đề xuất lộ trình mô hình hóa dự báo: SARIMA dạng nhân, Holt-Winters Multiplicative, ML XGBoost     │
 └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
========================================================================================================================
```

---

## 2. MA TRẬN PHÂN CÔNG & HỢP ĐỒNG ĐẦU VÀO / ĐẦU RA (I/O CONTRACT MATRIX)

| Giai đoạn | Mã CV | Phụ trách | Tệp thực thi / Notebook | Hợp đồng Đầu vào (Input Contract) | Hợp đồng Đầu ra (Output Contract) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **1. Khởi tạo & Audit Dữ liệu** | **CV01** | **Hưng (Lead)** | `src/visualize_data_characteristics.py` | • `data/raw/AirPassengers.csv` | • `outputs/tables/summary_statistics.csv`<br>• `outputs/figures/data_characteristics_overview.png`<br>• `data/processed/AirPassengers_resampled.csv` |
| **2. Tự tương quan ACF & PACF** | **CV02** | **Ngọc Trâm** | • `src/bt1_acf_pacf.py`<br>• `notebooks/bt1_acf_pacf.ipynb` | • `data/raw/AirPassengers.csv` | • `outputs/figures/bai1_acf_pacf.png`<br>*(ACF suy giảm chậm, PACF cut-off tại lag 1, đỉnh mùa vụ tại lag 12, 24, 36)* |
| **3. Kiểm định Nhiễu trắng** | **CV03** | **Thanh Phương** | • `src/bt2_white_noise.py`<br>• `notebooks/bt2_white_noise.ipynb` | • `data/raw/AirPassengers.csv` | • `outputs/figures/bai2_white_noise_comparison.png`<br>• `outputs/tables/ljung_box_test_results.csv`<br>*(p-value < 1e-10 khẳng định chuỗi chứa tín hiệu dự báo)* |
| **4. Phân rã Chuỗi Thời Gian** | **CV04** | **Hiệp** | • `src/bt3_decomposition.py`<br>• `notebooks/bt3_decomposition.ipynb` | • `data/raw/AirPassengers.csv` | • `outputs/figures/bai3_1_additive_decomposition.png`<br>• `outputs/figures/bai3_2_multiplicative_decomposition.png`<br>• `outputs/figures/bai3_3_stl_decomposition.png`<br>• `outputs/tables/decomposition_summary.csv` |
| **5. Làm sạch Chất lượng Dữ liệu**| **CV05** | **Hoàng** | • `src/bt4_data_quality.py`<br>• `notebooks/bt4_data_quality.ipynb` | • `data/raw/AirPassengers.csv` | • `data/processed/AirPassengers_cleaned.csv`<br>• `outputs/tables/data_cleaning_summary.csv`<br>• `outputs/figures/bai4_data_cleaning_before_after.png` |
| **6. Trực quan hóa Chuyên sâu** | **CV06** | **Khải** | • `src/bt5_visualization.py`<br>• `notebooks/bt5_visualization.ipynb` | • `data/raw/AirPassengers.csv` | • `outputs/figures/bai5_1_rolling_statistics.png`<br>• `outputs/figures/bai5_2_seasonal_plot.png`<br>• `outputs/figures/bai5_3_seasonality_heatmap.png`<br>• `outputs/tables/monthly_seasonal_summary.csv` |
| **7. Tổng kết & Nghiệm thu** | **CV07** | **Cả nhóm (6 thành viên)** | Báo cáo Markdown & Slide | • Tất cả tệp `outputs/tables/`<br>• Tất cả tệp `outputs/figures/` | • `outputs/tables/ket_qua_bai_tap_tong_hop.md`<br>• Báo cáo kỹ thuật tổng kết nghiệm thu |

---

## 3. THỨ TỰ THỰC THI LIÊN TỤC QUA CLI (EXECUTION COMMANDS)

Khởi chạy terminal tại thư mục `time-series-lab1/`:

```bash
# 1. Chạy Bài tập 1: Tính toán và vẽ biểu đồ ACF & PACF 40 lags
python src/bt1_acf_pacf.py

# 2. Chạy Bài tập 2: Sinh White Noise và chạy kiểm định Ljung-Box Test
python src/bt2_white_noise.py

# 3. Chạy Bài tập 3: Phân rã chuỗi thời gian (Additive, Multiplicative, STL)
python src/bt3_decomposition.py

# 4. Chạy Bài tập 4: Thử nghiệm xử lý Missing (Time Interpolation) & Outlier (Hampel Filter)
python src/bt4_data_quality.py

# 5. Chạy Bài tập 5: Xuất biểu đồ Rolling Stats, Monthly Seasonal Plot và Seasonality Heatmap
python src/bt5_visualization.py
```

---

## 4. BỐN NGUYÊN TẮC KỸ THUẬT CỐT LÕI (PIPELINE GOLDEN RULES)

1. **Bảo tồn tính liên tục thời gian (Strict Datetime Indexing):** Toàn bộ dữ liệu phải được ép kiểu `pd.DatetimeIndex` với tần suất bắt đầu tháng (`freq='MS'`), sắp xếp tăng dần và đảm bảo không có mốc thời gian trùng lặp.
2. **Kháng biến dạng dữ liệu gốc (Immutability):** Khi thực nghiệm tiêm lỗi ở Bài 4, bắt buộc thực hiện trên bản sao độc lập (`df.copy()`), dữ liệu nguyên bản `AirPassengers.csv` trong `data/raw/` không bao giờ bị ghi đè.
3. **Phù hợp mô hình mùa vụ (Multiplicative Fit):** Chuỗi `AirPassengers` có biên độ mùa vụ tăng theo xu hướng, nên mọi kỹ thuật phân tích mùa vụ phải ưu tiên mô hình Nhân ($Y = T \times S \times R$) hoặc biến đổi Logarithm trước khi áp dụng mô hình Cộng.
4. **Xử lý ngoại lai theo ngữ cảnh cục bộ (Local Context Outlier Filtering):** Không dùng các ngưỡng tĩnh toàn cục (như Mean $\pm 3\sigma$) vì sẽ nhầm các đỉnh mùa vụ hè là ngoại lai; bắt buộc sử dụng **Hampel Filter** trượt cục bộ dựa trên Trung vị (Median) và Sai lệch tuyệt đối trung vị (MAD).

