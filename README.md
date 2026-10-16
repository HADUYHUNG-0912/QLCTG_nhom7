# QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - NHÓM 7
## BÀI THỰC HÀNH LAB 2: KHÁM PHÁ ĐẶC TÍNH DỮ LIỆU & LÀM SẠCH CHẤT LƯỢNG TIME SERIES

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Course](https://img.shields.io/badge/Course-QLCTG%20--%20UTH-orange.svg)](https://ut.edu.vn/)
[![Status](https://img.shields.io/badge/Status-Completed%20100%25-brightgreen.svg)]()
[![Repository](https://img.shields.io/badge/GitHub-QLCTG__nhom7-lightgrey.svg)](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git)

> **Cơ sở đào tạo:** Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Khoa:** Công nghệ Thông tin  
> **Giảng viên phụ trách:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Nhóm thực hiện:** Nhóm 7 (Trưởng nhóm: Hà Duy Hưng)  
> **GitHub Repository:** [https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git)

---

## 1. CẤU TRÚC THƯ MỤC CHUẨN HÓA DỰ ÁN (`time-series-lab/`)

Toàn bộ mã nguồn, dữ liệu thực nghiệm, tài liệu đặc tả và kết quả xuất ra được chuẩn hóa theo kiến trúc module hóa:

```text
time-series-lab/
├── README.md               # Giới thiệu, cài đặt, thứ tự chạy
├── requirements.txt        # Danh sách các thư viện phụ thuộc
│
├── data/
│   ├── raw/                # Dữ liệu gốc
│   │   └── AirPassengers.csv
│   └── processed/          # Dữ liệu sau xử lý
│       ├── AirPassengers_cleaned.csv
│       └── AirPassengers_resampled.csv
│
├── docs/
│   ├── lab2_description.md # Yêu cầu cụ thể của cả 6 bài thực hành Lab 2
│   ├── lab1_description.md # Tài liệu liên kết đối chiếu (chuẩn hóa tương thích)
│   ├── data_introduction.md# Nguồn Box & Jenkins (1976), mô tả trường dữ liệu, bản quyền Public Domain
│   ├── data_processing.md  # Quyết định xử lý dữ liệu và lý do kỹ thuật (Interpolation, Hampel Filter)
│   ├── theory.md           # Lý thuyết chia mục theo 5 bài tập + 12 câu hỏi ôn tập cuối chương
│   ├── lecture_notes.md    # Ghi chú bài giảng từ slide của giảng viên
│   ├── bt6_report.md       # Báo cáo kỹ thuật tổng kết nghiệm thu hoàn chỉnh
│   └── project_management/ # Kế hoạch phân công (.md, .xlsx) và Báo cáo Word (.docx)
│
├── notebooks/
│   ├── bt1_acf_pacf.ipynb      # Bài 1: Tính và phân tích biểu đồ ACF & PACF 40 lags
│   ├── bt2_white_noise.ipynb   # Bài 2: Kiểm định White Noise vs AirPassengers (Ljung-Box)
│   ├── bt3_decomposition.ipynb # Bài 3: Phân rã Additive, Multiplicative và STL
│   ├── bt4_data_quality.ipynb  # Bài 4: Kiểm tra & làm sạch chất lượng dữ liệu Time Series
│   └── bt5_visualization.ipynb # Bài 5: Trực quan hóa chuyên sâu (Rolling Stats, Seasonal Plot, Heatmap)
│
├── outputs/
│   ├── figures/            # 9 biểu đồ PNG độ phân giải 300 DPI xuất ra từ 5 bài tập
│   │   ├── bai1_acf_pacf.png
│   │   ├── bai2_white_noise_comparison.png
│   │   ├── bai3_1_additive_decomposition.png
│   │   ├── bai3_2_multiplicative_decomposition.png
│   │   ├── bai3_3_stl_decomposition.png
│   │   ├── bai4_data_cleaning_before_after.png
│   │   ├── bai5_1_rolling_statistics.png
│   │   ├── bai5_2_seasonal_plot.png
│   │   └── bai5_3_seasonality_heatmap.png
│   └── tables/             # 5 bảng CSV số liệu định lượng + file Markdown tổng hợp
│       ├── summary_statistics.csv
│       ├── ljung_box_test_results.csv
│       ├── decomposition_summary.csv
│       ├── data_cleaning_summary.csv
│       ├── monthly_seasonal_summary.csv
│       └── ket_qua_bai_tap_tong_hop.md
│
└── src/                    # Bản script Python (.py) thực thi tự động qua CLI
    ├── bt1_acf_pacf.py
    ├── bt2_white_noise.py
    ├── bt3_decomposition.py
    ├── bt4_data_quality.py
    └── bt5_visualization.py
```

---

## 2. HƯỚNG DẪN CÀI ĐẶT & THỨ TỰ CHẠY (QUICK START)

### 2.1. Cài đặt thư viện phụ thuộc
Khởi chạy terminal từ thư mục gốc của repository:
```bash
# Cài đặt toàn bộ thư viện cần thiết
pip install -r time-series-lab/requirements.txt
```

### 2.2. Chạy tương tác qua Jupyter Notebooks
Khởi động Jupyter và mở từng notebook trong thư mục [`time-series-lab/notebooks/`](time-series-lab/notebooks/):
```bash
jupyter notebook time-series-lab/notebooks/
```
Thứ tự thực thi khuyến nghị:
1. [`bt1_acf_pacf.ipynb`](time-series-lab/notebooks/bt1_acf_pacf.ipynb): Khám phá tự tương quan ACF/PACF đến 40 lags.
2. [`bt2_white_noise.ipynb`](time-series-lab/notebooks/bt2_white_noise.ipynb): Kiểm định Ljung-Box Test đối chứng White Noise vs AirPassengers.
3. [`bt3_decomposition.ipynb`](time-series-lab/notebooks/bt3_decomposition.ipynb): Phân rã mô hình Nhân và STL Loess.
4. [`bt4_data_quality.ipynb`](time-series-lab/notebooks/bt4_data_quality.ipynb): Nội suy bù khuyết và làm sạch bằng Hampel Filter.
5. [`bt5_visualization.ipynb`](time-series-lab/notebooks/bt5_visualization.ipynb): Trực quan hóa Rolling Stats, Seasonal Plot và Seasonality Heatmap.

### 2.3. Chạy tự động dòng lệnh qua Python Scripts (`src/`)
Chạy lần lượt các script để tái tạo toàn bộ biểu đồ và bảng kết quả:
```bash
python time-series-lab/src/bt1_acf_pacf.py
python time-series-lab/src/bt2_white_noise.py
python time-series-lab/src/bt3_decomposition.py
python time-series-lab/src/bt4_data_quality.py
python time-series-lab/src/bt5_visualization.py
```

---

## 3. TỔNG KẾT KẾT QUẢ THỰC NGHIỆM CHÍNH (KEY FINDINGS)

Trích xuất từ báo cáo nghiệm thu [`time-series-lab/docs/bt6_report.md`](time-series-lab/docs/bt6_report.md) và bảng số liệu [`time-series-lab/outputs/tables/ket_qua_bai_tap_tong_hop.md`](time-series-lab/outputs/tables/ket_qua_bai_tap_tong_hop.md):

| Bài tập | Nội dung thực nghiệm | Kết quả định lượng nổi bật | Kết luận học thuật |
| :---: | :--- | :--- | :--- |
| **Bài 1** | Phân tích ACF / PACF (40 lags) | $\text{ACF}(\text{Lag 1}) = 0.9480$, đỉnh tại Lag 12 ($0.7604$), 24, 36 | Chuỗi không dừng, mang tính nhớ dài hạn và tính chu kỳ mùa vụ 12 tháng rõ nét. |
| **Bài 2** | Kiểm định White Noise | Ljung-Box: White Noise $p = 0.6571 > 0.05$; AirPassengers $p < 10^{-177} \ll 0.05$ | White Noise không thể dự báo; AirPassengers chứa tín hiệu quy luật mạnh mẽ, rất lý tưởng để dự báo. |
| **Bài 3** | Phân rã chuỗi thời gian | Tăng trưởng xu hướng: $+274.7\%$; Đỉnh mùa vụ: Tháng 7; Đáy: Tháng 11 | Mô hình Nhân ($Y = T \times S \times R$) tối ưu vượt trội dạng Cộng; phần dư ổn định quanh 1.0. |
| **Bài 4** | Kiểm tra & làm sạch dữ liệu | Time Interpolation ($\text{RMSE}=18.83 < 28.59$ của ffill); Hampel Filter phát hiện đúng 3/3 spikes | RMSE sau làm sạch giảm từ $99.18 \rightarrow 5.90$ (giảm ~17 lần), khôi phục dữ liệu gần như nguyên bản. |
| **Bài 5** | Trực quan hóa chuyên sâu | Rolling Std tăng gấp 5.67 lần (từ 13.72 lên 77.74); 95.5% điểm nằm trong dải $\pm 2$ Std | Phương sai thay đổi theo thời gian (Heteroskedasticity); mùa vụ đồng dạng qua 12 năm. |
| **Bài 6** | Đánh giá tiềm năng dự báo | Bác bỏ giả thuyết ngẫu nhiên, xu hướng ổn định, cấu trúc mùa vụ chuẩn xác | **Bộ dữ liệu hoàn toàn phù hợp để xây dựng mô hình dự báo (SARIMA, Prophet, XGBoost).** |

---

## 4. DANH MỤC TÀI LIỆU KỸ THUẬT CHI TIẾT (`docs/`)

- 📘 [**Mô tả Đề bài & Hướng dẫn Thực hành (lab2_description.md)**](time-series-lab/docs/lab2_description.md)
- 📗 [**Giới thiệu & Quyền sử dụng Dữ liệu (data_introduction.md)**](time-series-lab/docs/data_introduction.md)
- 📙 [**Quyết định Xử lý Dữ liệu & Lý do Kỹ thuật (data_processing.md)**](time-series-lab/docs/data_processing.md)
- 📕 [**Cơ sở Lý thuyết & 12 Câu hỏi Ôn tập Cuối chương 2 (theory.md)**](time-series-lab/docs/theory.md)
- 📓 [**Ghi chú Bài giảng Từ Slide Giảng viên (lecture_notes.md)**](time-series-lab/docs/lecture_notes.md)
- 📑 [**Báo cáo Kỹ thuật Tổng kết Nghiệm thu Bài 6 (bt6_report.md)**](time-series-lab/docs/bt6_report.md)
- 📁 [**Hồ sơ Quản lý Dự án & Báo cáo Word (project_management/)**](time-series-lab/docs/project_management/)

---

## 5. PHÂN CÔNG THỰC HIỆN NHÓM 7

| Mã CV | Thành viên phụ trách | Nhiệm vụ chính | Trạng thái |
| :---: | :--- | :--- | :---: |
| **CV01** | Hà Duy Hưng (Nhóm trưởng) | Kiến trúc dự án, điều phối pipeline, cấu trúc Git | **Hoàn thành (Done)** |
| **CV02** | Thành viên 1 | Bài tập 1: Tính toán & phân tích biểu đồ ACF / PACF | **Hoàn thành (Done)** |
| **CV03** | Thành viên 2 | Bài tập 2: Sinh White Noise & kiểm định Ljung-Box Test | **Hoàn thành (Done)** |
| **CV04** | Thành viên 3 | Bài tập 3: Phân rã chuỗi thời gian (Additive, Multiplicative, STL) | **Hoàn thành (Done)** |
| **CV05** | Thành viên 4 | Bài tập 4: Kiểm tra và làm sạch chất lượng dữ liệu Time Series | **Hoàn thành (Done)** |
| **CV06** | Bùi Trần Ngọc Khải | Bài tập 5: Trực quan hóa chuyên sâu (Rolling Stats, Seasonal Plot, Heatmap) | **Hoàn thành (Done)** |
| **CV07** | Nhóm 7 (Hà Duy Hưng tổng hợp) | Bài tập 6: Báo cáo kỹ thuật tổng kết nghiệm thu & nộp bài | **Hoàn thành (Done)** |

---
*Dự án được thực hiện phục vụ học phần Quản lý & Khai phá Chuỗi Thời gian (QLCTG) - Đại học Giao thông Vận tải TP.HCM (UTH), Học kỳ 1 Năm học 2026.*
