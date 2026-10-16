# TIME SERIES LAB - QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG)
## BÀI THỰC HÀNH SỐ 2: KHÁM PHÁ ĐẶC TÍNH DỮ LIỆU & LÀM SẠCH CHẤT LƯỢNG TIME SERIES

> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin - Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Nhóm thực hiện:** Nhóm 7  
> **GitHub Repository:** [https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git)

---

## 1. CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT DIRECTORY TREE)

Cấu trúc thư mục được thiết kế và module hóa chuẩn mực theo quy chuẩn kỹ thuật:

```text
time-series-lab/
├── README.md               # Giới thiệu, cài đặt, thứ tự chạy
├── requirements.txt        # Danh sách thư viện phụ thuộc
│
├── data/
│   ├── raw/                # Dữ liệu gốc (AirPassengers.csv)
│   └── processed/          # Dữ liệu sau xử lý (AirPassengers_cleaned.csv, AirPassengers_resampled.csv)
│
├── docs/
│   ├── lab2_description.md # Yêu cầu cụ thể của cả 6 bài thực hành Lab 2
│   ├── lab1_description.md # Tài liệu đối chiếu / bản sao liên kết
│   ├── data_introduction.md# Nguồn, mô tả, quyền sử dụng dữ liệu AirPassengers
│   ├── data_processing.md  # Quyết định xử lý dữ liệu và lý do kỹ thuật
│   ├── theory.md           # Lý thuyết chia mục theo 5 bài tập
│   ├── bt6_report.md       # Báo cáo kỹ thuật tổng hợp hoàn chỉnh (Bài tập 6)
│   └── project_management/ # Kế hoạch phân công (.md, .xlsx) và Báo cáo Word (.docx)
│
├── notebooks/
│   ├── bt1_acf_pacf.ipynb      # Bài 1: Tính và phân tích biểu đồ ACF & PACF 40 lags
│   ├── bt2_white_noise.ipynb   # Bài 2: Kiểm định chuỗi White Noise vs AirPassengers
│   ├── bt3_decomposition.ipynb # Bài 3: Phân rã Additive, Multiplicative và STL
│   ├── bt4_data_quality.ipynb  # Bài 4: Kiểm tra & làm sạch chất lượng dữ liệu Time Series
│   └── bt5_visualization.ipynb # Bài 5: Trực quan hóa chuyên sâu (Rolling Stats, Seasonal Plot, Heatmap)
│
├── outputs/
│   ├── figures/            # Toàn bộ biểu đồ xuất ra (.png)
│   │   ├── bai1_acf_pacf.png
│   │   ├── bai2_white_noise_comparison.png
│   │   ├── bai3_1_additive_decomposition.png
│   │   ├── bai3_2_multiplicative_decomposition.png
│   │   ├── bai3_3_stl_decomposition.png
│   │   ├── bai4_data_cleaning_before_after.png
│   │   ├── bai5_1_rolling_statistics.png
│   │   ├── bai5_2_seasonal_plot.png
│   │   └── bai5_3_seasonality_heatmap.png
│   └── tables/             # Các bảng kết quả xuất ra (.csv và .md)
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

## 2. YÊU CẦU MÔI TRƯỜNG & HƯỚNG DẪN CÀI ĐẶT

### 2.1. Yêu cầu hệ thống
- **Python:** Phiên bản `>= 3.9` (khuyến nghị Python 3.10, 3.11, 3.12, 3.13 hoặc 3.14).
- Hệ điều hành: Windows, macOS, hoặc Linux.

### 2.2. Các bước cài đặt
Khởi chạy terminal tại thư mục `time-series-lab/`:

```bash
# Di chuyển vào thư mục dự án
cd time-series-lab

# (Tùy chọn) Khởi tạo và kích hoạt môi trường ảo
python -m venv .venv
# Trên Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Trên Linux / macOS:
source .venv/bin/activate

# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

---

## 3. THỨ TỰ CHẠY MÃ NGUỒN (EXECUTION WORKFLOW)

Dự án cung cấp 2 phương thức thực thi linh hoạt:

### Cách 1: Chạy tương tác qua Jupyter Notebooks (Khuyến nghị cho chấm điểm)
Khởi động Jupyter Lab hoặc mở trực tiếp trên VS Code:
```bash
jupyter notebook
```
Mở và chạy lần lượt 5 notebook trong thư mục `notebooks/`:
1. `notebooks/bt1_acf_pacf.ipynb`: Phân tích tự tương quan ACF/PACF 40 độ trễ.
2. `notebooks/bt2_white_noise.ipynb`: Kiểm định Ljung-Box đối chứng White Noise và AirPassengers.
3. `notebooks/bt3_decomposition.ipynb`: Bóc tách thành phần xu hướng, mùa vụ và phần dư.
4. `notebooks/bt4_data_quality.ipynb`: Kiểm toán khuyết thiếu, nội suy và lọc Hampel Filter.
5. `notebooks/bt5_visualization.ipynb`: Khám phá Rolling Stats, Seasonal Plot và bản đồ nhiệt.

---

### Cách 2: Chạy tự động dòng lệnh qua Python Scripts (`src/`)
Chạy theo thứ tự tuần tự để tự động tái tạo toàn bộ biểu đồ và báo cáo:

```bash
# 1. Chạy bài 1: Tính ACF & PACF
python src/bt1_acf_pacf.py

# 2. Chạy bài 2: Kiểm định White Noise
python src/bt2_white_noise.py

# 3. Chạy bài 3: Phân rã chuỗi thời gian
python src/bt3_decomposition.py

# 4. Chạy bài 4: Kiểm tra & làm sạch chất lượng dữ liệu
python src/bt4_data_quality.py

# 5. Chạy bài 5: Trực quan hóa chuyên sâu
python src/bt5_visualization.py
```
> **Mẹo:** Thêm cờ `--show` nếu muốn hiển thị cửa sổ đồ thị trực tiếp trên màn hình (ví dụ: `python src/bt1_acf_pacf.py --show`).

---

## 4. DANH MỤC TÀI LIỆU KỸ THUẬT (`docs/`)

| Tên tệp | Nội dung chi tiết |
| :--- | :--- |
| [`docs/lab2_description.md`](docs/lab2_description.md) | Yêu cầu chi tiết của 6 bài tập, hướng dẫn kỹ thuật và thang điểm đánh giá. |
| [`docs/lab1_description.md`](docs/lab1_description.md) | Bản sao liên kết đối chiếu theo cấu trúc chuẩn. |
| [`docs/data_introduction.md`](docs/data_introduction.md) | Nguồn gốc dữ liệu Box & Jenkins (1976), mô tả trường thông tin, bản quyền Public Domain. |
| [`docs/data_processing.md`](docs/data_processing.md) | Quyết định kỹ thuật: Datetime indexing, so sánh bù khuyết dữ liệu, thuật toán Hampel Filter. |
| [`docs/theory.md`](docs/theory.md) | Cơ sở lý thuyết toàn diện chia theo 5 bài tập (ACF/PACF, White Noise, Decomposition, Outliers, Rolling Stats). |
| [`docs/bt6_report.md`](docs/bt6_report.md) | Báo cáo kỹ thuật tổng hợp Bài 6 đầy đủ hình ảnh, bảng số liệu và đánh giá tiềm năng dự báo. |
| [`docs/project_management/`](docs/project_management/) | Kế hoạch phân công công việc nhóm 7 (`.md`, `.xlsx`) và báo cáo bản Word (`BAO_CAO_LAB2.docx`). |

---

## 5. TỔNG KẾT KẾT QUẢ THỰC NGHIỆM

| Tiêu chí phân tích | Kết quả định lượng chính | Đánh giá học thuật |
| :--- | :---: | :--- |
| **Tính dừng (Stationarity)** | $\rho_1 = 0.9480$, suy giảm rất chậm | Chuỗi không dừng, cần sai phân bậc 1 ($d=1$). |
| **Chu kỳ mùa vụ** | Đỉnh nhô cao tại Lags 12, 24, 36 | Mùa vụ 12 tháng mạnh mẽ; đỉnh tháng 7, đáy tháng 11. |
| **Kiểm định White Noise** | Ljung-Box $p < 10^{-177} \ll 0.05$ | Bác bỏ $H_0$; chuỗi chứa tín hiệu quy luật dồi dào, dự báo tốt. |
| **Mô hình phân rã** | Nhân ($Y = T \times S \times R$) | Tối ưu vượt trội so với dạng Cộng vì biên độ mùa mở rộng theo xu hướng (+274.7%). |
| **Bù khuyết dữ liệu thiếu** | Time Interpolation ($\text{RMSE} = 18.83$) | Tốt hơn Forward-fill ($\text{RMSE} = 28.59$) nhờ bám theo độ dốc. |
| **Phát hiện ngoại lai** | Hampel Filter (3/3 chuẩn xác) | Kháng nhiễu tuyệt đối, không nhầm lẫn đỉnh du lịch hè. |
| **Khôi phục toàn diện** | RMSE giảm từ $99.18 \rightarrow 5.90$ | Giảm sai số ~17 lần, khôi phục bám sát chuỗi gốc. |
