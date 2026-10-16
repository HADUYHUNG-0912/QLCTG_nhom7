# TIME SERIES LAB 3 - DỰ ĐOÁN CHUỖI THỜI GIAN BẰNG MACHINE LEARNING MODELS
## BÀI THỰC HÀNH: DỰ BÁO TIÊU THỤ NĂNG LƯỢNG HÀNG GIỜ BẰNG XGBOOST REGRESSOR

> **Học phần:** Quản lý & Khai phá Chuỗi Thời gian (QLCTG)  
> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin - Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Nhóm thực hiện:** Nhóm 7 (`QLCTG_nhom7`) - 6 thành viên  
> **Kho lưu trữ GitHub:** [HADUYHUNG-0912/QLCTG_nhom7](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git)  
> **Kế hoạch phân công chi tiết:** [docs/KE_HOACH_PHAN_CONG_NHOM7.md](docs/KE_HOACH_PHAN_CONG_NHOM7.md)  
> **Đề bài chi tiết:** [DE_BAI_LAB.md](DE_BAI_LAB.md)

---

### 1. MA TRẬN PHÂN CÔNG CÔNG VIỆC 6 THÀNH VIÊN

| Mã CV | Hạng mục công việc | Mô tả & Trách nhiệm kỹ thuật | File mã nguồn phụ trách | Người phụ trách & Nhánh |
| :---: | :--- | :--- | :--- | :---: |
| **CV01** | **Tiền xử lý dữ liệu & Sinh Lag Features (Bước 1, 2, 3)** | Xử lý `AEP_hourly.csv`, tạo 24 Lags, phân chia Train/Test 80/20 theo thời gian (No Shuffle), lưu `data/processed/`. | [`src/cv01_data_and_lags.py`](src/cv01_data_and_lags.py) | **Hà Duy Hưng (Lead)**<br>`main` |
| **CV02** | **Huấn luyện mô hình cơ sở XGBoost (Bước 4, 5)** | Khởi tạo `XGBRegressor`, áp dụng early stopping 50 rounds, vẽ learning loss curve, lưu checkpoint `outputs/models/`. | [`src/cv02_xgboost_train.py`](src/cv02_xgboost_train.py) | **Ngọc Trâm**<br>`feat/cv02-xgb-train` |
| **CV03** | **Đánh giá chỉ số & Feature Importance (Bước 6, 7)** | Tính toán MAE, RMSE, MAPE, $R^2$; vẽ Actual vs Predicted và biểu đồ độ quan trọng đặc trưng (Top Lags). | [`src/cv03_evaluation_importance.py`](src/cv03_evaluation_importance.py) | **Thanh Phương**<br>`feat/cv03-evaluation` |
| **CV04** | **Nâng cao 1 & 2: Rolling Features & Multi-Step** | Tạo Rolling Mean/Std (24h, 168h) bằng `.shift(1)`; dự báo đệ quy đa bước 24h và phân tích sai số tích lũy. | [`src/cv04_rolling_and_multistep.py`](src/cv04_rolling_and_multistep.py) | **Hoàng Hiệp**<br>`feat/cv04-rolling-multistep` |
| **CV05** | **Nâng cao 3 & 4: Tuning TimeSeriesSplit & Random Forest** | Tinh chỉnh siêu tham số bằng `TimeSeriesSplit` (CV 3 folds); huấn luyện benchmark đối chứng với `RandomForestRegressor`. | [`src/cv05_tuning_random_forest.py`](src/cv05_tuning_random_forest.py) | **Hoàng**<br>`feat/cv05-tuning-benchmark` |
| **CV06** | **Lý luận chuyên sâu & Báo cáo kỹ thuật tổng hợp** | Trả lời 3 câu hỏi lý luận chuyên sâu; tổng hợp Dashboard đối chứng 4 mô hình; xuất Báo cáo kỹ thuật Markdown. | [`src/cv06_report_synthesis.py`](src/cv06_report_synthesis.py) | **Bùi Trần Ngọc Khải**<br>`feat/cv06-report-theory` |

---

### 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
time-series-lab2/
├── DE_BAI_LAB.md                      # Đề bài thực hành chính thức của Lab
├── README.md                          # Tài liệu giới thiệu và hướng dẫn thực thi
├── requirements.txt                   # Danh sách gói thư viện Python
│
├── data/
│   ├── raw/                           # Dữ liệu gốc: AEP_hourly.csv (121,273 dòng)
│   └── processed/                     # Dữ liệu train.csv, test.csv sau khi tạo lag
│
├── docs/
│   ├── KE_HOACH_PHAN_CONG_NHOM7.md   # Bảng phân công chi tiết 6 thành viên Nhóm 7
│   ├── BAO_CAO_LAB3_NHOM7.md          # Báo cáo kỹ thuật tổng kết toàn diện nộp Giảng viên
│   ├── lab_description.md             # Đề bài sao lưu
│   └── lab3_description.md            # Đề bài sao lưu
│
├── outputs/
│   ├── figures/                       # Trọn bộ đồ thị PNG xuất bản (DPI 150)
│   ├── models/                        # Checkpoint mô hình XGBoost (xgboost_baseline.json)
│   └── tables/                        # Bảng số liệu thống kê và đối chứng định lượng (.csv)
│
└── src/                               # 6 module mã nguồn chuẩn hóa của 6 thành viên
    ├── cv01_data_and_lags.py          # CV01: Hà Duy Hưng (Lead)
    ├── cv02_xgboost_train.py          # CV02: Ngọc Trâm
    ├── cv03_evaluation_importance.py  # CV03: Thanh Phương
    ├── cv04_rolling_and_multistep.py  # CV04: Hoàng Hiệp
    ├── cv05_tuning_random_forest.py   # CV05: Hoàng
    └── cv06_report_synthesis.py       # CV06: Bùi Trần Ngọc Khải
```

---

### 3. HƯỚNG DẪN CÀI ĐẶT & THỰC THI TOÀN HỆ THỐNG

1. **Cài đặt thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy tuần tự từng nhiệm vụ (mỗi thành viên kiểm thử độc lập):**
   ```bash
   # Bước 1 (CV01 - Hà Duy Hưng): Tiền xử lý dữ liệu & sinh Lag 1-24
   python src/cv01_data_and_lags.py

   # Bước 2 (CV02 - Ngọc Trâm): Huấn luyện XGBoost cơ sở & Early Stopping
   python src/cv02_xgboost_train.py

   # Bước 3 (CV03 - Thanh Phương): Đánh giá MAE/RMSE/MAPE & Feature Importance
   python src/cv03_evaluation_importance.py

   # Bước 4 (CV04 - Hoàng Hiệp): Rolling Features & Dự báo Đa bước 24h
   python src/cv04_rolling_and_multistep.py

   # Bước 5 (CV05 - Hoàng): Tinh chỉnh TimeSeriesSplit & So sánh Random Forest
   python src/cv05_tuning_random_forest.py

   # Bước 6 (CV06 - Bùi Trần Ngọc Khải): Tổng hợp Dashboard & Cập nhật Báo cáo
   python src/cv06_report_synthesis.py
   ```

3. **Chỉ thị sử dụng AI Agent (Cursor, Antigravity, Copilot):**
   Mỗi tệp `.py` trong `src/` đều tích hợp sẵn khối cấu hình `AI_AGENT_INSTRUCTIONS` ở phần đầu. Thành viên chỉ cần mở file bằng công cụ AI và yêu cầu thực hiện / tinh chỉnh, AI Agent sẽ đọc các chỉ thị và tuân thủ các ràng buộc (không shuffle, shift 1 trước khi tính rolling, xuất đúng định dạng file).
