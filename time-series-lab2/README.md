# TIME SERIES LAB 2 — NHÓM 7 (Phân tích dữ liệu chuỗi thời gian và dự báo)
## BÀI THỰC HÀNH SỐ 2: MÔ HÌNH HÓA VÀ DỰ BÁO CHUỖI THỜI GIAN

> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Nhóm thực hiện:** Nhóm 7 (6 thành viên)  
> **GitHub Repository:** [https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git)  

---

## 1. CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT DIRECTORY TREE)

Cấu trúc thư mục được thiết kế và module hóa chuẩn mực theo quy chuẩn kỹ thuật đồng nhất với các bài Lab trong hệ thống:

```text
time-series-lab2/
├── README.md               # Giới thiệu tổng quan, cài đặt, hướng dẫn thực thi
├── requirements.txt        # Danh sách các thư viện phụ thuộc
├── DE_BAI_LAB.md           # Đề bài và yêu cầu kỹ thuật chi tiết
│
├── data/
│   ├── raw/                # Dữ liệu gốc ban đầu (.csv)
│   └── processed/          # Dữ liệu sau khi làm sạch, chuẩn hóa, tạo lag features
│
├── docs/
│   ├── DE_BAI_LAB.md       # Bản sao đề bài lưu trữ kỹ thuật
│   ├── KE_HOACH_PHAN_CONG_NHOM7.md # Ma trận phân công nhiệm vụ 6 thành viên
│   └── PIPELINE.md         # Kiến trúc quy trình kỹ thuật & luồng dữ liệu
│
├── notebooks/              # Jupyter Notebooks tương ứng từng thành viên thực nghiệm
│
├── outputs/
│   ├── figures/            # Biểu đồ trực quan hóa xuất bản (PNG, DPI 150)
│   ├── models/             # Trọng số và file nhị phân mô hình đã huấn luyện (.pkl, .json)
│   └── tables/             # Bảng kết quả định lượng, metrics đối chuẩn (.csv, .txt)
│
├── src/                    # Mã nguồn thực thi Python module hóa (CV01 -> CV06)
│
└── tests/                  # Kiểm thử đơn vị (Unit Tests) & kiểm toán dữ liệu
```

---

## 2. PHÂN CÔNG THÀNH VIÊN (6 THÀNH VIÊN)

| Mã CV | Thành viên | Nhánh làm việc | Nhiệm vụ chính | Trạng thái |
| :---: | :--- | :---: | :--- | :---: |
| **CV01** | **Hà Duy Hưng (Lead)** | `main` | Quản trị Pipeline, EDA & Preprocessing | ⏳ Kế hoạch |
| **CV02** | **Hiệp** | `main` | Xây dựng Mô hình Baseline | ⏳ Kế hoạch |
| **CV03** | **Phương** | `main` | Huấn luyện & Tối ưu mô hình | ⏳ Kế hoạch |
| **CV04** | **Châm** | `cham-lab02` | Mở rộng tính năng & Đánh giá nâng cao | ⏳ Kế hoạch |
| **CV05** | **Hoàng** | `hoang-lab02` | Kiểm thử Walk-forward & Đối chuẩn mô hình | ⏳ Kế hoạch |
| **CV06** | **Khải** | `khai-lab02` | Tổng hợp báo cáo kỹ thuật & Phân tích lý luận | ⏳ Kế hoạch |

---

## 3. CÀI ĐẶT MÔI TRƯỜNG & HƯỚNG DẪN THỰC THI

```bash
# 1. Điều hướng vào thư mục bài thực hành
cd time-series-lab2

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 3. Thứ tự thực thi pipeline tuần tự
python src/cv01_eda_and_preprocessing.py
python src/cv02_baseline_model.py
python src/cv03_model_training.py
python src/cv04_feature_engineering.py
python src/cv05_backtesting_comparison.py
python src/cv06_report_synthesis.py
```

---

## 4. TÀI LIỆU KỸ THUẬT LIÊN QUAN

- [Đề bài chi tiết](DE_BAI_LAB.md)
- [Kế hoạch phân công nhiệm vụ](docs/KE_HOACH_PHAN_CONG_NHOM7.md)
- [Kiến trúc Pipeline xử lý](docs/PIPELINE.md)
