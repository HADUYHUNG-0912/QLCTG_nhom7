# KẾ HOẠCH THỰC HIỆN & PHÂN CÔNG NHIỆM VỤ — NHÓM 7
## BÀI THỰC HÀNH 2: TIME SERIES FORECASTING & MODELING
### Môn học: Phân tích dữ liệu chuỗi thời gian và dự báo

> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Nhóm trưởng:** Hà Duy Hưng (CV01 — Lead)  
> **Tổng số thành viên:** 6 người  
> **Kho mã nguồn:** `e:\Chuỗi_TG\time-series-lab2\`  

---

## 👥 1. MA TRẬN PHÂN CÔNG CHI TIẾT (6 THÀNH VIÊN)

| Mã CV | Thành viên | Nhánh Git | Vai trò phụ trách | Tệp thực thi dự kiến | Trạng thái |
| :---: | :--- | :---: | :--- | :--- | :---: |
| **CV01** | **Hà Duy Hưng (Lead)** | `main` | EDA, Tiền xử lý, Trích xuất Lag Features | `src/cv01_eda_and_preprocessing.py` | ⏳ Kế hoạch |
| **CV02** | **Hiệp** | `main` | Xây dựng Mô hình Baseline Thống kê | `src/cv02_baseline_model.py` | ⏳ Kế hoạch |
| **CV03** | **Phương** | `main` | Huấn luyện Mô hình Machine Learning | `src/cv03_model_training.py` | ⏳ Kế hoạch |
| **CV04** | **Châm** | `cham-lab02` | Mở rộng Feature Engineering & Ablation | `src/cv04_feature_engineering.py` | ⏳ Kế hoạch |
| **CV05** | **Hoàng** | `hoang-lab02` | Walk-forward Backtesting & Đối chuẩn Benchmark | `src/cv05_backtesting_comparison.py` | ⏳ Kế hoạch |
| **CV06** | **Khải** | `khai-lab02` | Tổng hợp Báo cáo Kỹ thuật & Lý luận Chuyên sâu | `src/cv06_report_synthesis.py` | ⏳ Kế hoạch |

---

## 📦 2. HỢP ĐỒNG ĐẦU VÀO / ĐẦU RA (I/O CONTRACTS)

- **CV01:** Nhận `data/raw/`, xuất `data/processed/train.csv`, `val.csv`, `test.csv`, biểu đồ EDA và bảng audit chất lượng.
- **CV02:** Nhận `data/processed/`, huấn luyện mô hình cơ sở, xuất `outputs/models/baseline_model.*` và metrics.
- **CV03:** Huấn luyện mô hình học máy nâng cao, tinh chỉnh siêu tham số, xuất mô hình và learning curve.
- **CV04:** Tạo đặc trưng thống kê cuộn, kiểm tra tính đóng góp của đặc trưng (Feature Importance).
- **CV05:** Thực hiện kiểm thử cuộn Walk-forward trên tập Test, đối chuẩn đa mô hình, xuất `outputs/tables/final_comparison.csv`.
- **CV06:** Hợp nhất toàn bộ biểu đồ, bảng dữ liệu và hoàn thành báo cáo `docs/BAO_CAO_LAB2_NHOM7.md`.
