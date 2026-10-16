# TIME SERIES LAB 2 - DỰ ĐOÁN CHUỖI THỜI GIAN BẰNG MACHINE LEARNING MODELS
## BÀI THỰC HÀNH: DỰ BÁO TIÊU THỤ NĂNG LƯỢNG HÀNG GIỜ BẰNG XGBOOST REGRESSOR

> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin - Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Nhóm thực hiện:** Nhóm 7  
> **GitHub Repository:** [https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git)  
> **Đề bài chi tiết:** [DE_BAI_LAB.md](DE_BAI_LAB.md) hoặc [docs/lab_description.md](docs/lab_description.md)

---

## 1. MỤC TIÊU VÀ ĐỀ BÀI THỰC HÀNH

- **Mục tiêu chính:**
  1. Chuyển bài toán chuỗi thời gian đơn biến thành bài toán học máy có giám sát (Supervised Learning).
  2. Huấn luyện mô hình XGBoost Regressor (`xgboost.XGBRegressor`).
  3. Đánh giá chất lượng dự báo và phân tích Feature Importance (ảnh hưởng của các độ trễ).

- **Dataset thực nghiệm:** [Hourly Energy Consumption (PJM) - AEP_hourly.csv](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv)  
  *Dữ liệu tiêu thụ năng lượng hàng giờ trong hơn 10 năm từ American Electric Power (AEP) thuộc PJM, tính bằng Megawatt (MW).*

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN (`time-series-lab2/`)

```text
time-series-lab2/
├── DE_BAI_LAB.md           # Đề bài thực hành chi tiết (chuyển từ file PDF/ảnh của giảng viên)
├── README.md               # Giới thiệu, hướng dẫn cài đặt và thứ tự chạy
├── requirements.txt        # Danh sách thư viện phụ thuộc (xgboost, scikit-learn, pandas, numpy,...)
│
├── data/
│   ├── raw/                # Dữ liệu gốc: AEP_hourly.csv
│   └── processed/          # Dữ liệu sau khi tạo lag features, train/test sets
│
├── docs/
│   ├── lab_description.md  # Toàn văn đề bài gốc và yêu cầu kỹ thuật chi tiết
│   ├── lab3_description.md # Bản sao theo tên bài thực hành 3
│   ├── data_introduction.md# Giới thiệu chi tiết dataset Hourly Energy Consumption (PJM)
│   ├── data_processing.md  # Kỹ thuật tạo Lag Features, Rolling Features, Time-based Split
│   ├── theory.md           # Cơ sở lý thuyết XGBoost, TimeSeriesSplit, Multi-step Forecast
│   └── bt_report.md        # Báo cáo kết quả dự báo và trả lời 3 câu hỏi lý luận
│
├── notebooks/
│   ├── 01_data_preparation_and_lags.ipynb # Bước 1-3: Chuẩn bị dữ liệu & tạo Lag Features
│   ├── 02_xgboost_training_eval.ipynb     # Bước 4-7: Huấn luyện XGBoost, Đánh giá & Feature Importance
│   └── 03_advanced_experiments.ipynb      # Nâng cao: Rolling Stats, Multi-step, TimeSeriesSplit, Random Forest
│
├── outputs/
│   ├── figures/            # Biểu đồ trích xuất: Actual vs Predicted, Feature Importance, Rolling Stats
│   └── tables/             # Bảng so sánh sai số định lượng (MAE, RMSE, MAPE, R2)
│
└── src/                    # Mã nguồn script Python (.py) thực thi tự động qua CLI
    ├── prepare_lags.py
    ├── train_xgboost.py
    └── advanced_tuning.py
```

---

## 3. NỘI DUNG THỰC HÀNH CỐT LÕI (7 BƯỚC)

1. **Bước 1. Chuẩn bị dữ liệu:** Nạp `AEP_hourly.csv`, parse `DatetimeIndex`, sort thời gian, kiểm tra missing.
2. **Bước 2. Tạo lag features:** Dùng `.shift(1)` đến `.shift(p)` chuyển đổi sang Supervised Learning.
3. **Bước 3. Chia $X - y$:** Tách ma trận đặc trưng $X = [y_{t-1}, \dots, y_{t-p}]$ và biến mục tiêu $y = y_t$.
4. **Bước 4. Chia train / test theo thời gian:** Tách theo chuỗi thời gian tuần tự (không shuffle để tránh rò rỉ dữ liệu).
5. **Bước 5. Huấn luyện mô hình XGBoost:** Khởi tạo `XGBRegressor` và fit trên tập huấn luyện.
6. **Bước 6. Dự đoán và đánh giá:** Đo MAE, RMSE, MAPE, $R^2$ và vẽ biểu đồ Actual vs Predicted.
7. **Bước 7. Phân tích feature importance:** Xác định các lag có ảnh hưởng mạnh nhất đến tải điện (Lag 1h, Lag 24h, Lag 168h).

---

## 4. BÀI TẬP NÂNG CAO (4 BÀI TOÁN MỞ RỘNG)

1. **Thêm rolling mean / rolling std:** Bổ sung trung bình trượt 24h, 168h và độ lệch chuẩn trượt để nắm bắt xu thế.
2. **Dự báo multi-step:** Triển khai dự báo nhiều giờ tiếp theo bằng chiến lược Recursive hoặc Direct.
3. **Tune hyperparameter bằng `TimeSeriesSplit`:** Tối ưu hóa siêu tham số bằng Cross-Validation chuyên biệt cho chuỗi thời gian.
4. **So sánh với Random Forest:** Đánh giá đối chứng hiệu năng giữa XGBoost và Random Forest.

---

## 5. HỆ THỐNG 3 CÂU HỎI LÝ LUẬN

1. **So sánh XGBoost với Linear Regression trong dự báo chuỗi thời gian?**
2. **Điều gì xảy ra khi tăng số lượng độ trễ $p$?** (Trade-off thông tin vs Overfitting và mất dữ liệu đầu chuỗi).
3. **Khi nào XGBoost phù hợp hơn mô hình truyền thống ARIMA?**

Chi tiết hướng dẫn giải và phân tích học thuật được trình bày tại [DE_BAI_LAB.md](DE_BAI_LAB.md).
