# TIME SERIES LAB 2 — NHÓM 7 (Phân tích dữ liệu chuỗi thời gian và dự báo)
## BÀI THỰC HÀNH 2: KIỂM ĐỊNH ADF & XÂY DỰNG MÔ HÌNH ARIMA / SARIMA
### Bộ dữ liệu: Monthly Retail Sales (`a10.csv`)

> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Nhóm thực hiện:** Nhóm 7 (Hà Duy Hưng (Lead), Thành viên 2, Thành viên 3, Thành viên 4, Thành viên 5, Thành viên 6)  
> **GitHub Repository:** [https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git)  

---

## 1. PHÂN CÔNG KHỐI LƯỢNG CÔNG VIỆC (6 THÀNH VIÊN)

| Mã CV | Thành viên | Nhiệm vụ chính trong Pipeline | Tệp thực thi dự kiến |
| :---: | :--- | :--- | :--- |
| **CV01** | **Hà Duy Hưng (Lead)** | Quản trị Pipeline, Tải dữ liệu, Phân tích EDA, Phân chia Train/Test 80/20 | `src/cv01_eda_and_data_prep.py` |
| **CV02** | **Thành viên 2** | Kiểm định tính dừng ADF trên chuỗi gốc, thực hiện sai phân bậc 1 và kiểm định ADF lần 2 | `src/cv02_adf_and_differencing.py` |
| **CV03** | **Thành viên 3** | Phân tích đồ thị ACF & PACF, quét lưới AIC/BIC, đề xuất các cấu hình $(p, d, q)$ | `src/cv03_acf_pacf_order_selection.py` |
| **CV04** | **Thành viên 4** | Huấn luyện mô hình ARIMA, chẩn đoán phần dư (Residual Diagnostics), dự báo Test và tính MAE/RMSE/MAPE | `src/cv04_arima_model_evaluation.py` |
| **CV05** | **Thành viên 5** | Mở rộng mô hình SARIMA xử lý tính mùa vụ chu kỳ $s = 12$, đối chuẩn trực tiếp với ARIMA | `src/cv05_sarima_seasonal_modeling.py` |
| **CV06** | **Thành viên 6** | Tổng hợp đối chuẩn toàn diện, kiểm định Walk-forward, xuất bản báo cáo kỹ thuật hoàn chỉnh | `src/cv06_report_synthesis.py` |

---

## 2. HƯỚNG DẪN THỰC THI PIPELINE

```bash
cd time-series-lab2
pip install -r requirements.txt

# Thứ tự thực thi tuần tự
python src/cv01_eda_and_data_prep.py
python src/cv02_adf_and_differencing.py
python src/cv03_acf_pacf_order_selection.py
python src/cv04_arima_model_evaluation.py
python src/cv05_sarima_seasonal_modeling.py
python src/cv06_report_synthesis.py
```

---

## 3. TÀI LIỆU LIÊN QUAN
- [Đặc tả Đề bài Lab 02](docs/DE_BAI_LAB02.md)
- [Kiến trúc Pipeline chi tiết](PIPELINE.md)
