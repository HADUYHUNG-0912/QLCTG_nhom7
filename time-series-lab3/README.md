# TIME SERIES LAB 3 — NHÓM 7 (QLCTG)
## BÀI THỰC HÀNH 4: MULTIVARIATE TIME SERIES FORECASTING
### (VAR vs XGBoost vs LSTM)

> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Nhóm thực hiện:** Nhóm 7 (6 thành viên)  
> **Bộ dữ liệu:** Electricity Load Diagrams Dataset (UCI) + Weather Data  

---

## 1. PHÂN CÔNG THÀNH VIÊN

| Mã CV | Thành viên | Nhiệm vụ | Deadline |
| :---: | :--- | :--- | :---: |
| CV01 | **Hà Duy Hưng (Lead)** | EDA + Preprocessing + Lag Features | 05/11 |
| CV02 | Khải | Mô hình VAR (Statistical Baseline) | 08/11 |
| CV03 | Hoàng | Mô hình XGBoost (Machine Learning) | 08/11 |
| CV04 | Trâm | Mô hình LSTM (Deep Learning) | 12/11 |
| CV05 | Hiệp | Walk-forward Backtesting + So sánh | 12/11 |
| CV06 | Phương | Tổng hợp báo cáo + 6 câu hỏi lý luận | 14/11 |

---

## 2. CÀI ĐẶT MÔI TRƯỜNG

```bash
cd time-series-lab3
pip install -r requirements.txt
```

## 3. THỨ TỰ CHẠY

```bash
python src/cv01_eda_and_preprocessing.py
python src/cv02_var_model.py
python src/cv03_xgboost_model.py
python src/cv04_lstm_model.py
python src/cv05_backtesting_comparison.py
python src/cv06_report_synthesis.py
```

## 4. TÀI LIỆU

- [Đề bài đầy đủ](docs/DE_BAI_LAB.md)
- [Kế hoạch phân công](docs/KE_HOACH_PHAN_CONG_NHOM7.md)
- [Báo cáo kỹ thuật](docs/BAO_CAO_LAB4_NHOM7.md) *(CV06 sẽ tạo)*

