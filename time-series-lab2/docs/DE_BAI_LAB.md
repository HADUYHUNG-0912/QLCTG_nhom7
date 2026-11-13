# PRACTICE 2 – TIME SERIES FORECASTING & MODELING
## BÀI THỰC HÀNH 2: MÔ HÌNH HÓA VÀ DỰ BÁO CHUỖI THỜI GIAN

> **Môn học:** Phân tích dữ liệu chuỗi thời gian và dự báo  
> **Đơn vị đào tạo:** Khoa Công nghệ Thông tin — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Nhóm thực hiện:** Nhóm 7  

---

## 🎯 1. MỤC TIÊU CỦA BÀI THỰC HÀNH (OBJECTIVES)

1. **Khám phá và chuẩn bị dữ liệu (Data Preparation):** Kiểm tra tính dừng, khử xu hướng, khử tính mùa vụ và tạo các biến trễ (Lag features).
2. **Xây dựng mô hình dự báo:** Triển khai các phương pháp dự báo chuỗi thời gian từ mô hình thống kê cổ điển đến mô hình học máy.
3. **Đánh giá hiệu năng:** Đo lường sai số dự báo định lượng qua các thước đo chuẩn: RMSE, MAE, MAPE.
4. **Phân tích kết quả:** So sánh chất lượng dự báo giữa các mô hình và đề xuất giải pháp tối ưu.

---

## 📊 2. QUY TRÌNH KỸ THUẬT (PIPELINE)

- **Bước 1 (CV01):** Khám phá dữ liệu (EDA), phân chia Train/Val/Test theo thứ tự thời gian (không shuffle) và tạo ma trận Lag Features.
- **Bước 2 (CV02):** Huấn luyện mô hình Baseline và thiết lập chuẩn so sánh ban đầu.
- **Bước 3 (CV03):** Huấn luyện mô hình học máy nâng cao và tinh chỉnh siêu tham số.
- **Bước 4 (CV04):** Mở rộng đặc trưng (Rolling Statistics, Calendar features) và phân tích lỗi.
- **Bước 5 (CV05):** Kiểm thử cuộn (Walk-forward Validation) và lập bảng so sánh hiệu năng.
- **Bước 6 (CV06):** Tổng hợp báo cáo kỹ thuật hoàn chỉnh và trả lời các câu hỏi lý luận chuyên sâu.
