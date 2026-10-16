# Quản lý & Khai phá Chuỗi Thời gian (QLCTG) - Nhóm 7

Repository lưu trữ toàn bộ mã nguồn, tài liệu và báo cáo thực hành môn học **Quản lý & Khai phá Chuỗi Thời gian (QLCTG)**.

- **Đơn vị đào tạo:** Trường Đại học Giao thông Vận tải TP.HCM (UTH)
- **Giảng viên phụ trách:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)
- **Nhóm sinh viên thực hiện:** Nhóm 7
- **GitHub Repository:** [https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git)

---

## Danh mục Dự án & Bài Thực hành

### [🚀 TIME-SERIES-LAB: Khám phá Đặc tính Dữ liệu & Kiến trúc Triển khai](time-series-lab/README.md)
Thư mục dự án thực hành chuẩn hóa theo quy chuẩn khoa học:
- **`data/`**: Chứa dữ liệu gốc [`raw/AirPassengers.csv`](time-series-lab/data/raw/AirPassengers.csv) và dữ liệu sau xử lý [`processed/`](time-series-lab/data/processed/).
- **`docs/`**: Toàn bộ tài liệu yêu cầu ([`lab2_description.md`](time-series-lab/docs/lab2_description.md)), mô tả dữ liệu ([`data_introduction.md`](time-series-lab/docs/data_introduction.md)), quyết định xử lý ([`data_processing.md`](time-series-lab/docs/data_processing.md)), lý thuyết 5 bài ([`theory.md`](time-series-lab/docs/theory.md)), bài giảng ([`lecture_notes.md`](time-series-lab/docs/lecture_notes.md)), báo cáo tổng hợp ([`bt6_report.md`](time-series-lab/docs/bt6_report.md)) và quản lý dự án ([`project_management/`](time-series-lab/docs/project_management/)).
- **`notebooks/`**: 5 Jupyter Notebooks (`.ipynb`) từ Bài 1 đến Bài 5 chuẩn mực, tương tác trực quan.
- **`outputs/`**: Thư mục lưu trữ biểu đồ trích xuất ([`figures/`](time-series-lab/outputs/figures/)) và bảng số liệu định lượng ([`tables/`](time-series-lab/outputs/tables/)).
- **`src/`**: Mã nguồn script Python độc lập (`.py`) cho phép chạy tự động hóa toàn bộ qua CLI.

---

### Tóm tắt Nội dung Thực hành:
1. **Kiến trúc Triển khai Dự án:** Luồng xử lý dữ liệu chuẩn (End-to-End Pipeline), Cấu trúc thư mục module hóa và Ngăn xếp công nghệ.
2. **Cơ sở Lý thuyết & Phương pháp luận Toán học:**
   - Hàm tự tương quan (ACF), Phương sai & Hiệp phương sai.
   - Hướng dẫn đọc biểu đồ ACF / PACF và lựa chọn mô hình ARIMA/SARIMA.
   - Bản chất và kiểm định Nhiễu trắng (White Noise).
   - Phân rã chuỗi thời gian (Classical vs STL Decomposition, Additive vs Multiplicative).
   - 11 Vấn đề chất lượng dữ liệu Time Series và giải pháp kỹ thuật triệt để (Hampel Filter).
   - Trực quan hóa dữ liệu (Rolling Stats, Seasonal Plot, Heatmap) và định hướng mô hình hóa XGBoost.
3. **Đề bài & Hướng dẫn Thực hành Chi tiết:**
   - Bài tập 1: Tính và phân tích biểu đồ ACF/PACF (40 lags).
   - Bài tập 2: Kiểm định chuỗi White Noise & so sánh chuỗi thực tế.
   - Bài tập 3: Phân rã chuỗi thời gian (Additive, Multiplicative, STL).
   - Bài tập 4: Kiểm tra và làm sạch dữ liệu (Missing values & Outlier filtering).
   - Bài tập 5: Trực quan hóa chuyên sâu (Rolling stats, Seasonal plots, Seasonality Heatmap).
   - Bài tập 6: Báo cáo kỹ thuật tổng kết đánh giá tiềm năng dự báo.
4. **Hệ thống 12 Câu hỏi Ôn tập Cuối chương có hướng dẫn trả lời chi tiết.**
5. **Quy trình Git Workflow và cộng tác nhóm.**


