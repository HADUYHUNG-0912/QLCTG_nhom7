# ĐỀ BÀI VÀ HƯỚNG DẪN THỰC HÀNH LAB 2
## KHÁM PHÁ ĐẶC TÍNH DỮ LIỆU CHUỖI THỜI GIAN
**(Time Series Exploratory Data Analysis & Data Quality Engineering)**

- **Môn học:** Quản lý & Khai phá Chuỗi Thời Gian (QLCTG)
- **Đơn vị đào tạo:** Khoa Công nghệ Thông tin - Trường Đại học Giao thông Vận tải TP.HCM (UTH)
- **Giảng viên phụ trách:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)
- **Nhóm thực hiện:** Nhóm 7
- **Bộ dữ liệu thực nghiệm:** `AirPassengers.csv` (144 quan sát tháng, từ 01/1949 đến 12/1960)

---

## MỤC TIÊU TỔNG QUAN CỦA BÀI THỰC HÀNH

Sau khi hoàn thành 6 bài tập trong đợt thực hành này, sinh viên phải nắm vững:
1. Bản chất toán học của hàm tự tương quan (ACF) và tự tương quan riêng phần (PACF), ý nghĩa của các độ trễ (lags) và cách suy luận tính dừng, xu hướng, mùa vụ.
2. Khái niệm và phương pháp kiểm định thống kê chuỗi Nhiễu trắng (White Noise), phân biệt chuỗi có thể dự báo và chuỗi ngẫu nhiên thuần túy.
3. Kỹ thuật phân rã chuỗi thời gian (Time Series Decomposition) theo mô hình Cộng (Additive), Nhân (Multiplicative) và STL (Loess Smoothing).
4. Các vấn đề chất lượng dữ liệu Time Series: Kỹ thuật xử lý missing values (bằng Time Interpolation) và phát hiện, làm sạch ngoại lai (Outliers) bằng thuật toán Hampel Filter.
5. Kỹ thuật trực quan hóa chuyên sâu: Thống kê trượt (Rolling Statistics), đồ thị mùa vụ theo tháng (Seasonal Plot) và bản đồ nhiệt chu kỳ (Seasonality Heatmap).
6. Kỹ năng tổng hợp báo cáo kỹ thuật nghiệm thu và xây dựng pipeline xử lý dữ liệu chuẩn công nghiệp.

---

## NỘI DUNG CHI TIẾT 6 BÀI TẬP THỰC HÀNH

### BÀI TẬP 1: TÍNH VÀ PHÂN TÍCH BIỂU ĐỒ ACF & PACF
- **Mã nguồn:** `notebooks/bt1_acf_pacf.ipynb` hoặc `src/bt1_acf_pacf.py`
- **Tệp xuất ra:** `outputs/figures/bai1_acf_pacf.png`
- **Nội dung yêu cầu:**
  1. Tải bộ dữ liệu `AirPassengers.csv`, chuyển cột `Month` thành kiểu `DatetimeIndex` với tần suất tháng (`MS`).
  2. Tính toán hệ số tự tương quan (ACF) và tự tương quan riêng phần (PACF) đến tối thiểu 40 lags.
  3. Vẽ biểu đồ ACF và PACF trên cùng một giao diện hình ảnh (2 subplots xếp chồng dọc), hiển thị rõ dải tin cậy 95% Bartlett:
     $$\text{CI}_{95\%} = \pm \frac{1.96}{\sqrt{n}}$$
  4. Trả lời và phân tích 3 câu hỏi nghiệm thu:
     - Tự tương quan của chuỗi có mạnh không? Có suy giảm chậm (tail-off) không?
     - Có dấu hiệu của xu hướng (Trend) và tính mùa vụ (Seasonality) hay không?
     - Những lag nào có ý nghĩa thống kê quan trọng nhất? (Gợi ý: Lag 1 phản ánh xu hướng, Lag 12 phản ánh đỉnh mùa vụ chu kỳ 1 năm).

---

### BÀI TẬP 2: KIỂM ĐỊNH CHUỖI NHIỄU TRẮNG (WHITE NOISE TESTING)
- **Mã nguồn:** `notebooks/bt2_white_noise.ipynb` hoặc `src/bt2_white_noise.py`
- **Tệp xuất ra:** `outputs/figures/bai2_white_noise_comparison.png`, `outputs/tables/ljung_box_test_results.csv`
- **Nội dung yêu cầu:**
  1. Sinh chuỗi Gaussian White Noise ngẫu nhiên với kích thước mẫu $N = 1000$, kỳ vọng $\mu = 0$, độ lệch chuẩn $\sigma = 1$ (cố định `seed=42`).
  2. Vẽ biểu đồ đối chứng dạng sóng (Waveform) và biểu đồ ACF (30 lags) của White Noise đặt cạnh chuỗi dữ liệu thực tế `AirPassengers`.
  3. Thực hiện kiểm định thống kê Ljung-Box Test tại các độ trễ $h = 10$ và $h = 20$:
     $$Q(h) = n(n+2)\sum_{k=1}^h \frac{\hat{\rho}_k^2}{n-k}$$
  4. Trình bày đoạn văn nhận xét so sánh (tối thiểu 5 dòng):
     - Biểu đồ ACF của White Noise có nằm trong dải tin cậy 95% không?
     - Trị số p-value của kiểm định Ljung-Box trên White Noise so với AirPassengers như thế nào?
     - Ý nghĩa thực tế: Chuỗi nào có thể dự báo được, chuỗi nào không thể dự báo được và vì sao?

---

### BÀI TẬP 3: PHÂN RÃ CHUỖI THỜI GIAN (TIME-SERIES DECOMPOSITION)
- **Mã nguồn:** `notebooks/bt3_decomposition.ipynb` hoặc `src/bt3_decomposition.py`
- **Tệp xuất ra:** `outputs/figures/bai3_1_additive_decomposition.png`, `bai3_2_multiplicative_decomposition.png`, `bai3_3_stl_decomposition.png`, `outputs/tables/decomposition_summary.csv`
- **Nội dung yêu cầu:**
  1. Thực hiện phân rã chuỗi theo phương pháp cổ điển (Classical Decomposition):
     - Dạng Cộng: $Y_t = T_t + S_t + R_t$
     - Dạng Nhân: $Y_t = T_t \times S_t \times R_t$
  2. Thực hiện phân rã nâng cao bằng phương pháp STL (Seasonal and Trend decomposition using Loess) với tính năng `robust=True`.
  3. Trích xuất và trực quan hóa 4 thành phần: Dữ liệu quan sát (`Observed`), Xu hướng (`Trend`), Mùa vụ (`Seasonal`), Phần dư (`Residual`).
  4. Phân tích học thuật:
     - Xu hướng dài hạn tăng hay giảm? Tốc độ tăng trưởng qua các năm là bao nhiêu phần trăm?
     - Tính mùa vụ diễn ra như thế nào? Tháng nào là đỉnh cao điểm (Peak), tháng nào là đáy (Trough)?
     - So sánh giữa dạng Cộng và dạng Nhân: Dạng nào phù hợp với dữ liệu `AirPassengers` hơn? Vì sao?
     - Đánh giá thành phần phần dư (Residual): Có mang tính chất của White Noise không?

---

### BÀI TẬP 4: KIỂM TRA VÀ XỬ LÝ CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY CLEANING)
- **Mã nguồn:** `notebooks/bt4_data_quality.ipynb` hoặc `src/bt4_data_quality.py`
- **Tệp xuất ra:** `outputs/figures/bai4_data_cleaning_before_after.png`, `outputs/tables/data_cleaning_summary.csv`, `data/processed/AirPassengers_cleaned.csv`
- **Nội dung yêu cầu:**
  1. Kiểm tra tính toàn vẹn của chuỗi dữ liệu gốc (số lượng missing, kiểm tra trùng lặp chỉ mục).
  2. Tạo bản sao khuyết tật mô phỏng:
     - Tạo ngẫu nhiên 10% giá trị bị thiếu (Missing NaN).
     - Tiêm 3 giá trị ngoại lai đột biến (Outlier spikes nhân 3.5 lần giá trị gốc) tại các vị trí chỉ số `[25, 75, 110]`.
  3. So sánh 2 phương pháp bù khuyết dữ liệu thiếu:
     - Phương pháp Forward-fill (`ffill`).
     - Phương pháp Nội suy theo thời gian (`interpolate(method='time')`).
     - Đo sai số MAE và RMSE trên các điểm bị khuyết.
  4. So sánh 3 phương pháp phát hiện ngoại lai:
     - Rolling Z-Score (dựa trên Rolling Mean & Std).
     - Rolling IQR (dựa trên phân vị $Q_1, Q_3$).
     - Hampel Filter (dựa trên Rolling Median và Median Absolute Deviation - MAD).
  5. Xây dựng pipeline hoàn chỉnh: Nội suy thời gian $\rightarrow$ Lọc Hampel Filter $\rightarrow$ Khôi phục chuỗi dữ liệu sạch. So sánh sai số toàn diện Trước và Sau khi làm sạch.

---

### BÀI TẬP 5: TRỰC QUAN HÓA CHUYÊN SÂU CHUỖI THỜI GIAN (ADVANCED EDA)
- **Mã nguồn:** `notebooks/bt5_visualization.ipynb` hoặc `src/bt5_visualization.py`
- **Tệp xuất ra:** `outputs/figures/bai5_1_rolling_statistics.png`, `bai5_2_seasonal_plot.png`, `bai5_3_seasonality_heatmap.png`, `outputs/tables/monthly_seasonal_summary.csv`
- **Nội dung yêu cầu:**
  1. Trực quan hóa Thống kê trượt (Rolling Statistics):
     - Vẽ đường trung bình trượt 12 tháng (`Rolling Mean`, $w=12$).
     - Vẽ dải biến thiên $\pm 2$ độ lệch chuẩn trượt (`Rolling Std`).
     - Đánh giá hiện tượng biến thiên phương sai (Heteroskedasticity).
  2. Trực quan hóa Mùa vụ theo tháng qua các năm (Monthly Seasonal Plot):
     - Vẽ 12 đường cong đại diện cho 12 năm quan sát trên cùng một hệ trục tọa độ 12 tháng.
     - Quan sát tính đồng dạng của các bước sóng và xác định tháng đỉnh hè, tháng đáy cuối năm.
  3. Xây dựng Bản đồ nhiệt Mùa vụ (Seasonality Heatmap):
     - Tái cấu trúc ma trận Pivot (Năm $\times$ Tháng, kích thước $12 \times 12$).
     - Tô màu theo thang đo mật độ hành khách (`YlGnBu`), nhận diện "vùng nhiệt cao" và "vùng nhiệt thấp".
  4. Trả lời 5 câu hỏi nghiệm thu chuẩn mực của đề bài.

---

### BÀI TẬP 6: BÁO CÁO KỸ THUẬT TỔNG KẾT (COMPREHENSIVE LAB REPORT)
- **Tệp báo cáo:** `docs/bt6_report.md`
- **Tệp Word nộp bài:** `docs/project_management/BAO_CAO_LAB2.docx`
- **Nội dung yêu cầu:**
  1. Tổng hợp toàn diện kết quả thực nghiệm từ Bài tập 1 đến Bài tập 5.
  2. Bổ sung bảng biểu số liệu, công thức toán học minh chứng cho mọi kết luận.
  3. Nhúng đầy đủ các hình ảnh biểu đồ chất lượng cao xuất ra từ thư mục `outputs/figures/`.
  4. Đưa ra đánh giá chuyên sâu về tiềm năng áp dụng các mô hình dự báo tương lai (SARIMA, Holt-Winters, Prophet, XGBoost with Lag Features).

---

## TIÊU CHÍ ĐÁNH GIÁ NGHIỆM THU (RUBRICS)

| Hạng mục | Trọng số | Tiêu chí đạt điểm tối đa |
| :--- | :---: | :--- |
| **Cấu trúc thư mục chuẩn hóa** | 15% | Đầy đủ thư mục `data/`, `docs/`, `notebooks/`, `outputs/`, `src/` theo đúng thiết kế khoa học. |
| **Chất lượng mã nguồn** | 30% | Mã nguồn viết module hóa, có chú thích giải thích rõ ràng, chạy thông suốt không lỗi, tương thích cả CLI và Jupyter Notebooks. |
| **Phân tích học thuật & Trực quan** | 30% | Biểu đồ sắc nét, phông chữ hiển thị chuẩn tiếng Việt, đồ thị đầy đủ nhãn trục, chú giải và tiêu đề. Phân tích sâu sắc, lập luận logic. |
| **Báo cáo tổng hợp** | 25% | Báo cáo Markdown & Word đầy đủ các phần, công thức toán học rõ ràng, bảng kết quả định lượng chuẩn xác và liên kết chéo thông minh. |
