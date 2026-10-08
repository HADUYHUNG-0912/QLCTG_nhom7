# BẢNG PHÂN CÔNG NHIỆM VỤ NHÓM 7 - LAB 2 (QLCTG)
## TIME SERIES LAB 2: KHÁM PHÁ ĐẶC TÍNH DỮ LIỆU CHUỖI THỜI GIAN

> **Học phần:** Quản lý & Khai phá Chuỗi Thời gian (QLCTG) - UTH  
> **Nhóm thực hiện:** Nhóm 7 (`QLCTG_nhom7`)  
> **Nhóm trưởng:** Hưng  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (tienntk@ut.edu.vn)  
> **Tệp Excel:** [ke_hoach_phan_cong_nhom7.xlsx](file:///e:/Chuỗi_TG/LAB2/ke_hoach_phan_cong_nhom7.xlsx)  
> **GitHub Repository:** [HADUYHUNG-0912/QLCTG_nhom7](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git)

---

### MA TRẬN PHÂN CÔNG NHIỆM VỤ CHI TIẾT (ĐỒNG BỘ 100% CẤU TRÚC DỰ ÁN)

| MÃ CV | HẠNG MỤC CÔNG VIỆC | MÔ TẢ CHI TIẾT & YÊU CẦU THỰC HIỆN | TỆP BÀN GIAO (DELIVERABLES) | PHỤ TRÁCH | DEADLINE | TRẠNG THÁI |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **CV01** | **Quản trị dự án, Git Workflow & Báo cáo kỹ thuật tổng hợp (Bài 6)** | • Thiết lập cấu trúc repo đồng bộ 5 bài tập, cấu hình `.gitignore` và `requirements.txt`.<br>• Xây dựng kiến trúc module chuẩn 4 bước và chỉ thị AI Agent Prompt cho 5 bài tập.<br>• Quản lý nhánh Git, phối hợp nhóm 6 người và soát xét code (Code Review).<br>• Chủ trì viết Báo cáo kỹ thuật tổng kết 7 phần (Bài tập 6) và bộ slide bảo vệ.<br>• Kiểm thử luồng dữ liệu End-to-End từ Dữ liệu gốc $\rightarrow$ Xử lý $\rightarrow$ Phân rã $\rightarrow$ Dự báo. | • [LAB2/README.md](README.md)<br>• [LAB2/data/raw/AirPassengers.csv](data/raw/AirPassengers.csv)<br>• `LAB2/bai_tap_6/BAO_CAO_LAB2_NHOM7.md`<br>• Lời giải 12 câu hỏi ôn tập cuối chương<br>• Slide thuyết trình Lab 2 | **Hưng (Lead)** | *(Để trống)* | 🟡 **Đang thực hiện** |
| **CV02** | **Tính toán & phân tích hàm tự tương quan ACF / PACF (Bài tập 1)** | • Nạp chuỗi `AirPassengers` từ `data/raw/`, thiết lập DatetimeIndex tần suất tháng (`MS`).<br>• Tính toán hệ số ACF và PACF đến 40 lags, xác định dải tin cậy 95% ($\pm 1.96/\sqrt{N}$).<br>• Phân tích tính tự tương quan mạnh, suy giảm chậm (trend) và đỉnh mùa vụ lặp lại tại các bội số 12 (Lag 12, 24, 36).<br>• Giải thích hiện tượng cắt cụt (cut-off) của PACF sau Lag 1 và đề xuất mô hình SARIMA.<br>• Trả lời 3 câu hỏi nghiệm thu Bài 1 & câu hỏi lý thuyết 1, 2, 10. | • Thư mục: `LAB2/bai_tap_1/`<br>• [bai_tap_1.py](bai_tap_1/bai_tap_1.py) (Chạy độc lập, chuẩn 4 bước)<br>• [ket_qua_bai_1.txt](bai_tap_1/ket_qua_bai_1.txt) (Báo cáo phân tích)<br>• *(Không lưu file ảnh .png ra đĩa để giữ repo nhẹ)* | **Thành viên 2** | *(Để trống)* | 🔵 **Review** |
| **CV03** | **Khảo sát & kiểm định chuỗi Nhiễu trắng - White Noise (Bài tập 2)** | • Sinh 1000 điểm Gaussian White Noise bằng NumPy (`np.random.normal`, seed=42, $\mu=0, \sigma=1$).<br>• Tính ACF của White Noise, đối chiếu toàn bộ các lag có nằm trọn trong dải tin cậy 95%.<br>• Thực thi kiểm định thống kê Ljung-Box Test (`acorr_ljungbox`) tại lag 10 và 20.<br>• So sánh trực quan giữa White Noise và chuỗi thực tế `AirPassengers`.<br>• Soạn thảo đúng 5 dòng nhận xét học thuật & trả lời câu hỏi lý thuyết 3, 4, 11. | • Thư mục: `LAB2/bai_tap_2/`<br>• [bai_tap_2.py](bai_tap_2/bai_tap_2.py)<br>• [ket_qua_bai_2.txt](bai_tap_2/ket_qua_bai_2.txt) (5 dòng chuẩn đề bài)<br>• Bảng p-value kiểm định Ljung-Box | **Thành viên 3** | *(Để trống)* | 🔵 **Review** |
| **CV04** | **Phân rã cấu trúc chuỗi thời gian - Time Series Decomposition (Bài tập 3)** | • Phân rã chuỗi `AirPassengers` (period=12) thành 4 thành phần: Observed, Trend, Seasonal, Residual.<br>• So sánh mô hình Cộng (Additive) vs Nhân (Multiplicative), lập luận khoa học vì sao dạng Nhân vượt trội do biên độ mùa vụ tăng theo xu hướng.<br>• Áp dụng phương pháp phân rã hiện đại STL (Seasonal and Trend decomposition using Loess, `robust=True`).<br>• Kiểm tra tính chất của Residual: Dao động quanh 1.0, tiệm cận White Noise.<br>• Trả lời 4 câu hỏi nghiệm thu Bài 3 & câu hỏi lý thuyết 5, 6. | • Thư mục: `LAB2/bai_tap_3/`<br>• [bai_tap_3.py](bai_tap_3/bai_tap_3.py)<br>• [ket_qua_bai_3.txt](bai_tap_3/ket_qua_bai_3.txt)<br>• Đánh giá định lượng tăng trưởng xu hướng (+274.7%) | **Thành viên 4** | *(Để trống)* | 🔵 **Review** |
| **CV05** | **Kiểm tra & xử lý chất lượng dữ liệu - Data Cleaning (Bài tập 4)** | • Bắt buộc tạo bản sao độc lập (`series.copy()`), tuyệt đối không sửa trực tiếp chuỗi gốc.<br>• Giả lập khuyết tật dữ liệu: 10% Missing (NaN ngẫu nhiên, seed=101) và 3 điểm Spikes ngoại lai phóng đại 3.5 lần.<br>• So sánh định lượng 2 kỹ thuật xử lý Missing: Forward-fill (`ffill`) vs Time-based Interpolation.<br>• Phát hiện và làm sạch điểm ngoại lai bằng bộ lọc kháng nhiễu Hampel Filter (Rolling Median $\pm 3 \times \text{MAD}$).<br>• Tính toán sai số khôi phục tổng thể (MAE, RMSE) & trả lời câu hỏi lý thuyết 7, 8. | • Thư mục: `LAB2/bai_tap_4/`<br>• [bai_tap_4.py](bai_tap_4/bai_tap_4.py)<br>• [ket_qua_bai_4.txt](bai_tap_4/ket_qua_bai_4.txt)<br>• Bảng so sánh sai số MAE/RMSE trước và sau làm sạch | **Thành viên 5** | *(Để trống)* | 🔵 **Review** |
| **CV06** | **Trực quan hóa chuyên sâu & Bản đồ nhiệt Mùa vụ (Bài tập 5)** | • Vẽ biểu đồ đường toàn bộ chuỗi `AirPassengers` (1949–1960).<br>• Tính 12-Month Rolling Mean làm mịn xu hướng và dải biên độ dao động $\pm 2$ Rolling Std.<br>• Chứng minh hiện tượng phương sai thay đổi (Heteroskedasticity: độ lệch chuẩn tăng 5.67 lần từ 1949 đến 1960).<br>• Vẽ Seasonal Plot theo tháng qua các năm (12 đường cong đồng dạng tịnh tiến dần lên trên).<br>• Xây dựng Bản đồ nhiệt Seasonality Heatmap (Năm $\times$ Tháng) bằng `seaborn.heatmap`.<br>• Kết luận đỉnh tháng 7, đáy tháng 11 & trả lời câu hỏi lý thuyết 9, 12. | • Thư mục: `LAB2/bai_tap_5/`<br>• [bai_tap_5.py](bai_tap_5/bai_tap_5.py)<br>• [ket_qua_bai_5.txt](bai_tap_5/ket_qua_bai_5.txt)<br>• Báo cáo phân tích quy luật mùa vụ và phương sai | **Thành viên 6** | *(Để trống)* | 🔵 **Review** |
| **CV07** | **Tổng kết toàn diện, Code Review chéo & Bảo vệ trước Giảng viên** | • Toàn bộ 6 thành viên rà soát chéo mã nguồn và báo cáo kết quả của nhau.<br>• Kiểm thử đồng bộ toàn bộ 5 script python: Chạy độc lập từ console thoát với mã 0.<br>• Tổng hợp thành Báo cáo kỹ thuật hoàn chỉnh 7 phần cho Giảng viên.<br>• Chuẩn bị câu trả lời cho 12 câu hỏi ôn tập cuối chương 2 để sẵn sàng vấn đáp bảo vệ. | • Trọn bộ 5 thư mục bài tập chuẩn mực<br>• Báo cáo kỹ thuật tổng kết (Bài 6)<br>• Slide báo cáo nhóm 7<br>• Toàn bộ nhóm nắm vững kiến thức | **Cả nhóm (6 TV)** | *(Để trống)* | 🟡 **Đang thực hiện** |

> **Quy chuẩn dữ liệu xác thực (Data Validation) trong tệp Excel:**  
> Cột **TRẠNG THÁI** trong tệp Excel đã được cài đặt danh sách Dropdown tự động đổi màu:  
> - 🟡 **Đang thực hiện:** Màu vàng nhạt (`#FFF2CC`) - Dành cho các hạng mục đang trong tiến trình triển khai.  
> - 🔵 **Review:** Màu xanh lam pastel (`#DDEBF7`) - Dành cho bài tập đã hoàn thành mã nguồn, đang chờ kiểm tra/nghiệm thu.  
> - 🟢 **Done:** Màu xanh lục tươi (`#E2EFDA`) - Dành cho bài tập đã được nhóm trưởng nghiệm thu đạt chuẩn.  
> Cột **DEADLINE** được để trống hoàn toàn để nhóm trưởng chủ động điền mốc thời gian phù hợp với tiến độ lớp học.

---

### DANH MỤC TÀI NGUYÊN & QUY CHUẨN KỸ THUẬT DÀNH CHO THÀNH VIÊN

1. **Kho lưu trữ GitHub chính thức:** [HADUYHUNG-0912/QLCTG_nhom7](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git) (Nhánh làm việc chính: `main`).
2. **Tài liệu hướng dẫn kỹ thuật Lab 2:** [LAB2/README.md](README.md) (Chứa toàn bộ lý thuyết, công thức toán và hướng dẫn từng bài).
3. **Bộ dữ liệu chuẩn mực:** [LAB2/data/raw/AirPassengers.csv](data/raw/AirPassengers.csv) (144 dòng quan sát, từ 1949-01 đến 1960-12, dùng chung cho cả 5 bài).
4. **Quy định kiến trúc mã nguồn:** Mỗi bài tập bắt buộc tuân theo 4 hàm:
   `step1_load_data` $\rightarrow$ `step2_execute_analysis` $\rightarrow$ `step3_visualize` $\rightarrow$ `step4_conclude_and_report`.
5. **Chỉ thị AI Agent Guardrails:** Đã tích hợp sẵn `AI_AGENT_INSTRUCTIONS` ở đầu mỗi tệp `.py` để chống sai lệch cấu trúc khi dùng Cursor/Copilot.
