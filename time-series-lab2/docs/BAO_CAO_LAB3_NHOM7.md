# BÁO CÁO KỸ THUẬT TỔNG KẾT LAB 3 - NHÓM 7
## DỰ BÁO CHUỖI THỜI GIAN SỬ DỤNG XGBOOST REGRESSOR

> **Học phần:** Quản lý & Khai phá Chuỗi Thời gian (QLCTG) - Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Nhóm thực hiện:** Nhóm 7 (`QLCTG_nhom7`)  
> **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
> **Bộ dữ liệu thực nghiệm:** [Hourly Energy Consumption - AEP_hourly.csv](file:///e:/Chuỗi_TG/time-series-lab2/data/raw/AEP_hourly.csv) (121,273 mẫu quan sát từ 2004 đến 2018)  
> **Kho lưu trữ mã nguồn:** [HADUYHUNG-0912/QLCTG_nhom7](https://github.com/HADUYHUNG-0912/QLCTG_nhom7.git)

---

### PHẦN 1. BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ĐỐI CHỨNG ĐA MÔ HÌNH

| STT | Phương pháp / Mô hình triển khai | MAE (MW) | RMSE (MW) | MAPE (%) | $R^2$ Score | Ghi chú & Đánh giá |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **XGBoost Baseline (24 Lags)** | 167.78 | 223.69 | 1.16% | 0.9916 | Thiết lập chuẩn đề bài, early stopping tại điểm dừng tối ưu |
| **2** | **XGBoost + Rolling Features** | 170.05 | 226.50 | 1.18% | 0.9914 | Bổ sung `rolling_mean_24`, `rolling_std_24`, `rolling_mean_168` |
| **3** | **XGBoost Tuned (TimeSeriesSplit)** | **142.40** | **193.87** | **0.98%** | **0.9937** | **Mô hình xuất sắc nhất**, tối ưu hóa qua Expanding Window CV |
| **4** | **Benchmark Random Forest** | 152.06 | 207.54 | 1.04% | 0.9928 | Huấn luyện theo Bagging, thời gian chạy lâu hơn đáng kể |

#### Dashboard Tổng kết Định lượng & So sánh Hiệu năng:
![Dashboard Tổng Kết](../outputs/figures/cv06_summary_dashboard.png)

---

### PHẦN 2. HÌNH ẢNH TRỰC QUAN HÓA TOÀN BỘ TIẾN TRÌNH THỰC NGHIỆM

#### 1. Dữ liệu gốc & Phân chia Train/Test theo thời gian (CV01 - Hà Duy Hưng Lead):
![Phân đoạn Dữ liệu và Lags](../outputs/figures/cv01_raw_and_lags.png)

#### 2. Đường cong học tập Learning Loss Curve & Early Stopping (CV02 - Ngọc Trâm):
![Learning Curve XGBoost](../outputs/figures/cv02_learning_curve.png)

#### 3. Đối chứng Thực tế vs Dự báo và Độ quan trọng đặc trưng Top Lags (CV03 - Thanh Phương):
![Actual vs Predicted](../outputs/figures/cv03_actual_vs_predicted.png)
![Feature Importance](../outputs/figures/cv03_feature_importance.png)

#### 4. Nghiên cứu Rolling Features & Dự báo Đa bước 24h Đệ quy (CV04 - Hoàng Hiệp):
![Tác động của Rolling Features](../outputs/figures/cv04_rolling_features_impact.png)
![Dự báo Đa bước và Sai số Tích lũy](../outputs/figures/cv04_multistep_forecast.png)

#### 5. Tinh chỉnh Siêu tham số & So sánh Thực nghiệm với Random Forest (CV05 - Hoàng):
![Benchmark XGBoost vs Random Forest](../outputs/figures/cv05_benchmark_comparison.png)

---

### PHẦN 3. LỜI GIẢI CHI TIẾT 3 CÂU HỎI LÝ LUẬN CHUYÊN SÂU (CV06 - BÙI TRẦN NGỌC KHẢI)

#### Câu 1: So sánh XGBoost với Linear Regression trong bài toán dự báo chuỗi thời gian?
1. **Khả năng mô hình hóa quan hệ phi tuyến và tương tác bậc cao:**
   - *Linear Regression:* Giả định mối quan hệ giữa biến phụ thuộc $y_t$ và các biến độ trễ là tuyến tính cộng tính: $y_t = \beta_0 + \sum \beta_i y_{t-i} + \epsilon$. Mô hình hoàn toàn bất lực trước các cấu trúc phụ tải điện phi tuyến tính (như hiệu ứng bão hòa lúc nửa đêm, ngưỡng chịu tải đột ngột khi thời tiết quá nóng).
   - *XGBoost:* Sử dụng tổ hợp các cây quyết định tăng cường độ dốc (Gradient Boosted Trees). Mỗi nhánh chia phân đoạn không gian đặc trưng thành các vùng siêu phẳng cục bộ, tự động nhận diện tương tác phức tạp (ví dụ: nếu $y_{t-24}$ cao VÀ $y_{t-1}$ đang tăng $\rightarrow$ dự báo đỉnh phụ tải) mà không cần lập trình tương tác thủ công.
2. **Khả năng kháng hiện tượng đa cộng tuyến (Multicollinearity):**
   - Trong chuỗi thời gian phụ tải, các độ trễ kế cận ($y_{t-1}, y_{t-2}, y_{t-3}$) tương quan cực kỳ cao ($r > 0.95$). Trong Hồi quy tuyến tính, ma trận biến độc lập trở nên gần như suy biến (Near-singular), dẫn đến phương sai của các trọng số bị phóng đại dữ dội (Variance Inflation), khiến mô hình cực kỳ nhạy cảm với nhiễu.
   - Ngược lại, XGBoost tại mỗi nút phân nhánh chỉ chọn một đặc trưng duy nhất đem lại độ giảm tạp chất (Gain) lớn nhất. Các đặc trưng tương quan còn lại không làm thổi phồng trọng số, giúp mô hình ổn định tuyệt đối.
3. **Ngoại suy xu hướng (Extrapolation Limitation):**
   - Đây là nhược điểm cốt lõi cần lưu ý của các mô hình dựa trên cây (Decision Trees/XGBoost): Cây quyết định chỉ có thể gán giá trị dự báo bằng trung bình mẫu trong các lá đã thấy, nghĩa là giá trị dự báo nằm trong miền giá trị cực trị $[y_{\min}, y_{\max}]$ của tập huấn luyện. Nếu chuỗi thời gian có xu hướng dốc lên mạnh (Strong Trend) vượt ra ngoài lịch sử, XGBoost sẽ dự báo nằm ngang (Flatline). Để khắc phục, cần khử xu hướng (Detrending hoặc Differencing) trước khi huấn luyện cây.

---

#### Câu 2: Điều gì xảy ra khi tăng số lượng độ trễ (lags $p$)?
- **Lợi ích:**
  - Giúp mô hình bao quát được các nhịp tuần hoàn xa trong quá khứ. Đặc biệt với dữ liệu năng lượng theo giờ, nếu chỉ dùng $p=6$ thì mô hình chỉ thấy 6 giờ trước, không thể biết được nhịp điệu cùng giờ ngày hôm qua ($p=24$) hay cùng giờ tuần trước ($p=168$).
- **Hệ quả tiêu cực và rủi ro:**
  1. *Quá khớp (Overfitting):* Tăng $p$ lên quá lớn làm phình to không gian đặc trưng với nhiều lag không mang tín hiệu dự báo, khiến mô hình học thuộc lòng cả các nhiễu ngẫu nhiên trong tập huấn luyện.
  2. *Mất mát dữ liệu ở đầu chuỗi:* Khi tạo $p$ đặc trưng trễ, ta bắt buộc phải xóa bỏ $p$ quan sát đầu tiên (`dropna()`). Nếu chuỗi dữ liệu ngắn mà chọn $p$ lớn sẽ làm suy giảm đáng kể kích thước mẫu.
  3. *Lời nguyền số chiều (Curse of Dimensionality):* Tăng thời gian huấn luyện và độ phức tạp tính toán khi cây phải duyệt qua quá nhiều ngưỡng phân chia (split candidates).
- **Khuyến nghị thực tế:** Nên dựa vào biểu đồ tự tương quan riêng phần (PACF) để chọn các điểm lag có ý nghĩa thống kê cao nhất (ví dụ: các lag 1, 2, 3 và các bội số mùa vụ 24, 48, 168) thay vì lấy tràn lan liên tục hàng trăm lags.

---

#### Câu 3: Khi nào mô hình XGBoost phù hợp hơn mô hình truyền thống ARIMA/SARIMA?
- **Nên chọn XGBoost khi:**
  1. *Tính mùa vụ phức tạp đan xen (Multi-Seasonal Cycles):* Dữ liệu giờ điện lực PJM chứa cùng lúc 3 chu kỳ mùa vụ: Chu kỳ ngày (24 giờ), Chu kỳ tuần (168 giờ), và Chu kỳ năm (8760 giờ). Mô hình SARIMA chỉ thiết kế xử lý tốt 1 chu kỳ mùa vụ chính; khi ép SARIMA chạy với chu kỳ $s=168$, ma trận tham số AR/MA khổng lồ sẽ khiến việc ước lượng khả dĩ cực đại (MLE) bị tắc nghẽn bộ nhớ hoặc không thể hội tụ.
  2. *Kích thước dữ liệu lớn (Big Data):* Với hơn 120,000 quan sát, XGBoost huấn luyện bằng thuật toán Histogram-based chỉ mất vài giây đến vài phút. Trong khi đó, việc tối ưu hóa hàm khả dĩ ARIMA trên hàng trăm nghìn mẫu có thể mất hàng giờ.
  3. *Sự hiện diện của các biến ngoại sinh (Exogenous Features):* XGBoost dễ dàng tiếp nhận hàng chục biến ngoại sinh khác nhau: nhiệt độ, chỉ số kinh tế, ngày lễ tết, cờ cuối tuần, biến vị trí thời gian mà không cần thỏa mãn các giả định khắt khe về tính dừng hay phương sai đồng nhất.
- **Nên chọn ARIMA/SARIMA khi:**
  1. Chuỗi thời gian đơn biến có xu hướng và tính mùa vụ tuyến tính rõ rệt, kích thước mẫu nhỏ hoặc trung bình ($N < 1,000$).
  2. Yêu cầu bắt buộc về tính diễn giải toán học chặt chẽ và khoảng tin cậy lý thuyết (Statistical Confidence Intervals) theo phân phối chuẩn.
  3. Cần mô hình hóa thuần túy bằng các giả định chuỗi dừng (Stationarity) mà không có sẵn dữ liệu biến giải thích ngoại sinh.

---

### PHẦN 4. TỔNG KẾT VÀ BÀN GIAO MÃ NGUỒN NHÓM 7

Toàn bộ hệ thống mã nguồn được chuẩn hóa 100% độc lập, hỗ trợ AI Agent và sẵn sàng vận hành:
- [src/cv01_data_and_lags.py](file:///e:/Chuỗi_TG/time-series-lab2/src/cv01_data_and_lags.py): Chuẩn bị dữ liệu và tạo Lag 1-24.
- [src/cv02_xgboost_train.py](file:///e:/Chuỗi_TG/time-series-lab2/src/cv02_xgboost_train.py): Huấn luyện XGBoost cơ sở với Early Stopping.
- [src/cv03_evaluation_importance.py](file:///e:/Chuỗi_TG/time-series-lab2/src/cv03_evaluation_importance.py): Đánh giá 4 chỉ số và vẽ biểu đồ Feature Importance.
- [src/cv04_rolling_and_multistep.py](file:///e:/Chuỗi_TG/time-series-lab2/src/cv04_rolling_and_multistep.py): Rolling Statistics và Dự báo đa bước 24h.
- [src/cv05_tuning_random_forest.py](file:///e:/Chuỗi_TG/time-series-lab2/src/cv05_tuning_random_forest.py): Tinh chỉnh TimeSeriesSplit và Benchmark Random Forest.
- [src/cv06_report_synthesis.py](file:///e:/Chuỗi_TG/time-series-lab2/src/cv06_report_synthesis.py): Tổng hợp số liệu và nghiệm thu toàn diện.
