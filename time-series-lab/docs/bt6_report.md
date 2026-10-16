# BÁO CÁO KỸ THUẬT TỔNG KẾT LAB 2 (BÀI TẬP 6)
## ĐÁNH GIÁ ĐẶC TÍNH DỮ LIỆU CHUỖI THỜI GIAN & TIỀM NĂNG DỰ BÁO

- **Môn học:** Quản lý & Khai phá Chuỗi Thời Gian (QLCTG)  
- **Giảng viên hướng dẫn:** PhD. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  
- **Nhóm thực hiện:** Nhóm 07 (Trưởng nhóm: Hà Duy Hưng)  
- **Bộ dữ liệu thực nghiệm:** `data/raw/AirPassengers.csv` (144 quan sát tháng, 1949 - 1960)  
- **Bảng số liệu tổng hợp đính kèm:** [`../outputs/tables/ket_qua_bai_tap_tong_hop.md`](../outputs/tables/ket_qua_bai_tap_tong_hop.md)

---

## 1. MÔ TẢ ĐẶC TÍNH DATASET (BÀI TẬP 1)

Bộ dữ liệu **AirPassengers** ghi lại số lượng hành khách đi máy bay trên các tuyến bay quốc tế từ tháng 1 năm 1949 đến tháng 12 năm 1960. Cứ mỗi tháng, hãng hàng không tổng kết lại một lần, do đó xuyên suốt 12 năm liên tục ta có đúng 144 mốc thời gian quan sát. Dữ liệu gồm hai cột thông tin chuẩn hóa: cột `Month` lưu trữ mốc thời gian theo từng tháng (`pd.DatetimeIndex`, tần suất `MS`), và cột `Passengers` lưu số lượng hành khách trong tháng đó (đơn vị tính: nghìn người, ví dụ 112 tương đương $112,000$ hành khách).

Các thông số thống kê mô tả cơ bản của bộ dữ liệu gồm:
- **Giá trị trung bình và phân tán:** Mỗi tháng có trung bình khoảng $280.30$ nghìn lượt khách bay, với độ lệch chuẩn là $119.97$ nghìn người thể hiện mức độ biến động khá lớn quanh giá trị trung bình.
- **Biên độ dao động cực trị:** Tháng thấp nhất chỉ có $104.00$ nghìn người (Tháng 11/1949), trong khi tháng cao nhất đạt tới $622.00$ nghìn người (Tháng 07/1960).
- **Độ hoàn chỉnh dữ liệu:** Toàn bộ 144 tháng quan sát đều có số liệu đầy đủ, không bị thiếu sót hay khuyết tật bất kỳ điểm nào.

Công thức tính giá trị trung bình mẫu:
$$
\bar{y} = \frac{1}{n} \sum_{t=1}^{n} y_t = 280.30 \text{ (nghìn người)}
$$

Công thức tính độ lệch chuẩn mẫu:
$$
s = \sqrt{\frac{1}{n-1} \sum_{t=1}^{n} (y_t - \bar{y})^2} \approx 119.97 \text{ (nghìn người)}
$$

Nhìn vào chuỗi số liệu này bằng mắt thường, ta thấy ngay hai quy luật thực tế gắn liền với đời sống: một là xu hướng tăng trưởng dài hạn sau Chiến tranh thế giới thứ hai khi ngành hàng không mở rộng và thu nhập người dân nâng cao; hai là thói quen du lịch mùa hè khiến lượng khách tháng 7 và tháng 8 luôn tăng vọt, trong khi tháng 11 nhu cầu bay lại giảm xuống thấp nhất năm.

---

## 2. PHÂN TÍCH TỰ TƯƠNG QUAN ACF VÀ PACF (BÀI TẬP 2)

Hàm tự tương quan (ACF) đo lường mức độ phụ thuộc giữa số lượng khách ở tháng hiện tại với các tháng trước đó. Nhóm tính toán tự tương quan trên 40 độ trễ (tương đương với khoảng thời gian hơn 3 năm về trước) để xem dữ liệu có tính ghi nhớ như thế nào.

Công thức tính hệ số tự tương quan mẫu tại độ trễ $k$:
$$
\rho_k = \frac{\sum_{t=k+1}^{n} (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{n} (y_t - \bar{y})^2}
$$

Khoảng tin cậy 95% Bartlett:
$$
\text{CI}_{95\%} = \pm \frac{1.96}{\sqrt{n}} = \pm \frac{1.96}{\sqrt{144}} \approx \pm 0.1633 \text{ (không thứ nguyên)}
$$

![Hình 1: Đồ thị hàm tự tương quan ACF và tự tương quan riêng phần PACF của chuỗi AirPassengers](../outputs/figures/bai1_acf_pacf.png)
*Hình 1: Đồ thị hàm tự tương quan (ACF) và tự tương quan riêng phần (PACF) của chuỗi AirPassengers qua 40 độ trễ.*

Từ kết quả phân tích biểu đồ, nhóm rút ra ba nhận xét quan trọng:
1. **Tính tự tương quan rất mạnh và suy giảm chậm:** Tại độ trễ 1 tháng (lag 1), hệ số đạt mức rất cao là $0.9480$. Toàn bộ 40 độ trễ đều vượt ra ngoài dải tin cậy và giảm rất từ từ, chứng tỏ chuỗi bị chi phối bởi xu hướng tăng và chưa dừng về trung bình. Để đưa chuỗi về ổn định, cần lấy sai phân bậc 1 $\Delta y_t = y_t - y_{t-1}$.
2. **Tính mùa vụ chu kỳ 12 tháng rõ nét:** Biểu đồ ACF xuất hiện các đỉnh nhô cao đột biến lặp lại đều đặn sau mỗi 12 tháng, cụ thể tại độ trễ 12 đạt $0.7604$, độ trễ 24 đạt $0.7100$, độ trễ 36 đạt $0.6500$. Hiện tượng này khẳng định lượng khách tháng này có mối liên hệ mật thiết với cùng kỳ năm trước.
3. **Định hướng cấu trúc mô hình qua PACF:** Biểu đồ PACF cắt cụt ngay sau độ trễ 1 và xuất hiện xung đáng chú ý tại độ trễ 12, gợi ý mô hình SARIMA với bậc tự hồi quy phi mùa vụ $p = 1$ và mùa vụ $P = 1$ sẽ rất phù hợp với dữ liệu.

---

## 3. ĐỐI CHỨNG VỚI CHUỖI NHIỄU TRẮNG (WHITE NOISE) (BÀI TẬP 3)

Nhiễu trắng (White Noise) là chuỗi dữ liệu hoàn toàn ngẫu nhiên, trong đó các số liệu ở từng thời điểm độc lập với nhau, không hề có quy luật hay tính ghi nhớ quá khứ. Nhóm tiến hành tạo ra 1000 điểm nhiễu trắng giả lập theo phân phối chuẩn Gaussian ($\mu = 0, \sigma = 1$) để đặt cạnh đối chứng trực tiếp với chuỗi AirPassengers.

Kiểm định Ljung-Box được sử dụng để kiểm tra thống kê xem một chuỗi có phải là nhiễu trắng hay không:
$$
Q(h) = n(n + 2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n - k}
$$

![Hình 2: Đối chứng dạng sóng và đồ thị ACF giữa White Noise và AirPassengers](../outputs/figures/bai2_white_noise_comparison.png)
*Hình 2: Đối chứng dạng sóng và đồ thị ACF giữa Gaussian White Noise và chuỗi dữ liệu thực tế AirPassengers.*

Bảng kết quả kiểm định Ljung-Box Test:

| Chuỗi dữ liệu | Độ trễ ($h$) | Thống kê $Q$ | Trị số p-value | Mức ý nghĩa $\alpha$ | Kết luận thống kê |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gaussian White Noise** | 10 | 7.71 | 0.6571 | 0.05 | Chấp nhận $H_0$ (Là nhiễu trắng thuần túy) |
| **Gaussian White Noise** | 20 | 15.16 | 0.7675 | 0.05 | Chấp nhận $H_0$ (Là nhiễu trắng thuần túy) |
| **AirPassengers** | 10 | 857.07 | $1.10 \times 10^{-177}$ | 0.05 | Bác bỏ $H_0$ (Chứa tín hiệu quy luật mạnh) |
| **AirPassengers** | 20 | 1434.15 | $5.30 \times 10^{-292}$ | 0.05 | Bác bỏ $H_0$ (Chứa tín hiệu quy luật mạnh) |

Kết quả đối chứng thể hiện rõ hai kết luận mang tính bản chất:
- **Chuỗi nhiễu trắng không thể dự báo:** Toàn bộ các cột ACF của nhiễu trắng đều nằm gọn trong dải tin cậy 95% và kiểm định Ljung-Box cho p-value lớn hơn 0.05 ($p = 0.6571$). Dữ liệu hoàn toàn ngẫu nhiên và không thể đoán trước được tương lai.
- **Chuỗi AirPassengers chứa tín hiệu quy luật mạnh mẽ:** Toàn bộ các cột ACF của chuỗi hành khách đều vượt ra ngoài dải tin cậy và p-value xấp xỉ bằng 0 tuyệt đối. Điều này bác bỏ hoàn toàn giả thuyết ngẫu nhiên, chứng minh dữ liệu rất giàu thông tin và phù hợp để xây dựng mô hình dự báo.

---

## 4. PHÂN RÃ CHUỖI THỜI GIAN (BÀI TẬP 4)

Phân rã chuỗi thời gian là phương pháp bóc tách dãy số hành khách thực tế thành các mảnh ghép thành phần riêng biệt, bao gồm: Xu hướng dài hạn ($T_t$ - Trend), Mùa vụ lặp lại hàng năm ($S_t$ - Seasonal) và Phần dư còn sót lại ($R_t$ - Residual).

Về mặt toán học, có hai dạng mô hình phân rã chính:
- **Mô hình dạng Cộng:** $Y_t = T_t + S_t + R_t$, trong đó tất cả các thành phần đều có cùng đơn vị đo là nghìn người.
- **Mô hình dạng Nhân:** $Y_t = T_t \times S_t \times R_t$, trong đó lượng khách thực tế $Y_t$ và xu hướng $T_t$ có đơn vị là nghìn người, còn thành phần mùa vụ $S_t$ và phần dư $R_t$ là hệ số tỷ lệ không có đơn vị (dao động xung quanh mức 1.0).

Công thức tính tốc độ tăng trưởng của đường xu hướng:
$$
\text{Growth} = \frac{T_{\text{cuối}} - T_{\text{đầu}}}{T_{\text{đầu}}} \times 100\% \approx +274.7\%
$$

![Hình 3: Phân rã chuỗi thời gian AirPassengers theo mô hình nhân](../outputs/figures/bai3_2_multiplicative_decomposition.png)
*Hình 3: Phân rã chuỗi thời gian AirPassengers theo mô hình Nhân (Classical Multiplicative Decomposition).*

![Hình 3b: Phân rã chuỗi bằng thuật toán STL Loess](../outputs/figures/bai3_3_stl_decomposition.png)
*Hình 3b: Phân rã chuỗi thời gian AirPassengers bằng phương pháp STL Robust to Outliers.*

Kết quả phân rã cho thấy các đặc tính cấu trúc nổi bật:
- **Xu hướng tăng trưởng dài hạn:** Đường xu hướng tăng đều đặn qua từng năm, đưa lượng hành khách từ $120.21$ nghìn người vào năm 1949 lên tới $450.42$ nghìn người vào năm 1960, tương ứng mức tăng trưởng $274.7\%$.
- **Tính mùa vụ ổn định:** Đỉnh cao nhất luôn cố định ở tháng 7 và tháng 8 (với hệ số khoảng 1.25, tức cao hơn mức trung bình năm 25%), còn đáy thấp nhất luôn rơi vào tháng 11 (hệ số khoảng 0.83, tức thấp hơn trung bình 17%).
- **Ưu thế của mô hình dạng Nhân:** Khoảng cách chênh lệch giữa mùa hè và mùa đông ngày càng phình to theo thời gian (từ 44 nghìn khách năm 1949 lên 232 nghìn khách năm 1960). Do đó mô hình Nhân vượt trội hơn hẳn dạng Cộng vì phần dư của nó dao động ổn định quanh mức 1.0 và tiệm cận tính chất nhiễu trắng, chứng minh các quy luật lớn đã được trích xuất trọn vẹn.

---

## 5. VẤN ĐỀ CHẤT LƯỢNG DỮ LIỆU & LÀM SẠCH (BÀI TẬP 5)

Để kiểm tra năng lực làm sạch dữ liệu trong điều kiện môi trường thực tế, nhóm đã tạo ra một bản sao thử nghiệm chứa 13 điểm bị mất ngẫu nhiên (tỷ lệ 9.03%) và cấy thêm 3 điểm ngoại lai bị phóng đại gấp 3.5 lần tại các vị trí tháng thứ 25, 75 và 110.

Để đánh giá độ chính xác khi phục hồi dữ liệu, ta sử dụng hai chỉ số sai số:
$$
\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|, \qquad \text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2}
$$

Để phát hiện các điểm ngoại lai cục bộ, thuật toán Hampel Filter xét một cửa sổ trượt $w=13$ tháng bao trọn chu kỳ mùa vụ 12 tháng xung quanh mỗi điểm. Thuật toán tính giá trị trung vị $M_t$ và độ lệch tuyệt đối trung vị $\text{MAD}_t$:
$$
M_t = \operatorname{median}(y_{t-k}, \dots, y_{t+k}), \quad \text{MAD}_t = 1.4826 \times \operatorname{median}(\{|y_{t+j} - M_t|\}_{j=-k}^{k})
$$

![Hình 4: Dữ liệu trước và sau khi làm sạch bằng Time Interpolation và Hampel Filter](../outputs/figures/bai4_data_cleaning_before_after.png)
*Hình 4: Trực quan hóa đối chứng chuỗi dữ liệu trước và sau khi làm sạch khuyết tật.*

Bảng so sánh hiệu quả xử lý dữ liệu khuyết thiếu:

| Phương pháp xử lý | MAE (nghìn người) | RMSE (nghìn người) | Nhận xét chuyên môn |
| :--- | :---: | :---: | :--- |
| **Forward-fill (`ffill`)** | 23.31 | 28.59 | Kém hiệu quả; tạo các bậc phẳng nằm ngang trễ so với xu hướng tăng |
| **Time-based Interpolation** | **16.43** | **18.83** | **Tối ưu;** nối thẳng theo mốc thời gian giúp bảo toàn độ dốc tăng trưởng |

Bảng so sánh thuật toán phát hiện ngoại lai:

| Thuật toán | Vị trí phát hiện | Đánh giá độ chuẩn xác |
| :--- | :---: | :--- |
| **Rolling Z-score** | `[25, 75, 110]` | Phát hiện đúng 3/3 điểm nhưng nhạy cảm với giả định phân phối chuẩn |
| **Rolling IQR** | `[25, 43, 75, 110]` | Báo nhầm đỉnh mùa hè tự nhiên tại điểm 43 (tháng 7) là ngoại lai |
| **Hampel Filter** | **`[25, 75, 110]`** | **Chính xác tuyệt đối 3/3 điểm**, độ trơ cao, không nhận nhầm đỉnh mùa hè |

Quá trình làm sạch dữ liệu mang lại hai kết quả nổi bật:
- **Bù khuyết dữ liệu mất mát:** Phương pháp nội suy theo thời gian (Time Interpolation) tỏ ra tối ưu hơn hẳn so với việc lấy giá trị tháng trước đè vào (Forward-fill), vì nó bám sát theo độ dốc tăng trưởng tự nhiên của chuỗi.
- **Lọc sạch ngoại lai đột biến:** Thuật toán Hampel Filter bắt trúng 100% các điểm lỗi mà không nhầm lẫn đỉnh mùa hè; sau khi thay thế bằng giá trị trung vị trượt, sai số toàn chuỗi giảm gần 17 lần (RMSE từ $99.18$ xuống chỉ còn $5.90$ nghìn người).

---

## 6. Ý NGHĨA TRỰC QUAN HÓA CHUYÊN SÂU (BÀI TẬP 6)

Trực quan hóa dữ liệu bằng các biểu đồ chuyên sâu mang lại cái nhìn trực giác sinh động:

![Hình 5: Biểu đồ Rolling Mean và dải biên độ Rolling Std](../outputs/figures/bai5_1_rolling_statistics.png)
*Hình 5: Biểu đồ đường trung bình trượt 12 tháng và dải biến thiên độ lệch chuẩn trượt $\pm 2$ Std.*

![Hình 6: Biểu đồ mùa vụ theo tháng qua 12 năm](../outputs/figures/bai5_2_seasonal_plot.png)
*Hình 6: Biểu đồ mùa vụ theo tháng (Seasonal Plot) thể hiện sự tịnh tiến qua các năm từ 1949 đến 1960.*

![Hình 7: Bản đồ nhiệt Seasonality Heatmap](../outputs/figures/bai5_3_seasonality_heatmap.png)
*Hình 7: Bản đồ nhiệt (Heatmap) ma trận Năm x Tháng thể hiện mật độ hành khách hàng không.*

Hệ thống biểu đồ cung cấp ba góc nhìn trực quan giá trị:
1. **Làm phẳng mùa vụ để nhận diện xu hướng:** Đường trung bình trượt 12 tháng triệt tiêu hoàn toàn các bước sóng ngắn hạn, phơi bày xu thế tăng trưởng liên tục từ 1520 nghìn lên 5714 nghìn khách/năm; đồng thời độ lệch chuẩn trượt mở rộng gấp $5.67$ lần chứng minh chuỗi có hiện tượng phương sai thay đổi theo thời gian (Heteroskedasticity).
2. **Nhận diện tính đồng dạng pha sóng:** Biểu đồ mùa vụ vẽ 12 đường cong đại diện cho 12 năm đều có pha sóng giống hệt nhau (cùng tăng từ tháng 3, đạt đỉnh nhọn vào tháng 7-8 và rơi về đáy vào tháng 11) và tịnh tiến đều đặn lên phía trên.
3. **Quan sát mật độ qua bản đồ nhiệt:** Bản đồ Heatmap thể hiện rõ dải màu nóng sáng luôn tập trung liên tục ở cột tháng 7 và tháng 8, đồng thời toàn bộ ma trận chuyển màu sáng dần theo trục năm minh chứng cho quy mô phát triển dài hạn.

---

## 7. KẾT LUẬN: DỮ LIỆU CÓ PHÙ HỢP ĐỂ DỰ BÁO KHÔNG?

**KẾT LUẬN CUỐI CÙNG: BỘ DỮ LIỆU HOÀN TOÀN PHÙ HỢP VÀ RẤT LÝ TƯỞNG ĐỂ XÂY DỰNG MÔ HÌNH DỰ BÁO.**

Căn cứ khoa học:
1. Kiểm định Ljung-Box đã bác bỏ hoàn toàn giả thuyết nhiễu trắng ($p\text{-value} \approx 0 < 0.05$), chuỗi mang tính ghi nhớ quá khứ mạnh mẽ.
2. Chuỗi sở hữu xu hướng tăng trưởng dài hạn bền vững ($+274.7\%$), chu kỳ mùa vụ 12 tháng lặp lại cực kỳ chuẩn xác (đỉnh tháng 7, đáy tháng 11).
3. Thành phần phần dư sau phân rã nhân tiệm cận nhiễu trắng, chứng minh các đặc trưng cấu trúc lớn có thể được mô hình hóa trọn vẹn.
4. **Định hướng mô hình tối ưu:**
   - Khi áp dụng **SARIMA$(1,1,0) \times (1,1,0)_{12}$**, cần lấy logarit biến đổi $z_t = \ln(y_t)$ để ổn định phương sai và lấy sai phân mùa vụ bậc 1.
   - Khi áp dụng **XGBoost / Machine Learning:** Cần xây dựng các biến trễ (Lag 1, 12), biến thống kê trượt (Rolling Mean 12), và biến lịch (tháng trong năm).
