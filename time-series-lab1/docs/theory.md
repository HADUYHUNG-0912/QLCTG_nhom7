# CƠ SỞ LÝ THUYẾT VÀ PHƯƠNG PHÁP LUẬN CHUỖI THỜI GIAN
## (THEORETICAL FOUNDATIONS & METHODOLOGY)

Tài liệu hệ thống hóa toàn bộ kiến thức toán học và phương pháp luận thống kê được áp dụng trong 5 bài tập thực hành, chia mục tương ứng theo từng bài.

---

## MỤC 1 (BÀI TẬP 1): HÀM TỰ TƯƠNG QUAN (ACF) VÀ TỰ TƯƠNG QUAN RIÊNG PHẦN (PACF)

### 1.1. Phương sai và Hiệp phương sai trễ
- **Phương sai (Variance):** Đo mức độ phân tán của chuỗi xung quanh giá trị kỳ vọng trung bình:
  $$\operatorname{Var}(Y) = \gamma_0 = \frac{1}{n} \sum_{t=1}^{n} (y_t - \bar{y})^2$$
- **Hiệp phương sai tại độ trễ $k$ (Autocovariance at Lag $k$):** Đo lường mối quan hệ đồng biến hoặc nghịch biến giữa giá trị thời điểm hiện tại $y_t$ và giá trị quá khứ $y_{t-k}$:
  $$\operatorname{Cov}(y_t, y_{t-k}) = \gamma_k = \frac{1}{n} \sum_{t=k+1}^{n} (y_t - \bar{y})(y_{t-k} - \bar{y})$$

### 1.2. Hàm tự tương quan (Autocorrelation Function - ACF)
Hệ số tự tương quan $\rho_k$ chuẩn hóa hiệp phương sai về dải giá trị $[-1, 1]$ (không có thứ nguyên):
$$\rho_k = \frac{\operatorname{Cov}(y_t, y_{t-k})}{\operatorname{Var}(y_t)} = \frac{\gamma_k}{\gamma_0}$$
- $\rho_k \approx +1$: Tương quan thuận rất mạnh (giá trị quá khứ cao kéo theo hiện tại cao).
- $\rho_k \approx -1$: Tương quan nghịch rất mạnh.
- $\rho_k \approx 0$: Không có tương quan tuyến tính tại độ trễ $k$.

### 1.3. Hàm tự tương quan riêng phần (Partial Autocorrelation Function - PACF)
PACF tại độ trễ $k$ ($\phi_{kk}$) đo lường mối tương quan trực tiếp thuần túy giữa $y_t$ và $y_{t-k}$ sau khi đã loại bỏ hoàn toàn ảnh hưởng gián tiếp trung gian của tất cả các biến trễ ở giữa ($y_{t-1}, y_{t-2}, \dots, y_{t-(k-1)}$).

### 1.4. Khoảng tin cậy Bartlett và Quy tắc nhận diện mô hình:
- **Dải tin cậy 95% Bartlett:**
  $$\text{CI}_{95\%} = \pm \frac{1.96}{\sqrt{n}}$$
  Nếu một hệ số tự tương quan vượt ra ngoài dải tin cậy này, tương quan tại độ trễ đó có ý nghĩa thống kê ở mức $\alpha = 0.05$.
- **Quy tắc định hướng cấu trúc mô hình SARIMA:**
  - *Chuỗi không dừng (Non-stationary):* ACF bắt đầu ở mức rất cao ($\rho_1 \approx 0.95$) và suy giảm rất từ từ (Tail-off / Slow decay) qua nhiều lag $\rightarrow$ Cần lấy sai phân bậc 1 ($d = 1$).
  - *Tính mùa vụ (Seasonality):* ACF xuất hiện các đỉnh sóng lặp lại theo chu kỳ tại các bội số của chu kỳ mùa vụ (Lag $12, 24, 36, \dots$).
  - *Mô hình AR(p):* ACF suy giảm dần theo hàm mũ, PACF bị cắt cụt đột ngột (Cut-off) sau đúng $p$ lags.
  - *Mô hình MA(q):* PACF suy giảm dần, ACF bị cắt cụt sau đúng $q$ lags.

---

## MỤC 2 (BÀI TẬP 2): KHÁI NIỆM VÀ KIỂM ĐỊNH NHIỄU TRẮNG (WHITE NOISE)

### 2.1. Định nghĩa toán học của chuỗi White Noise
Một chuỗi thời gian $\{\epsilon_t\}$ được định nghĩa là một chuỗi Nhiễu trắng thuần túy nếu thỏa mãn đồng thời 3 điều kiện:
1. **Kỳ vọng bằng 0:** $E[\epsilon_t] = 0, \quad \forall t$
2. **Phương sai là hằng số hữu hạn không đổi:** $\operatorname{Var}(\epsilon_t) = \sigma^2 < \infty, \quad \forall t$
3. **Không có bất kỳ sự tự tương quan nào ở mọi độ trễ khác 0:**
   $$\operatorname{Cov}(\epsilon_t, \epsilon_{t-k}) = 0, \quad \forall k \neq 0$$

### 2.2. Khả năng dự báo của White Noise
Vì các quan sát hoàn toàn độc lập và không chứa tính ghi nhớ quá khứ, dự báo tốt nhất trong tương lai cho chuỗi White Noise tại mọi thời điểm chỉ là giá trị kỳ vọng trung bình $\hat{\epsilon}_{t+h} = 0$. Sai số dự báo chính bằng phương sai $\sigma^2$ và không có bất kỳ thuật toán máy học nào có thể cải thiện kết quả này.

### 2.3. Kiểm định Ljung-Box Test
Kiểm định Ljung-Box kiểm tra đồng thời toàn bộ các hệ số tự tương quan từ lag $1$ đến lag $h$:
$$Q(h) = n(n + 2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n - k}$$
- **Giả thuyết $H_0$:** Dữ liệu là nhiễu trắng độc lập (toàn bộ $\rho_1 = \dots = \rho_h = 0$).
- **Giả thuyết $H_1$:** Dữ liệu có sự tự tương quan (chứa quy luật nội tại).
- **Quy tắc quyết định:** Thống kê $Q$ tuân theo phân phối Chi-bình phương $\chi^2(h)$. Nếu trị số $\text{p-value} < 0.05$, ta bác bỏ $H_0$ và khẳng định dữ liệu có cấu trúc thông tin dồi dào để xây dựng mô hình dự báo.

---

## MỤC 3 (BÀI TẬP 3): PHÂN RÃ CHUỖI THỜI GIAN (TIME-SERIES DECOMPOSITION)

Phân rã chuỗi thời gian là kỹ thuật bóc tách chuỗi quan sát $Y_t$ thành các mảnh ghép cấu trúc độc lập:
1. **Xu hướng dài hạn ($T_t$ - Trend):** Thể hiện hướng đi tổng thể (tăng trưởng hoặc suy thoái dài hạn).
2. **Mùa vụ tuần hoàn ($S_t$ - Seasonal):** Các dao động lặp lại đều đặn theo chu kỳ cố định (như 12 tháng/năm).
3. **Phần dư ngẫu nhiên ($R_t$ - Residual):** Nhiễu còn lại sau khi đã tách bỏ xu hướng và mùa vụ.

### 3.1. Mô hình Phân rã dạng Cộng (Additive Model)
$$Y_t = T_t + S_t + R_t$$
- Các thành phần $T_t, S_t, R_t$ đều có cùng đơn vị đo lường với $Y_t$.
- *Điều kiện áp dụng:* Biên độ dao động mùa vụ giữ nguyên không đổi bất kể đường xu hướng tăng hay giảm.

### 3.2. Mô hình Phân rã dạng Nhân (Multiplicative Model)
$$Y_t = T_t \times S_t \times R_t$$
- $Y_t$ và $T_t$ có cùng đơn vị; thành phần mùa vụ $S_t$ và phần dư $R_t$ là các hệ số tỷ lệ không thứ nguyên (dao động quanh giá trị $1.0$).
- *Điều kiện áp dụng:* Biên độ dao động mùa vụ phình rộng ra hoặc thu hẹp lại tỷ lệ thuận với độ lớn của xu hướng (trường hợp của bộ dữ liệu `AirPassengers`).

### 3.3. Phương pháp phân rã STL (Seasonal and Trend decomposition using Loess)
STL là phương pháp phi tham số hiện đại sử dụng kỹ thuật làm mịn cục bộ hồi quy có trọng số (Loess). Ưu điểm nổi bật:
- Thích ứng tốt với mùa vụ có thể biến đổi nhẹ qua các năm.
- Có khả năng kháng nhiễu và ngoại lai cực cao nhờ cơ chế tính trọng số lặp (Robust STL).

---

## MỤC 4 (BÀI TẬP 4): CÁC VẤN ĐỀ CHẤT LƯỢNG DỮ LIỆU TIME SERIES

### 4.1. Cơ chế khuyết thiếu dữ liệu (Missing Data)
- **MCAR (Missing Completely at Random):** Dữ liệu mất ngẫu nhiên không phụ thuộc vào bất kỳ yếu tố nào.
- **MAR (Missing at Random):** Dữ liệu mất phụ thuộc vào biến quan sát khác.
- **MNAR (Missing Not at Random):** Dữ liệu mất phụ thuộc vào chính giá trị của nó (ví dụ: cảm biến hỏng khi tải quá cao).

Trong chuỗi thời gian, **Nội suy theo thời gian (Time-based Interpolation)** là lựa chọn vượt trội vì bám theo đường dốc tuyến tính nối giữa $y_{t-1}$ và $y_{t+1}$, duy trì liên tục độ dốc của xu hướng mà không tạo bước nhảy bậc thang như `ffill`.

### 4.2. Thuật toán Hampel Filter phát hiện ngoại lai
Hampel Filter là phương pháp phát hiện ngoại lai mạnh mẽ nhất cho chuỗi thời gian:
1. Tại mỗi thời điểm $t$, xét cửa sổ lân cận kích thước $w = 2k + 1$ đối xứng: $[t-k, \dots, t, \dots, t+k]$.
2. Tính Trung vị trượt: $\text{Med}_t = \operatorname{median}(y_{t-k}, \dots, y_{t+k})$.
3. Tính Độ lệch tuyệt đối trung vị (MAD - Median Absolute Deviation):
   $$\text{MAD}_t = \operatorname{median}(|y_i - \text{Med}_t|)$$
4. Xác định ngưỡng giới hạn: $\text{Threshold}_t = 3 \times 1.4826 \times \text{MAD}_t$.
5. Nếu $|y_t - \text{Med}_t| > \text{Threshold}_t$, điểm $y_t$ bị coi là ngoại lai và được thay thế bằng $\text{Med}_t$.

Nhờ sử dụng hàm Trung vị (Median) thay vì Trung bình (Mean), Hampel Filter không bị chính các giá trị ngoại lai kéo lệch ngưỡng kiểm tra, và không bị nhầm đỉnh cao điểm mùa hè tự nhiên thành lỗi.

---

## MỤC 5 (BÀI TẬP 5): TRỰC QUAN HÓA CHUYÊN SÂU CHUỖI THỜI GIAN

### 5.1. Thống kê trượt (Rolling Statistics)
- **Rolling Mean (window=$s$):** Đóng vai trò là bộ lọc thông thấp (Low-pass filter), triệt tiêu hoàn toàn các dao động sóng mùa vụ nội năm ngắn hạn để làm lộ ra đường xu hướng thực chất.
- **Dải $\pm 2$ Rolling Std:** Cho phép kiểm chứng giả thuyết về tính đồng nhất phương sai. Nếu dải này mở rộng theo thời gian, chuỗi vi phạm giả định phương sai đồng nhất (Heteroskedasticity).

### 5.2. Biểu đồ Mùa vụ (Seasonal Plot) và Bản đồ nhiệt (Heatmap)
- **Monthly Seasonal Plot:** Giúp quan sát tính đồng dạng của các bước sóng qua từng năm, xác định chính xác tháng cao điểm du lịch và tháng chạm đáy nhu cầu.
- **Seasonality Heatmap ($12 \times 12$):** Cung cấp ma trận nhiệt trực quan 2 chiều, cho phép phát hiện đồng thời xu hướng liên năm theo trục dọc và mùa vụ nội năm theo trục ngang.

### 5.3. Định hướng cho mô hình máy học (XGBoost / Random Forest)
Từ các đặc tính khám phá được, khi chuyển sang mô hình hóa máy học bảng:
1. **Tạo biến trễ (Lag Features):** Tạo các biến $y_{t-1}, y_{t-2}$ (tự tương quan ngắn hạn) và $y_{t-12}, y_{t-24}$ (tự tương quan mùa vụ chu kỳ năm).
2. **Tạo biến cửa sổ trượt (Rolling Features):** Tạo `rolling_mean_12`, `rolling_std_12` để cung cấp thông tin mức nền và độ biến động.
3. **Tạo biến thời gian (Calendar Features):** Trích xuất tháng trong năm ($1 \dots 12$), quý ($1 \dots 4$) dưới dạng biến phân loại hoặc mã hóa sin/cos.

---

## MỤC 6: HỆ THỐNG 12 CÂU HỎI ÔN TẬP VÀ ĐÁP ÁN CHUẨN CUỐI CHƯƠNG 2

#### Câu 1. Tự tương quan (autocorrelation) là gì? Nó giúp nhận diện điều gì trong chuỗi thời gian?
- **Trả lời:** Tự tương quan là mức độ tương quan tuyến tính giữa một chuỗi thời gian với chính nó tại các độ trễ khác nhau ($k$). Nó giúp nhận diện: tính ghi nhớ ngắn hạn (AR), xu hướng dài hạn (ACF giảm chậm) và tính chu kỳ mùa vụ lặp lại (đỉnh nhọn tại lag chu kỳ).

#### Câu 2. Sự khác nhau giữa ACF và PACF là gì? Lag quan trọng được hiểu như thế nào?
- **Trả lời:** 
  - ACF đo lường tương quan tổng thể (bao gồm cả hiệu ứng trung gian gián tiếp), dùng nhận diện Trend, Seasonality và bậc MA($q$).
  - PACF đo tương quan trực tiếp thuần túy sau khi trừ bỏ các điểm trung gian, dùng nhận diện bậc AR($p$).
  - Lag quan trọng là lag có giá trị hệ số vượt ra khỏi dải tin cậy 95% Bartlett ($\pm 1.96/\sqrt{n}$), có ý nghĩa thống kê khác 0.

#### Câu 3. Thế nào là white noise? Làm sao biết một chuỗi có phải white noise?
- **Trả lời:** White noise là chuỗi ngẫu nhiên độc lập có kỳ vọng bằng 0, phương sai không đổi và không có tự tương quan ($\rho_k = 0, \forall k \ge 1$). Nhận biết bằng cách xem biểu đồ ACF (toàn bộ nằm trong dải tin cậy 95%) hoặc kiểm định Ljung-Box với $p\text{-value} > 0.05$.

#### Câu 4. Chuỗi white noise có dùng được để dự báo không? Vì sao?
- **Trả lời:** **Không thể dùng để dự báo.** Vì mọi điểm dữ liệu đều độc lập và ngẫu nhiên, không chứa bất kỳ quy luật quá khứ nào để suy diễn cho tương lai. Dự báo tốt nhất chỉ là giá trị trung bình $\mu = 0$.

#### Câu 5. Phân rã chuỗi thời gian gồm những thành phần nào? Mỗi thành phần mang ý nghĩa gì?
- **Trả lời:** Gồm 3 thành phần cấu trúc:
  1. *Trend ($T_t$):* Hướng đi dài hạn (tăng/giảm/ngang).
  2. *Seasonality ($S_t$):* Quy luật lặp lại theo chu kỳ cố định biết trước.
  3. *Residual ($R_t$):* Biến động ngẫu nhiên còn lại sau khi bóc tách trend và mùa vụ.

#### Câu 6. Phân biệt phân rã cộng và phân rã nhân (additive vs multiplicative).
- **Trả lời:** 
  - *Additive ($Y = T + S + R$):* Biên độ mùa vụ là hằng số cố định, không đổi khi xu hướng tăng lên.
  - *Multiplicative ($Y = T \times S \times R$):* Biên độ dao động mùa vụ tỉ lệ thuận với mức độ của xu hướng (càng lên cao dao động mùa vụ càng phình rộng).

#### Câu 7. Nêu 3 loại lỗi chất lượng dữ liệu phổ biến trong chuỗi thời gian.
- **Trả lời:** 
  1. Missing values (mất mát dữ liệu do gián đoạn cảm biến/mạng).
  2. Outliers (đột biến bất thường do lỗi nhập liệu hoặc sự kiện cú sốc).
  3. Non-stationarity (tính không dừng do trend và mùa vụ làm sai lệch giả định mô hình).

#### Câu 8. Có những cách nào để xử lý missing values trong dữ liệu chuỗi thời gian?
- **Trả lời:** Forward-fill (`ffill`), Backward-fill (`bfill`), Nội suy theo thời gian (`interpolate`), điền bằng giá trị cùng kỳ mùa vụ trước, hoặc dùng mô hình dự báo bù khuyết (KNN, Kalman Filter, Prophet).

#### Câu 9. Rolling mean được dùng để kiểm tra điều gì trong dữ liệu?
- **Trả lời:** Rolling mean giúp làm mượt các dao động ngắn hạn để làm nổi bật **đường xu hướng dài hạn (Trend)** và kiểm tra xem kỳ vọng của chuỗi có ổn định theo thời gian hay không (kiểm tra tính dừng).

#### Câu 10. Biểu đồ ACF giúp ta nhận biết mùa vụ như thế nào?
- **Trả lời:** Biểu đồ ACF xuất hiện các đỉnh nhô cao vượt dải tin cậy lặp lại có tính chu kỳ đều đặn tại các bội số của lag mùa vụ (ví dụ: các đỉnh xuất hiện tại lag 12, 24, 36 đối với dữ liệu tháng).

#### Câu 11. Khi residual của decomposition giống white noise, điều đó có ý nghĩa gì?
- **Trả lời:** Điều đó chứng minh phương pháp phân rã hoặc mô hình đã **trích xuất thành công toàn bộ thông tin có cấu trúc** (xu hướng và mùa vụ), phần còn lại thực sự chỉ là nhiễu ngẫu nhiên thuần túy.

#### Câu 12. Vì sao trực quan hóa dữ liệu là bước quan trọng trong phân tích chuỗi thời gian?
- **Trả lời:** Trực quan hóa giúp mắt thường phát hiện ngay lập tức các đặc tính then chốt khó thấy qua bảng số: xu hướng, chu kỳ mùa vụ, các điểm gãy cấu trúc, ngoại lai cực đoan và lựa chọn dạng mô hình phù hợp (Additive hay Multiplicative).

