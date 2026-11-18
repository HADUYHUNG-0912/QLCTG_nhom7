# BẢNG TỔNG KẾT KẾT QUẢ MÔ HÌNH HÓA ARIMA (ARIMA MODEL SUMMARY)
## DỰ BÁO NHIỆT ĐỘ HÀNG NGÀY (DAILY MINIMUM TEMPERATURES FORECASTING)

> **Dataset:** Daily Minimum Temperatures  
> **Date Range:** `1981-01-01 00:00:00` đến `1990-12-31 00:00:00`  
> **Số quan sát (Observations):** 3,650 mẫu  
> **Cấu trúc mô hình:** $\text{ARIMA}(1, 0, 1)$ — Không lấy sai phân ($d=0$), bậc tự hồi quy $p=1$, bậc trung bình trượt $q=1$  

---

## 1. Kết quả Nguyên bản từ Statsmodels (Raw Terminal Output)

```text
ARIMA Model Summary - Temperature Forecasting
==============================================================================
Data: Daily Minimum Temperatures
Date range: 1981-01-01 00:00:00 to 1990-12-31 00:00:00
Number of observations: 3650
AR order (p): 1
Differencing order (d): 0
MA order (q): 1
Model: ARIMA(1, 0, 1)
==============================================================================
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                   Temp   No. Observations:                 3650
Model:                  ARIMA(1, 0, 1)   Log Likelihood               -8605.425
Date:                Sat, 20 Dec 2025   AIC                          17218.851
Time:                        08:49:00   BIC                          17243.661
Sample:                             0   HQIC                         17227.686
                                - 3650                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          11.1965      0.240     46.721      0.000      10.727      11.666
ar.L1           0.8649      0.011     81.948      0.000       0.844       0.886
ma.L1          -0.2414      0.019    -12.604      0.000      -0.279      -0.204
sigma2          6.5349      0.143     45.767      0.000       6.255       6.815
==============================================================================
Ljung-Box (L1) (Q):              8.97   Jarque-Bera (JB):                15.27
Prob(Q):                         0.00   Prob(JB):                         0.00
Heteroskedasticity (H):          0.85   Skew:                             0.05
Prob(H) (two-sided):             0.00   Kurtosis:                         3.30
==============================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
==============================================================================
```

---

## 2. Các Bảng Định lượng Chi tiết (Structured Markdown Tables)

### 2.1. Thông tin Cấu hình & Độ khớp Mô hình (Model Fit Statistics)

| Chỉ số / Thuộc tính | Giá trị | Ý nghĩa kinh tế lượng / Thống kê |
| :--- | :---: | :--- |
| **Biến phụ thuộc (Dep. Variable)** | `Temp` | Nhiệt độ tối thiểu hàng ngày ($^\circ\text{C}$) |
| **Cấu trúc mô hình (Model)** | $\text{ARIMA}(1, 0, 1)$ | Mô hình $\text{ARMA}(1, 1)$ tĩnh trên dữ liệu gốc ($d=0$) |
| **Số lượng quan sát ($N$)** | $3,650$ | 10 năm dữ liệu liên tục theo ngày (1981–1990) |
| **Log-Likelihood** | **$-8,605.425$** | Độ hợp lý logarit cực đại của hàm ước lượng |
| **AIC (Akaike Info Criterion)** | **$17,218.851$** | Tiêu chuẩn thông tin phạt số tham số |
| **BIC (Bayesian Info Criterion)** | **$17,243.661$** | Phạt phức tạp mô hình chặt chẽ hơn ($\ln(N) \cdot k$) |
| **HQIC (Hannan-Quinn Criterion)** | **$17,227.686$** | Tiêu chuẩn tiệm cận giữa AIC và BIC |
| **Loại hiệp phương sai (Covariance)** | `opg` | Outer Product of Gradients (xấp xỉ ma trận thông tin Fisher) |

---

### 2.2. Bảng Ước lượng Tham số & Mức Ý nghĩa Thống kê (Coefficients & Significance)

| Tham số | Hệ số ước lượng ($\beta$) | Sai số chuẩn (Std Err) | Thống kê $z$ | Giá trị $P > |z|$ | Khoảng tin cậy 95% $[0.025 - 0.975]$ | Kết luận |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hằng số (`const`)** | **$11.1965$** | $0.240$ | $46.721$ | **$< 0.001$** | $[10.727, 11.666]$ | Có ý nghĩa thống kê rất cao |
| **Bậc AR1 (`ar.L1`)** | **$0.8649$** | $0.011$ | $81.948$ | **$< 0.001$** | $[0.844, 0.886]$ | Quán tính nhiệt độ hôm trước rất mạnh ($|\phi_1| < 1 \Rightarrow$ dừng) |
| **Bậc MA1 (`ma.L1`)** | **$-0.2414$** | $0.019$ | $-12.604$ | **$< 0.001$** | $[-0.279, -0.204]$ | Hấp thụ cú sốc ngẫu nhiên ngày hôm trước ($|\theta_1| < 1 \Rightarrow$ khả nghịch) |
| **Phương sai nhiễu (`sigma2`)** | **$6.5349$** | $0.143$ | $45.767$ | **$< 0.001$** | $[6.255, 6.815]$ | Độ lệch chuẩn sai số $\sigma \approx 2.556^\circ\text{C}$ |

---

### 2.3. Kiểm định Chẩn đoán Phần dư (Residual Diagnostics)

| Phép kiểm định | Giá trị thống kê | Giá trị xác suất ($p$-value) | Ngưỡng $\alpha = 0.05$ | Diễn giải & Kết luận |
| :--- | :---: | :---: | :---: | :--- |
| **Ljung-Box (L1) ($Q$)** | $8.97$ | **$0.000$** | $p < 0.05$ | Phần dư **vẫn còn tự tương quan** ở lag bậc 1 $\Rightarrow$ Cần xem xét thêm lag mùa vụ hoặc bậc mô hình cao hơn. |
| **Jarque-Bera ($JB$)** | $15.27$ | **$0.000$** | $p < 0.05$ | Bác bỏ giả thuyết phân phối chuẩn của phần dư $\Rightarrow$ Phần dư có đuôi dày hơn chuẩn. |
| **Độ bất đối xứng (Skewness)** | $0.05$ | — | $\approx 0$ | Phân phối phần dư gần như đối xứng hoàn hảo quanh $0$. |
| **Độ nhọn (Kurtosis)** | $3.30$ | — | Chuẩn $= 3.0$ | Đỉnh hơi nhọn hơn chuẩn nhẹ (Leptokurtic). |
| **Phương sai thay đổi (Heteroskedasticity $H$)** | $0.85$ | **$0.000$** | $p < 0.05$ | Tồn tại hiện tượng phương sai phần dư thay đổi theo thời gian giữa nửa đầu và nửa sau mẫu. |

---

## 3. Phương trình Toán học của Mô hình

Từ bảng hệ số, phương trình của mô hình $\text{ARIMA}(1, 0, 1)$ có dạng:

$$(y_t - \mu) = \phi_1 (y_{t-1} - \mu) + \varepsilon_t + \theta_1 \varepsilon_{t-1}$$

Thay các giá trị ước lượng cụ thể:
$$(y_t - 11.1965) = 0.8649 (y_{t-1} - 11.1965) + \varepsilon_t - 0.2414 \varepsilon_{t-1}$$

Hoặc viết theo toán tử trễ $B$:
$$(1 - 0.8649 B) (y_t - 11.1965) = (1 - 0.2414 B) \varepsilon_t, \quad \varepsilon_t \sim \text{WN}(0, 6.5349)$$

* **Nhiệt độ trung bình dài hạn ($\mu$):** $11.1965^\circ\text{C}$.
* **Hệ số tự hồi quy $\phi_1 = 0.8649$:** Nhiệt độ hôm nay phụ thuộc tới $86.49\%$ vào độ lệch nhiệt độ hôm qua so với trung bình.
* **Hệ số trung bình trượt $\theta_1 = -0.2414$:** Sai số ngẫu nhiên hôm qua được điều chỉnh giảm $24.14\%$ vào hôm nay.
