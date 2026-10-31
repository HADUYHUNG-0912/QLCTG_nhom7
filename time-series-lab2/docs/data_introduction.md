# GIỚI THIỆU NGUỒN GỐC VÀ ĐẶC TÍNH BỘ DỮ LIỆU
## Bộ Dữ Liệu Tiêu Thụ Điện Năng Theo Giờ (AEP Hourly Energy Consumption)

---

## 1. NGUỒN GỐC DỮ LIỆU VÀ Ý NGHĨA THỰC TẾ

### 1.1. Dữ liệu này từ đâu mà có?
* **Tên gọi:** `AEP_hourly.csv` (American Electric Power Hourly Energy Consumption).
* **Nguồn công bố:** Được thu thập từ mạng lưới truyền tải điện **PJM Interconnection** (Hoa Kỳ) và phát hành mở trên nền tảng [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) bởi tác giả Rob Mulla.
* **Đơn vị thu thập:** **AEP (American Electric Power)** — một trong những tập đoàn năng lượng và công ty điện lực lớn nhất nước Mỹ, phụ trách cấp điện cho hơn 5 triệu khách hàng tại 11 bang (như Ohio, Texas, Virginia, Tennessee, Indiana,...).
* **Vị trí lưu trữ trong dự án:** [`data/raw/AEP_hourly.csv`](../data/raw/AEP_hourly.csv).

---

### 1.2. Tại sao bài toán dự báo điện năng lại quan trọng?
Điện năng có một đặc thù rất khác so với hàng hóa thông thường: **chúng ta không thể dễ dàng cất trữ điện vào kho với số lượng lớn**. Điện phát ra từ các nhà máy phải cân bằng ngay lập tức với lượng điện mà người dân và nhà máy đang tiêu thụ trên lưới điện:
* **Nếu phát điện thiếu:** Hệ thống bị quá tải, gây tụt áp hoặc mất điện diện rộng (sập lưới / blackout), gây thiệt hại hàng triệu đô la cho sản xuất và sinh hoạt.
* **Nếu phát điện thừa:** Gây lãng phí nhiên liệu (than đá, khí đốt), tăng khí thải ô nhiễm môi trường và làm hư hại đường dây.

> **Mục tiêu của bài toán:** Sử dụng máy học (cụ thể là mô hình XGBoost) để nhìn vào lịch sử tiêu thụ điện của những giờ trước, từ đó dự đoán chính xác lượng điện người dân sẽ dùng trong các giờ tiếp theo.

---

## 2. CẤU TRÚC VÀ ĐƠN VỊ ĐO CỦA BỘ DỮ LIỆU

Bộ dữ liệu rất ngắn gọn và trực tiếp, gồm 2 cột chính:

| Tên cột | Ý nghĩa đời thường | Ví dụ giá trị | Giải thích dễ hiểu |
| :--- | :--- | :---: | :--- |
| `Datetime` | Mốc thời gian ghi nhận | `2004-12-31 01:00:00` | Đúng 1 giờ sáng ngày 31/12/2004. Cứ cách 1 tiếng đồng hồ lại đo một lần. |
| `AEP_MW` | Lượng điện tiêu thụ trong giờ đó | `13,478.0` | 13,478 Megawatt (MW). |

### Đơn vị Megawatt (MW) là bao nhiêu?
* $1 \text{ MW} = 1.000 \text{ Kilowatt (kW)} = 1.000.000 \text{ Watt}$.
* Một chiếc điều hòa nhiệt độ gia đình thường tiêu thụ khoảng $1 \text{ kW}$ mỗi giờ. Như vậy, $1 \text{ MW}$ có thể đủ cho khoảng $1.000$ chiếc điều hòa chạy đồng thời, hoặc cấp điện sinh hoạt cho khoảng $600 - 1.000$ hộ gia đình bình thường.

---

## 3. CÁC CON SỐ THỐNG KÊ CƠ BẢN (DỄ NHỚ)

Dữ liệu được thống kê chi tiết tại bảng [`outputs/tables/cv01_data_summary.csv`](../outputs/tables/cv01_data_summary.csv):

| Tiêu chí | Con số thực tế | Nhận xét nhanh |
| :--- | :---: | :--- |
| **Thời gian theo dõi** | Từ `01/10/2004` đến `03/08/2018` | Gần **14 năm liên tục**, phản ánh chu kỳ dài hạn của xã hội. |
| **Tổng số dòng dữ liệu** | **121.269 dòng** | Mỗi dòng là 1 giờ đo đạc $\rightarrow$ Tập dữ liệu đủ lớn để huấn luyện mô hình máy học. |
| **Dữ liệu bị thiếu (NaN)** | **0 dòng (0%)** | Dữ liệu cực kỳ "sạch" và đầy đủ, không bị đứt quãng hay mất mẩu dữ liệu nào. |
| **Mức tiêu thụ trung bình** | **15.499 MW** | Bình quân lúc nào lưới điện AEP cũng phải gánh khoảng 15.500 MW. |
| **Mức tiêu thụ thấp nhất (Min)** | **9.581 MW** | Rơi vào đêm khuya những ngày thời tiết mát mẻ (mọi người ngủ, không bật điều hòa hay sưởi). |
| **Mức tiêu thụ cao nhất (Max)** | **25.695 MW** | Cao gấp 2.7 lần mức thấp nhất, thường rơi vào những ngày cao điểm nắng nóng mùa hè hoặc rét đậm mùa đông. |

---

## 4. 4 ĐẶC TÍNH TỰ NHIÊN NỔI BẬT CỦA DỮ LIỆU ĐIỆN NĂNG

Dữ liệu tiêu thụ điện không biến đổi ngẫu nhiên mà tuân theo các thói quen sinh hoạt và thời tiết rất rõ ràng:

### 4.1. Chu kỳ ngày đêm (24 giờ)
* **Ban đêm (23h - 5h sáng):** Lượng điện dùng giảm xuống đáy vì các nhà máy nghỉ ca, người dân đi ngủ, đèn đường và thiết bị gia dụng tắt bớt.
* **Ban ngày (8h - 18h):** Lượng điện tăng vọt khi công xưởng vận hành máy móc, tòa nhà văn phòng bật điện và máy tính.
* **Chiều tối (18h - 21h):** Xuất hiện đỉnh giờ cao điểm sinh hoạt gia đình (nấu nướng, bật tivi, bình nóng lạnh, máy lạnh).

### 4.2. Chu kỳ ngày trong tuần (7 ngày)
* **Từ thứ 2 đến thứ 6:** Lượng điện tiêu thụ ở mức cao và đều đặn do các cơ quan, nhà máy, trường học hoạt động 100% công suất.
* **Thứ 7 và Chủ nhật:** Lượng điện sụt giảm rõ rệt vì các công sở đóng cửa, nhiều dây chuyền sản xuất nghỉ ngơi.

### 4.3. Tính mùa vụ trong năm (4 mùa)
Khác với nhiều ngành kinh doanh chỉ có 1 mùa cao điểm cuối năm, ngành điện tại Mỹ có **2 đỉnh cao điểm (đỉnh kép)** trong năm:
1. **Đỉnh Mùa Hè (Tháng 7 - Tháng 8):** Nắng nóng gay gắt khiến hàng triệu hộ gia đình và tòa nhà đồng loạt bật điều hòa hết công suất.
2. **Đỉnh Mùa Đông (Tháng 1 - Tháng 2):** Bão tuyết và nhiệt độ dưới $0^\circ\text{C}$ khiến nhu cầu sử dụng máy sưởi điện tăng đột biến.
3. **Mùa Xuân và Mùa Thu (Tháng 4 - 5 và Tháng 9 - 10):** Thời tiết ôn hòa, dễ chịu nhất trong năm, không cần máy sưởi lẫn điều hòa $\rightarrow$ Mức tiêu thụ điện rơi xuống đáy thấp nhất.

### 4.4. Tính "quán tính" (Độ trễ - Lag Features)
Lượng điện tiêu thụ ở giờ này rất phụ thuộc vào những giờ liền trước:
* Nếu lúc 14h trời đang nóng $38^\circ\text{C}$ và điện đang dùng rất cao, thì khả năng rất cao lúc 15h lượng điện vẫn sẽ tiếp tục ở mức cao.
* Lượng điện lúc 14h hôm nay thường có hình thái rất giống với lượng điện lúc 14h hôm qua (chu kỳ 24 giờ).

> 💡 **Đây chính là lý do vì sao nhóm tạo các đặc trưng trễ ($y_{t-1}, y_{t-2}, \dots, y_{t-24}$)** trong bài làm: để dạy mô hình học được nhịp điệu sinh hoạt quen thuộc này!

---

## 5. TỔNG KẾT BỎ TÚI DÀNH CHO BẠN KHI DEMO

Khi thầy cô hoặc người nghe hỏi: *"Dữ liệu của nhóm có đặc điểm gì đặc biệt?"*, bạn chỉ cần trả lời 3 ý chính gọn gàng:

1. **Về quy mô:** Dữ liệu đo lượng điện tiêu thụ theo từng giờ của hãng AEP (Mỹ) trong gần 14 năm với hơn 121.000 dòng, hoàn toàn sạch sẽ (0% lỗi/thiếu).
2. **Về quy luật:** Dữ liệu có 3 nhịp điệu chính lặp đi lặp lại: **theo ngày** (ngày cao hơn đêm), **theo tuần** (ngày làm việc cao hơn cuối tuần) và **theo mùa** (hè và đông cao điểm hơn xuân và thu).
3. **Về ứng dụng:** Việc dùng mô hình XGBoost để học lại các nhịp điệu này giúp công ty điện dự báo được nhu cầu sắp tới, tránh tình trạng phát thiếu gây sập lưới điện hoặc phát thừa gây lãng phí hàng triệu USD.
