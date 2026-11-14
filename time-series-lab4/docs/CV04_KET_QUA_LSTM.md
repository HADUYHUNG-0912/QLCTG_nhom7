# CV04 — LSTM dự báo phụ tải điện trước 1 giờ

Chạy với dữ liệu đã chuẩn hóa của CV01 trên nhánh `cham-lab04`. Mỗi dự báo sử dụng 24 giờ quan sát gần nhất của phụ tải, nhiệt độ, độ ẩm và tốc độ gió. Tập validation dùng 24 giờ cuối của tập train làm lịch sử ban đầu. Tập test dùng 24 giờ cuối của tập validation. Các cửa sổ chỉ chứa dữ liệu có trước giờ dự báo.

## Kết quả trên tập test

Cả hai mô hình được đánh giá trên 3.947 mốc giờ của tập test. RMSE và MAE tính trên thang phụ tải gốc. MAPE tính theo phần trăm.

| Mô hình | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: |
| LSTM | 475,7041 | 326,9603 | 2,1518% |
| XGBoost | 477,4474 | 328,8135 | 2,1620% |

Nguồn số liệu XGBoost là bảng kết quả CV03 trên nhánh. [LSTM](../outputs/tables/cv04_lstm_metrics.csv) · [XGBoost](../outputs/tables/xgb_metrics.csv)

## Q4. LSTM học được những chu kỳ nào?

Cửa sổ 24 giờ cho LSTM quan sát biến động trong một chu kỳ ngày. Các cổng của LSTM điều tiết thông tin được giữ lại trong trạng thái ẩn trong cửa sổ đó. Dữ liệu đầu vào hiện tại chỉ cung cấp lịch sử 24 giờ cho từng dự báo, nên kết quả này chưa xác nhận mô hình đã học chu kỳ tuần 168 giờ. Có thể kiểm tra giả thuyết tuần bằng thử nghiệm cửa sổ 168 giờ hoặc thêm đặc trưng ngày trong tuần, rồi so sánh trên cùng các mốc validation và test.

## Q5. LSTM có bị overfitting không?

Validation MSE thấp nhất tại epoch 36 (0,038600). Đến epoch 46, train MSE giảm từ 0,046251 xuống 0,045251, trong khi validation MSE tăng lên 0,043198. Diễn biến này cho thấy dấu hiệu overfitting sau epoch 36. EarlyStopping với patience 10 dừng tại epoch 46 và khôi phục trọng số epoch 36. Hai lớp Dropout với tỷ lệ 0,2 được dùng trong huấn luyện. [Learning curve](../outputs/figures/cv04_lstm_learning_curve.png)

## Q6. XGBoost và LSTM khác nhau thế nào?

XGBoost nhận một vector phẳng gồm các lag và thống kê rolling được tạo trước. Mô hình học các quan hệ phi tuyến bằng cây quyết định. LSTM nhận chuỗi có thứ tự 24 × 4 và học cách tóm tắt lịch sử qua trạng thái ẩn. Trong lần chạy này, LSTM giảm RMSE khoảng 0,37%, MAE khoảng 0,56% và MAPE 0,0102 điểm phần trăm so với XGBoost. Đây là mức chênh lệch nhỏ trên tập test này.

## Chạy lại và bàn giao

Từ thư mục `time-series-lab4`, cài thư viện trong `requirements.txt` và chạy `python src/cv04_lstm_model.py`. Lượt chạy này dùng Python 3.12 và TensorFlow 2.19.1.

- [Mô hình Keras](../outputs/models/lstm_model.keras)
- [Checkpoint H5 cho CV05](../outputs/models/lstm_model_best.h5)
- [Biểu đồ dự báo và thực tế](../outputs/figures/cv04_lstm_actual_vs_predicted.png)
