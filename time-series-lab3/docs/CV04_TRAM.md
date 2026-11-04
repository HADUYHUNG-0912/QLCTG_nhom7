# CV04 — Trâm: Rolling Features và dự báo 24 giờ

Bản phân tích đầy đủ: [CV04_PHAN_TICH_CHI_TIET.md](CV04_PHAN_TICH_CHI_TIET.md).
Notebook đã chạy kiểm chứng: [cv04_tram_analyst.ipynb](../notebooks/cv04_tram_analyst.ipynb).
Bản đầy đủ có ablation từng cửa sổ, hai seasonal-naive, sai số theo nhóm, bootstrap và phần bảo vệ bài.

## Phạm vi và căn cứ

Theo bảng phân công Google Sheets và ma trận trên GitHub, Trâm phụ trách CV04.
Đã đối chiếu `README.md`, `DE_BAI_LAB.md`, `docs/KE_HOACH_PHAN_CONG_NHOM7.md`
và khung `src/cv04_rolling_and_multistep.py` trên nhánh main.
Bảng phân công chi tiết yêu cầu cả Recursive và Direct; phần đầu file cũ còn ghi Hoàng Hiệp.
Bài bàn giao này triển khai cả hai chiến lược, dùng tên đầu ra CV04 theo ma trận hiện hành.

## Cách chạy

Chạy tại thư mục `time-series-lab2` sau khi có `data/processed/train.csv` và `test.csv` của CV01:

```bash
python -m pip install -r requirements.txt
python src/cv04_rolling_and_multistep.py
python -m unittest discover -s tests -v
```

Trên macOS, XGBoost cần OpenMP; nếu báo thiếu `libomp.dylib`, cài `brew install libomp`.
Lần kiểm chứng này dùng Python 3.9, XGBoost 2.1.4 và runtime OpenMP có sẵn trong môi trường thử nghiệm.
Không cần chạy CV02: CV04 tự huấn luyện mô hình thực nghiệm của riêng mình.

## Phương pháp và chống rò rỉ dữ liệu

- Giữ mốc test từ CV01: **2015-10-28 17:00:00**. Có 96,736 hàng train sau khi bỏ warm-up và 24,221 hàng test thực tế để đánh giá một bước.
- Tái tạo lag 1–24 và lag 168 trên lưới giờ liên tục, thêm 7 biến lịch. Cả hai thí nghiệm tác động rolling dùng cùng hàng, cùng lags, cùng cấu hình và quy trình chọn số cây. Đây là đối chứng riêng của CV04, không phải checkpoint baseline của CV02.
- Tạo đủ rolling mean/std 24 và 168 giờ bằng `shift(1).rolling(window)`. Std là độ lệch chuẩn mẫu (`ddof=1`).
- Dữ liệu CV01 có 27 giờ thiếu trong khoảng thời gian đã đọc. Bổ sung lưới giờ và forward-fill từ quá khứ; không nội suy từ tương lai. Chỉ chấm điểm trên target quan sát thật; loại cả cửa sổ dự báo 24h nếu có target được điền. Dữ liệu huấn luyện có dùng các giờ forward-fill nên cần lưu ý ảnh hưởng của DST và khuyết thiếu.
- XGBoost: tối đa 1.000 cây, learning_rate=0.01, max_depth=5, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50, random_state=42. Validation là 10% cuối train; sau chọn số cây, huấn luyện lại trên toàn train. Test không chọn số cây.
- Recursive: dự báo liên tiếp 24 lần, cập nhật toàn bộ lags và rolling bằng dự báo vừa sinh. Không lấy target thật trong cùng cửa sổ 24h.
- Direct: huấn luyện 24 mô hình độc lập, mô hình h ánh xạ cùng đặc trưng tại mốc bắt đầu sang target ở h giờ tiếp theo. Loại nhãn vượt ranh giới train; chừa gap h−1 giữa train và validation để nhãn train không chạm validation.
- Backtest: 1006 mốc dự báo hợp lệ, cách nhau 24h. Tại mốc mới được dùng các quan sát đã xảy ra trước mốc đó. Cả hai chiến lược chấm trên đúng cùng mốc và cùng target. Đoạn test cuối không đủ 24h không được đánh giá.

## Tác động của rolling — dự báo một bước

| model | MAE | RMSE | n_test | n_estimators |
|---|---|---|---|---|
| Lags + calendar | 163.235 | 212.102 | 24221 | 1000 |
| Lags + calendar + rolling | 158.548 | 206.720 | 24221 | 1000 |

MAE thay đổi theo hướng giảm **2.87%** khi thêm rolling (giá trị âm nghĩa là MAE tăng).
Đây là kết quả một lần chia thời gian; không khẳng định rolling luôn cải thiện trên mọi chuỗi hay mọi cấu hình.

![Rolling](../outputs/figures/cv04_rolling_features_impact.png)

## Dự báo nhiều bước và tích lũy sai số

| strategy | horizon | n_origins | MAE | RMSE | cumulative_MAE | cumulative_RMSE |
|---|---|---|---|---|---|---|
| Recursive | 1 | 1006 | 135.622 | 173.149 | 135.622 | 173.149 |
| Recursive | 6 | 1006 | 545.507 | 701.228 | 391.401 | 535.516 |
| Recursive | 12 | 1006 | 645.679 | 897.746 | 493.268 | 683.837 |
| Recursive | 24 | 1006 | 844.448 | 1086.833 | 613.825 | 847.288 |
| Direct | 1 | 1006 | 135.622 | 173.149 | 135.622 | 173.149 |
| Direct | 6 | 1006 | 486.765 | 630.685 | 364.909 | 492.568 |
| Direct | 12 | 1006 | 592.859 | 780.267 | 445.880 | 604.295 |
| Direct | 24 | 1006 | 802.703 | 1005.696 | 568.330 | 772.732 |

MAE/RMSE ở từng h được tính qua tất cả mốc dự báo. Cumulative MAE/RMSE gộp các residual
của bước 1..h; cumulative RMSE là căn trung bình bình phương residual, không phải trung bình các RMSE.
Đơn vị các sai số: MW. Trong biểu đồ, ví dụ 24h là cửa sổ hợp lệ đầu tiên, không lựa chọn cửa sổ đẹp nhất.

Recursive có nguy cơ lan truyền sai số vì dự báo trước trở thành đầu vào tiếp theo; Direct tránh phản hồi này
nhưng cần 24 lần huấn luyện và có thể thiếu nhất quán giữa các bước. Sai số thực nghiệm không bắt buộc
tăng đơn điệu theo h: chu kỳ phụ tải và khả năng dự báo khác nhau theo giờ có thể làm đường sai số giảm cục bộ.
Vì vậy cần đọc toàn bộ bảng và đường cong, không ép số liệu để chứng minh sai số luôn tăng.

![Multi-step](../outputs/figures/cv04_multistep_forecast.png)

## Tệp bàn giao

- `src/cv04_rolling_and_multistep.py`: mã thực thi độc lập.
- `outputs/figures/cv04_rolling_features_impact.png`: rolling và đối chứng một bước.
- `outputs/figures/cv04_multistep_forecast.png`: ví dụ dự báo, sai số từng h và sai số gộp.
- `outputs/tables/cv04_multistep_metrics.csv`: 24 hàng Recursive, giữ tên bàn giao gốc.
- `outputs/tables/cv04_direct_metrics.csv`: 24 hàng Direct.
- `outputs/tables/cv04_multistep_predictions.csv`, `cv04_direct_predictions.csv`: dự báo từng mốc để kiểm toán.
- `outputs/tables/cv04_rolling_features_impact.csv`, `cv04_run_metadata.json`: đối chứng và cấu hình chạy.
- `outputs/models/cv04_xgboost_rolling.json`: mô hình Recursive; không ghi đè model CV02.
- `data/processed/AEP_features_rolling.csv`: đặc trưng đã tái tạo, gồm target và cả train/test; khi dùng lại phải tách theo mốc test trong metadata.
- `tests/test_cv04.py`: kiểm tra đặc trưng không nhìn target hiện tại/tương lai, phản hồi dự báo và rolling, dữ liệu thiếu giờ, công thức gộp sai số.

Các bảng có thể nạp bằng `pd.read_csv(...)` để Phương ghép vào báo cáo CV06.
