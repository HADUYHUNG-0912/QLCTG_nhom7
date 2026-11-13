# PIPELINE QUY TRÌNH KỸ THUẬT: BÀI THỰC HÀNH LAB 2
## MÔN HỌC: PHÂN TÍCH DỮ LIỆU CHUỖI THỜI GIAN VÀ DỰ BÁO

> **Kho lưu trữ:** `time-series-lab2`  
> **Nhóm thực hiện:** Nhóm 7 (Hưng, Hiệp, Phương, Châm, Hoàng, Khải)  
> **Kiến trúc hệ thống:** Phân chia thành hai giai đoạn độc lập (**Training Pipeline** & **Inference Pipeline**)  

---

## 1. SƠ ĐỒ PIPELINE TỔNG THỂ

```text
====================================================================================================
                        SƠ ĐỒ PIPELINE KỸ THUẬT CHUẨN HÓA (LAB 2)
====================================================================================================

 [ DATA INGESTION ]
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ CV01 (Hưng Lead): Data Cleaning & Temporal Split        │
 │ • Kiểm tra Missing, Outliers, tính mùa vụ               │
 │ • Phân chia Train / Val / Test TUẦN TỰ (Không shuffle)  │
 │ • Trích xuất đặc trưng trễ (Lag Features)               │
 └────────────────────────────┬────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
 ┌───────────────────────────┐ ┌───────────────────────────┐
 │ CV02 (Hiệp): Baseline     │ │ CV03 (Phương): ML Model   │
 │ • Mô hình thống kê cơ sở  │ │ • Huấn luyện mô hình ML   │
 │ • Thiết lập ngưỡng chuẩn  │ │ • Tối ưu hóa siêu tham số │
 └─────────────┬─────────────┘ └─────────────┬─────────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ CV04 (Châm): Advanced Feature Engineering               │
 │ • Rolling Statistics & Xu hướng động                    │
 │ • Trích xuất Feature Importance & Ablation Study        │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ CV05 (Hoàng): Walk-Forward Backtesting                  │
 │ • Kiểm thử cuộn đa bước trên Test stream                │
 │ • Đánh giá RMSE, MAE, MAPE trên thang đo gốc            │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ CV06 (Khải): Synthesis Report & Final Submission        │
 │ • Tổng hợp số liệu từ outputs/tables/ & outputs/figures/│
 │ • Trả lời các câu hỏi lý luận chuyên sâu                │
 └─────────────────────────────────────────────────────────┘
```

---

## 2. NĂM NGUYÊN TẮC KỸ THUẬT BẮT BUỘC (GUARDRAILS)

1. **Bảo tồn tính tuần tự thời gian (No Shuffle):** Tuyệt đối không xáo trộn dữ liệu khi phân chia tập dữ liệu.
2. **Chống rò rỉ dữ liệu (Anti-Data Leakage):** Mọi bộ chuẩn hóa (Scaler) chỉ được fit trên tập Train, sau đó áp dụng transform cho Val và Test.
3. **Tránh Lookahead Bias:** Các biến trượt (Rolling features) bắt buộc áp dụng độ trễ `.shift(1)`.
4. **Đánh giá trên đơn vị gốc:** Tính toán sai số (RMSE, MAE, MAPE) sau khi đã nghịch đảo chuẩn hóa về đơn vị thực tế.
5. **Định dạng chuẩn:** Tất cả biểu đồ xuất ở định dạng PNG với độ phân giải DPI $\ge 150$.
