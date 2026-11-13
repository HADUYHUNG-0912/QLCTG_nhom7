# =============================================================================
# AI_AGENT_INSTRUCTIONS — ĐỌC TRƯỚC KHI THỰC HIỆN BẤT KỲ THAY ĐỔI NÀO
# =============================================================================
# MODULE     : CV04 — Xây dựng mô hình tiêu chuẩn ARIMA(p, d, q)
# PHỤ TRÁCH  : Hoàng
# MÔ TẢ TASK : Huấn luyện 1–2 mô hình ARIMA tiêu chuẩn từ danh sách ứng viên
#              đề xuất bởi CV03. Thực hiện chẩn đoán phần dư nghiêm ngặt
#              (Residual Diagnostics) gồm kiểm định White Noise qua Ljung-Box
#              test. Dự báo ngoài mẫu (Out-of-sample) 41 tháng trên tập Test.
#              Đánh giá hiệu năng bằng MAE, RMSE, MAPE.
#
# DELIVERABLES (ĐẦU RA):
#   • outputs/models/arima_model.pkl              – mô hình ARIMA đã fit tốt nhất
#   • outputs/tables/cv04_arima_metrics.csv        – AIC, BIC, MAE, RMSE, MAPE
#   • outputs/tables/cv04_residual_stats.csv       – thống kê White Noise phần dư
#   • outputs/figures/cv04_arima_residual_diag.png – chẩn đoán phần dư
#   • outputs/figures/cv04_arima_forecast_vs_actual.png
#
# RÀNG BUỘC BẮT BUỘC (HARD CONSTRAINTS):
#   [1] Chỉ FIT mô hình trên data/processed/train.csv (163 tháng).
#   [2] KHÔNG dùng bất kỳ thông tin từ test.csv để chọn tham số (p, d, q).
#   [3] Dự báo Out-of-sample: forecast(steps=41) — không rolling refit.
#   [4] Chẩn đoán phần dư: Mean ≈ 0, kiểm định Ljung-Box p-value > 0.05.
#   [5] Lưu model bằng pickle để CV05 tái sử dụng metrics.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---------------------------------------------------------------------------
# 0. THIẾT LẬP ĐƯỜNG DẪN GỐC
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(ROOT, "data", "processed")
FIG_DIR   = os.path.join(ROOT, "outputs", "figures")
TABLE_DIR = os.path.join(ROOT, "outputs", "tables")
MODEL_DIR = os.path.join(ROOT, "outputs", "models")

TRAIN_CSV = os.path.join(PROC_DIR, "train.csv")
TEST_CSV  = os.path.join(PROC_DIR, "test.csv")

for d in [FIG_DIR, TABLE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. NẠP TẬP TRAIN & TEST
# ---------------------------------------------------------------------------
def load_data(train_path: str, test_path: str):
    """Đọc train.csv và test.csv, trả về 2 pd.Series với DatetimeIndex."""
    # TODO: pd.read_csv(path, index_col='date', parse_dates=True)['value']
    raise NotImplementedError("Thành viên 4 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 2. HUẤN LUYỆN MÔ HÌNH ARIMA(p, d, q)
# ---------------------------------------------------------------------------
def fit_arima(train: pd.Series, order: tuple) -> object:
    """
    Fit mô hình ARIMA với bộ tham số (p, d, q) trên tập Train.

    Mô hình đề xuất khởi điểm cho a10.csv:
      ARIMA(1, 1, 1) — Điểm khởi đầu kinh điển theo Box-Jenkins.
      ARIMA(2, 1, 1) — Nếu PACF cắt cụt tại lag 2.

    Args:
        train : pd.Series — Chuỗi Train gốc (163 tháng).
        order : tuple (p, d, q).

    Returns:
        Đối tượng ARIMAResultsWrapper đã fit.
    """
    # TODO: model = ARIMA(train, order=order)
    #       result = model.fit()
    #       print(result.summary())
    raise NotImplementedError("Thành viên 4 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 3. CHẨN ĐOÁN PHẦN DƯ (RESIDUAL DIAGNOSTICS)
# ---------------------------------------------------------------------------
def diagnose_residuals(result, model_label: str = "ARIMA") -> dict:
    """
    Chẩn đoán phần dư của mô hình đã fit:
      (a) Vẽ 4-panel: Residual Plot | Histogram | Q-Q Plot | ACF residuals.
      (b) Kiểm định Ljung-Box test: H₀ = phần dư là White Noise.
          → p-value > 0.05: không đủ cơ sở bác bỏ White Noise ✓

    Lưu: outputs/figures/cv04_arima_residual_diag.png
    Returns:
        dict: {'mean_residual', 'ljungbox_stat', 'ljungbox_pvalue', 'is_white_noise'}
    """
    # TODO: residuals = result.resid
    #       lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
    raise NotImplementedError("Thành viên 4 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 4. DỰ BÁO NGOÀI MẪU (OUT-OF-SAMPLE FORECAST)
# ---------------------------------------------------------------------------
def forecast_test(result, steps: int = 41) -> pd.Series:
    """
    Dự báo ngoài mẫu (Out-of-sample) 41 bước thời gian.

    Lưu ý quan trọng:
      - Dùng result.forecast(steps=steps) — đây là dự báo thực sự ngoài mẫu.
      - KHÔNG được dùng result.predict() với dữ liệu Test đã biết trước.

    Returns:
        pd.Series — Giá trị dự báo cho 41 tháng tập Test.
    """
    # TODO: forecast = result.forecast(steps=steps)
    raise NotImplementedError("Thành viên 4 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 5. TÍNH CHỈ SỐ ĐÁNH GIÁ
# ---------------------------------------------------------------------------
def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Tính MAE, RMSE, MAPE trên tập Test.

    Công thức:
      MAE  = mean(|y_t − ŷ_t|)
      RMSE = sqrt(mean((y_t − ŷ_t)²))
      MAPE = mean(|y_t − ŷ_t| / |y_t|) × 100%

    Returns:
        dict: {'MAE', 'RMSE', 'MAPE', 'AIC', 'BIC'}
    """
    # TODO: errors = actual.values - predicted.values
    #       mae  = np.mean(np.abs(errors))
    #       rmse = np.sqrt(np.mean(errors**2))
    #       mape = np.mean(np.abs(errors / actual.values)) * 100
    raise NotImplementedError("Thành viên 4 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 6. VẼ BIỂU ĐỒ DỰ BÁO VS THỰC TẾ
# ---------------------------------------------------------------------------
def plot_forecast(train: pd.Series, test: pd.Series, forecast: pd.Series) -> None:
    """
    Vẽ biểu đồ:
      - Chuỗi Train (màu xanh đậm)
      - Giá trị thực tế tập Test (màu xanh lá)
      - Giá trị dự báo ARIMA (màu đỏ, nét đứt)
    Lưu: outputs/figures/cv04_arima_forecast_vs_actual.png
    """
    raise NotImplementedError("Thành viên 4 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 7. LƯU MÔ HÌNH & KẾT QUẢ
# ---------------------------------------------------------------------------
def save_model(result, path: str) -> None:
    """Lưu mô hình đã fit bằng pickle để CV05 đọc lại metrics."""
    with open(path, 'wb') as f:
        pickle.dump(result, f)


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("CV04 — Mô hình ARIMA(p, d, q) (Thành viên 4)")
    print("=" * 70)

    # Bộ tham số đề xuất (Thành viên 4 điều chỉnh sau khi đọc kết quả CV03)
    ARIMA_ORDER = (1, 1, 1)  # TODO: Điều chỉnh theo phân tích ACF/PACF từ CV03

    # Bước 1: Nạp dữ liệu
    train, test = load_data(TRAIN_CSV, TEST_CSV)
    print(f"[OK] Train: {len(train)} tháng | Test: {len(test)} tháng")

    # Bước 2: Fit mô hình ARIMA
    result = fit_arima(train, order=ARIMA_ORDER)
    print(f"[OK] ARIMA{ARIMA_ORDER} đã fit | AIC={result.aic:.2f} | BIC={result.bic:.2f}")

    # Bước 3: Chẩn đoán phần dư
    diag = diagnose_residuals(result, model_label=f"ARIMA{ARIMA_ORDER}")
    print(f"[Residuals] Mean={diag['mean_residual']:.4f} | Ljung-Box p={diag['ljungbox_pvalue']:.4f} | White Noise: {diag['is_white_noise']}")

    # Bước 4: Dự báo 41 tháng
    forecast = forecast_test(result, steps=len(test))

    # Bước 5: Tính chỉ số
    metrics = compute_metrics(test, forecast)
    print(f"[Metrics] MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | MAPE={metrics['MAPE']:.4f}%")

    # Bước 6: Vẽ và lưu
    plot_forecast(train, test, forecast)
    save_model(result, os.path.join(MODEL_DIR, "arima_model.pkl"))
    pd.DataFrame([metrics]).to_csv(os.path.join(TABLE_DIR, "cv04_arima_metrics.csv"), index=False)

    print("[DONE] CV04 hoàn tất. Chuyển arima_model.pkl & cv04_arima_metrics.csv cho CV05.")

