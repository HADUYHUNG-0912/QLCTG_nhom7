# =============================================================================
# AI_AGENT_INSTRUCTIONS — ĐỌC TRƯỚC KHI THỰC HIỆN BẤT KỲ THAY ĐỔI NÀO
# =============================================================================
# MODULE     : CV05 — Mô hình mùa vụ SARIMA(p,d,q)(P,D,Q)[12] & Benchmark
# PHỤ TRÁCH  : Khải
# MÔ TẢ TASK : Xây dựng mô hình SARIMA với chu kỳ mùa vụ s=12, sai phân
#              mùa vụ D=1 trên chuỗi a10.csv. Chẩn đoán phần dư nghiêm ngặt
#              (kiểm định White Noise tại lag mùa vụ 12, 24). Dự báo ngoài mẫu
#              41 tháng trên tập Test. Lập bảng Benchmark đối chuẩn trực diện
#              với mô hình ARIMA từ CV04.
#
# DELIVERABLES (ĐẦU RA):
#   • outputs/models/sarima_model.pkl
#   • outputs/tables/cv05_sarima_metrics.csv
#   • outputs/tables/cv05_model_benchmark_table.csv  – ARIMA vs SARIMA
#   • outputs/figures/cv05_sarima_forecast_vs_actual.png
#   • outputs/figures/cv05_arima_vs_sarima_comparison.png
#
# RÀNG BUỘC BẮT BUỘC (HARD CONSTRAINTS):
#   [1] Chỉ FIT mô hình trên data/processed/train.csv (163 tháng).
#   [2] KHÔNG dùng bất kỳ thông tin từ test.csv để chọn tham số SARIMA.
#   [3] Dự báo Out-of-sample: forecast(steps=41) — không rolling refit.
#   [4] Chu kỳ mùa vụ BẮT BUỘC: s=12 (12 tháng trong năm), D=1.
#   [5] Chẩn đoán phần dư: kiểm định Ljung-Box tại lag 12, 24 → p-value > 0.05.
#   [6] Bảng Benchmark đọc cv04_arima_metrics.csv — KHÔNG chạy lại ARIMA.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---------------------------------------------------------------------------
# 0. THIẾT LẬP ĐƯỜNG DẪN GỐC
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(ROOT, "data", "processed")
FIG_DIR   = os.path.join(ROOT, "outputs", "figures")
TABLE_DIR = os.path.join(ROOT, "outputs", "tables")
MODEL_DIR = os.path.join(ROOT, "outputs", "models")

TRAIN_CSV        = os.path.join(PROC_DIR,  "train.csv")
TEST_CSV         = os.path.join(PROC_DIR,  "test.csv")
ARIMA_METRICS    = os.path.join(TABLE_DIR, "cv04_arima_metrics.csv")

for d in [FIG_DIR, TABLE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. NẠP TẬP TRAIN & TEST
# ---------------------------------------------------------------------------
def load_data(train_path: str, test_path: str):
    """Đọc train.csv và test.csv, trả về 2 pd.Series với DatetimeIndex."""
    # TODO: pd.read_csv(path, index_col='date', parse_dates=True)['value']
    raise NotImplementedError("Thành viên 5 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 2. HUẤN LUYỆN MÔ HÌNH SARIMA
# ---------------------------------------------------------------------------
def fit_sarima(train: pd.Series, order: tuple, seasonal_order: tuple) -> object:
    """
    Fit mô hình SARIMA(p,d,q)(P,D,Q)[s] trên tập Train.

    Mô hình đề xuất khởi điểm cho a10.csv (Doanh số thuốc — trend + mùa vụ):
      order          = (1, 1, 1)         — Thành phần phi mùa vụ
      seasonal_order = (0, 1, 1, 12)     — Mùa vụ 12 tháng, sai phân D=1
    Có thể thử:
      seasonal_order = (1, 1, 1, 12)     — Nếu PACF gai nhọn tại lag 12

    Args:
        train          : pd.Series — Chuỗi Train gốc (163 tháng).
        order          : tuple (p, d, q).
        seasonal_order : tuple (P, D, Q, s) — s PHẢI bằng 12.

    Returns:
        Đối tượng SARIMAXResultsWrapper đã fit.
    """
    # TODO: model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
    #                       enforce_stationarity=False, enforce_invertibility=False)
    #       result = model.fit(disp=False)
    #       print(result.summary())
    raise NotImplementedError("Thành viên 5 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 3. CHẨN ĐOÁN PHẦN DƯ SARIMA (TẬP TRUNG VÀO LAG MÙA VỤ)
# ---------------------------------------------------------------------------
def diagnose_residuals_sarima(result, model_label: str = "SARIMA") -> dict:
    """
    Chẩn đoán phần dư của mô hình SARIMA:
      (a) Vẽ 4-panel: Residual Plot | Histogram | Q-Q Plot | ACF residuals.
      (b) Kiểm định Ljung-Box tại lag 12 VÀ lag 24 — phải sạch tự tương quan mùa vụ.
          → p-value > 0.05 tại lag 12 và 24: phần dư đạt White Noise ✓

    Lưu: outputs/figures/cv05_sarima_residual_diag.png (tùy chọn)
    Returns:
        dict: {'mean_residual', 'lb_pvalue_lag12', 'lb_pvalue_lag24', 'is_white_noise'}
    """
    # TODO: residuals = result.resid
    #       lb = acorr_ljungbox(residuals, lags=[12, 24], return_df=True)
    raise NotImplementedError("Thành viên 5 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 4. DỰ BÁO NGOÀI MẪU 41 THÁNG
# ---------------------------------------------------------------------------
def forecast_test(result, steps: int = 41) -> pd.Series:
    """
    Dự báo ngoài mẫu (Out-of-sample) 41 bước thời gian.
    KHÔNG được dùng get_prediction() với dữ liệu Test đã biết trước.

    Returns:
        pd.Series — Giá trị dự báo cho 41 tháng tập Test.
    """
    # TODO: forecast_obj = result.get_forecast(steps=steps)
    #       return forecast_obj.predicted_mean
    raise NotImplementedError("Thành viên 5 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 5. TÍNH CHỈ SỐ ĐÁNH GIÁ
# ---------------------------------------------------------------------------
def compute_metrics(actual: pd.Series, predicted: pd.Series, aic: float, bic: float) -> dict:
    """
    Tính MAE, RMSE, MAPE trên tập Test (cùng định nghĩa với CV04).

    Returns:
        dict: {'MAE', 'RMSE', 'MAPE', 'AIC', 'BIC'}
    """
    # TODO: (giống CV04 — tái sử dụng công thức)
    raise NotImplementedError("Thành viên 5 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 6. LẬP BẢNG BENCHMARK ĐỐI CHUẨN ARIMA VS SARIMA
# ---------------------------------------------------------------------------
def build_benchmark_table(arima_metrics_path: str, sarima_metrics: dict) -> pd.DataFrame:
    """
    Đọc cv04_arima_metrics.csv và ghép với sarima_metrics để tạo bảng
    đối chuẩn trực diện 2 mô hình.

    Cột bảng: [Model, AIC, BIC, MAE, RMSE, MAPE, Nhận xét]
    Lưu: outputs/tables/cv05_model_benchmark_table.csv

    Returns:
        pd.DataFrame — Bảng Benchmark hoàn chỉnh.
    """
    # TODO: arima_df = pd.read_csv(arima_metrics_path)
    #       Tạo DataFrame 2 hàng: ARIMA và SARIMA
    raise NotImplementedError("Thành viên 5 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 7. VẼ BIỂU ĐỒ DỰ BÁO VS THỰC TẾ & ĐỐI CHUẨN 2 MÔ HÌNH
# ---------------------------------------------------------------------------
def plot_sarima_forecast(train: pd.Series, test: pd.Series, sarima_fc: pd.Series) -> None:
    """
    Vẽ biểu đồ:
      - Train (màu xanh đậm)
      - Thực tế Test (màu xanh lá)
      - Dự báo SARIMA (màu đỏ, nét đứt)
    Lưu: outputs/figures/cv05_sarima_forecast_vs_actual.png
    """
    raise NotImplementedError("Thành viên 5 cài đặt tại đây")


def plot_model_comparison(test: pd.Series, arima_metrics_path: str, sarima_fc: pd.Series) -> None:
    """
    Vẽ biểu đồ so sánh 2 mô hình trên tập Test (Actual vs ARIMA vs SARIMA).
    Lưu: outputs/figures/cv05_arima_vs_sarima_comparison.png
    """
    # TODO: Đọc dự báo ARIMA từ arima_model.pkl để vẽ so sánh
    raise NotImplementedError("Thành viên 5 cài đặt tại đây")


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("CV05 — Mô hình SARIMA(p,d,q)(P,D,Q)[12] & Benchmark (Thành viên 5)")
    print("=" * 70)

    # Bộ tham số đề xuất (điều chỉnh sau khi tham khảo kết quả CV03)
    SARIMA_ORDER          = (1, 1, 1)         # TODO: Điều chỉnh theo CV03
    SARIMA_SEASONAL_ORDER = (0, 1, 1, 12)     # TODO: Thử (1,1,1,12) nếu cần

    # Bước 1: Nạp dữ liệu
    train, test = load_data(TRAIN_CSV, TEST_CSV)
    print(f"[OK] Train: {len(train)} tháng | Test: {len(test)} tháng")

    # Bước 2: Fit SARIMA
    result = fit_sarima(train, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
    model_name = f"SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER}"
    print(f"[OK] {model_name} đã fit | AIC={result.aic:.2f} | BIC={result.bic:.2f}")

    # Bước 3: Chẩn đoán phần dư mùa vụ
    diag = diagnose_residuals_sarima(result, model_label=model_name)
    print(f"[Residuals] LB p-value lag12={diag['lb_pvalue_lag12']:.4f} | lag24={diag['lb_pvalue_lag24']:.4f} | White Noise: {diag['is_white_noise']}")

    # Bước 4: Dự báo 41 tháng
    sarima_forecast = forecast_test(result, steps=len(test))

    # Bước 5: Tính chỉ số
    metrics = compute_metrics(test, sarima_forecast, aic=result.aic, bic=result.bic)
    print(f"[Metrics] MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | MAPE={metrics['MAPE']:.4f}%")

    # Bước 6: Bảng Benchmark
    benchmark = build_benchmark_table(ARIMA_METRICS, metrics)
    print(f"\n[Benchmark]\n{benchmark.to_string()}")

    # Bước 7: Vẽ và lưu
    plot_sarima_forecast(train, test, sarima_forecast)
    plot_model_comparison(test, ARIMA_METRICS, sarima_forecast)
    with open(os.path.join(MODEL_DIR, "sarima_model.pkl"), 'wb') as f:
        pickle.dump(result, f)
    pd.DataFrame([metrics]).to_csv(os.path.join(TABLE_DIR, "cv05_sarima_metrics.csv"), index=False)

    print("[DONE] CV05 hoàn tất. Chuyển cv05_model_benchmark_table.csv cho CV06 (Lead).")

