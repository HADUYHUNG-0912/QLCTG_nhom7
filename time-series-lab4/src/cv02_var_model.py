# ============================================================
# AI_AGENT_INSTRUCTIONS
# ============================================================
# THÀNH VIÊN  : CV02 – Khải
# DEADLINE    : 08/11
# NHIỆM VỤ   : Xây dựng mô hình VAR (Statistical Baseline)
#
# INPUT CONTRACT:
#   - data/processed/train.csv, val.csv, test.csv   (scaled, DatetimeIndex)
#   - data/processed/scaler.pkl                     (StandardScaler fit trên train)
#
# OUTPUT CONTRACT:
#   - outputs/models/var_model.pkl                  (model đã fit)
#   - outputs/tables/var_metrics.csv                (RMSE, MAE, MAPE trên test)
#   - outputs/tables/var_lag_selection.csv          (AIC/BIC theo p)
#   - outputs/figures/var_aic_bic.png
#   - outputs/figures/var_forecast_vs_actual.png
#   - outputs/figures/var_adf_results.png           (tùy chọn)
#
# LƯU Ý QUAN TRỌNG:
#   - VAR yêu cầu tất cả chuỗi phải DỪNG (stationary)
#   - Nếu cần sai phân, phải inverse differencing khi dự báo
#   - Chỉ đánh giá biến TARGET = 'load'
#   - Dự báo 1 bước trước (one-step-ahead forecasting)
# ============================================================

import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

# ----------------------------------------------------------
# Định nghĩa đường dẫn
# ----------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR    = PROJECT_ROOT / "outputs" / "tables"

for d in [MODELS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FEATURES     = ["load", "temp", "humidity", "wind_speed"]
TARGET       = "load"
RANDOM_STATE = 42


# ----------------------------------------------------------
# STEP 1: Kiểm định tính dừng (ADF Test)
# ----------------------------------------------------------
def step1_stationarity_test(df: pd.DataFrame):
    """
    Kiểm định ADF (Augmented Dickey-Fuller) cho tất cả 4 chuỗi.

    Nếu chuỗi KHÔNG dừng (p-value > 0.05) → lấy sai phân bậc 1 (diff(1)).
    Sau đó dropna().

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã scale với 4 cột FEATURES.

    Returns
    -------
    df_stationary : pd.DataFrame
        DataFrame dừng (sau sai phân nếu cần).
    differenced : bool
        True nếu đã lấy sai phân bậc 1.
    adf_results : dict
        Kết quả ADF test cho từng cột (trước sai phân).
    """
    print("=" * 60)
    print("[STEP 1] Kiểm định tính dừng – ADF Test")
    print("=" * 60)

    adf_results   = {}
    non_stationary = []

    for col in FEATURES:
        series   = df[col].dropna()
        adf_stat, p_value, lags_used, n_obs, crit_vals, _ = adfuller(series, autolag="AIC")
        is_stationary = p_value < 0.05
        adf_results[col] = {
            "ADF Statistic": round(adf_stat, 4),
            "p-value"      : round(p_value, 4),
            "lags_used"    : lags_used,
            "Stationary?"  : is_stationary,
        }
        status = "DỪNG ✓" if is_stationary else "KHÔNG DỪNG ✗"
        print(f"  {col:15s} → ADF={adf_stat:.4f}, p={p_value:.4f}  [{status}]")

        if not is_stationary:
            non_stationary.append(col)

    differenced = False
    if non_stationary:
        print(f"\n  Các chuỗi không dừng: {non_stationary}")
        print("  → Áp dụng sai phân bậc 1 cho TOÀN BỘ DataFrame")
        df_stationary = df.diff(1).dropna()
        differenced   = True

        # Kiểm tra lại sau sai phân
        print("\n  Kiểm tra lại sau sai phân bậc 1:")
        for col in FEATURES:
            _, p2, _, _, _, _ = adfuller(df_stationary[col].dropna(), autolag="AIC")
            status = "DỪNG ✓" if p2 < 0.05 else "VẪN KHÔNG DỪNG ✗"
            print(f"  {col:15s} → p={p2:.4f}  [{status}]")
    else:
        print("  Tất cả chuỗi đã dừng, không cần sai phân.")
        df_stationary = df.copy()

    print(f"\n  [STEP 1 DONE] differenced={differenced}, shape={df_stationary.shape}\n")
    return df_stationary, differenced, adf_results


# ----------------------------------------------------------
# STEP 2: Chọn bậc tối ưu cho VAR
# ----------------------------------------------------------
def step2_select_lag_order(df_stationary: pd.DataFrame, max_lags: int = 24):
    """
    Thử bậc lag p từ 1 đến max_lags, ghi lại AIC và BIC.
    Chọn p tối ưu theo AIC (thường dùng cho dự báo).

    Vẽ biểu đồ AIC và BIC theo p, lưu vào outputs/figures/.

    Parameters
    ----------
    df_stationary : pd.DataFrame
        DataFrame đã dừng.
    max_lags : int
        Số lag tối đa thử (mặc định 24).

    Returns
    -------
    p_opt : int
        Bậc lag tối ưu theo AIC.
    lag_df : pd.DataFrame
        Bảng AIC/BIC theo p.
    """
    print("=" * 60)
    print("[STEP 2] Chọn bậc tối ưu cho VAR (AIC/BIC)")
    print("=" * 60)

    model_selector = VAR(df_stationary)

    # statsmodels cung cấp phương thức select_order tiện lợi
    results = model_selector.select_order(maxlags=max_lags)
    print(results.summary())

    # Trích xuất AIC và BIC theo p
    aic_vals = results.ics["aic"]   # dict {p: aic_value}
    bic_vals = results.ics["bic"]

    lag_df = pd.DataFrame({
        "p"  : list(aic_vals.keys()),
        "AIC": list(aic_vals.values()),
        "BIC": list(bic_vals.values()),
    }).set_index("p")

    lag_df.to_csv(TABLES_DIR / "var_lag_selection.csv")
    print(f"\n  Bảng AIC/BIC đã lưu tại: {TABLES_DIR / 'var_lag_selection.csv'}")

    # p tối ưu theo AIC
    p_opt = int(lag_df["AIC"].idxmin())
    print(f"  p tối ưu (AIC) = {p_opt}")
    print(f"  p tối ưu (BIC) = {int(lag_df['BIC'].idxmin())}")

    # Vẽ biểu đồ AIC vs BIC
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lag_df.index, lag_df["AIC"], marker="o", label="AIC", color="steelblue")
    ax.plot(lag_df.index, lag_df["BIC"], marker="s", label="BIC", color="tomato")
    ax.axvline(x=p_opt, color="green", linestyle="--", label=f"p_opt (AIC) = {p_opt}")
    ax.set_xlabel("Bậc lag p")
    ax.set_ylabel("Thông tin tiêu chuẩn")
    ax.set_title("Chọn bậc VAR theo AIC/BIC")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "var_aic_bic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {FIGURES_DIR / 'var_aic_bic.png'}")

    print(f"[STEP 2 DONE] p_opt={p_opt}\n")
    return p_opt, lag_df


# ----------------------------------------------------------
# STEP 3: Fit mô hình VAR(p)
# ----------------------------------------------------------
def step3_fit_var(df_train_stationary: pd.DataFrame, p_opt: int):
    """
    Fit mô hình VAR với bậc p_opt tối ưu trên tập train.

    Parameters
    ----------
    df_train_stationary : pd.DataFrame
        Tập train đã dừng.
    p_opt : int
        Bậc lag tối ưu.

    Returns
    -------
    fitted_model : VARResultsWrapper
        Mô hình VAR đã fit.
    """
    print("=" * 60)
    print(f"[STEP 3] Fit mô hình VAR(p={p_opt})")
    print("=" * 60)

    var_model  = VAR(df_train_stationary)
    fitted_model = var_model.fit(maxlags=p_opt, ic=None)

    # In summary
    summary = fitted_model.summary()
    print(summary)

    # Lưu summary ra file text
    summary_path = TABLES_DIR / "var_model_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(summary))
    print(f"\n  Đã lưu model summary: {summary_path}")

    # Lưu model
    model_path = MODELS_DIR / "var_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(fitted_model, f)
    print(f"  Đã lưu model: {model_path}")

    print("[STEP 3 DONE]\n")
    return fitted_model


# ----------------------------------------------------------
# STEP 4: Dự báo & Inverse Transform
# ----------------------------------------------------------
def step4_forecast_and_inverse(fitted_model, df_train_stationary: pd.DataFrame,
                                df_test: pd.DataFrame,
                                scaler, differenced: bool,
                                df_train_original: pd.DataFrame = None):
    """
    Dự báo one-step-ahead cho toàn bộ tập test.

    Quy trình:
      1. Dùng cửa sổ trượt = p_opt bước cuối của lịch sử làm đầu vào
      2. Nếu differenced=True: inverse difference bằng cách cộng dồn
      3. Inverse StandardScaler
      4. Chỉ lấy cột 'load' để đánh giá

    Parameters
    ----------
    fitted_model : VARResultsWrapper
    df_train_stationary : pd.DataFrame   (tập train đã dừng)
    df_test : pd.DataFrame               (tập test đã scale, CHƯA sai phân)
    scaler : StandardScaler
    differenced : bool
    df_train_original : pd.DataFrame     (tập train chưa sai phân, cần nếu differenced=True)

    Returns
    -------
    y_true : np.ndarray
        Giá trị thực tế load (original scale).
    y_pred : np.ndarray
        Giá trị dự báo load (original scale).
    """
    print("=" * 60)
    print("[STEP 4] Dự báo & Inverse Transform")
    print("=" * 60)

    p = fitted_model.k_ar   # bậc lag của model
    col_idx_load = FEATURES.index(TARGET)

    # Lịch sử ban đầu = p bước cuối của train (dạng stationary)
    history = df_train_stationary.values[-p:]  # shape (p, 4)

    y_pred_scaled_diff = []

    # TODO (CV02): Implement vòng lặp walk-forward one-step-ahead
    # GỢI Ý:
    # for i in range(len(df_test)):
    #     forecast = fitted_model.forecast(y=history, steps=1)  # shape (1, 4)
    #     y_pred_scaled_diff.append(forecast[0])
    #
    #     # Cập nhật history:
    #     if differenced:
    #         # Giá trị stationary thực tế của bước i
    #         actual_stat = df_test_stationary.iloc[i].values  # ← cần tính trước
    #     else:
    #         actual_stat = df_test.iloc[i].values
    #     history = np.vstack([history[1:], actual_stat])

    y_pred_scaled_diff = np.array(y_pred_scaled_diff)  # shape (n_test, 4)

    # --- Inverse differencing (nếu cần) ---
    if differenced:
        # TODO (CV02): Cộng dồn lại để khôi phục giá trị scaled (không sai phân)
        # GỢI Ý: last_train_val = df_train_original.values[-1]  (giá trị cuối train, đã scale)
        # y_pred_scaled = np.cumsum(y_pred_scaled_diff, axis=0) + last_train_val
        pass
    else:
        y_pred_scaled = y_pred_scaled_diff

    # --- Inverse StandardScaler ---
    # GỢI Ý: y_pred_original = scaler.inverse_transform(y_pred_scaled)
    #         y_true_original = scaler.inverse_transform(df_test.values)
    # Sau đó chỉ lấy cột load:
    #         y_pred = y_pred_original[:, col_idx_load]
    #         y_true = y_true_original[:, col_idx_load]

    # Placeholder (CV02 thay thế bằng kết quả thực):
    y_true = np.zeros(len(df_test))
    y_pred = np.zeros(len(df_test))

    print(f"  y_true shape: {y_true.shape}")
    print(f"  y_pred shape: {y_pred.shape}")
    print("[STEP 4 DONE]\n")
    return y_true, y_pred


# ----------------------------------------------------------
# Helper: Tính Metrics
# ----------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Tính RMSE, MAE, MAPE cho dự báo.

    Parameters
    ----------
    y_true, y_pred : np.ndarray

    Returns
    -------
    metrics : dict  {"RMSE": ..., "MAE": ..., "MAPE": ...}
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # Tránh chia cho 0 khi y_true = 0
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE": round(mape, 4)}
    print(f"  RMSE = {metrics['RMSE']}")
    print(f"  MAE  = {metrics['MAE']}")
    print(f"  MAPE = {metrics['MAPE']} %")
    return metrics


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    """
    Pipeline chính cho CV02: VAR Baseline.
    """
    print("\n" + "=" * 60)
    print("  CV02 – VAR MODEL PIPELINE")
    print("=" * 60 + "\n")

    # --- Load dữ liệu ---
    df_train = pd.read_csv(PROCESSED_DIR / "train.csv", index_col=0, parse_dates=True)
    df_val   = pd.read_csv(PROCESSED_DIR / "val.csv",   index_col=0, parse_dates=True)
    df_test  = pd.read_csv(PROCESSED_DIR / "test.csv",  index_col=0, parse_dates=True)

    with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print(f"  Train shape : {df_train.shape}")
    print(f"  Val shape   : {df_val.shape}")
    print(f"  Test shape  : {df_test.shape}\n")

    # Train + Val gộp lại để fit VAR (CV02 có thể thử cả 2 cách)
    df_train_full = pd.concat([df_train, df_val])

    # --- Step 1: Stationarity test ---
    df_train_stat, differenced, adf_res = step1_stationarity_test(df_train_full)

    # --- Step 2: Lag order selection ---
    p_opt, lag_df = step2_select_lag_order(df_train_stat, max_lags=24)

    # --- Step 3: Fit VAR ---
    fitted_model = step3_fit_var(df_train_stat, p_opt)

    # --- Step 4: Forecast ---
    y_true, y_pred = step4_forecast_and_inverse(
        fitted_model, df_train_stat, df_test,
        scaler, differenced, df_train_original=df_train_full
    )

    # --- Metrics ---
    print("=" * 60)
    print("[METRICS] VAR – Tập Test")
    print("=" * 60)
    metrics = compute_metrics(y_true, y_pred)
    pd.DataFrame([metrics]).to_csv(TABLES_DIR / "var_metrics.csv", index=False)
    print(f"  Đã lưu: {TABLES_DIR / 'var_metrics.csv'}")

    # --- Vẽ Forecast vs Actual ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true, label="Actual Load",   color="steelblue", linewidth=1.0)
    ax.plot(y_pred, label="VAR Forecast",  color="tomato",    linewidth=0.8, linestyle="--")
    ax.set_title("VAR: Dự báo vs Thực tế (Load) – Tập Test")
    ax.set_xlabel("Bước thời gian")
    ax.set_ylabel("Load (original scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "var_forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {FIGURES_DIR / 'var_forecast_vs_actual.png'}")

    print("\n" + "=" * 60)
    print("  [CV02 DONE] VAR pipeline hoàn tất.")
    print("=" * 60)


if __name__ == "__main__":
    main()
