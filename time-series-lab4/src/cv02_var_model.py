# ============================================================
# AI_AGENT_INSTRUCTIONS
# ============================================================
# THÀNH VIÊN  : CV02 – Hiệp
# DEADLINE    : 08/11/2026
# NHIỆM VỤ   : Xây dựng mô hình VAR (Statistical Baseline)
#
# INPUT CONTRACT:
#   - data/processed/train.csv, val.csv, test.csv   (scaled, DatetimeIndex)
#   - data/processed/scaler.pkl                     (StandardScaler fit trên train)
#
# OUTPUT CONTRACT:
#   - outputs/models/var_model.pkl                  (model đã fit)
#   - outputs/tables/cv02_var_metrics.csv           (RMSE, MAE, MAPE trên test)
#   - outputs/tables/var_metrics.csv
#   - outputs/tables/cv02_var_lag_selection.csv     (AIC/BIC theo p)
#   - outputs/tables/var_lag_selection.csv
#   - outputs/tables/cv02_adf_test_results.csv
#   - outputs/figures/cv02_aic_bic_lag_selection.png
#   - outputs/figures/var_aic_bic.png
#   - outputs/figures/cv02_var_actual_vs_predicted.png
#   - outputs/figures/var_forecast_vs_actual.png
#   - outputs/figures/cv02_adf_results.png
#
# LƯU Ý QUAN TRỌNG:
#   - VAR yêu cầu tất cả chuỗi phải DỪNG (stationary)
#   - Nếu cần sai phân, phải inverse differencing khi dự báo
#   - Chỉ đánh giá biến TARGET = 'load'
#   - Dự báo 1 bước trước (one-step-ahead walk-forward forecasting)
# ============================================================

import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

# Cấu hình thẩm mỹ đồ thị
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

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
    Nếu có chuỗi KHÔNG dừng (p-value > 0.05) → lấy sai phân bậc 1 (diff(1)).
    """
    print("=" * 60)
    print("[STEP 1] Kiểm định tính dừng – ADF Test")
    print("=" * 60)

    adf_results = {}
    adf_records = []
    non_stationary = []

    for col in FEATURES:
        series = df[col].dropna()
        adf_stat, p_value, lags_used, n_obs, crit_vals, _ = adfuller(series, autolag="AIC")
        is_stationary = p_value < 0.05
        adf_results[col] = {
            "ADF Statistic": round(adf_stat, 4),
            "p-value"      : round(p_value, 4),
            "lags_used"    : lags_used,
            "Stationary?"  : is_stationary,
        }
        adf_records.append({
            "Feature": col,
            "ADF_Statistic": round(adf_stat, 4),
            "p_value": round(p_value, 4),
            "Lags_Used": lags_used,
            "Critical_Value_5%": round(crit_vals["5%"], 4),
            "Stationary": "YES" if is_stationary else "NO",
        })
        status = "DỪNG ✓" if is_stationary else "KHÔNG DỪNG ✗ (p > 0.05)"
        print(f"  {col:15s} → ADF={adf_stat:8.4f}, p={p_value:.4e}  [{status}]")

        if not is_stationary:
            non_stationary.append(col)

    # Lưu bảng kết quả ADF Test
    adf_df = pd.DataFrame(adf_records)
    adf_df.to_csv(TABLES_DIR / "cv02_adf_test_results.csv", index=False)
    print(f"\n  Đã lưu bảng ADF test: {TABLES_DIR / 'cv02_adf_test_results.csv'}")

    # Vẽ biểu đồ p-values so với ngưỡng 0.05
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(adf_df["Feature"], adf_df["p_value"], color=["seagreen" if p < 0.05 else "crimson" for p in adf_df["p_value"]])
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.2, label="Ngưỡng p = 0.05")
    ax.set_title("Kiểm định ADF: Giá trị p-value của 4 biến số", fontsize=12, fontweight="bold")
    ax.set_ylabel("p-value")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cv02_adf_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu biểu đồ ADF: {FIGURES_DIR / 'cv02_adf_results.png'}")

    differenced = False
    if non_stationary:
        print(f"\n  Các chuỗi không dừng: {non_stationary}")
        print("  → Áp dụng sai phân bậc 1 cho TOÀN BỘ DataFrame")
        df_stationary = df.diff(1).dropna()
        differenced = True

        print("\n  Kiểm tra lại sau sai phân bậc 1:")
        for col in FEATURES:
            _, p2, _, _, _, _ = adfuller(df_stationary[col].dropna(), autolag="AIC")
            status = "DỪNG ✓" if p2 < 0.05 else "VẪN KHÔNG DỪNG ✗"
            print(f"  {col:15s} → p={p2:.4e}  [{status}]")
    else:
        print("  Tất cả chuỗi đã dừng, không cần sai phân.")
        df_stationary = df.copy()

    print(f"\n  [STEP 1 DONE] differenced={differenced}, shape={df_stationary.shape}\n")
    return df_stationary, differenced, adf_results


# ----------------------------------------------------------
# STEP 2: Chọn bậc tối ưu cho VAR (AIC/BIC)
# ----------------------------------------------------------
def step2_select_lag_order(df_stationary: pd.DataFrame, max_lags: int = 24):
    """
    Thử nghiệm bậc lag p từ 1 đến max_lags, ghi lại AIC và BIC.
    Chọn p tối ưu theo AIC cho bài toán dự báo.
    """
    print("=" * 60)
    print(f"[STEP 2] Chọn bậc tối ưu cho VAR (AIC/BIC) [max_lags={max_lags}]")
    print("=" * 60)

    model_selector = VAR(df_stationary)
    results = model_selector.select_order(maxlags=max_lags)
    print(results.summary())

    aic_vals = results.ics["aic"]
    bic_vals = results.ics["bic"]
    p_indices = list(range(len(aic_vals)))

    lag_df = pd.DataFrame({
        "p"  : p_indices,
        "AIC": aic_vals,
        "BIC": bic_vals,
    }).set_index("p")

    # Lưu bảng kết quả AIC/BIC
    lag_df.to_csv(TABLES_DIR / "cv02_var_lag_selection.csv")
    lag_df.to_csv(TABLES_DIR / "var_lag_selection.csv")
    print(f"\n  Bảng AIC/BIC đã lưu tại: {TABLES_DIR / 'cv02_var_lag_selection.csv'}")

    p_opt = int(results.selected_orders.get("aic", lag_df["AIC"].idxmin()))
    p_bic = int(results.selected_orders.get("bic", lag_df["BIC"].idxmin()))
    if p_opt == 0:
        p_opt = 1
    print(f"  p tối ưu (AIC) = {p_opt}")
    print(f"  p tối ưu (BIC) = {p_bic}")

    # Vẽ biểu đồ AIC vs BIC
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(lag_df.index, lag_df["AIC"], marker="o", markersize=4, label="AIC", color="#1f77b4")
    ax.plot(lag_df.index, lag_df["BIC"], marker="s", markersize=4, label="BIC", color="#d62728")
    ax.axvline(x=p_opt, color="seagreen", linestyle="--", linewidth=1.5, label=f"Tối ưu AIC: p = {p_opt}")
    ax.set_xlabel("Bậc trễ p (Lags)", fontsize=11)
    ax.set_ylabel("Information Criterion", fontsize=11)
    ax.set_title("Lựa chọn bậc trễ tối ưu cho VAR (AIC vs BIC)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(FIGURES_DIR / "cv02_aic_bic_lag_selection.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "var_aic_bic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu biểu đồ: {FIGURES_DIR / 'cv02_aic_bic_lag_selection.png'}")

    print(f"[STEP 2 DONE] p_opt={p_opt}\n")
    return p_opt, lag_df


# ----------------------------------------------------------
# STEP 3: Fit mô hình VAR(p)
# ----------------------------------------------------------
def step3_fit_var(df_train_stationary: pd.DataFrame, p_opt: int):
    """
    Fit mô hình VAR với bậc p_opt tối ưu trên tập dữ liệu huấn luyện đã dừng.
    """
    print("=" * 60)
    print(f"[STEP 3] Huấn luyện mô hình VAR(p={p_opt})")
    print("=" * 60)

    var_model = VAR(df_train_stationary)
    fitted_model = var_model.fit(maxlags=p_opt, ic=None)

    summary = fitted_model.summary()
    summary_path = TABLES_DIR / "var_model_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(summary))
    print(f"  Đã lưu tóm tắt mô hình: {summary_path}")

    # Lưu model artifact
    model_path = MODELS_DIR / "var_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(fitted_model, f)
    print(f"  Đã lưu model artifact: {model_path}")

    print("[STEP 3 DONE]\n")
    return fitted_model


# ----------------------------------------------------------
# STEP 4: Dự báo One-Step-Ahead Walk-Forward & Inverse Transform
# ----------------------------------------------------------
def step4_forecast_and_inverse(fitted_model, df_train_stationary: pd.DataFrame,
                                df_test: pd.DataFrame,
                                scaler, differenced: bool,
                                df_train_original: pd.DataFrame = None):
    """
    Dự báo One-Step-Ahead Walk-Forward cho toàn bộ tập test:
      - Sử dụng cửa sổ trượt p bước quá khứ
      - Nếu differenced=True: bù sai phân để phục hồi thang đo chuẩn hóa
      - Nghịch đảo chuẩn hóa (Inverse transform) về thang đo thực tế (MW)
      - Trích xuất biến TARGET = 'load'
    """
    print("=" * 60)
    print("[STEP 4] Dự báo One-Step-Ahead Walk-Forward & Inverse Transform")
    print("=" * 60)

    p = fitted_model.k_ar
    col_idx_load = FEATURES.index(TARGET)
    n_test = len(df_test)

    # History khởi tạo dạng stationary (p x 4)
    history = df_train_stationary.values[-p:].copy()

    if differenced:
        last_train_point = df_train_original.values[-1:]  # (1, 4)
        combined_series = np.vstack([last_train_point, df_test.values])  # (n_test + 1, 4)
        test_diff_actual = np.diff(combined_series, axis=0)  # (n_test, 4)

    y_pred_scaled = np.zeros((n_test, len(FEATURES)))

    # Vòng lặp Walk-Forward One-Step-Ahead
    for i in range(n_test):
        pred_diff_step = fitted_model.forecast(y=history, steps=1)[0]

        if differenced:
            prev_actual_scaled = combined_series[i]
            y_pred_scaled[i] = prev_actual_scaled + pred_diff_step
            # Cập nhật history với sai phân thực tế quan sát được tại bước i
            history = np.vstack([history[1:], test_diff_actual[i]])
        else:
            y_pred_scaled[i] = pred_diff_step
            # Cập nhật history với quan sát thực tế tại bước i
            history = np.vstack([history[1:], df_test.values[i]])

    # Nghịch đảo chuẩn hóa về đơn vị vật lý gốc
    y_true_original = scaler.inverse_transform(df_test.values)
    y_pred_original = scaler.inverse_transform(y_pred_scaled)

    y_true = y_true_original[:, col_idx_load]
    y_pred = y_pred_original[:, col_idx_load]

    print(f"  y_true shape : {y_true.shape} (Min={y_true.min():.1f}, Max={y_true.max():.1f})")
    print(f"  y_pred shape : {y_pred.shape} (Min={y_pred.min():.1f}, Max={y_pred.max():.1f})")
    print("[STEP 4 DONE]\n")
    return y_true, y_pred


# ----------------------------------------------------------
# Helper: Tính Metrics
# ----------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {
        "Model": "VAR",
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MAPE": round(float(mape), 4),
    }
    print(f"  RMSE = {metrics['RMSE']:.4f} MW")
    print(f"  MAE  = {metrics['MAE']:.4f} MW")
    print(f"  MAPE = {metrics['MAPE']:.4f} %")
    return metrics


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  CV02 – VAR MODEL PIPELINE (Hiệp)")
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

    # Huấn luyện trên Train + Val theo chuẩn Walk-forward
    df_train_full = pd.concat([df_train, df_val])

    # --- Step 1: Kiểm định tính dừng ---
    df_train_stat, differenced, adf_res = step1_stationarity_test(df_train_full)

    # --- Step 2: Lựa chọn bậc trễ p tối ưu ---
    p_opt, lag_df = step2_select_lag_order(df_train_stat, max_lags=24)

    # --- Step 3: Huấn luyện mô hình VAR ---
    fitted_model = step3_fit_var(df_train_stat, p_opt)

    # --- Step 4: Dự báo và nghịch đảo chuẩn hóa ---
    y_true, y_pred = step4_forecast_and_inverse(
        fitted_model, df_train_stat, df_test,
        scaler, differenced, df_train_original=df_train_full
    )

    # --- Metrics ---
    print("=" * 60)
    print("[METRICS] VAR – Đánh giá trên tập Test")
    print("=" * 60)
    metrics = compute_metrics(y_true, y_pred)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(TABLES_DIR / "cv02_var_metrics.csv", index=False)
    metrics_df.to_csv(TABLES_DIR / "var_metrics.csv", index=False)
    print(f"  Đã lưu: {TABLES_DIR / 'cv02_var_metrics.csv'}")

    # --- Trực quan hóa Actual vs Forecast ---
    fig, ax = plt.subplots(figsize=(14, 5))
    test_dates = df_test.index
    ax.plot(test_dates, y_true, label="Phụ tải thực tế (Actual Load)", color="#1f77b4", linewidth=1.2)
    ax.plot(test_dates, y_pred, label=f"VAR(p={p_opt}) Dự báo (Forecast)", color="#d62728", linewidth=1.0, linestyle="--", alpha=0.9)
    ax.set_title(f"Mô hình Thống kê VAR(p={p_opt}): Dự báo vs Thực tế trên tập Test (3,947 giờ)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thời gian", fontsize=11)
    ax.set_ylabel("Phụ tải (Megawatt - MW)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(FIGURES_DIR / "cv02_var_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "var_forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu đồ thị: {FIGURES_DIR / 'cv02_var_actual_vs_predicted.png'}")

    print("\n" + "=" * 60)
    print("  [CV02 HOÀN THÀNH] Toàn bộ pipeline VAR đã hoàn tất thành công.")
    print("=" * 60)


if __name__ == "__main__":
    main()
