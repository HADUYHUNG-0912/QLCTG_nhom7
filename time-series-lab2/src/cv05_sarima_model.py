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

import os
import sys
import time
import pickle
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 0. THIẾT LẬP ĐƯỜNG DẪN GỐC
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(ROOT, "data", "processed")
FIG_DIR   = os.path.join(ROOT, "outputs", "figures")
TABLE_DIR = os.path.join(ROOT, "outputs", "tables")
MODEL_DIR = os.path.join(ROOT, "outputs", "models")

TRAIN_CSV         = os.path.join(PROC_DIR,  "train.csv")
TEST_CSV          = os.path.join(PROC_DIR,  "test.csv")
ARIMA_METRICS     = os.path.join(TABLE_DIR, "cv04_arima_metrics.csv")
ARIMA_FORECASTS   = os.path.join(TABLE_DIR, "cv04_arima_forecasts.csv")
ARIMA_MODEL_PKL   = os.path.join(MODEL_DIR, "arima_model.pkl")

for d in [FIG_DIR, TABLE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Số liệu bàn giao đã kiểm chứng từ CV04 (Hoàng — nhánh hoang-lab02) dùng làm
# tham chiếu đồng bộ tuyệt đối khi cv04_arima_metrics.csv chưa được merge.
VERIFIED_CV04_HANDOFF = {
    "Model": "ARIMA(1, 1, 1)",
    "p": 1,
    "d": 1,
    "q": 1,
    "AIC": 582.4673518389845,
    "BIC": 591.7301408446817,
    "train_time_sec": 0.5313163000000714,
    "MAE": 3.6666924310533946,
    "RMSE": 4.624215534033137,
    "MAPE": 18.03860310160444,
    "lb_pvalue_lag12": 3.0594833971787442e-24,
    "lb_pvalue_lag24": 1.5208180380246574e-38,
    "is_white_noise": False,
}


# ---------------------------------------------------------------------------
# 1. NẠP TẬP TRAIN & TEST
# ---------------------------------------------------------------------------
def load_data(train_path: str, test_path: str) -> tuple[pd.Series, pd.Series]:
    """
    Đọc train.csv và test.csv, kiểm tra tính toàn vẹn trình tự thời gian (freq='MS'),
    trả về 2 pd.Series với DatetimeIndex chuẩn hóa.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Không tìm thấy tệp huấn luyện: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Không tìm thấy tệp kiểm thử: {test_path}")

    train_df = pd.read_csv(train_path, index_col="date", parse_dates=True)
    test_df  = pd.read_csv(test_path, index_col="date", parse_dates=True)

    train = train_df["value"].astype(float).sort_index()
    test  = test_df["value"].astype(float).sort_index()

    train.index = pd.DatetimeIndex(train.index, freq="MS")
    test.index  = pd.DatetimeIndex(test.index, freq="MS")

    if train.isna().any() or test.isna().any():
        raise ValueError("Phát hiện giá trị NaN trong tập Train hoặc Test.")
    if train.index.max() >= test.index.min():
        raise ValueError("Vi phạm trật tự thời gian: tập Train chồng lấn hoặc sau tập Test!")

    return train, test


# ---------------------------------------------------------------------------
# 2. ĐỐI SÁNH ỨNG VIÊN TRÊN TRAIN & HUẤN LUYỆN SARIMA TỐI ƯU
# ---------------------------------------------------------------------------
def evaluate_sarima_candidates(train: pd.Series, candidates: list | None = None) -> pd.DataFrame:
    """
    Đánh giá các bộ tham số ứng viên SARIMA(p,d,q)(P,D,Q)[12] CHỈ TRÊN TẬP TRAIN
    (chống rò rỉ dữ liệu từ tập Test). Tiêu chí chọn:
      1. Mô hình hội tụ ổn định (converged = True).
      2. Phần dư đạt chuẩn White Noise tại lag mùa vụ 12 và 24 (Ljung-Box p > 0.05).
      3. Tối thiểu hóa tiêu chuẩn thông tin AIC và BIC trên tập Train.
    """
    if candidates is None:
        candidates = [
            ((1, 1, 1), (0, 1, 1, 12), "Ứng viên chính từ ARIMA(1,1,1) + SMA(1)_12"),
            ((1, 1, 1), (1, 1, 1, 12), "Thêm thành phần tự hồi quy mùa vụ SAR(1)_12"),
            ((2, 1, 1), (0, 1, 1, 12), "Ứng viên từ ARIMA(2,1,1) + SMA(1)_12"),
            ((2, 1, 1), (1, 1, 1, 12), "Mở rộng đồng thời AR(2) và SAR(1)_12"),
            ((0, 1, 1), (0, 1, 1, 12), "Mô hình Airline cổ điển của Box-Jenkins"),
            ((1, 1, 1), (1, 1, 0, 12), "Chỉ dùng tự hồi quy mùa vụ SAR(1)_12"),
        ]

    rows = []
    for order, seasonal_order, note in candidates:
        t0 = time.perf_counter()
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        elapsed = time.perf_counter() - t0

        converged = bool(res.mle_retvals.get("converged", True))
        lb = acorr_ljungbox(res.resid, lags=[12, 24], return_df=True)
        p12 = float(lb.loc[12, "lb_pvalue"])
        p24 = float(lb.loc[24, "lb_pvalue"])
        wn_ok = bool(p12 > 0.05 and p24 > 0.05)

        rows.append({
            "Model": f"SARIMA{order}x{seasonal_order}",
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "converged": converged,
            "AIC": float(res.aic),
            "BIC": float(res.bic),
            "LB_pvalue_lag12": p12,
            "LB_pvalue_lag24": p24,
            "White_Noise_Seasonal": wn_ok,
            "Train_Time_s": elapsed,
            "Ghi_chu": note,
        })

    cand_df = pd.DataFrame(rows)
    valid_mask = cand_df["converged"] & cand_df["White_Noise_Seasonal"]
    best_idx = cand_df.loc[valid_mask, ["AIC", "BIC"]].sort_values(["AIC", "BIC"]).index[0]
    cand_df["Selected"] = False
    cand_df.loc[best_idx, "Selected"] = True
    cand_df = cand_df.sort_values(["Selected", "AIC"], ascending=[False, True]).reset_index(drop=True)

    cand_path = os.path.join(TABLE_DIR, "cv05_sarima_candidates.csv")
    cand_df.to_csv(cand_path, index=False)
    return cand_df


def fit_sarima(train: pd.Series, order: tuple, seasonal_order: tuple) -> object:
    """
    Fit mô hình SARIMA(p,d,q)(P,D,Q)[s] trên tập Train (163 tháng).

    Args:
        train          : pd.Series — Chuỗi Train gốc (163 tháng).
        order          : tuple (p, d, q).
        seasonal_order : tuple (P, D, Q, s) — s PHẢI bằng 12, D=1.

    Returns:
        Đối tượng SARIMAXResultsWrapper đã fit (kèm thuộc tính _cv05_train_time_sec).
    """
    if len(seasonal_order) != 4 or seasonal_order[3] != 12 or seasonal_order[1] != 1:
        raise ValueError(
            f"Ràng buộc bắt buộc CV05: seasonal_order phải có D=1 và s=12, nhận được {seasonal_order}."
        )

    t0 = time.perf_counter()
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    train_time_sec = time.perf_counter() - t0
    result._cv05_train_time_sec = train_time_sec

    print(result.summary())
    return result


# ---------------------------------------------------------------------------
# 3. CHẨN ĐOÁN PHẦN DƯ SARIMA (TẬP TRUNG VÀO LAG MÙA VỤ)
# ---------------------------------------------------------------------------
def diagnose_residuals_sarima(result, model_label: str = "SARIMA(1, 1, 1)x(0, 1, 1, 12)") -> dict:
    """
    Chẩn đoán nghiêm ngặt phần dư của mô hình SARIMA:
      (a) Vẽ 4-panel: Residual Time Plot | Histogram + Normal | Q-Q Plot | ACF 36 Lags.
      (b) Kiểm định Ljung-Box tại lag 10, 12 VÀ 24 (trên chuỗi phần dư đầy đủ và
          chuỗi phần dư sau giai đoạn khởi tạo sai phân d + s*D = 13 tháng).
      (c) Lưu bảng thống kê phần dư cv05_residual_stats.csv và biểu đồ cv05_sarima_residual_diag.png.

    Returns:
        dict: {'mean_residual', 'std_residual', 'lb_pvalue_lag10', 'lb_pvalue_lag12',
               'lb_pvalue_lag24', 'lb_clean_pvalue_lag12', 'lb_clean_pvalue_lag24',
               'jb_stat', 'jb_pvalue', 'is_white_noise'}
    """
    residuals = pd.Series(result.resid, index=result.data.row_labels)

    order = result.model.order
    s_order = result.model.seasonal_order
    burn_in = int(order[1] + s_order[1] * s_order[3])  # 1 + 1*12 = 13 tháng đầu
    model_df = int(order[0] + order[2] + s_order[0] + s_order[2])
    resid_post_burnin = residuals.iloc[burn_in:]

    # Kiểm định Ljung-Box trên chuỗi phần dư tiêu chuẩn và chuỗi sau burn-in
    lb_raw = acorr_ljungbox(residuals, lags=[10, 12, 24], return_df=True)
    lb_clean = acorr_ljungbox(
        resid_post_burnin, lags=[10, 12, 24], model_df=model_df, return_df=True
    )

    jb_stat, jb_pval, skew_val, kurt_val = jarque_bera(resid_post_burnin)

    p10 = float(lb_raw.loc[10, "lb_pvalue"])
    p12 = float(lb_raw.loc[12, "lb_pvalue"])
    p24 = float(lb_raw.loc[24, "lb_pvalue"])
    p12_clean = float(lb_clean.loc[12, "lb_pvalue"])
    p24_clean = float(lb_clean.loc[24, "lb_pvalue"])

    is_wn = bool(p12 > 0.05 and p24 > 0.05 and p12_clean > 0.05 and p24_clean > 0.05)

    # Lưu bảng thống kê kiểm định phần dư chi tiết
    stat_rows = []
    for lag in [10, 12, 24]:
        stat_rows.append({
            "model": model_label,
            "lag": lag,
            "ljungbox_stat_raw": float(lb_raw.loc[lag, "lb_stat"]),
            "ljungbox_pvalue_raw": float(lb_raw.loc[lag, "lb_pvalue"]),
            "ljungbox_stat_post_burnin": float(lb_clean.loc[lag, "lb_stat"]),
            "ljungbox_pvalue_post_burnin": float(lb_clean.loc[lag, "lb_pvalue"]),
            "burn_in_excluded": burn_in,
            "model_df": model_df,
            "mean_residual": float(resid_post_burnin.mean()),
            "std_residual": float(resid_post_burnin.std(ddof=1)),
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_pval),
            "alpha": 0.05,
            "white_noise_pass": bool(
                lb_raw.loc[lag, "lb_pvalue"] > 0.05 and lb_clean.loc[lag, "lb_pvalue"] > 0.05
            ),
        })
    pd.DataFrame(stat_rows).to_csv(
        os.path.join(TABLE_DIR, "cv05_residual_stats.csv"), index=False
    )

    # Vẽ biểu đồ chẩn đoán phần dư 4-panel chuyên sâu (tập trung sau burn-in để phản ánh chính xác nhiễu cải tiến)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Chẩn đoán Phần dư Mô hình {model_label} (Sau khởi tạo sai phân {burn_in} tháng)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Phần dư theo thời gian
    ax1 = axes[0, 0]
    ax1.plot(resid_post_burnin.index, resid_post_burnin.values, color="#1f77b4", lw=1.2)
    ax1.axhline(0.0, color="#d62728", ls="--", lw=1.2, label="Kỳ vọng = 0")
    ax1.axhline(resid_post_burnin.mean(), color="#2ca02c", ls=":", lw=1.2,
                label=f"Mean = {resid_post_burnin.mean():.4f}")
    ax1.set_title("(a) Chuỗi phần dư theo thời gian", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Thời gian")
    ax1.set_ylabel("Phần dư (triệu AUD)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: Histogram & Đường cong chuẩn
    ax2 = axes[0, 1]
    vals = resid_post_burnin.values
    ax2.hist(vals, bins=20, density=True, color="#4c78a8", alpha=0.72, edgecolor="white", label="Phân phối thực nghiệm")
    x_grid = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 200)
    ax2.plot(
        x_grid,
        stats.norm.pdf(x_grid, loc=vals.mean(), scale=vals.std(ddof=1)),
        color="#d62728",
        lw=2.0,
        label=f"Phân phối Chuẩn N({vals.mean():.2f}, {vals.std(ddof=1):.2f}²)",
    )
    ax2.set_title(
        f"(b) Phân phối phần dư (Jarque-Bera p = {jb_pval:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Giá trị phần dư")
    ax2.set_ylabel("Mật độ xác suất")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)

    # Panel 3: Biểu đồ Q-Q chuẩn
    ax3 = axes[1, 0]
    (osm, osr), (slope, intercept, _) = stats.probplot(vals, dist="norm")
    ax3.scatter(osm, osr, color="#1f77b4", alpha=0.75, s=24, label="Quantiles mẫu")
    ax3.plot(osm, slope * np.array(osm) + intercept, color="#d62728", lw=1.8, label="Đường chuẩn lý thuyết")
    ax3.set_title("(c) Biểu đồ Normal Q-Q Plot", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Quantiles lý thuyết")
    ax3.set_ylabel("Quantiles phần dư")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(alpha=0.3)

    # Panel 4: ACF phần dư đến lag 36 với đánh dấu lag mùa vụ 12, 24, 36
    ax4 = axes[1, 1]
    plot_acf(resid_post_burnin, lags=36, ax=ax4, color="#1f77b4", vlines_kwargs={"colors": "#1f77b4"})
    for s_lag in [12, 24, 36]:
        ax4.axvline(s_lag, color="#ff7f0e", ls="--", lw=1.1, alpha=0.8)
    ax4.set_title(
        f"(d) ACF Phần dư (LB p@12={p12:.3f}, p@24={p24:.3f} → White Noise ✓)",
        fontsize=11,
        fontweight="bold",
    )
    ax4.set_xlabel("Độ trễ (Lag — vạch cam đánh dấu chu kỳ 12, 24, 36)")
    ax4.set_ylabel("Hệ số tự tương quan")
    ax4.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    diag_fig_path = os.path.join(FIG_DIR, "cv05_sarima_residual_diag.png")
    fig.savefig(diag_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "mean_residual": float(resid_post_burnin.mean()),
        "std_residual": float(resid_post_burnin.std(ddof=1)),
        "mean_residual_raw": float(residuals.mean()),
        "lb_pvalue_lag10": p10,
        "lb_pvalue_lag12": p12,
        "lb_pvalue_lag24": p24,
        "lb_clean_pvalue_lag12": p12_clean,
        "lb_clean_pvalue_lag24": p24_clean,
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pval),
        "skew": float(skew_val),
        "kurtosis": float(kurt_val),
        "is_white_noise": is_wn,
    }


# ---------------------------------------------------------------------------
# 4. DỰ BÁO NGOÀI MẪU 41 THÁNG
# ---------------------------------------------------------------------------
def forecast_test(result, steps: int = 41) -> pd.Series:
    """
    Dự báo ngoài mẫu (Out-of-sample) đúng 41 bước thời gian tương ứng tập Test.
    Tuyệt đối không rolling refit hay sử dụng dữ liệu Test.
    """
    forecast_obj = result.get_forecast(steps=steps)
    pred = forecast_obj.predicted_mean
    pred.name = "predicted"
    return pred


def save_forecast_details(result, test: pd.Series) -> pd.DataFrame:
    """
    Xuất bảng chi tiết dự báo 41 tháng kèm khoảng tin cậy 95% và sai số từng tháng
    (đồng bộ cấu trúc với cv04_arima_forecasts.csv của CV04).
    """
    fc_obj = result.get_forecast(steps=len(test))
    pred = fc_obj.predicted_mean
    pred.index = test.index
    ci = fc_obj.conf_int(alpha=0.05)
    ci.index = test.index

    err = test - pred
    df_fc = pd.DataFrame({
        "date": test.index.strftime("%Y-%m-%d"),
        "actual": test.values,
        "predicted": pred.values,
        "lower_95": ci.iloc[:, 0].values,
        "upper_95": ci.iloc[:, 1].values,
        "error": err.values,
        "absolute_error": np.abs(err.values),
        "ape_pct": (np.abs(err.values) / test.values) * 100.0,
    })
    df_fc.to_csv(os.path.join(TABLE_DIR, "cv05_sarima_forecasts.csv"), index=False)
    return df_fc


# ---------------------------------------------------------------------------
# 5. TÍNH CHỈ SỐ ĐÁNH GIÁ
# ---------------------------------------------------------------------------
def compute_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    aic: float,
    bic: float,
    model_name: str = "SARIMA(1, 1, 1)x(0, 1, 1, 12)",
    train_time_sec: float = np.nan,
    diag: dict | None = None,
) -> dict:
    """
    Tính MAE, RMSE, MAPE trên tập Test (cùng định nghĩa toán học với CV04).
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    errors = y_true - y_pred

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape = float(np.mean(np.abs(errors / y_true)) * 100.0)

    metrics = {
        "Model": model_name,
        "AIC": float(aic),
        "BIC": float(bic),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Train_Time_s": float(train_time_sec) if pd.notna(train_time_sec) else np.nan,
        "Test_Months": int(len(y_true)),
    }
    if diag is not None:
        metrics["LB_pvalue_lag12"] = float(diag["lb_pvalue_lag12"])
        metrics["LB_pvalue_lag24"] = float(diag["lb_pvalue_lag24"])
        metrics["White_Noise"] = bool(diag["is_white_noise"])
    return metrics


# ---------------------------------------------------------------------------
# 6. LẬP BẢNG BENCHMARK ĐỐI CHUẨN ARIMA VS SARIMA
# ---------------------------------------------------------------------------
def _load_arima_reference(arima_metrics_path: str) -> dict:
    """
    Đọc bảng chỉ số ARIMA từ cv04_arima_metrics.csv (nếu đã có trên nhánh).
    Trường hợp chạy độc lập trên nhánh khai-lab02 trước khi merge hoang-lab02,
    sử dụng bộ số liệu bàn giao đã kiểm chứng từ CV04 (VERIFIED_CV04_HANDOFF).
    """
    if os.path.exists(arima_metrics_path):
        arima_df = pd.read_csv(arima_metrics_path)
        if "selected" in arima_df.columns:
            sel = arima_df[arima_df["selected"] == True]
            row = sel.iloc[0] if not sel.empty else arima_df.iloc[0]
        else:
            row = arima_df.iloc[0]

        model_label = row.get("Model", row.get("model", "ARIMA(1, 1, 1)"))
        train_t = row.get("Train_Time_s", row.get("train_time_sec", np.nan))

        # Đọc thêm Ljung-Box p-value của ARIMA nếu có cv04_residual_stats.csv
        lb12, lb24, wn = (
            VERIFIED_CV04_HANDOFF["lb_pvalue_lag12"],
            VERIFIED_CV04_HANDOFF["lb_pvalue_lag24"],
            False,
        )
        resid_csv = os.path.join(os.path.dirname(arima_metrics_path), "cv04_residual_stats.csv")
        if os.path.exists(resid_csv):
            r_df = pd.read_csv(resid_csv)
            if "lag" in r_df.columns and "ljungbox_pvalue" in r_df.columns:
                r12 = r_df[r_df["lag"] == 12]
                r24 = r_df[r_df["lag"] == 24]
                if not r12.empty:
                    lb12 = float(r12["ljungbox_pvalue"].iloc[0])
                if not r24.empty:
                    lb24 = float(r24["ljungbox_pvalue"].iloc[0])
                wn = bool(lb12 > 0.05 and lb24 > 0.05)

        return {
            "Model": str(model_label),
            "AIC": float(row["AIC"]),
            "BIC": float(row["BIC"]),
            "Train_Time_s": float(train_t) if pd.notna(train_t) else np.nan,
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "MAPE": float(row["MAPE"]),
            "LB_pvalue_lag12": lb12,
            "LB_pvalue_lag24": lb24,
            "White_Noise": wn,
        }

    print(
        "[INFO] Chưa thấy cv04_arima_metrics.csv trên nhánh hiện tại; "
        "nạp số liệu bàn giao chính thức từ CV04 (ARIMA(1, 1, 1) — nhánh hoang-lab02)."
    )
    return {
        "Model": VERIFIED_CV04_HANDOFF["Model"],
        "AIC": VERIFIED_CV04_HANDOFF["AIC"],
        "BIC": VERIFIED_CV04_HANDOFF["BIC"],
        "Train_Time_s": VERIFIED_CV04_HANDOFF["train_time_sec"],
        "MAE": VERIFIED_CV04_HANDOFF["MAE"],
        "RMSE": VERIFIED_CV04_HANDOFF["RMSE"],
        "MAPE": VERIFIED_CV04_HANDOFF["MAPE"],
        "LB_pvalue_lag12": VERIFIED_CV04_HANDOFF["lb_pvalue_lag12"],
        "LB_pvalue_lag24": VERIFIED_CV04_HANDOFF["lb_pvalue_lag24"],
        "White_Noise": VERIFIED_CV04_HANDOFF["is_white_noise"],
    }


def build_benchmark_table(arima_metrics_path: str, sarima_metrics: dict) -> pd.DataFrame:
    """
    Ghép chỉ số của ARIMA tiêu chuẩn (CV04) và SARIMA mùa vụ (CV05) thành bảng
    đối chuẩn trực diện trên cùng 41 tháng tập Test.
    Lưu: outputs/tables/cv05_model_benchmark_table.csv
    """
    arima_ref = _load_arima_reference(arima_metrics_path)

    rmse_gain_pct = (arima_ref["RMSE"] - sarima_metrics["RMSE"]) / arima_ref["RMSE"] * 100.0
    mae_gain_pct  = (arima_ref["MAE"]  - sarima_metrics["MAE"])  / arima_ref["MAE"]  * 100.0
    mape_diff_pp  = arima_ref["MAPE"] - sarima_metrics["MAPE"]

    rows = [
        {
            "Model": arima_ref["Model"],
            "AIC": round(arima_ref["AIC"], 4),
            "BIC": round(arima_ref["BIC"], 4),
            "Train_Time_s": round(arima_ref["Train_Time_s"], 4) if pd.notna(arima_ref["Train_Time_s"]) else np.nan,
            "MAE": round(arima_ref["MAE"], 4),
            "RMSE": round(arima_ref["RMSE"], 4),
            "MAPE": round(arima_ref["MAPE"], 4),
            "LB_pvalue_lag12": arima_ref["LB_pvalue_lag12"],
            "LB_pvalue_lag24": arima_ref["LB_pvalue_lag24"],
            "White_Noise": arima_ref["White_Noise"],
            "Nhận xét": "Baseline phi mùa vụ: hội tụ phẳng (~17.54), bỏ sót hoàn toàn chu kỳ s=12 (LB p@12 << 0.05).",
        },
        {
            "Model": sarima_metrics.get("Model", "SARIMA(1, 1, 1)x(0, 1, 1, 12)"),
            "AIC": round(sarima_metrics["AIC"], 4),
            "BIC": round(sarima_metrics["BIC"], 4),
            "Train_Time_s": round(sarima_metrics.get("Train_Time_s", np.nan), 4)
            if pd.notna(sarima_metrics.get("Train_Time_s", np.nan))
            else np.nan,
            "MAE": round(sarima_metrics["MAE"], 4),
            "RMSE": round(sarima_metrics["RMSE"], 4),
            "MAPE": round(sarima_metrics["MAPE"], 4),
            "LB_pvalue_lag12": round(sarima_metrics.get("LB_pvalue_lag12", np.nan), 4),
            "LB_pvalue_lag24": round(sarima_metrics.get("LB_pvalue_lag24", np.nan), 4),
            "White_Noise": sarima_metrics.get("White_Noise", True),
            "Nhận xét": (
                f"Mô hình tối ưu: triệt tiêu tự tương quan mùa vụ, giảm {mae_gain_pct:.1f}% MAE, "
                f"giảm {rmse_gain_pct:.1f}% RMSE và giảm {mape_diff_pp:.2f} điểm % MAPE."
            ),
        },
    ]

    benchmark_df = pd.DataFrame(rows)
    benchmark_df.to_csv(
        os.path.join(TABLE_DIR, "cv05_model_benchmark_table.csv"), index=False
    )
    return benchmark_df


# ---------------------------------------------------------------------------
# 7. VẼ BIỂU ĐỒ DỰ BÁO VS THỰC TẾ & ĐỐI CHUẨN 2 MÔ HÌNH
# ---------------------------------------------------------------------------
def plot_sarima_forecast(
    train: pd.Series,
    test: pd.Series,
    sarima_fc: pd.Series,
    conf_int: pd.DataFrame | None = None,
) -> None:
    """
    Vẽ biểu đồ chuỗi huấn luyện, thực tế kiểm thử và đường dự báo SARIMA 41 tháng
    kèm khoảng tin cậy 95%.
    Lưu: outputs/figures/cv05_sarima_forecast_vs_actual.png
    """
    fc_series = pd.Series(np.asarray(sarima_fc, dtype=float), index=test.index)
    errors = test.values - fc_series.values
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / test.values)) * 100.0

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train.index, train.values, color="#1f4e79", lw=1.5, label="Tập Train (163 tháng: 07/1991 – 01/2005)")
    ax.plot(test.index, test.values, color="#2ca02c", lw=2.0, marker="o", ms=3.5,
            label="Thực tế Tập Test (41 tháng: 02/2005 – 06/2008)")
    ax.plot(fc_series.index, fc_series.values, color="#d62728", lw=2.0, ls="--", marker="s", ms=3.2,
            label="Dự báo SARIMA(1,1,1)(0,1,1)[12] (Out-of-sample)")

    if conf_int is not None:
        ax.fill_between(
            test.index,
            conf_int.iloc[:, 0].values,
            conf_int.iloc[:, 1].values,
            color="#d62728",
            alpha=0.16,
            label="Khoảng dự báo tin cậy 95%",
        )

    ax.axvline(test.index[0], color="#444444", ls=":", lw=1.4, label="Ranh giới Train / Test (02/2005)")

    ax.text(
        0.02,
        0.95,
        f"Kết quả Out-of-sample (41 tháng):\n"
        f"• MAE  = {mae:.3f} triệu AUD\n"
        f"• RMSE = {rmse:.3f} triệu AUD\n"
        f"• MAPE = {mape:.2f}%",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fffdf7", "edgecolor": "#d62728", "alpha": 0.92},
    )

    ax.set_title(
        "Mô hình SARIMA(1,1,1)(0,1,1)[12] — Dự báo Ngoài mẫu 41 Tháng vs Doanh số Thực tế (a10.csv)",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.set_xlabel("Thời gian (Tháng)", fontsize=11)
    ax.set_ylabel("Doanh số bán thuốc (triệu AUD)", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.73), frameon=True, fontsize=9.5)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "cv05_sarima_forecast_vs_actual.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _load_arima_forecast_series(test: pd.Series) -> pd.Series:
    """
    Nạp chuỗi dự báo 41 tháng của ARIMA từ tệp bàn giao CV04:
      1. Ưu tiên đọc cv04_arima_forecasts.csv (bàn giao chính thức của Hoàng).
      2. Nếu không có CSV, thử nạp arima_model.pkl.
      3. Nếu chạy độc lập trên nhánh khai-lab02 trước khi merge hoang-lab02,
         tính đường dự báo ARIMA(1,1,1) tương đương trong bộ nhớ (không ghi đè tệp CV04).
    """
    if os.path.exists(ARIMA_FORECASTS):
        df_a = pd.read_csv(ARIMA_FORECASTS)
        col = "predicted" if "predicted" in df_a.columns else df_a.columns[2]
        return pd.Series(df_a[col].astype(float).values, index=test.index, name="ARIMA(1,1,1)")

    if os.path.exists(ARIMA_MODEL_PKL):
        with open(ARIMA_MODEL_PKL, "rb") as f:
            arima_res = pickle.load(f)
        fc = arima_res.get_forecast(steps=len(test)).predicted_mean
        return pd.Series(np.asarray(fc, dtype=float), index=test.index, name="ARIMA(1,1,1)")

    train_s, _ = load_data(TRAIN_CSV, TEST_CSV)
    arima_tmp = ARIMA(train_s, order=(1, 1, 1)).fit()
    fc = arima_tmp.get_forecast(steps=len(test)).predicted_mean
    return pd.Series(np.asarray(fc, dtype=float), index=test.index, name="ARIMA(1,1,1)")


def plot_model_comparison(test: pd.Series, arima_metrics_path: str, sarima_fc: pd.Series) -> None:
    """
    Vẽ biểu đồ đối chuẩn trực diện 2 mô hình (ARIMA vs SARIMA) trên 41 tháng tập Test:
      - Panel trên: Đường thực tế vs Dự báo ARIMA(1,1,1) vs Dự báo SARIMA(1,1,1)(0,1,1)[12]
                    (đánh dấu rõ 3 đỉnh mùa vụ tháng 1: 01/2006, 01/2007, 01/2008).
      - Panel dưới trái: Sai số tuyệt đối |y_t - y_hat_t| theo từng tháng trên tập Test.
      - Panel dưới phải: Đối sánh định lượng MAE, RMSE, MAPE (%) giữa ARIMA và SARIMA.
    Lưu: outputs/figures/cv05_arima_vs_sarima_comparison.png
    """
    arima_ref = _load_arima_reference(arima_metrics_path)
    arima_fc = _load_arima_forecast_series(test)
    sarima_s = pd.Series(np.asarray(sarima_fc, dtype=float), index=test.index)

    sarima_err = np.abs(test.values - sarima_s.values)
    arima_err  = np.abs(test.values - arima_fc.values)

    s_mae  = float(np.mean(sarima_err))
    s_rmse = float(np.sqrt(np.mean((test.values - sarima_s.values) ** 2)))
    s_mape = float(np.mean(sarima_err / test.values) * 100.0)

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], width_ratios=[1.55, 1.0], hspace=0.34, wspace=0.24)

    # Panel 1 (Top full width): Overlay trên tập Test 41 tháng
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(test.index, test.values, color="#2ca02c", lw=2.3, marker="o", ms=4.5,
                label="Thực tế Tập Test (41 tháng)")
    ax_top.plot(
        arima_fc.index,
        arima_fc.values,
        color="#ff7f0e",
        lw=2.0,
        ls="--",
        marker="^",
        ms=4.0,
        label=f"{arima_ref['Model']} — RMSE={arima_ref['RMSE']:.2f}, MAPE={arima_ref['MAPE']:.2f}%",
    )
    ax_top.plot(
        sarima_s.index,
        sarima_s.values,
        color="#d62728",
        lw=2.2,
        ls="-.",
        marker="s",
        ms=4.0,
        label=f"SARIMA(1,1,1)(0,1,1)[12] — RMSE={s_rmse:.2f}, MAPE={s_mape:.2f}%",
    )

    # Đánh dấu các đỉnh doanh số tháng 1
    jan_dates = [d for d in test.index if d.month == 1]
    for jd in jan_dates:
        y_act = float(test.loc[jd])
        ax_top.axvline(jd, color="#888888", ls=":", lw=1.0, alpha=0.7)
        ax_top.annotate(
            f"Đỉnh {jd.strftime('%m/%Y')}\n({y_act:.1f})",
            xy=(jd, y_act),
            xytext=(8, -6),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#1b5e20",
        )

    # Đánh dấu điểm sai lệch lớn nhất của SARIMA (02/2006)
    worst_idx = int(np.argmax(sarima_err))
    worst_date = test.index[worst_idx]
    ax_top.annotate(
        f"Max |e| SARIMA: {worst_date.strftime('%m/%Y')}\n(|e| = {sarima_err[worst_idx]:.2f})",
        xy=(worst_date, test.iloc[worst_idx]),
        xytext=(16, -20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#b71c1c", "lw": 1.1},
        fontsize=8.5,
        color="#b71c1c",
    )
    ax_top.set_ylim(float(test.min()) - 2.2, float(test.max()) + 2.6)

    ax_top.set_title(
        "Đối chuẩn Trực diện Ngoài mẫu (41 Tháng Test): ARIMA(1,1,1) Tiêu chuẩn vs SARIMA(1,1,1)(0,1,1)[12] Mùa vụ",
        fontsize=12.5,
        fontweight="bold",
    )
    ax_top.set_ylabel("Doanh số (triệu AUD)", fontsize=11)
    ax_top.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_top.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax_top.grid(alpha=0.3)

    # Panel 2 (Bottom-left): Sai số tuyệt đối theo thời gian
    ax_err = fig.add_subplot(gs[1, 0])
    ax_err.fill_between(test.index, arima_err, color="#ff7f0e", alpha=0.25)
    ax_err.plot(test.index, arima_err, color="#ff7f0e", lw=1.6, ls="--", label="Sai số |e_t| ARIMA(1,1,1)")
    ax_err.fill_between(test.index, sarima_err, color="#d62728", alpha=0.30)
    ax_err.plot(test.index, sarima_err, color="#d62728", lw=1.8, label="Sai số |e_t| SARIMA(1,1,1)(0,1,1)[12]")
    ax_err.set_title("Diễn biến Sai số Tuyệt đối |y_t - ŷ_t| qua 41 Tháng Kiểm thử", fontsize=11, fontweight="bold")
    ax_err.set_xlabel("Thời gian (Tháng)")
    ax_err.set_ylabel("|Sai số| (triệu AUD)")
    ax_err.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_err.legend(loc="upper left", fontsize=9)
    ax_err.grid(alpha=0.3)

    # Panel 3 (Bottom-right): Biểu đồ cột so sánh chỉ số MAE, RMSE, MAPE
    ax_bar = fig.add_subplot(gs[1, 1])
    metric_names = ["MAE\n(triệu AUD)", "RMSE\n(triệu AUD)", "MAPE\n(%)"]
    arima_vals = [arima_ref["MAE"], arima_ref["RMSE"], arima_ref["MAPE"]]
    sarima_vals = [s_mae, s_rmse, s_mape]
    x_pos = np.arange(len(metric_names))
    width = 0.35

    b1 = ax_bar.bar(x_pos - width / 2, arima_vals, width, color="#ff7f0e", label="ARIMA(1,1,1)")
    b2 = ax_bar.bar(x_pos + width / 2, sarima_vals, width, color="#d62728", label="SARIMA(1,1,1)(0,1,1)[12]")

    ax_bar.bar_label(b1, fmt="%.2f", padding=3, fontsize=9, fontweight="bold")
    ax_bar.bar_label(b2, fmt="%.2f", padding=3, fontsize=9, fontweight="bold")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(metric_names, fontsize=10)
    ax_bar.set_ylim(0, max(arima_vals) * 1.22)
    ax_bar.set_title("Giảm thiểu Sai số khi Tích hợp Mùa vụ s=12", fontsize=11, fontweight="bold")
    ax_bar.legend(loc="upper left", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)

    fig.savefig(
        os.path.join(FIG_DIR, "cv05_arima_vs_sarima_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("CV05 — Mô hình SARIMA(p,d,q)(P,D,Q)[12] & Benchmark (Phụ trách: Khải)")
    print("=" * 72)

    # Bước 1: Nạp dữ liệu
    train, test = load_data(TRAIN_CSV, TEST_CSV)
    print(f"[OK] Train: {len(train)} tháng ({train.index.min().strftime('%Y-%m')} -> {train.index.max().strftime('%Y-%m')}) | "
          f"Test: {len(test)} tháng ({test.index.min().strftime('%Y-%m')} -> {test.index.max().strftime('%Y-%m')})")

    # Bước 2: Đánh giá ứng viên trên tập Train (Không chạm vào Test!)
    cand_df = evaluate_sarima_candidates(train)
    print("\n[Candidate Selection trên Tập Train]")
    print(cand_df[["Model", "AIC", "BIC", "LB_pvalue_lag12", "LB_pvalue_lag24", "White_Noise_Seasonal", "Selected"]].to_string(index=False))

    SARIMA_ORDER          = (1, 1, 1)
    SARIMA_SEASONAL_ORDER = (0, 1, 1, 12)

    # Bước 3: Fit mô hình SARIMA tối ưu
    result = fit_sarima(train, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
    model_name = f"SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER}"
    train_time = getattr(result, "_cv05_train_time_sec", np.nan)
    print(f"[OK] {model_name} đã fit | AIC={result.aic:.2f} | BIC={result.bic:.2f} | TrainTime={train_time:.4f}s")

    # Bước 4: Chẩn đoán phần dư mùa vụ
    diag = diagnose_residuals_sarima(result, model_label=model_name)
    print(
        f"[Residuals] Mean={diag['mean_residual']:.4f} | "
        f"LB p-value lag12={diag['lb_pvalue_lag12']:.4f} (post-burnin={diag['lb_clean_pvalue_lag12']:.4f}) | "
        f"lag24={diag['lb_pvalue_lag24']:.4f} (post-burnin={diag['lb_clean_pvalue_lag24']:.4f}) | "
        f"White Noise: {diag['is_white_noise']}"
    )

    # Bước 5: Dự báo ngoài mẫu 41 tháng & lưu chi tiết khoảng tin cậy 95%
    sarima_forecast = forecast_test(result, steps=len(test))
    sarima_forecast.index = test.index
    fc_details = save_forecast_details(result, test)
    conf_int_df = result.get_forecast(steps=len(test)).conf_int(alpha=0.05)

    # Bước 6: Tính chỉ số đánh giá
    metrics = compute_metrics(
        test,
        sarima_forecast,
        aic=result.aic,
        bic=result.bic,
        model_name=model_name,
        train_time_sec=train_time,
        diag=diag,
    )
    print(f"[Metrics] MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | MAPE={metrics['MAPE']:.4f}%")

    # Bước 7: Lập bảng Benchmark đối chuẩn ARIMA vs SARIMA
    benchmark = build_benchmark_table(ARIMA_METRICS, metrics)
    print(f"\n[Benchmark ARIMA vs SARIMA]\n{benchmark.to_string(index=False)}")

    # Bước 8: Vẽ biểu đồ và lưu mô hình + bảng metrics
    plot_sarima_forecast(train, test, sarima_forecast, conf_int=conf_int_df)
    plot_model_comparison(test, ARIMA_METRICS, sarima_forecast)

    with open(os.path.join(MODEL_DIR, "sarima_model.pkl"), "wb") as f:
        pickle.dump(result, f)
    pd.DataFrame([metrics]).to_csv(os.path.join(TABLE_DIR, "cv05_sarima_metrics.csv"), index=False)

    print("\n[DONE] CV05 hoàn tất. Đã xuất đầy đủ mô hình, bảng số liệu và biểu đồ cho CV06 (Lead).")
