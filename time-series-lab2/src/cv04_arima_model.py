"""CV04 — ARIMA trên a10.csv, chẩn đoán phần dư và dự báo 41 tháng."""

from pathlib import Path
from time import perf_counter
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"
TRAIN_CSV = PROC_DIR / "train.csv"
TEST_CSV = PROC_DIR / "test.csv"
DEFAULT_ORDERS = [(1, 1, 1), (2, 1, 1)]
ALPHA = 0.05


def read_monthly(path):
    series = pd.read_csv(path, index_col="date", parse_dates=True)["value"]
    series = pd.to_numeric(series, errors="raise")
    expected = pd.date_range(series.index[0], periods=len(series), freq="MS")
    if not series.index.equals(expected) or not np.isfinite(series).all():
        raise ValueError(f"{path}: cần dữ liệu hữu hạn, theo tháng liên tục và đúng thứ tự.")
    return series.asfreq("MS")


def load_data(train_path=TRAIN_CSV, test_path=TEST_CSV):
    """Ưu tiên dữ liệu CV01, tự chia a10.csv khi chạy nhánh độc lập."""
    if Path(train_path).exists() or Path(test_path).exists():
        train, test = read_monthly(train_path), read_monthly(test_path)
    else:
        if (Path(train_path), Path(test_path)) != (TRAIN_CSV, TEST_CSV):
            raise FileNotFoundError(f"Thiếu {train_path} và {test_path}")
        raw = read_monthly(ROOT / "data" / "raw" / "a10.csv")
        split = int(len(raw) * 0.8)
        train, test = raw.iloc[:split], raw.iloc[split:]
    if (len(train), len(test)) != (163, 41):
        raise ValueError("CV04 cần đúng 163 tháng Train và 41 tháng Test.")
    if test.index[0] != train.index[-1] + pd.offsets.MonthBegin():
        raise ValueError("Tập Test phải bắt đầu ngay sau tập Train.")
    return train, test


def load_candidate_orders(path):
    """Lấy tối đa hai cấu hình theo thứ tự ưu tiên trong bảng CV03."""
    if not path.exists():
        return DEFAULT_ORDERS, "fallback_cv04"
    candidates = pd.read_csv(path)[["p", "d", "q"]].apply(pd.to_numeric)
    if candidates.empty or (candidates < 0).any().any() or not (
        candidates % 1 == 0
    ).all().all():
        raise ValueError(f"{path}: p, d và q phải là số nguyên không âm.")
    top = candidates.drop_duplicates().head(2).astype(int)
    return list(top.itertuples(index=False, name=None)), "cv03_candidate_orders"


def fit_arima(train, order):
    return ARIMA(train, order=order).fit()


def select_model(train, orders, source):
    rows = []
    eligible = []
    for order in orders:
        start = perf_counter()
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            result = fit_arima(train, order)
        duration = perf_counter() - start
        converged = bool(result.mle_retvals.get("converged", False)) and not any(
            issubclass(message.category, ConvergenceWarning) for message in messages
        )
        row = {
            "model": f"ARIMA{order}",
            "p": order[0], "d": order[1], "q": order[2],
            "AIC": result.aic, "BIC": result.bic,
            "train_time_sec": duration, "converged": converged,
            "candidate_source": source,
            "fit_warnings": " | ".join(str(message.message) for message in messages),
        }
        rows.append(row)
        if converged and np.isfinite([result.aic, result.bic]).all():
            eligible.append((result, row))
    if not eligible:
        raise RuntimeError("Không có ứng viên ARIMA hội tụ với AIC và BIC hữu hạn.")
    result, selected = min(eligible, key=lambda pair: (pair[1]["AIC"], pair[1]["BIC"]))
    for row in rows:
        row["selected"] = row is selected
    return result, selected, pd.DataFrame(rows)


def diagnose_residuals(result, model_label="ARIMA"):
    burn = result.loglikelihood_burn
    residuals = result.resid.iloc[burn:].dropna()
    p, _, q = result.model.order
    lags = sorted({max(10, p + q + 1), 12, 24} & set(range(p + q + 1, len(residuals))))
    lb = acorr_ljungbox(residuals, lags=lags, model_df=p + q, return_df=True)
    stats = lb.rename_axis("lag").reset_index().rename(
        columns={"lb_stat": "ljungbox_stat", "lb_pvalue": "ljungbox_pvalue"}
    )
    stats["model"] = model_label
    stats["mean_residual"] = residuals.mean()
    stats["std_residual"] = residuals.std(ddof=1)
    stats["n_residuals"] = len(residuals)
    stats["initial_residuals_excluded"] = burn
    stats["model_df"] = p + q
    stats["alpha"] = ALPHA
    stats["reject_no_autocorrelation"] = stats["ljungbox_pvalue"] <= ALPHA

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    axes[0, 0].plot(residuals, color="#155e75", linewidth=1)
    axes[0, 0].axhline(0, color="#64748b", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Phần dư theo thời gian")
    axes[0, 1].hist(residuals, bins=18, color="#0e7490", edgecolor="white")
    axes[0, 1].set_title("Phân phối phần dư")
    qqplot(residuals, fit=True, line="45", ax=axes[1, 0])
    axes[1, 0].set_title("Q–Q so với phân phối chuẩn")
    plot_acf(residuals, lags=min(36, len(residuals) - 1), zero=False, ax=axes[1, 1])
    for lag in (12, 24):
        axes[1, 1].axvline(lag, color="#ea580c", alpha=0.45, linestyle="--")
    axes[1, 1].set_title("ACF phần dư")
    fig.suptitle(f"CV04 | Chẩn đoán phần dư {model_label}", fontsize=15)
    fig.savefig(FIG_DIR / "cv04_arima_residual_diag.png", dpi=150)
    plt.close(fig)
    return stats


def forecast_test(result, steps=41):
    return result.get_forecast(steps=steps)


def compute_metrics(actual, predicted):
    if not actual.index.equals(predicted.index):
        raise ValueError("Mốc tháng dự báo phải khớp với tập Test.")
    if (actual == 0).any():
        raise ValueError("MAPE không xác định khi giá trị thực bằng 0.")
    errors = actual - predicted
    return {
        "MAE": errors.abs().mean(),
        "RMSE": np.sqrt(errors.pow(2).mean()),
        "MAPE": errors.div(actual).abs().mean() * 100,
    }


def plot_forecast(train, test, forecasts, model_label):
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    ax.plot(train.iloc[-48:], color="#334155", label="Thực tế Train (48 tháng cuối)")
    ax.plot(test, color="#047857", label="Thực tế Test", linewidth=1.8)
    ax.plot(forecasts["predicted"], "--", color="#c2410c", label=model_label, linewidth=2)
    ax.fill_between(
        forecasts.index, forecasts["lower_95"], forecasts["upper_95"],
        alpha=0.15, color="#c2410c", label="Khoảng dự báo 95%",
    )
    ax.axvline(test.index[0], color="#64748b", linewidth=1, linestyle=":")
    ax.set(title="CV04 | Dự báo ngoài mẫu 41 tháng", xlabel="Tháng", ylabel="Doanh số a10")
    ax.grid(axis="y", color="#e2e8f0")
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(FIG_DIR / "cv04_arima_forecast_vs_actual.png", dpi=150)
    plt.close(fig)


def main():
    train, test = load_data()
    candidates, source = load_candidate_orders(TABLE_DIR / "cv03_candidate_orders.csv")
    result, selected, comparison = select_model(train, candidates, source)
    for directory in (TABLE_DIR, FIG_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    diagnostics = diagnose_residuals(result, selected["model"])
    prediction = forecast_test(result, steps=len(test))
    forecast = prediction.predicted_mean
    scores = compute_metrics(test, forecast)
    interval = prediction.conf_int(alpha=ALPHA)
    forecasts = pd.DataFrame({
        "actual": test, "predicted": forecast,
        "lower_95": interval.iloc[:, 0], "upper_95": interval.iloc[:, 1],
        "error": test - forecast, "absolute_error": (test - forecast).abs(),
    })
    metrics = pd.DataFrame([{
        **selected, **scores, "train_n": len(train), "test_n": len(test),
        "test_start": test.index[0].date(), "test_end": test.index[-1].date(),
    }])
    plot_forecast(train, test, forecasts, selected["model"])
    result.save(MODEL_DIR / "arima_model.pkl")
    metrics.to_csv(TABLE_DIR / "cv04_arima_metrics.csv", index=False)
    comparison.to_csv(TABLE_DIR / "cv04_arima_candidate_comparison.csv", index=False)
    diagnostics.to_csv(TABLE_DIR / "cv04_residual_stats.csv", index=False)
    forecasts.to_csv(TABLE_DIR / "cv04_arima_forecasts.csv", index_label="date")
    print(metrics[["model", "AIC", "BIC", "MAE", "RMSE", "MAPE"]].to_string(index=False))
    return metrics, comparison, diagnostics, forecasts


if __name__ == "__main__":
    main()
