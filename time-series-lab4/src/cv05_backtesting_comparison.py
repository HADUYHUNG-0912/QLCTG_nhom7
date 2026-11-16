"""CV05 — benchmark one-step-ahead trên cùng tập test cho VAR, XGBoost, LSTM."""

import pickle
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from cv03_xgboost_model import FEATURES, LAG, TARGET, step1_build_flat_features


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "outputs" / "models"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
TARGET_IDX = FEATURES.index(TARGET)
METRICS_FILES = {
    "VAR": "var_metrics.csv",
    "XGBoost": "xgb_metrics.csv",
    "LSTM": "lstm_metrics.csv",
}
COLORS = {"VAR": "#2573a7", "XGBoost": "#c36a16", "LSTM": "#884aa0"}


def load_splits():
    splits = [
        pd.read_csv(PROCESSED / f"{name}.csv", index_col=0, parse_dates=True)[FEATURES]
        for name in ("train", "val", "test")
    ]
    for name, split in zip(("train", "val", "test"), splits):
        if split.empty or not np.isfinite(split.to_numpy()).all():
            raise ValueError(f"Tập {name} rỗng hoặc có giá trị không hợp lệ.")
    index = pd.concat(splits).index
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans
            or not ((index[1:] - index[:-1]) == pd.Timedelta(hours=1)).all()):
        raise ValueError("Train, val và test phải nối tiếp theo giờ, không trùng hoặc khuyết mốc.")
    return splits


def load_lstm():
    import tensorflow as tf

    for filename in ("lstm_model.keras", "lstm_model_best.h5"):
        path = MODELS / filename
        if path.is_file():
            return tf, tf.keras.models.load_model(path, compile=False)
    raise FileNotFoundError("Thiếu mô hình LSTM. Chạy src/cv04_lstm_model.py trước.")


def var_inputs(model, history, test):
    """Nhận dạng sai phân từ dữ liệu đã fit, xác minh với train + val."""
    fitted = np.asarray(model.endog)
    if list(model.names) != FEATURES:
        raise ValueError("Thứ tự biến của VAR khác dữ liệu CV01.")
    if fitted.shape == history.shape and np.allclose(fitted, history):
        differenced = False
    elif len(fitted) == len(history) - 1 and np.allclose(fitted, np.diff(history, axis=0)):
        differenced = True
    else:
        raise ValueError("VAR không khớp train + val hiện tại. Chạy lại CV02.")

    levels = np.concatenate((history, test))
    stationary = np.diff(levels, axis=0) if differenced else levels
    order = model.k_ar
    if order < 1 or len(fitted) < order:
        raise ValueError("Bậc trễ VAR không đủ để chạy walk-forward.")
    offset = len(history) - int(differenced)
    windows = (stationary[end - order:end] for end in range(offset, offset + len(test)))
    anchors = levels[len(history) - 1:-1, TARGET_IDX] if differenced else np.zeros(len(test))
    return windows, anchors


def timed_predictions(predict_one, windows, count):
    """Warm-up bằng cửa sổ đầu, sau đó đo từng lần gọi dự báo một mẫu."""
    first = next(windows)
    predict_one(first)
    predictions = np.empty(count)
    total = 0.0
    start = perf_counter()
    predictions[0] = predict_one(first)
    total += perf_counter() - start
    for i, window in enumerate(windows, start=1):
        start = perf_counter()
        predictions[i] = predict_one(window)
        total += perf_counter() - start
    return predictions, total


def backtest(train, val, test, scaler):
    with (MODELS / "var_model.pkl").open("rb") as handle:
        var = pickle.load(handle)
    xgb = XGBRegressor()
    xgb.load_model(MODELS / "xgb_model.json")
    tf, lstm = load_lstm()

    history = pd.concat([train, val])
    n_test = len(test)
    lookback = int(lstm.input_shape[1])
    if lstm.input_shape[2] != len(FEATURES) or len(history) < max(LAG + 1, lookback):
        raise ValueError("Lịch sử hoặc kích thước đầu vào của LSTM/XGBoost không hợp lệ.")

    # Với nhãn test tại i, mọi cửa sổ kết thúc ở i-1. Giá trị thực ở i
    # chỉ được đưa vào cửa sổ cho i+1. Trọng số mô hình cố định qua test.
    var_windows, anchors = var_inputs(var, history.to_numpy(), test.to_numpy())
    xgb_context = pd.concat([history.tail(LAG + 1), test])
    xgb_features = step1_build_flat_features(xgb_context, lag=LAG).loc[test.index]
    trained_columns = xgb.get_booster().feature_names or list(xgb_features.drop(columns="load_t0"))
    xgb_matrix = xgb_features[trained_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(xgb_matrix).all() or "load_t0" in trained_columns:
        raise ValueError("Đặc trưng XGBoost không hợp lệ cho benchmark.")

    sequence_data = pd.concat([history.tail(lookback), test]).to_numpy(dtype=np.float32)
    lstm_windows = (sequence_data[i:i + lookback][None, :, :] for i in range(n_test))
    lstm_infer = tf.function(lambda window: lstm(window, training=False))

    predictors = {
        "VAR": (lambda window: var.forecast(window, steps=1)[0, TARGET_IDX], var_windows),
        "XGBoost": (lambda window: xgb.predict(window)[0], (row[None, :] for row in xgb_matrix)),
        "LSTM": (lambda window: float(lstm_infer(window).numpy()[0, 0]), lstm_windows),
    }
    actual = scaler.inverse_transform(test.to_numpy())[:, TARGET_IDX]
    forecasts = pd.DataFrame({"Thực tế": actual}, index=test.index)
    timings = {}
    for name, (predict_one, windows) in predictors.items():
        print(f"Dự báo {name}: {n_test} bước...", flush=True)
        scaled, timings[name] = timed_predictions(predict_one, windows, n_test)
        if name == "VAR":
            scaled += anchors
        forecasts[name] = scaled * scaler.scale_[TARGET_IDX] + scaler.mean_[TARGET_IDX]
    if not np.isfinite(forecasts.to_numpy()).all():
        raise ValueError("Benchmark gặp dự báo hoặc phụ tải không hữu hạn.")
    return forecasts, timings


def summarize(forecasts, timings):
    actual = forecasts["Thực tế"]
    valid_mape = actual != 0
    rows = []
    for name, filename in METRICS_FILES.items():
        path = TABLES / filename
        train_seconds = np.nan
        if path.is_file():
            metrics = pd.read_csv(path)
            if not metrics.empty and "TrainTime(s)" in metrics:
                train_seconds = metrics["TrainTime(s)"].iloc[0]
        if pd.isna(train_seconds):
            print(f"{name}: chạy lại script huấn luyện để ghi TrainTime(s).")
        error = forecasts[name] - actual
        rows.append({
            "Model": name,
            "RMSE": np.sqrt((error ** 2).mean()),
            "MAE": error.abs().mean(),
            "MAPE (%)": (error[valid_mape] / actual[valid_mape]).abs().mean() * 100,
            "TrainTime(s)": train_seconds,
            "InferTime(s)": timings[name],
            "InferTime(ms/step)": timings[name] * 1000 / len(forecasts),
            "TestSamples": len(forecasts),
        })
    return pd.DataFrame(rows).sort_values("RMSE", ignore_index=True)


def save_figure(figure, filename, legacy=None):
    path = FIGURES / filename
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    if legacy:
        copyfile(path, FIGURES / legacy)


def plot_overlay(forecasts):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), layout="constrained")
    styles = {"Thực tế": "-", "VAR": "--", "XGBoost": "-.", "LSTM": ":"}
    colors = {"Thực tế": "#20262e", **COLORS}
    for ax, frame, title in zip(
        axes, (forecasts, forecasts.head(168)), ("Toàn bộ tập test", "7 ngày đầu tập test")
    ):
        for name, color in colors.items():
            ax.plot(frame.index, frame[name], label=name, color=color,
                    linestyle=styles[name], linewidth=1.2)
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set(title=title, ylabel="Phụ tải (MW)")
        ax.grid(alpha=0.2)
    axes[0].legend(ncols=4, frameon=False)
    fig.suptitle("CV05 — Dự báo trước 1 giờ và phụ tải thực tế", fontweight="bold")
    save_figure(fig, "cv05_actual_vs_all_models.png", "comparison_overlay.png")


def plot_heatmap(forecasts):
    error = forecasts[list(COLORS)].sub(forecasts["Thực tế"], axis=0).abs()
    by_hour = error.groupby(error.index.hour).mean().reindex(range(24)).T
    fig, ax = plt.subplots(figsize=(13, 3), layout="constrained")
    image = ax.imshow(by_hour.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(24))
    ax.set_yticks(range(len(by_hour)), by_hour.index)
    ax.set(title="CV05 — Sai số tuyệt đối trung bình theo giờ", xlabel="Giờ trong ngày")
    for row, values in enumerate(by_hour.to_numpy()):
        for hour, value in enumerate(values):
            if np.isfinite(value):
                color = "white" if image.norm(value) > 0.65 else "#222222"
                ax.text(hour, row, f"{value:.0f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, label="MAE (MW)")
    save_figure(fig, "cv05_error_heatmap_by_hour.png", "error_heatmap_by_hour.png")


def plot_benchmark(comparison):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    panels = [
        ("RMSE", "RMSE (MW)", 1), ("MAE", "MAE (MW)", 1), ("MAPE (%)", "MAPE (%)", 2),
        ("TrainTime(s)", "Huấn luyện (s)", 3),
        ("InferTime(ms/step)", "Dự báo (ms/bước)", 3),
    ]
    for ax, (column, title, decimals) in zip(axes.flat, panels):
        values = comparison[column]
        bars = ax.bar(comparison["Model"], values, color=comparison["Model"].map(COLORS))
        labels = [f"{v:.{decimals}f}" if pd.notna(v) else "Chưa đo" for v in values]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
        ax.set_title(title)
        ax.margins(y=0.23)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    axes.flat[-1].axis("off")
    fig.suptitle("CV05 — Benchmark VAR, XGBoost và LSTM", fontweight="bold")
    save_figure(fig, "cv05_model_comparison_bar.png")


def main():
    train, val, test = load_splits()
    with (PROCESSED / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    forecasts, timings = backtest(train, val, test, scaler)
    comparison = summarize(forecasts, timings)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = TABLES / "cv05_final_comparison.csv"
    comparison.to_csv(table, index=False, float_format="%.6f")
    copyfile(table, TABLES / "comparison_table.csv")
    plot_overlay(forecasts)
    plot_heatmap(forecasts)
    plot_benchmark(comparison)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
