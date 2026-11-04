# ============================================================
# AI_AGENT_INSTRUCTIONS
# ============================================================
# THÀNH VIÊN  : CV05 – Hiệp
# DEADLINE    : 12/11
# NHIỆM VỤ   : Walk-forward Backtesting và so sánh 3 mô hình
#
# INPUT CONTRACT:
#   - data/processed/train.csv, val.csv, test.csv   (scaled, DatetimeIndex)
#   - data/processed/scaler.pkl
#   - outputs/models/var_model.pkl
#   - outputs/models/xgb_model.json
#   - outputs/models/lstm_model_best.h5
#   - outputs/tables/var_metrics.csv, xgb_metrics.csv, lstm_metrics.csv
#
# OUTPUT CONTRACT:
#   - outputs/tables/comparison_table.csv           (bảng so sánh 3 mô hình)
#   - outputs/figures/comparison_overlay.png        (4 đường: Actual, VAR, XGBoost, LSTM)
#   - outputs/figures/error_heatmap_by_hour.png     (heatmap sai số theo giờ)
#   - outputs/figures/backtesting_walkforward.png   (tùy chọn: kết quả từng fold)
#
# LƯU Ý QUAN TRỌNG:
#   - Walk-forward: KHÔNG nhìn trước – history chỉ gồm dữ liệu đã xảy ra
#   - Đo thời gian train (TrainTime) và inference (InferTime) cho mỗi mô hình
#   - Đánh giá theo từng giờ trong ngày để phát hiện pattern lỗi
# ============================================================

import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# statsmodels
from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper

# XGBoost
from xgboost import XGBRegressor

# TensorFlow / Keras
import tensorflow as tf

# ----------------------------------------------------------
# Định nghĩa đường dẫn
# ----------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR    = PROJECT_ROOT / "outputs" / "tables"

for d in [FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FEATURES   = ["load", "temp", "humidity", "wind_speed"]
TARGET     = "load"
TARGET_IDX = FEATURES.index(TARGET)
LOOKBACK   = 24   # phải khớp với cv04
LAG_XGB    = 24   # phải khớp với cv03


# ----------------------------------------------------------
# Helpers chung
# ----------------------------------------------------------
def _inv_load(arr_1d: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse transform cho mảng 1D (chỉ cột load).
    Tạo dummy array 4 cột, đặt giá trị vào cột TARGET_IDX, inverse, lấy lại.
    """
    dummy = np.zeros((len(arr_1d), len(FEATURES)))
    dummy[:, TARGET_IDX] = arr_1d
    return scaler.inverse_transform(dummy)[:, TARGET_IDX]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính RMSE, MAE, MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE": round(mape, 4)}


def _build_xgb_features_single(history_df: pd.DataFrame, lag: int = 24) -> pd.Series:
    """
    Tạo feature vector cho XGBoost từ lịch sử gần nhất (dùng trong walk-forward).

    Cần ít nhất `lag+25` hàng trong history_df.

    Parameters
    ----------
    history_df : pd.DataFrame  (scaled, 4 cột FEATURES)
    lag        : int

    Returns
    -------
    feat_series : pd.Series  (vector feature 1 bước)
    """
    feat = {}

    # Lag features
    for col in FEATURES:
        for k in range(1, lag + 1):
            feat[f"{col}_lag{k}"] = history_df[col].iloc[-k]

    # Rolling features (dùng shift=1 → tính trên lịch sử thực)
    load_hist = history_df[TARGET]
    feat["rolling_mean_6h"]  = load_hist.iloc[-7:-1].mean()   # 6 giá trị trước t
    feat["rolling_std_24h"]  = load_hist.iloc[-25:-1].std()   # 24 giá trị trước t

    # Trend
    if len(history_df) >= 26:
        feat["load_trend"] = float(history_df[TARGET].iloc[-1]) - float(history_df[TARGET].iloc[-25])
    else:
        feat["load_trend"] = 0.0

    return pd.Series(feat)


# ----------------------------------------------------------
# Walk-forward VAR
# ----------------------------------------------------------
def walk_forward_var(fitted_var: VARResultsWrapper,
                     df_train_stat: pd.DataFrame,
                     df_test: pd.DataFrame,
                     scaler,
                     differenced: bool,
                     df_train_original: pd.DataFrame = None):
    """
    Walk-forward one-step-ahead forecasting với VAR.

    Tại mỗi bước t:
      1. Dùng p bước cuối của history (dạng stationary)
      2. Dự báo 1 bước
      3. Cập nhật history với giá trị THỰC tế (không phải dự báo)

    Parameters
    ----------
    fitted_var      : VARResultsWrapper
    df_train_stat   : pd.DataFrame  (train đã stationary, làm history ban đầu)
    df_test         : pd.DataFrame  (test chưa sai phân, đã scale)
    scaler          : StandardScaler
    differenced     : bool
    df_train_original : pd.DataFrame  (cần nếu differenced=True)

    Returns
    -------
    y_true, y_pred : np.ndarray  (original scale)
    infer_time     : float  (tổng thời gian inference, giây)
    """
    print("[walk_forward_var] Đang chạy...")

    p = fitted_var.k_ar
    history_stat = df_train_stat.values[-p:].copy()   # shape (p, 4)

    y_pred_stat = []
    t0 = time.time()

    for i in range(len(df_test)):
        # Dự báo 1 bước
        forecast = fitted_var.forecast(y=history_stat, steps=1)  # (1, 4)
        y_pred_stat.append(forecast[0])

        # Cập nhật history với giá trị THỰC tế (teacher forcing)
        if differenced:
            # TODO (CV05): Tính giá trị stationary thực tế của bước i
            # actual_stat = df_test_stat.iloc[i].values  ← cần df_test_stat
            actual_stat = np.zeros(len(FEATURES))   # placeholder
        else:
            actual_stat = df_test.iloc[i].values

        history_stat = np.vstack([history_stat[1:], actual_stat])

    infer_time = time.time() - t0
    y_pred_stat = np.array(y_pred_stat)   # (N, 4)

    # TODO (CV05): Inverse differencing nếu differenced=True
    if differenced:
        # last_val = df_train_original.values[-1]
        # y_pred_scaled = np.cumsum(y_pred_stat, axis=0) + last_val
        pass
    else:
        y_pred_scaled = y_pred_stat

    # Inverse transform → original scale, chỉ lấy load
    y_pred = _inv_load(y_pred_scaled[:, TARGET_IDX], scaler)
    y_true = _inv_load(df_test[TARGET].values, scaler)

    print(f"  VAR walk-forward: {len(y_pred)} bước, inference time={infer_time:.2f}s")
    return y_true, y_pred, infer_time


# ----------------------------------------------------------
# Walk-forward XGBoost
# ----------------------------------------------------------
def walk_forward_xgboost(model_xgb: XGBRegressor,
                          df_train: pd.DataFrame,
                          df_test: pd.DataFrame,
                          scaler,
                          lag: int = LAG_XGB):
    """
    Walk-forward one-step-ahead forecasting với XGBoost.

    Xây dựng feature vector từ lịch sử (train + các bước test đã xảy ra),
    predict 1 bước, cập nhật history với giá trị THỰC (teacher forcing).

    Parameters
    ----------
    model_xgb : XGBRegressor
    df_train  : pd.DataFrame  (scaled, làm history ban đầu)
    df_test   : pd.DataFrame  (scaled)
    scaler    : StandardScaler
    lag       : int

    Returns
    -------
    y_true, y_pred : np.ndarray  (original scale)
    infer_time     : float
    """
    print("[walk_forward_xgboost] Đang chạy...")

    # Khởi tạo history từ cuối train (cần ít nhất lag+25 hàng)
    min_hist = lag + 25
    history_df = df_train.iloc[-min_hist:].copy()

    y_pred_scaled = []
    t0 = time.time()

    for i in range(len(df_test)):
        # Tạo feature vector
        feat = _build_xgb_features_single(history_df, lag=lag)
        X_single = feat.values.reshape(1, -1)

        # Dự báo
        pred = model_xgb.predict(X_single)[0]
        y_pred_scaled.append(pred)

        # Cập nhật history với giá trị THỰC
        new_row = df_test.iloc[[i]].copy()
        history_df = pd.concat([history_df, new_row]).iloc[-min_hist:]

    infer_time = time.time() - t0
    y_pred_scaled = np.array(y_pred_scaled)

    y_pred = _inv_load(y_pred_scaled, scaler)
    y_true = _inv_load(df_test[TARGET].values, scaler)

    print(f"  XGBoost walk-forward: {len(y_pred)} bước, inference time={infer_time:.2f}s")
    return y_true, y_pred, infer_time


# ----------------------------------------------------------
# Walk-forward LSTM
# ----------------------------------------------------------
def walk_forward_lstm(model_lstm,
                       df_train: pd.DataFrame,
                       df_test: pd.DataFrame,
                       scaler,
                       lookback: int = LOOKBACK):
    """
    Walk-forward one-step-ahead forecasting với LSTM.

    Tại mỗi bước t:
      1. Lấy lookback bước cuối của history → tensor (1, lookback, 4)
      2. Predict → 1 giá trị load_scaled
      3. Cập nhật history với giá trị THỰC

    Parameters
    ----------
    model_lstm : keras Model
    df_train   : pd.DataFrame  (scaled, làm history ban đầu)
    df_test    : pd.DataFrame  (scaled)
    scaler     : StandardScaler
    lookback   : int

    Returns
    -------
    y_true, y_pred : np.ndarray  (original scale)
    infer_time     : float
    """
    print("[walk_forward_lstm] Đang chạy...")

    # History khởi tạo từ lookback bước cuối của train
    history = df_train[FEATURES].values[-lookback:].copy()  # (lookback, 4)

    y_pred_scaled = []
    t0 = time.time()

    for i in range(len(df_test)):
        # Tensor input: (1, lookback, n_features)
        X_input = history[-lookback:][np.newaxis, :, :]   # (1, lookback, 4)

        # Predict
        pred_scaled = model_lstm.predict(X_input, verbose=0)[0, 0]
        y_pred_scaled.append(pred_scaled)

        # Cập nhật history với giá trị THỰC (teacher forcing)
        actual_row = df_test[FEATURES].iloc[i].values   # shape (4,)
        history = np.vstack([history[1:], actual_row])  # cuộn cửa sổ

    infer_time = time.time() - t0
    y_pred_scaled = np.array(y_pred_scaled)

    y_pred = _inv_load(y_pred_scaled, scaler)
    y_true = _inv_load(df_test[TARGET].values, scaler)

    print(f"  LSTM walk-forward: {len(y_pred)} bước, inference time={infer_time:.2f}s")
    return y_true, y_pred, infer_time


# ----------------------------------------------------------
# So sánh 3 mô hình
# ----------------------------------------------------------
def compare_models(results_var: dict, results_xgb: dict, results_lstm: dict) -> pd.DataFrame:
    """
    Tạo bảng so sánh tổng hợp 3 mô hình: RMSE, MAE, MAPE, InferTime.

    Parameters
    ----------
    results_var, results_xgb, results_lstm : dict
        Mỗi dict chứa: {"y_true", "y_pred", "infer_time", "train_time"}

    Returns
    -------
    comparison_df : pd.DataFrame
    """
    print("=" * 60)
    print("[compare_models] Tạo bảng so sánh")
    print("=" * 60)

    rows = []
    for model_name, res in [("VAR", results_var), ("XGBoost", results_xgb), ("LSTM", results_lstm)]:
        m = _compute_metrics(res["y_true"], res["y_pred"])
        rows.append({
            "Model"      : model_name,
            "RMSE"       : m["RMSE"],
            "MAE"        : m["MAE"],
            "MAPE (%)"   : m["MAPE"],
            "InferTime(s)": round(res.get("infer_time", 0.0), 3),
            "TrainTime(s)": round(res.get("train_time", 0.0), 3),
        })

    comparison_df = pd.DataFrame(rows).set_index("Model")
    print(comparison_df.to_string())
    comparison_df.to_csv(TABLES_DIR / "comparison_table.csv")
    print(f"\n  Đã lưu: {TABLES_DIR / 'comparison_table.csv'}")
    return comparison_df


# ----------------------------------------------------------
# Vẽ Overlay Plot
# ----------------------------------------------------------
def plot_comparison(results_dict: dict, df_test_index) -> None:
    """
    Vẽ overlay: Actual, VAR, XGBoost, LSTM trên cùng 1 trục.

    Parameters
    ----------
    results_dict : dict  {"VAR": res_var, "XGBoost": res_xgb, "LSTM": res_lstm}
    df_test_index : DatetimeIndex  (trục X thời gian)
    """
    print("=" * 60)
    print("[plot_comparison] Vẽ overlay comparison")
    print("=" * 60)

    colors = {"Actual": "black", "VAR": "steelblue", "XGBoost": "darkorange", "LSTM": "purple"}
    linestyles = {"Actual": "-", "VAR": "--", "XGBoost": "-.", "LSTM": ":"}
    linewidths = {"Actual": 1.2, "VAR": 0.8, "XGBoost": 0.8, "LSTM": 0.8}

    fig, ax = plt.subplots(figsize=(16, 5))

    # Actual (từ VAR – cùng y_true)
    y_true = list(results_dict.values())[0]["y_true"]
    idx = df_test_index[:len(y_true)]

    ax.plot(idx, y_true, label="Actual", color=colors["Actual"],
            linewidth=linewidths["Actual"], linestyle=linestyles["Actual"])

    for model_name, res in results_dict.items():
        ax.plot(idx, res["y_pred"], label=model_name,
                color=colors[model_name],
                linewidth=linewidths[model_name],
                linestyle=linestyles[model_name])

    ax.set_title("Backtesting Walk-Forward: So sánh 3 mô hình vs Thực tế (Load)")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Load (original scale)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "comparison_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {out}")


# ----------------------------------------------------------
# Heatmap sai số theo giờ trong ngày
# ----------------------------------------------------------
def plot_error_heatmap_by_hour(results_dict: dict, df_test: pd.DataFrame) -> None:
    """
    Vẽ heatmap MAE trung bình theo giờ trong ngày (0-23) cho mỗi mô hình.

    Heatmap giúp phát hiện mô hình nào yếu vào giờ cao điểm (6-9h, 18-22h).

    Parameters
    ----------
    results_dict : dict  {"VAR": res, "XGBoost": res, "LSTM": res}
    df_test      : pd.DataFrame  (cần DatetimeIndex để lấy .hour)
    """
    print("=" * 60)
    print("[plot_error_heatmap_by_hour] Vẽ heatmap sai số theo giờ")
    print("=" * 60)

    n = len(list(results_dict.values())[0]["y_true"])
    hours = df_test.index[:n].hour   # Series giờ trong ngày

    heatmap_data = {}
    for model_name, res in results_dict.items():
        abs_err = np.abs(res["y_true"] - res["y_pred"])
        hour_df = pd.DataFrame({"hour": hours, "abs_err": abs_err})
        heatmap_data[model_name] = hour_df.groupby("hour")["abs_err"].mean()

    heatmap_df = pd.DataFrame(heatmap_data)   # shape (24, 3)

    fig, ax = plt.subplots(figsize=(10, 5))
    import seaborn as sns
    sns.heatmap(heatmap_df.T, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=range(24), yticklabels=heatmap_df.columns,
                ax=ax, cbar_kws={"label": "MAE trung bình"})
    ax.set_title("Heatmap sai số trung bình (MAE) theo giờ trong ngày")
    ax.set_xlabel("Giờ trong ngày (0–23)")
    ax.set_ylabel("Mô hình")
    fig.tight_layout()
    out = FIGURES_DIR / "error_heatmap_by_hour.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {out}")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    """
    Pipeline chính cho CV05: Walk-forward Backtesting & Comparison.
    """
    print("\n" + "=" * 60)
    print("  CV05 – BACKTESTING & COMPARISON PIPELINE")
    print("=" * 60 + "\n")

    # --- Load dữ liệu ---
    df_train = pd.read_csv(PROCESSED_DIR / "train.csv", index_col=0, parse_dates=True)
    df_val   = pd.read_csv(PROCESSED_DIR / "val.csv",   index_col=0, parse_dates=True)
    df_test  = pd.read_csv(PROCESSED_DIR / "test.csv",  index_col=0, parse_dates=True)

    with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df_train_full = pd.concat([df_train, df_val])
    print(f"  Train (full) shape : {df_train_full.shape}")
    print(f"  Test shape         : {df_test.shape}\n")

    # --- Load models ---
    print("[INFO] Load các mô hình đã train...")

    with open(MODELS_DIR / "var_model.pkl", "rb") as f:
        fitted_var = pickle.load(f)
    print("  ✓ VAR model loaded")

    model_xgb = XGBRegressor()
    model_xgb.load_model(str(MODELS_DIR / "xgb_model.json"))
    print("  ✓ XGBoost model loaded")

    model_lstm = tf.keras.models.load_model(str(MODELS_DIR / "lstm_model_best.h5"))
    print("  ✓ LSTM model loaded\n")

    # --- Load train times từ metrics (nếu đã lưu) ---
    # TODO (CV05): Lưu train time vào CSV khi train mỗi mô hình để load ở đây
    train_times = {"VAR": 0.0, "XGBoost": 0.0, "LSTM": 0.0}   # placeholder

    # --- Walk-forward backtesting ---

    # VAR (cần df_train_stationary)
    # TODO (CV05): Chạy lại ADF test trên df_train_full để có df_train_full_stat
    # Tạm thời dùng df_train_full không sai phân:
    df_train_full_stat = df_train_full.copy()
    differenced        = False

    y_true_var, y_pred_var, infer_var = walk_forward_var(
        fitted_var, df_train_full_stat, df_test,
        scaler, differenced, df_train_full
    )

    y_true_xgb, y_pred_xgb, infer_xgb = walk_forward_xgboost(
        model_xgb, df_train_full, df_test, scaler, lag=LAG_XGB
    )

    y_true_lstm, y_pred_lstm, infer_lstm = walk_forward_lstm(
        model_lstm, df_train_full, df_test, scaler, lookback=LOOKBACK
    )

    # Kết quả
    results_var  = {"y_true": y_true_var,  "y_pred": y_pred_var,
                    "infer_time": infer_var,  "train_time": train_times["VAR"]}
    results_xgb  = {"y_true": y_true_xgb,  "y_pred": y_pred_xgb,
                    "infer_time": infer_xgb,  "train_time": train_times["XGBoost"]}
    results_lstm = {"y_true": y_true_lstm, "y_pred": y_pred_lstm,
                    "infer_time": infer_lstm, "train_time": train_times["LSTM"]}

    # --- Bảng so sánh ---
    comparison_df = compare_models(results_var, results_xgb, results_lstm)

    # --- Vẽ overlay plot ---
    results_dict = {"VAR": results_var, "XGBoost": results_xgb, "LSTM": results_lstm}
    plot_comparison(results_dict, df_test.index)

    # --- Heatmap sai số theo giờ ---
    plot_error_heatmap_by_hour(results_dict, df_test)

    print("\n" + "=" * 60)
    print("  [CV05 DONE] Backtesting & Comparison hoàn tất.")
    print("=" * 60)


if __name__ == "__main__":
    main()
