# ============================================================
# AI_AGENT_INSTRUCTIONS
# ============================================================
# THÀNH VIÊN  : CV04 – Trâm
# DEADLINE    : 12/11
# NHIỆM VỤ   : Xây dựng mô hình LSTM (Deep Learning)
#
# INPUT CONTRACT:
#   - data/processed/train.csv, val.csv, test.csv   (scaled, DatetimeIndex)
#   - data/processed/scaler.pkl
#
# OUTPUT CONTRACT:
#   - outputs/models/lstm_model.h5                  (model đã train)
#   - outputs/models/lstm_model_best.h5             (best checkpoint)
#   - outputs/tables/lstm_metrics.csv               (RMSE, MAE, MAPE trên test)
#   - outputs/figures/lstm_learning_curve.png
#   - outputs/figures/lstm_forecast_vs_actual.png
#
# LƯU Ý QUAN TRỌNG:
#   - lookback=24: mỗi sample là cửa sổ 24 bước (24 giờ) × 4 features
#   - Target: load tại bước t+1 (one-step-ahead)
#   - EarlyStopping patience=10 monitor='val_loss'
#   - Inverse transform chỉ cột load (dùng dummy array trick như XGBoost)
#   - KHÔNG shuffle khi tạo sequences (giữ thứ tự thời gian)
# ============================================================

import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Cố định seed cho tính tái lập
tf.random.set_seed(42)
np.random.seed(42)

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
LOOKBACK     = 24    # số bước lịch sử (24 giờ)
N_FEATURES   = 4     # số biến đầu vào
TARGET_IDX   = FEATURES.index(TARGET)   # = 0


# ----------------------------------------------------------
# STEP 1: Tạo tensor sequences 3D
# ----------------------------------------------------------
def step1_build_sequences(data_scaled: np.ndarray, lookback: int = 24):
    """
    Chuyển đổi mảng 2D (T, F) thành tensor 3D (samples, timesteps, features).

    Sample thứ i:
      X[i] = data_scaled[i : i+lookback, :]     → shape (lookback, n_features)
      y[i] = data_scaled[i+lookback, TARGET_IDX] → scalar (giá trị load kế tiếp)

    KHÔNG shuffle để giữ thứ tự thời gian.

    Parameters
    ----------
    data_scaled : np.ndarray  shape (T, n_features)
    lookback    : int         số bước lịch sử (mặc định 24)

    Returns
    -------
    X : np.ndarray  shape (T - lookback, lookback, n_features)
    y : np.ndarray  shape (T - lookback,)
    """
    X_list, y_list = [], []

    for i in range(len(data_scaled) - lookback):
        X_list.append(data_scaled[i : i + lookback, :])          # cửa sổ lookback bước
        y_list.append(data_scaled[i + lookback, TARGET_IDX])     # load bước tiếp theo

    X = np.array(X_list, dtype=np.float32)   # (N, lookback, n_features)
    y = np.array(y_list,  dtype=np.float32)  # (N,)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    return X, y


# ----------------------------------------------------------
# STEP 2: Xây dựng kiến trúc LSTM
# ----------------------------------------------------------
def step2_build_lstm(lookback: int = 24, n_features: int = 4) -> Sequential:
    """
    Định nghĩa và compile mô hình LSTM hai lớp.

    Kiến trúc:
      Input         → shape (lookback, n_features)
      LSTM(64)      → return_sequences=True
      Dropout(0.2)
      LSTM(32)      → return_sequences=False
      Dropout(0.2)
      Dense(1)      → dự báo load tại t+1

    Optimizer : Adam(lr=0.001)
    Loss      : MSE
    Metrics   : MAE

    Parameters
    ----------
    lookback   : int
    n_features : int

    Returns
    -------
    model : Sequential (chưa fit)
    """
    print("=" * 60)
    print("[STEP 2] Xây dựng kiến trúc LSTM")
    print("=" * 60)

    model = Sequential(name="LSTM_Forecast", layers=[
        LSTM(64, return_sequences=True,
             input_shape=(lookback, n_features),
             name="lstm_layer_1"),
        Dropout(0.2, name="dropout_1"),
        LSTM(32, return_sequences=False,
             name="lstm_layer_2"),
        Dropout(0.2, name="dropout_2"),
        Dense(1, name="output_layer"),
    ])

    model.compile(
        optimizer = Adam(learning_rate=0.001),
        loss      = "mse",
        metrics   = ["mae"],
    )

    model.summary()
    print("[STEP 2 DONE]\n")
    return model


# ----------------------------------------------------------
# STEP 3: Huấn luyện LSTM
# ----------------------------------------------------------
def step3_train(model: Sequential,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val:   np.ndarray, y_val:   np.ndarray):
    """
    Huấn luyện LSTM với EarlyStopping và ModelCheckpoint.

    Callbacks:
      - EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
      - ModelCheckpoint: lưu best model vào outputs/models/lstm_model_best.h5

    Parameters
    ----------
    model   : Sequential  (chưa fit)
    X_train, y_train : np.ndarray
    X_val,   y_val   : np.ndarray

    Returns
    -------
    model   : Sequential  (đã fit, best weights)
    history : History object  (lưu loss theo epoch)
    """
    print("=" * 60)
    print("[STEP 3] Huấn luyện LSTM")
    print("=" * 60)

    best_model_path = str(MODELS_DIR / "lstm_model_best.h5")

    callbacks = [
        EarlyStopping(
            monitor           = "val_loss",
            patience          = 10,
            restore_best_weights = True,
            verbose           = 1,
        ),
        ModelCheckpoint(
            filepath          = best_model_path,
            monitor           = "val_loss",
            save_best_only    = True,
            verbose           = 1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = 100,    # EarlyStopping sẽ dừng sớm nếu không cải thiện
        batch_size      = 64,
        callbacks       = callbacks,
        shuffle         = False,  # QUAN TRỌNG: KHÔNG shuffle với time-series
        verbose         = 1,
    )

    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_val   = min(history.history["val_loss"])
    print(f"\n  Best epoch    : {best_epoch}")
    print(f"  Best val_loss : {best_val:.6f}")

    # --- Vẽ Learning Curve ---
    step5_plot_learning_curve(history)

    print(f"  Model best đã lưu tại: {best_model_path}")
    print("[STEP 3 DONE]\n")
    return model, history


# ----------------------------------------------------------
# STEP 4: Đánh giá trên tập test
# ----------------------------------------------------------
def step4_evaluate(model: Sequential,
                   X_test: np.ndarray, y_test: np.ndarray,
                   scaler) -> dict:
    """
    Dự báo trên tập test, inverse transform về original scale, tính metrics.

    Trick inverse transform:
      Vì scaler fit trên 4 biến, tạo dummy array (N, 4),
      đặt cột load vào vị trí TARGET_IDX, inverse_transform, lấy cột đó.

    Parameters
    ----------
    model   : Sequential
    X_test  : np.ndarray  shape (N, lookback, n_features)
    y_test  : np.ndarray  shape (N,)  – giá trị scaled
    scaler  : StandardScaler

    Returns
    -------
    metrics : dict  {"RMSE": ..., "MAE": ..., "MAPE": ...}
    """
    print("=" * 60)
    print("[STEP 4] Evaluate LSTM – Tập Test")
    print("=" * 60)

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()   # shape (N,)

    # --- Inverse transform ---
    def _inv(arr_1d):
        dummy = np.zeros((len(arr_1d), len(FEATURES)))
        dummy[:, TARGET_IDX] = arr_1d
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]

    y_true = _inv(y_test)
    y_pred = _inv(y_pred_scaled)

    # --- Metrics ---
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE": round(mape, 4)}
    print(f"  RMSE = {metrics['RMSE']}")
    print(f"  MAE  = {metrics['MAE']}")
    print(f"  MAPE = {metrics['MAPE']} %")

    # --- Vẽ Forecast vs Actual ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true, label="Actual Load",      color="steelblue", linewidth=1.0)
    ax.plot(y_pred, label="LSTM Forecast",    color="purple",    linewidth=0.8, linestyle="--")
    ax.set_title("LSTM: Dự báo vs Thực tế – Tập Test")
    ax.set_xlabel("Bước thời gian")
    ax.set_ylabel("Load (original scale)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lstm_forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {FIGURES_DIR / 'lstm_forecast_vs_actual.png'}")

    print("[STEP 4 DONE]\n")
    return metrics, y_true, y_pred


# ----------------------------------------------------------
# STEP 5: Vẽ Learning Curve
# ----------------------------------------------------------
def step5_plot_learning_curve(history) -> None:
    """
    Vẽ train loss vs val loss theo epoch để phát hiện overfitting.

    Dấu hiệu overfitting: val_loss tăng trong khi train_loss vẫn giảm.

    Parameters
    ----------
    history : keras History object  (history.history dict)
    """
    train_loss = history.history["loss"]
    val_loss   = history.history["val_loss"]
    epochs     = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # --- Plot MSE Loss ---
    axes[0].plot(epochs, train_loss, label="Train Loss (MSE)", color="steelblue")
    axes[0].plot(epochs, val_loss,   label="Val Loss (MSE)",   color="tomato")
    best_epoch = int(np.argmin(val_loss)) + 1
    axes[0].axvline(x=best_epoch, color="green", linestyle="--",
                    label=f"Best epoch={best_epoch}")
    axes[0].set_title("LSTM Learning Curve – MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # --- Plot MAE ---
    if "mae" in history.history:
        train_mae = history.history["mae"]
        val_mae   = history.history["val_mae"]
        axes[1].plot(epochs, train_mae, label="Train MAE", color="steelblue")
        axes[1].plot(epochs, val_mae,   label="Val MAE",   color="tomato")
        axes[1].set_title("LSTM Learning Curve – MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE (scaled)")
        axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "lstm_learning_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu learning curve: {out}")

    # --- Phân tích overfitting ---
    gap = np.array(val_loss) - np.array(train_loss)
    max_gap_epoch = int(np.argmax(gap)) + 1
    print(f"  Khoảng cách val-train loss lớn nhất tại epoch {max_gap_epoch} "
          f"(gap={gap.max():.6f})")
    if gap[-1] > gap[0] * 2:
        print("  [CẢNH BÁO] Có dấu hiệu OVERFITTING rõ – xem xét tăng Dropout hoặc giảm model.")
    else:
        print("  [OK] Không có dấu hiệu overfitting nghiêm trọng.")


# ----------------------------------------------------------
# STEP 6: Câu hỏi lý luận về LSTM
# ----------------------------------------------------------
def step6_answer_questions() -> dict:
    """
    Skeleton câu hỏi lý luận liên quan LSTM.

    CV04 phải hoàn thiện nội dung.

    Returns
    -------
    answers : dict
    """
    answers = {}

    # --- Q1: LSTM học được chu kỳ nào trong dữ liệu? ---
    answers["Q1_LSTM_cycles"] = """
    TODO (CV04):
    - Chu kỳ ngày (24h): điện tiêu thụ cao buổi sáng/chiều, thấp ban đêm
    - Chu kỳ tuần (168h): cuối tuần vs ngày thường
    - LSTM với lookback=24 học được chu kỳ ngắn (24h)
    - Để học chu kỳ tuần cần lookback=168 hoặc thêm time features (giờ, ngày_trong_tuần)
    - ...
    """

    # --- Q2: LSTM có bị overfitting không và cách xử lý? ---
    answers["Q2_LSTM_overfitting"] = """
    TODO (CV04):
    - Quan sát learning curve: nếu val_loss tăng sau N epoch → overfitting
    - EarlyStopping đã giúp dừng tại best epoch
    - Các giải pháp: tăng Dropout, giảm LSTM units, thêm L2 regularization
    - Dữ liệu ít: data augmentation hoặc giảm độ phức tạp model
    - ...
    """

    # --- Q3: Sự khác biệt giữa ML (XGBoost) và DL (LSTM) ---
    answers["Q3_ML_vs_DL"] = """
    TODO (CV04):
    - XGBoost: flat vector feature, không có memory state, nhanh hơn
    - LSTM: sequence model, có hidden state (memory), học temporal dependency tự nhiên
    - DL cần nhiều dữ liệu và compute hơn ML
    - Trong bài toán này, ai tốt hơn? So sánh RMSE/MAE/MAPE từ Step 4 vs XGBoost
    - ...
    """

    for q, a in answers.items():
        print(f"\n  [{q}]\n{a}")

    return answers


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    """
    Pipeline chính cho CV04: LSTM.
    """
    print("\n" + "=" * 60)
    print("  CV04 – LSTM MODEL PIPELINE")
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

    # --- Step 1: Tạo sequences ---
    print("=" * 60)
    print("[STEP 1] Tạo sequences 3D")
    print("=" * 60)

    X_train, y_train = step1_build_sequences(df_train[FEATURES].values, lookback=LOOKBACK)
    X_val,   y_val   = step1_build_sequences(df_val[FEATURES].values,   lookback=LOOKBACK)
    X_test,  y_test  = step1_build_sequences(df_test[FEATURES].values,  lookback=LOOKBACK)

    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val  : X={X_val.shape},   y={y_val.shape}")
    print(f"  Test : X={X_test.shape},  y={y_test.shape}\n")

    # --- Step 2: Build LSTM ---
    model = step2_build_lstm(lookback=LOOKBACK, n_features=N_FEATURES)

    # --- Step 3: Train ---
    model, history = step3_train(model, X_train, y_train, X_val, y_val)

    # Lưu model cuối
    model_path = MODELS_DIR / "lstm_model.h5"
    model.save(str(model_path))
    print(f"  Đã lưu model cuối: {model_path}")

    # --- Step 4: Evaluate ---
    metrics, y_true, y_pred = step4_evaluate(model, X_test, y_test, scaler)
    pd.DataFrame([metrics]).to_csv(TABLES_DIR / "lstm_metrics.csv", index=False)
    print(f"  Đã lưu metrics: {TABLES_DIR / 'lstm_metrics.csv'}")

    # --- Step 6: Câu hỏi lý luận ---
    answers = step6_answer_questions()

    print("\n" + "=" * 60)
    print("  [CV04 DONE] LSTM pipeline hoàn tất.")
    print("=" * 60)


if __name__ == "__main__":
    main()
