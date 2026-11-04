# ============================================================
# AI_AGENT_INSTRUCTIONS
# ============================================================
# THÀNH VIÊN  : CV03 – Hoàng
# DEADLINE    : 08/11
# NHIỆM VỤ   : Xây dựng mô hình XGBoost (Machine Learning Baseline)
#
# INPUT CONTRACT:
#   - data/processed/train.csv, val.csv, test.csv   (scaled, DatetimeIndex)
#   - data/processed/scaler.pkl
#
# OUTPUT CONTRACT:
#   - outputs/models/xgb_model.json                 (model đã train)
#   - outputs/tables/xgb_metrics.csv                (RMSE, MAE, MAPE trên test)
#   - outputs/figures/xgb_learning_curve.png
#   - outputs/figures/xgb_feature_importance.png
#   - outputs/figures/xgb_forecast_vs_actual.png
#
# LƯU Ý QUAN TRỌNG:
#   - Rolling features PHẢI dùng .shift(1).rolling(w) để tránh lookahead bias
#   - random_state=42 cho XGBRegressor
#   - Dùng early_stopping_rounds với eval_set=[(X_val, y_val)]
#   - Feature target là 'load' (không scale lại – scaler đã được áp dụng ở CV01)
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
from xgboost import XGBRegressor

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
LAG          = 24


# ----------------------------------------------------------
# STEP 1: Xây dựng vector phẳng feature
# ----------------------------------------------------------
def step1_build_flat_features(df_scaled: pd.DataFrame, lag: int = 24) -> pd.DataFrame:
    """
    Tạo bộ feature phẳng gồm:
      - 96 lag features    : {col}_lag{k}  (k=1..24, 4 biến)
      - rolling_mean_6h    : mean của load 6 bước trước (shift(1) trước rolling)
      - rolling_std_24h    : std của load 24 bước trước (shift(1) trước rolling)
      - load_trend         : load_lag1 - load_lag25 (xu hướng 24h)

    QUAN TRỌNG: Tất cả rolling đều dùng .shift(1).rolling(w) để TRÁNH LOOKAHEAD BIAS.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        DataFrame đã scale (train, val hoặc test).
    lag : int
        Số lượng lag (mặc định 24).

    Returns
    -------
    df_feat : pd.DataFrame
        DataFrame chứa tất cả features + cột target 'load_t0' (nhãn dự báo).
        Sau dropna().
    """
    print("[STEP 1] Xây dựng flat features cho XGBoost...")

    df_feat = pd.DataFrame(index=df_scaled.index)

    # --- Lag features ---
    for col in FEATURES:
        for k in range(1, lag + 1):
            df_feat[f"{col}_lag{k}"] = df_scaled[col].shift(k)

    # --- Rolling features (shift(1) TRƯỚC rolling để tránh lookahead) ---
    load_shifted = df_scaled[TARGET].shift(1)

    df_feat["rolling_mean_6h"]  = load_shifted.rolling(window=6,  min_periods=1).mean()
    df_feat["rolling_std_24h"]  = load_shifted.rolling(window=24, min_periods=1).std()

    # --- Trend feature: load_lag1 - load_lag25 ---
    # Đại diện cho xu hướng 24h qua
    if lag >= 25:
        df_feat["load_trend"] = (
            df_scaled[TARGET].shift(1) - df_scaled[TARGET].shift(25)
        )
    else:
        # Nếu lag < 25, dùng sai phân 1 bước
        df_feat["load_trend"] = df_scaled[TARGET].shift(1).diff(1)

    # --- Target: giá trị load tại bước t (hiện tại) ---
    df_feat["load_t0"] = df_scaled[TARGET].values

    # --- Bỏ hàng NaN ---
    df_feat = df_feat.dropna()

    n_feat = len([c for c in df_feat.columns if c != "load_t0"])
    print(f"  Số features  : {n_feat}")
    print(f"  Shape sau dropna : {df_feat.shape}")
    return df_feat


# ----------------------------------------------------------
# STEP 2: Train XGBoost
# ----------------------------------------------------------
def step2_train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series):
    """
    Huấn luyện XGBRegressor với early stopping.

    Tham số cố định:
      n_estimators=1000, learning_rate=0.01, max_depth=5,
      subsample=0.8, colsample_bytree=0.8, random_state=42,
      early_stopping_rounds=50

    Vẽ learning curve (train RMSE vs val RMSE theo số cây).

    Parameters
    ----------
    X_train, X_val : pd.DataFrame    (ma trận features)
    y_train, y_val : pd.Series       (nhãn load_t0)

    Returns
    -------
    model : XGBRegressor  (đã fit)
    evals_result : dict   (lịch sử loss để vẽ learning curve)
    """
    print("=" * 60)
    print("[STEP 2] Train XGBoost")
    print("=" * 60)

    model = XGBRegressor(
        n_estimators       = 1000,
        learning_rate      = 0.01,
        max_depth          = 5,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 3,
        gamma              = 0,
        reg_alpha          = 0.0,
        reg_lambda         = 1.0,
        random_state       = RANDOM_STATE,
        n_jobs             = -1,
        objective          = "reg:squarederror",
        eval_metric        = "rmse",
        early_stopping_rounds = 50,
        verbosity          = 1,
    )

    evals_result = {}
    model.fit(
        X_train, y_train,
        eval_set        = [(X_train, y_train), (X_val, y_val)],
        verbose         = 100,
        callbacks       = None,     # early_stopping_rounds dùng trực tiếp trong fit
    )

    # TODO (CV03): Lấy evals_result từ model.evals_result() sau khi fit
    #   evals_result = model.evals_result()
    #   train_rmse   = evals_result["validation_0"]["rmse"]
    #   val_rmse     = evals_result["validation_1"]["rmse"]

    best_iter = model.best_iteration
    print(f"\n  Best iteration  : {best_iter}")
    print(f"  Best val RMSE   : {model.best_score:.4f}")

    # --- Vẽ Learning Curve ---
    # TODO (CV03): uncomment sau khi có evals_result
    # fig, ax = plt.subplots(figsize=(10, 4))
    # ax.plot(train_rmse, label="Train RMSE", color="steelblue")
    # ax.plot(val_rmse,   label="Val RMSE",   color="tomato")
    # ax.axvline(x=best_iter, color="green", linestyle="--", label=f"Best iter={best_iter}")
    # ax.set_xlabel("Số cây (n_estimators)")
    # ax.set_ylabel("RMSE (scaled)")
    # ax.set_title("XGBoost Learning Curve")
    # ax.legend(); ax.grid(alpha=0.3)
    # fig.tight_layout()
    # fig.savefig(FIGURES_DIR / "xgb_learning_curve.png", dpi=150, bbox_inches="tight")
    # plt.close(fig)
    # print(f"  Đã lưu: {FIGURES_DIR / 'xgb_learning_curve.png'}")

    print("[STEP 2 DONE]\n")
    return model, evals_result


# ----------------------------------------------------------
# STEP 3: Evaluate
# ----------------------------------------------------------
def step3_evaluate(model: XGBRegressor,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   scaler) -> dict:
    """
    Dự báo trên tập test, inverse transform về original scale, tính metrics.

    Parameters
    ----------
    model : XGBRegressor
    X_test : pd.DataFrame
    y_test : pd.Series          (giá trị load đã scale)
    scaler : StandardScaler     (để inverse transform)

    Returns
    -------
    metrics : dict  {"RMSE": ..., "MAE": ..., "MAPE": ...}
    """
    print("=" * 60)
    print("[STEP 3] Evaluate XGBoost – Tập Test")
    print("=" * 60)

    y_pred_scaled = model.predict(X_test)

    # --- Inverse Transform ---
    # XGBoost chỉ dự báo cột 'load', nhưng scaler được fit trên 4 biến.
    # Cần tạo dummy array 4 cột để inverse_transform đúng.
    col_idx = list(FEATURES).index(TARGET)

    def _inv(arr_1d):
        dummy = np.zeros((len(arr_1d), len(FEATURES)))
        dummy[:, col_idx] = arr_1d
        return scaler.inverse_transform(dummy)[:, col_idx]

    y_true = _inv(y_test.values)
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
    ax.plot(y_true, label="Actual Load",       color="steelblue", linewidth=1.0)
    ax.plot(y_pred, label="XGBoost Forecast",  color="darkorange", linewidth=0.8, linestyle="--")
    ax.set_title("XGBoost: Dự báo vs Thực tế – Tập Test")
    ax.set_xlabel("Bước thời gian")
    ax.set_ylabel("Load (original scale)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xgb_forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {FIGURES_DIR / 'xgb_forecast_vs_actual.png'}")

    print("[STEP 3 DONE]\n")
    return metrics, y_true, y_pred


# ----------------------------------------------------------
# STEP 4: Feature Importance
# ----------------------------------------------------------
def step4_feature_importance(model: XGBRegressor, feature_names: list) -> pd.DataFrame:
    """
    Trích xuất và vẽ top-20 feature importance theo Gain.

    Gain = mức độ cải thiện độ chính xác khi feature được dùng để split.

    Parameters
    ----------
    model : XGBRegressor
    feature_names : list  (tên cột trong X_train)

    Returns
    -------
    fi_df : pd.DataFrame  (bảng feature importance đã sắp xếp giảm dần)
    """
    print("=" * 60)
    print("[STEP 4] Feature Importance (XGBoost Gain)")
    print("=" * 60)

    # Lấy importance theo gain
    booster   = model.get_booster()
    scores    = booster.get_score(importance_type="gain")

    fi_df = pd.DataFrame({
        "feature"   : list(scores.keys()),
        "importance": list(scores.values()),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi_df["rank"] = fi_df.index + 1
    print(fi_df.head(20).to_string(index=False))

    # Vẽ top-20
    top20 = fi_df.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top20["feature"][::-1], top20["importance"][::-1], color="steelblue")
    ax.set_xlabel("Gain (XGBoost)")
    ax.set_title("Top-20 Feature Importance – XGBoost (Gain)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xgb_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu: {FIGURES_DIR / 'xgb_feature_importance.png'}")

    fi_df.to_csv(TABLES_DIR / "xgb_feature_importance.csv", index=False)
    print("[STEP 4 DONE]\n")
    return fi_df


# ----------------------------------------------------------
# STEP 5: Trả lời câu hỏi lý luận
# ----------------------------------------------------------
def step5_answer_questions() -> dict:
    """
    Skeleton trả lời 3 câu hỏi lý luận liên quan XGBoost.

    CV03 phải hoàn thiện nội dung các câu trả lời bên dưới.

    Returns
    -------
    answers : dict  {câu hỏi: câu trả lời}
    """
    answers = {}

    # --- Q1: XGBoost khác VAR ở điểm cốt lõi nào? ---
    answers["Q1_XGB_vs_VAR"] = """
    TODO (CV03):
    - VAR là mô hình tuyến tính, mô hình hoá mối quan hệ tuyến tính giữa các biến.
    - XGBoost là ensemble cây quyết định, có thể nắm bắt phi tuyến.
    - ...
    """

    # --- Q2: XGBoost học gì từ các biến thời tiết (temp, humidity, wind)? ---
    answers["Q2_XGB_weather_features"] = """
    TODO (CV03):
    - Dựa vào feature importance (Step 4), biến nào có Gain cao nhất?
    - Giải thích cơ chế: XGBoost dùng split tại các ngưỡng nhiệt độ/độ ẩm...
    - ...
    """

    # --- Q3: XGBoost có ưu điểm gì so với LSTM trong bài toán này? ---
    answers["Q3_XGB_vs_LSTM"] = """
    TODO (CV03):
    - Tốc độ train/inference nhanh hơn LSTM
    - Không cần nhiều dữ liệu để fit tốt
    - Feature importance dễ diễn giải (interpretability)
    - Nhược điểm: không mô hình hoá sequential dependency tự nhiên như LSTM
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
    Pipeline chính cho CV03: XGBoost.
    """
    print("\n" + "=" * 60)
    print("  CV03 – XGBOOST MODEL PIPELINE")
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

    # --- Step 1: Xây dựng features ---
    df_train_feat = step1_build_flat_features(df_train, lag=LAG)
    df_val_feat   = step1_build_flat_features(df_val,   lag=LAG)
    df_test_feat  = step1_build_flat_features(df_test,  lag=LAG)

    feature_cols = [c for c in df_train_feat.columns if c != "load_t0"]

    X_train, y_train = df_train_feat[feature_cols], df_train_feat["load_t0"]
    X_val,   y_val   = df_val_feat[feature_cols],   df_val_feat["load_t0"]
    X_test,  y_test  = df_test_feat[feature_cols],  df_test_feat["load_t0"]

    print(f"  X_train shape : {X_train.shape}")
    print(f"  X_val shape   : {X_val.shape}")
    print(f"  X_test shape  : {X_test.shape}\n")

    # --- Step 2: Train ---
    model, evals_result = step2_train_xgboost(X_train, y_train, X_val, y_val)

    # Lưu model
    model_path = MODELS_DIR / "xgb_model.json"
    model.save_model(str(model_path))
    print(f"  Đã lưu model: {model_path}")

    # --- Step 3: Evaluate ---
    metrics, y_true, y_pred = step3_evaluate(model, X_test, y_test, scaler)
    pd.DataFrame([metrics]).to_csv(TABLES_DIR / "xgb_metrics.csv", index=False)
    print(f"  Đã lưu metrics: {TABLES_DIR / 'xgb_metrics.csv'}")

    # --- Step 4: Feature Importance ---
    fi_df = step4_feature_importance(model, feature_cols)

    # --- Step 5: Câu hỏi lý luận ---
    answers = step5_answer_questions()

    print("\n" + "=" * 60)
    print("  [CV03 DONE] XGBoost pipeline hoàn tất.")
    print("=" * 60)


if __name__ == "__main__":
    main()
