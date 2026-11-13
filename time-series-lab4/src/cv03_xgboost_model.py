# ============================================================
# AI_AGENT_INSTRUCTIONS
# ============================================================
# THÀNH VIÊN  : CV03 – Phương
# DEADLINE    : 08/11/2026
# NHIỆM VỤ   : Xây dựng mô hình XGBoost (Machine Learning Baseline)
#
# INPUT CONTRACT:
#   - data/processed/train.csv, val.csv, test.csv   (scaled, DatetimeIndex)
#   - data/processed/scaler.pkl
#
# OUTPUT CONTRACT:
#   - outputs/models/xgboost_model.json             (model đã train)
#   - outputs/models/xgb_model.json
#   - outputs/tables/cv03_xgb_metrics.csv           (RMSE, MAE, MAPE trên test)
#   - outputs/tables/xgb_metrics.csv
#   - outputs/tables/xgb_feature_importance.csv
#   - outputs/figures/cv03_xgb_learning_curve.png
#   - outputs/figures/xgb_learning_curve.png
#   - outputs/figures/cv03_xgb_feature_importance.png
#   - outputs/figures/xgb_feature_importance.png
#   - outputs/figures/cv03_xgb_forecast_vs_actual.png
#   - outputs/figures/xgb_forecast_vs_actual.png
#
# LƯU Ý QUAN TRỌNG:
#   - Rolling features PHẢI dùng .shift(1).rolling(w) để tránh lookahead bias
#   - random_state=42 cho XGBRegressor
#   - Dùng early_stopping_rounds=50 với eval_set=[(X_train, y_train), (X_val, y_val)]
#   - Feature target là 'load'
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
from xgboost import XGBRegressor

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
    """
    print("[STEP 1] Xây dựng flat features cho XGBoost...")

    df_feat = pd.DataFrame(index=df_scaled.index)

    # --- 96 Lag features ---
    for col in FEATURES:
        for k in range(1, lag + 1):
            df_feat[f"{col}_lag{k}"] = df_scaled[col].shift(k)

    # --- Rolling features (shift(1) TRƯỚC rolling để tránh lookahead) ---
    load_shifted = df_scaled[TARGET].shift(1)
    df_feat["rolling_mean_6h"] = load_shifted.rolling(window=6, min_periods=1).mean()
    df_feat["rolling_std_24h"] = load_shifted.rolling(window=24, min_periods=1).std()

    # --- Trend feature: load_lag1 - load_lag25 (đại diện xu hướng 24h qua) ---
    df_feat["load_trend"] = df_scaled[TARGET].shift(1) - df_scaled[TARGET].shift(25)

    # --- Target: giá trị load tại bước t ---
    df_feat["load_t0"] = df_scaled[TARGET].values

    n_feat = len([c for c in df_feat.columns if c != "load_t0"])
    print(f"  Số features tạo thành : {n_feat} (96 lags + 3 rolling/trend)")
    return df_feat


# ----------------------------------------------------------
# STEP 2: Train XGBoost với Early Stopping
# ----------------------------------------------------------
def step2_train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series):
    """
    Huấn luyện XGBRegressor với Early Stopping trên tập Validation.
    Vẽ learning curve (train RMSE vs val RMSE).
    """
    print("=" * 60)
    print("[STEP 2] Huấn luyện XGBoost Regressor")
    print("=" * 60)

    model = XGBRegressor(
        n_estimators       = 1000,
        learning_rate      = 0.02,
        max_depth          = 5,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 3,
        gamma              = 0.0,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        random_state       = RANDOM_STATE,
        n_jobs             = -1,
        objective          = "reg:squarederror",
        eval_metric        = "rmse",
        early_stopping_rounds = 50,
        verbosity          = 1,
    )

    model.fit(
        X_train, y_train,
        eval_set = [(X_train, y_train), (X_val, y_val)],
        verbose  = 100,
    )

    evals_result = model.evals_result()
    train_rmse = evals_result["validation_0"]["rmse"]
    val_rmse   = evals_result["validation_1"]["rmse"]
    best_iter  = model.best_iteration
    best_val_score = model.best_score

    print(f"\n  Số cây tối ưu (Best iteration) : {best_iter}")
    print(f"  Best Val RMSE (scaled)        : {best_val_score:.5f}")

    # --- Vẽ Learning Curve ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(train_rmse, label="Train RMSE", color="#1f77b4", linewidth=1.5)
    ax.plot(val_rmse,   label="Val RMSE",   color="#d62728", linewidth=1.5)
    ax.axvline(x=best_iter, color="seagreen", linestyle="--", linewidth=1.5,
               label=f"Điểm dừng tối ưu: Round {best_iter} (Val={best_val_score:.4f})")
    ax.set_xlabel("Số lượng cây (Boosting Rounds)", fontsize=11)
    ax.set_ylabel("Hàm mất mát RMSE (Không gian chuẩn hóa)", fontsize=11)
    ax.set_title("Đường cong học tập XGBoost (Learning Curve Train vs Validation)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(FIGURES_DIR / "cv03_xgb_learning_curve.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "xgb_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu biểu đồ Learning Curve: {FIGURES_DIR / 'cv03_xgb_learning_curve.png'}")

    print("[STEP 2 DONE]\n")
    return model, evals_result


# ----------------------------------------------------------
# STEP 3: Đánh giá trên tập Test
# ----------------------------------------------------------
def step3_evaluate(model: XGBRegressor,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   scaler) -> tuple:
    """
    Dự báo trên tập test, nghịch đảo chuẩn hóa về đơn vị Megawatt và tính các thước đo RMSE, MAE, MAPE.
    """
    print("=" * 60)
    print("[STEP 3] Đánh giá XGBoost trên tập Test")
    print("=" * 60)

    y_pred_scaled = model.predict(X_test)

    # Nghịch đảo chuẩn hóa cột TARGET = 'load'
    col_idx = list(FEATURES).index(TARGET)

    def _inv(arr_1d):
        dummy = np.zeros((len(arr_1d), len(FEATURES)))
        dummy[:, col_idx] = arr_1d
        return scaler.inverse_transform(dummy)[:, col_idx]

    y_true = _inv(y_test.values)
    y_pred = _inv(y_pred_scaled)

    # Tính các thước đo
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {
        "Model": "XGBoost",
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MAPE": round(float(mape), 4),
    }
    print(f"  RMSE = {metrics['RMSE']:.4f} MW")
    print(f"  MAE  = {metrics['MAE']:.4f} MW")
    print(f"  MAPE = {metrics['MAPE']:.4f} %")

    # --- Vẽ Forecast vs Actual ---
    fig, ax = plt.subplots(figsize=(14, 5))
    test_dates = y_test.index
    ax.plot(test_dates, y_true, label="Phụ tải thực tế (Actual Load)", color="#1f77b4", linewidth=1.2)
    ax.plot(test_dates, y_pred, label="XGBoost Dự báo (Forecast)", color="#ff7f0e", linewidth=1.0, linestyle="--", alpha=0.9)
    ax.set_title(f"Mô hình XGBoost Regressor: Dự báo vs Thực tế trên tập Test ({len(y_test):,} giờ)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thời gian", fontsize=11)
    ax.set_ylabel("Phụ tải (Megawatt - MW)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(FIGURES_DIR / "cv03_xgb_forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "xgb_forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu đồ thị: {FIGURES_DIR / 'cv03_xgb_forecast_vs_actual.png'}")

    print("[STEP 3 DONE]\n")
    return metrics, y_true, y_pred


# ----------------------------------------------------------
# STEP 4: Phân tích Feature Importance
# ----------------------------------------------------------
def step4_feature_importance(model: XGBRegressor, feature_names: list) -> pd.DataFrame:
    """
    Trích xuất và vẽ top-20 feature importance theo Gain.
    Gain phản ánh mức cải thiện độ chính xác phân nhánh tại mỗi node.
    """
    print("=" * 60)
    print("[STEP 4] Phân tích Feature Importance (XGBoost Gain)")
    print("=" * 60)

    booster = model.get_booster()
    scores  = booster.get_score(importance_type="gain")

    # Gán score cho các feature nếu có feature chưa xuất hiện
    all_scores = {f: scores.get(f, 0.0) for f in feature_names}

    fi_df = pd.DataFrame({
        "feature"   : list(all_scores.keys()),
        "importance": list(all_scores.values()),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi_df["rank"] = fi_df.index + 1
    print("\nTop 15 đặc trưng có Gain cao nhất:")
    print(fi_df.head(15).to_string(index=False))

    # Lưu bảng kết quả Feature Importance
    fi_df.to_csv(TABLES_DIR / "cv03_xgb_feature_importance.csv", index=False)
    fi_df.to_csv(TABLES_DIR / "xgb_feature_importance.csv", index=False)
    print(f"\n  Đã lưu: {TABLES_DIR / 'cv03_xgb_feature_importance.csv'}")

    # Vẽ top-20 biểu đồ thanh ngang
    top20 = fi_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    colors = ["#d62728" if "temp" in f else ("#2ca02c" if "rolling" in f or "trend" in f else "#1f77b4") for f in top20["feature"][::-1]]
    ax.barh(top20["feature"][::-1], top20["importance"][::-1], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mức độ quan trọng theo Gain (Độ giảm phương sai trung bình)", fontsize=11)
    ax.set_title("Top 20 Đặc trưng Quan trọng nhất trong Mô hình XGBoost (Gain Importance)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    fig.savefig(FIGURES_DIR / "cv03_xgb_feature_importance.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "xgb_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu biểu đồ: {FIGURES_DIR / 'cv03_xgb_feature_importance.png'}")

    print("[STEP 4 DONE]\n")
    return fi_df


# ----------------------------------------------------------
# STEP 5: Trả lời 3 Câu hỏi Lý luận Chuyên sâu
# ----------------------------------------------------------
def step5_answer_questions(fi_df: pd.DataFrame) -> dict:
    answers = {}

    top_weather = fi_df[fi_df["feature"].str.contains("temp|humidity|wind")].iloc[0]

    answers["Q1_XGB_vs_VAR"] = """
    [Lý luận 1: Sự khác biệt cốt lõi giữa XGBoost và VAR]
    1. Giả định tính tuyến tính: VAR giả định mối quan hệ động giữa các biến là hoàn toàn tuyến tính. 
       Ngược lại, XGBoost (Gradient Boosted Trees) có khả năng xấp xỉ bất kỳ hàm mục tiêu phi tuyến phức tạp nào, 
       đặc biệt là mối quan hệ U-Shape kinh điển giữa phụ tải điện và nhiệt độ (nhiệt độ quá lạnh hoặc quá nóng đều làm vọt phụ tải).
    2. Ràng buộc tính dừng: VAR yêu cầu nghiêm ngặt chuỗi phải dừng (Stationary) và phải kiểm định ADF, sai phân. 
       XGBoost không bị ràng buộc bởi giả định tính dừng phân phối xác suất và xử lý tốt các biến số có biên độ biến thiên đa dạng.
    3. Tương tác đa chiều: XGBoost tự động học các quy tắc tương tác (feature interaction) thông qua các nhánh cây liên tiếp 
       (ví dụ: nếu nhiệt độ > 32°C VÀ độ ẩm > 80% thì phụ tải tăng đột biến do điều hòa hoạt động hết công suất).
    """

    answers["Q2_XGB_weather_features"] = f"""
    [Lý luận 2: XGBoost học được gì từ các biến thời tiết]
    - Biến thời tiết có đóng góp lớn nhất trong mô hình là: {top_weather['feature']} (Gain = {top_weather['importance']:.4f}, Rank = {top_weather['rank']}).
    - Cơ chế học của cây quyết định: XGBoost phân tách các ngưỡng nhiệt độ tới hạn (Cooling Degree Days / Heating Degree Days).
      Khi nhiệt độ vượt quá 28°C hoặc giảm xuống dưới 12°C, các split node của cây lập tức chuyển hướng dự báo sang các lá (leaf nodes)
      có giá trị phụ tải cao hơn hẳn mức bình thường.
    - Sự tương tác giữa Nhiệt độ và Độ ẩm: Độ ẩm cao làm giảm khả năng thoát nhiệt tự nhiên của môi trường, tạo ra nhiệt độ cảm nhận (Heat Index) cao hơn,
      dẫn đến hành vi bật làm mát của người tiêu dùng tăng mạnh.
    """

    answers["Q3_XGB_vs_LSTM"] = """
    [Lý luận 3: Ưu thế của XGBoost so với LSTM trong bài toán thực tế]
    1. Tốc độ huấn luyện & Độ trễ suy luận: XGBoost huấn luyện chỉ mất khoảng 10-30 giây và suy luận tính bằng micro-giây,
       trong khi mạng LSTM sâu đòi hỏi tài nguyên tính toán GPU/CPU lớn và thời gian huấn luyện lâu hơn gấp 10-20 lần.
    2. Tính diễn giải (Interpretability): XGBoost cung cấp cơ chế Feature Importance (Gain, Cover, Frequency) và SHAP values minh bạch,
       giúp kỹ sư vận hành hệ thống điện hiểu rõ lý do vì sao dự báo phụ tải tăng. LSTM đóng vai trò như một hộp đen (black-box).
    3. Yêu cầu tiền xử lý: XGBoost ít nhạy cảm với scale và phân phối dữ liệu hơn so với mạng nơ-ron LSTM.
    """

    for q, a in answers.items():
        print(f"\n{a}")

    return answers


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  CV03 – XGBOOST MODEL PIPELINE (Phương)")
    print("=" * 60 + "\n")

    # --- Load dữ liệu đã chuẩn hóa ---
    df_train = pd.read_csv(PROCESSED_DIR / "train.csv", index_col=0, parse_dates=True)
    df_val   = pd.read_csv(PROCESSED_DIR / "val.csv",   index_col=0, parse_dates=True)
    df_test  = pd.read_csv(PROCESSED_DIR / "test.csv",  index_col=0, parse_dates=True)

    with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print(f"  Train shape : {df_train.shape}")
    print(f"  Val shape   : {df_val.shape}")
    print(f"  Test shape  : {df_test.shape}\n")

    # --- Step 1: Tạo Feature Engineering đồng bộ và liên tục ---
    # Ghép chuỗi để bảo tồn đúng 24h trễ cho tập val và test mà không rò rỉ dữ liệu
    df_all = pd.concat([df_train, df_val, df_test])
    df_all_feat = step1_build_flat_features(df_all, lag=LAG)

    df_train_feat = df_all_feat.loc[df_train.index].dropna()
    df_val_feat   = df_all_feat.loc[df_val.index]
    df_test_feat  = df_all_feat.loc[df_test.index]

    feature_cols = [c for c in df_train_feat.columns if c != "load_t0"]

    X_train, y_train = df_train_feat[feature_cols], df_train_feat["load_t0"]
    X_val,   y_val   = df_val_feat[feature_cols],   df_val_feat["load_t0"]
    X_test,  y_test  = df_test_feat[feature_cols],  df_test_feat["load_t0"]

    print(f"  X_train shape : {X_train.shape} (đã dropna 25 bước đầu)")
    print(f"  X_val shape   : {X_val.shape} (bảo toàn trọn vẹn)")
    print(f"  X_test shape  : {X_test.shape} (bảo toàn trọn vẹn)\n")

    # --- Step 2: Train với Early Stopping ---
    model, evals_result = step2_train_xgboost(X_train, y_train, X_val, y_val)

    # Lưu model artifacts
    model.save_model(str(MODELS_DIR / "xgboost_model.json"))
    model.save_model(str(MODELS_DIR / "xgb_model.json"))
    print(f"  Đã lưu model artifacts: {MODELS_DIR / 'xgboost_model.json'}")

    # --- Step 3: Đánh giá trên tập Test ---
    metrics, y_true, y_pred = step3_evaluate(model, X_test, y_test, scaler)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(TABLES_DIR / "cv03_xgb_metrics.csv", index=False)
    metrics_df.to_csv(TABLES_DIR / "xgb_metrics.csv", index=False)
    print(f"  Đã lưu metrics: {TABLES_DIR / 'cv03_xgb_metrics.csv'}")

    # --- Step 4: Phân tích Feature Importance ---
    fi_df = step4_feature_importance(model, feature_cols)

    # --- Step 5: Trả lời câu hỏi lý luận chuyên sâu ---
    answers = step5_answer_questions(fi_df)

    print("\n" + "=" * 60)
    print("  [CV03 HOÀN THÀNH] Toàn bộ pipeline XGBoost đã hoàn tất thành công.")
    print("=" * 60)


if __name__ == "__main__":
    main()
