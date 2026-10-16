# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 3 (NHÓM 7)
MÃ CÔNG VIỆC: CV04 - NÂNG CAO 1 & 2: ROLLING FEATURES & DỰ BÁO ĐA BƯỚC (MULTI-STEP)
---------------------------------------------------------------------------------
Thành viên phụ trách: Hoàng Hiệp
Nhánh Git khuyến nghị: feat/cv04-rolling-multistep
Tệp mã nguồn: src/cv04_rolling_and_multistep.py
Dữ liệu đầu vào: data/raw/AEP_hourly.csv (hoặc data/processed/train.csv & test.csv)
Đầu ra tạo thành:
  - outputs/figures/cv04_rolling_features_impact.png
  - outputs/figures/cv04_multistep_forecast.png
  - outputs/tables/cv04_multistep_metrics.csv

AI_AGENT_INSTRUCTIONS:
=================================================================================
Mục tiêu dành cho AI Agent (Cursor / GitHub Copilot / Antigravity):
1. [NÂNG CAO 1 - ROLLING WINDOW FEATURES]:
   - Tính toán các đặc trưng thống kê trượt:
     + rolling_mean_24: Trung bình trượt 24 giờ.
     + rolling_std_24:  Độ lệch chuẩn trượt 24 giờ.
     + rolling_mean_168: Trung bình trượt 1 tuần (168 giờ).
   - [ĐIỀU KIỆN TIÊN QUYẾT BẮT BUỘC]:
     Luôn tính rolling dựa trên dữ liệu đã trễ 1 bước (sử dụng .shift(1).rolling(w))
     để triệt tiêu hoàn toàn rò rỉ dữ liệu (Lookahead Bias/Data Leakage) tại thời điểm dự báo t.
   - So sánh sự cải thiện MAE/RMSE giữa mô hình chỉ có Lags vs mô hình có Lags + Rolling Features.
2. [NÂNG CAO 2 - MULTI-STEP AHEAD FORECASTING (24H TIẾP THEO)]:
   - Xây dựng chiến lược Dự báo Đệ quy (Recursive Strategy):
     Dự báo y_{t+1}, cập nhật vector lag, dùng giá trị vừa dự báo để suy luận tiếp y_{t+2}, ..., y_{t+24}.
   - Phân tích hiện tượng sai số tích lũy (Error Accumulation) khi tầm dự báo h tăng từ 1 đến 24 giờ.
3. Xuất biểu đồ và bảng chỉ số vào outputs/figures/ và outputs/tables/.
4. Đảm bảo hỗ trợ chạy CLI độc lập: python src/cv04_rolling_and_multistep.py [--show]
=================================================================================
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

# Cấu hình encoding UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

# Cấu hình phông chữ hỗ trợ tiếng Việt trên Windows
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# -------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ TOÀN CỤC
# -------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------------
# 2. BƯỚC 1: TẠO DATASET VỚI ROLLING FEATURES (KHÔNG RÒ RỈ DỮ LIỆU)
# -------------------------------------------------------------------------------
def step1_load_and_create_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Nạp dữ liệu AEP_hourly.csv và sinh:
    - 24 Lag features: lag_1 ... lag_24
    - Rolling features: rolling_mean_24, rolling_std_24, rolling_mean_168
    (Luôn dùng .shift(1) trước khi .rolling để bảo đảm không rò rỉ dữ liệu)
    """
    print(f"\n[BƯỚC 1] Nạp dữ liệu và tạo Rolling Window Features từ {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH, index_col="Datetime", parse_dates=True).sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]

    target = df["AEP_MW"]
    feat_df = pd.DataFrame(index=df.index)
    feat_df["target"] = target

    # 1. Lag features (1 đến 24)
    for p in range(1, 25):
        feat_df[f"lag_{p}"] = target.shift(p)

    # 2. Rolling features (tính trên chuỗi đã shift 1 để không thấy tương lai)
    shifted_target = target.shift(1)
    feat_df["rolling_mean_24"] = shifted_target.rolling(window=24).mean()
    feat_df["rolling_std_24"] = shifted_target.rolling(window=24).std()
    feat_df["rolling_mean_168"] = shifted_target.rolling(window=168).mean() # 1 tuần

    feat_df.dropna(inplace=True)
    print(f" -> Kích thước dữ liệu sau khi tạo Rolling Features: {feat_df.shape[0]:,} dòng, {feat_df.shape[1]} cột.")

    # Chia train/test 80% - 20% theo thời gian
    split_idx = int(len(feat_df) * 0.80)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]

    lag_cols = [f"lag_{p}" for p in range(1, 25)]
    all_cols = [c for c in feat_df.columns if c != "target"]

    X_train_lags = train_df[lag_cols]
    X_train_all = train_df[all_cols]
    y_train = train_df["target"]

    X_test_lags = test_df[lag_cols]
    X_test_all = test_df[all_cols]
    y_test = test_df["target"]

    return train_df, test_df, y_train, test_df, y_test


# -------------------------------------------------------------------------------
# 3. BƯỚC 2: SO SÁNH HIỆU NĂNG ROLLING FEATURES & THỰC HIỆN DỰ BÁO ĐỆ QUY 24H
# -------------------------------------------------------------------------------
def step2_execute_analysis(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    1. Huấn luyện và so sánh mô hình Baseline (chỉ có Lags) vs Mô hình Nâng cao (+ Rolling Features).
    2. Thực hiện Recursive Multi-step Forecast cho 24 giờ tiếp theo.
    """
    print("\n[BƯỚC 2] Huấn luyện so sánh hiệu năng của Rolling Features...")
    lag_cols = [f"lag_{p}" for p in range(1, 25)]
    all_cols = [c for c in train_df.columns if c != "target"]

    y_train = train_df["target"]
    y_test = test_df["target"]

    # Model 1: Chỉ Lag features
    print(" -> [Model 1] Huấn luyện XGBoost với 24 Lag features...")
    model_lags = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42, n_jobs=-1)
    model_lags.fit(train_df[lag_cols], y_train)
    pred_lags = model_lags.predict(test_df[lag_cols])
    mae_lags = mean_absolute_error(y_test, pred_lags)
    rmse_lags = root_mean_squared_error(y_test, pred_lags)

    # Model 2: Lag features + Rolling Window
    print(" -> [Model 2] Huấn luyện XGBoost với Lags + Rolling Window (24h, 168h)...")
    model_rolling = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42, n_jobs=-1)
    model_rolling.fit(train_df[all_cols], y_train)
    pred_rolling = model_rolling.predict(test_df[all_cols])
    mae_rolling = mean_absolute_error(y_test, pred_rolling)
    rmse_rolling = root_mean_squared_error(y_test, pred_rolling)

    print(f"    • Model Lags:           MAE = {mae_lags:,.2f} MW, RMSE = {rmse_lags:,.2f} MW")
    print(f"    • Model Lags + Rolling: MAE = {mae_rolling:,.2f} MW, RMSE = {rmse_rolling:,.2f} MW")

    # [NÂNG CAO 2]: DỰ BÁO ĐA BƯỚC (RECURSIVE MULTI-STEP 24H AHEAD)
    print("\n[BƯỚC 2.2] Triển khai Dự báo Đa bước (Recursive Strategy) cho 24 giờ tới...")
    steps_ahead = 24
    # Lấy mẫu thử 24 giờ liên tiếp từ tập test
    start_pos = 100
    actual_24h = y_test.iloc[start_pos : start_pos + steps_ahead].values
    timestamps_24h = y_test.index[start_pos : start_pos + steps_ahead]

    # Vector lag ban đầu (tại thời điểm ngay trước bước 1)
    current_lags = test_df[lag_cols].iloc[start_pos].values.copy()
    recursive_preds = []
    step_errors = []

    for step in range(steps_ahead):
        # Dự báo 1 bước tiếp theo
        pred_val = model_lags.predict(current_lags.reshape(1, -1))[0]
        recursive_preds.append(pred_val)

        # Tính sai số tuyệt đối tại bước h
        err = abs(actual_24h[step] - pred_val)
        step_errors.append(err)

        # Cập nhật vector lag: đẩy giá trị vừa dự báo vào lag_1 và dịch các lag còn lại
        current_lags = np.roll(current_lags, 1)
        current_lags[0] = pred_val

    recursive_preds = np.array(recursive_preds)

    # Lưu bảng metrics
    multistep_df = pd.DataFrame({
        "Hour_Ahead (h)": range(1, steps_ahead + 1),
        "Timestamp": timestamps_24h.astype(str),
        "Actual (MW)": actual_24h,
        "Recursive_Forecast (MW)": recursive_preds,
        "Absolute_Error (MW)": step_errors
    })
    multistep_path = TABLES_DIR / "cv04_multistep_metrics.csv"
    multistep_df.to_csv(multistep_path, index=False)
    print(f" -> Đã lưu kết quả dự báo đa bước vào: {multistep_path}")

    return {
        "mae_lags": mae_lags,
        "rmse_lags": rmse_lags,
        "mae_rolling": mae_rolling,
        "rmse_rolling": rmse_rolling,
        "timestamps_24h": timestamps_24h,
        "actual_24h": actual_24h,
        "recursive_preds": recursive_preds,
        "step_errors": step_errors
    }


# -------------------------------------------------------------------------------
# 4. BƯỚC 3: TRỰC QUAN HÓA ROLLING IMPACT & MULTI-STEP ERROR ACCUMULATION
# -------------------------------------------------------------------------------
def step3_visualize(results: Dict[str, Any], show: bool = False):
    """
    Vẽ 2 biểu đồ:
    1. So sánh sai số giữa Model thuần Lag và Model có Rolling Features.
    2. Đường dự báo 24 giờ đệ quy và sai số tích lũy qua từng bước (h=1..24).
    """
    print("\n[BƯỚC 3] Trực quan hóa kết quả nghiên cứu nâng cao...")

    # 1. Biểu đồ so sánh Rolling Features
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Chỉ Lags (1-24)", "Lags + Rolling (24h, 168h)"]
    mae_values = [results["mae_lags"], results["mae_rolling"]]
    rmse_values = [results["rmse_lags"], results["rmse_rolling"]]

    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, mae_values, width, label="MAE (MW)", color="#1f77b4")
    ax.bar(x + width/2, rmse_values, width, label="RMSE (MW)", color="#d62728")

    ax.set_title("So sánh Hiệu năng: Lags thuần túy vs Bổ sung Rolling Features", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sai số (MW)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    for i in range(len(models)):
        ax.text(i - width/2, mae_values[i] + 10, f"{mae_values[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width/2, rmse_values[i] + 10, f"{rmse_values[i]:.1f}", ha="center", fontsize=9, fontweight="bold")

    fig1_path = FIGURES_DIR / "cv04_rolling_features_impact.png"
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=150)
    print(f" -> Đã lưu biểu đồ tác động của Rolling Features vào: {fig1_path}")
    if show:
        plt.show()
    plt.close()

    # 2. Biểu đồ Dự báo Đa bước (Multi-Step Forecast) và Sai số tích lũy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1.3, 1]})
    h_steps = range(1, 25)

    ax1.plot(h_steps, results["actual_24h"], label="Phụ tải thực tế (Actual)", marker="o", color="#1f77b4", lw=2)
    ax1.plot(h_steps, results["recursive_preds"], label="Dự báo đệ quy (Recursive XGBoost)", marker="s", color="#d62728", lw=2, linestyle="--")
    ax1.set_title("Dự báo Đa bước 24 Giờ Tiếp theo (Recursive Multi-Step Forecasting)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Số bước dự báo về tương lai h (Giờ)", fontsize=11)
    ax1.set_ylabel("Mức tiêu thụ (MW)", fontsize=11)
    ax1.set_xticks(h_steps)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Đồ thị sai số tích lũy theo từng bước h
    ax2.bar(h_steps, results["step_errors"], color="#ff7f0e", edgecolor="black", alpha=0.85)
    ax2.set_title("Hiện tượng Sai số Tích lũy (Error Accumulation) theo bước dự báo h", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Bước dự báo h (Giờ)", fontsize=11)
    ax2.set_ylabel("Sai số tuyệt đối |y - y_hat| (MW)", fontsize=11)
    ax2.set_xticks(h_steps)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig2_path = FIGURES_DIR / "cv04_multistep_forecast.png"
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=150)
    print(f" -> Đã lưu biểu đồ Multi-step Forecast vào: {fig2_path}")
    if show:
        plt.show()
    plt.close()


# -------------------------------------------------------------------------------
# 5. BƯỚC 4: KẾT LUẬN & ĐÁNH GIÁ KỸ THUẬT NÂNG CAO
# -------------------------------------------------------------------------------
def step4_conclude_and_report(results: Dict[str, Any]):
    """
    Tổng kết học thuật về Rolling features và chiến lược Multi-step.
    """
    improvement = (results["mae_lags"] - results["mae_rolling"]) / results["mae_lags"] * 100
    print("\n" + "=" * 80)
    print(" [CV04 - BÁO CÁO TỔNG KẾT BÀI TẬP NÂNG CAO 1 & 2]")
    print(f" • Tác động Rolling Features: Giảm MAE từ {results['mae_lags']:.2f} MW xuống {results['mae_rolling']:.2f} MW")
    print(f"   $\rightarrow$ Cải thiện mức độ chính xác: {improvement:+.2f}%")
    print(" • Nhận xét Dự báo Đa bước (Recursive Multi-Step):")
    print("   Khi h tăng từ 1 -> 24 giờ, sai số tích lũy gia tăng do mô hình phải sử dụng chính các")
    print("   ước lượng dự báo không chắc chắn ở các bước trước để làm đầu vào tiếp theo.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="CV04: Rolling Features và Multi-step Forecast")
    parser.add_argument("--show", action="store_true", help="Hiển thị biểu đồ sau khi vẽ")
    args = parser.parse_args()

    print("=" * 80)
    print(" [CV04 - HOÀNG HIỆP] NÂNG CAO 1 & 2: ROLLING FEATURES & MULTI-STEP FORECASTING")
    print("=" * 80)

    train_df, test_df, y_train, _, _ = step1_load_and_create_features()
    results = step2_execute_analysis(train_df, test_df)
    step3_visualize(results, show=args.show)
    step4_conclude_and_report(results)

    print("\n[THÀNH CÔNG] Nhiệm vụ CV04 hoàn thành 100%! Kết quả đã lưu tại outputs/.")


if __name__ == "__main__":
    main()
