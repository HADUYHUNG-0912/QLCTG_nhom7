# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 3 (NHÓM 7)
MÃ CÔNG VIỆC: CV02 - HUẤN LUYỆN MÔ HÌNH CƠ SỞ XGBOOST REGRESSOR (BASELINE TRAINING)
---------------------------------------------------------------------------------
Thành viên phụ trách: Ngọc Trâm
Nhánh Git khuyến nghị: feat/cv02-xgb-train
Tệp mã nguồn: src/cv02_xgboost_train.py
Dữ liệu đầu vào: data/processed/train.csv, data/processed/test.csv
Đầu ra tạo thành:
  - outputs/models/xgboost_baseline.json
  - outputs/figures/cv02_learning_curve.png
  - outputs/tables/cv02_train_history.csv

AI_AGENT_INSTRUCTIONS:
=================================================================================
Mục tiêu dành cho AI Agent (Cursor / GitHub Copilot / Antigravity):
1. Đọc dữ liệu tập Train và Test từ data/processed/train.csv và test.csv.
   (Nếu chưa có file processed, tự động kích hoạt hàm nạp dự phòng từ data/raw/AEP_hourly.csv).
2. Khởi tạo mô hình XGBRegressor từ thư viện xgboost với các siêu tham số chuẩn đề bài:
   - n_estimators: 1000
   - learning_rate: 0.01
   - max_depth: 5
   - early_stopping_rounds: 50
   - objective: 'reg:squarederror'
   - random_state: 42
3. Huấn luyện mô hình và truyền eval_set=[(X_train, y_train), (X_test, y_test)] với verbose=100.
4. Trích xuất lịch sử lỗi qua các vòng lặp (evals_result()) và vẽ đồ thị Learning Loss Curve.
5. Lưu mô hình huấn luyện vào file outputs/models/xgboost_baseline.json.
6. Xuất bảng chỉ số lịch sử huấn luyện vào outputs/tables/cv02_train_history.csv.
7. Đảm bảo hỗ trợ chạy độc lập từ CLI: python src/cv02_xgboost_train.py [--show]
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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIG = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "early_stopping_rounds": 50,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1
}


# -------------------------------------------------------------------------------
# 2. BƯỚC 1: NẠP DỮ LIỆU TRAIN / TEST (DATA LOADING VỚI FALLBACK)
# -------------------------------------------------------------------------------
def step1_load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Nạp dữ liệu Train/Test từ thư mục data/processed/.
    Nếu chưa có, tự động nạp từ data/raw/ và tạo lag features 24 giờ.
    """
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    if train_path.exists() and test_path.exists():
        print(f"\n[BƯỚC 1] Đang nạp dữ liệu đã xử lý từ: {PROCESSED_DIR}")
        train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
    else:
        print(f"\n[BƯỚC 1] Chưa tìm thấy data/processed/, đang tự động tiền xử lý từ {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH, index_col="Datetime", parse_dates=True).sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="first")]
        df_lags = pd.DataFrame(index=df.index)
        df_lags["target"] = df["AEP_MW"]
        for p in range(1, 25):
            df_lags[f"lag_{p}"] = df["AEP_MW"].shift(p)
        df_lags.dropna(inplace=True)

        split_idx = int(len(df_lags) * 0.80)
        train_df = df_lags.iloc[:split_idx]
        test_df = df_lags.iloc[split_idx:]

    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    print(f" -> Kích thước X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f" -> Kích thước X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    return X_train, y_train, X_test, y_test


# -------------------------------------------------------------------------------
# 3. BƯỚC 2: HUẤN LUYỆN MÔ HÌNH XGBOOST VỚI EARLY STOPPING
# -------------------------------------------------------------------------------
def step2_execute_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[XGBRegressor, Dict[str, Any]]:
    """
    Khởi tạo và huấn luyện mô hình XGBRegressor với eval_set và early_stopping_rounds=50.
    """
    print("\n[BƯỚC 2] Bắt đầu huấn luyện mô hình XGBoost Regressor...")
    print(f" -> Cấu hình siêu tham số: {MODEL_CONFIG}")

    model = XGBRegressor(**MODEL_CONFIG)
    eval_set = [(X_train, y_train), (X_test, y_test)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=100
    )

    best_iteration = model.best_iteration if hasattr(model, "best_iteration") else model.n_estimators
    print(f"\n -> Huấn luyện hoàn tất! Best Iteration: {best_iteration}")

    # Lấy lịch sử mất mát
    evals_result = model.evals_result()
    train_rmse = evals_result["validation_0"]["rmse"]
    test_rmse = evals_result["validation_1"]["rmse"]

    # Lưu checkpoint model
    model_save_path = MODELS_DIR / "xgboost_baseline.json"
    model.save_model(str(model_save_path))
    print(f" -> Đã lưu model checkpoint vào: {model_save_path}")

    # Lưu bảng lịch sử huấn luyện
    history_df = pd.DataFrame({
        "Iteration": range(1, len(train_rmse) + 1),
        "Train_RMSE": train_rmse,
        "Test_RMSE": test_rmse
    })
    history_path = TABLES_DIR / "cv02_train_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f" -> Đã lưu lịch sử huấn luyện vào: {history_path}")

    return model, {
        "best_iteration": best_iteration,
        "history_df": history_df,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse
    }


# -------------------------------------------------------------------------------
# 4. BƯỚC 3: TRỰC QUAN HÓA LEARNING CURVE & BÁO CÁO HỘI TỤ
# -------------------------------------------------------------------------------
def step3_visualize(eval_info: Dict[str, Any], show: bool = False):
    """
    Vẽ đồ thị Learning Loss Curve (Train RMSE vs Test RMSE) qua các vòng lặp.
    """
    print("\n[BƯỚC 3] Trực quan hóa đường cong học tập (Learning Curve)...")
    train_rmse = eval_info["train_rmse"]
    test_rmse = eval_info["test_rmse"]
    best_iter = eval_info["best_iteration"]

    plt.figure(figsize=(10, 6))
    iterations = range(1, len(train_rmse) + 1)
    plt.plot(iterations, train_rmse, label="Train RMSE Loss", color="#1f77b4", lw=2)
    plt.plot(iterations, test_rmse, label="Test RMSE Loss (Validation)", color="#d62728", lw=2)
    plt.axvline(best_iter, color="#2ca02c", linestyle="--", lw=1.8, label=f"Điểm dừng tối ưu (Best Iter = {best_iter})")

    plt.title("Đường cong huấn luyện XGBoost (Learning Curve with Early Stopping)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Số vòng lặp (Iterations / Trees)", fontsize=11)
    plt.ylabel("Root Mean Squared Error (RMSE - MW)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=10)

    fig_path = FIGURES_DIR / "cv02_learning_curve.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f" -> Đã lưu biểu đồ Learning Curve vào: {fig_path}")

    if show:
        plt.show()
    plt.close()


# -------------------------------------------------------------------------------
# 5. BƯỚC 4: KẾT LUẬN & ĐÁNH GIÁ TIẾN TRÌNH HỌC
# -------------------------------------------------------------------------------
def step4_conclude_and_report(eval_info: Dict[str, Any]):
    """
    In tổng kết đánh giá mức độ hội tụ và tránh overfitting của mô hình.
    """
    best_iter = eval_info["best_iteration"]
    best_test_rmse = eval_info["test_rmse"][best_iter]
    best_train_rmse = eval_info["train_rmse"][best_iter]

    print("\n" + "=" * 80)
    print(" [CV02 - BÁO CÁO TỔNG KẾT TIẾN TRÌNH HUẤN LUYỆN XGBOOST]")
    print(f" • Điểm dừng tối ưu (Best Iteration): {best_iter} cây quyết định")
    print(f" • Sai số Train RMSE tại điểm tối ưu: {best_train_rmse:,.2f} MW")
    print(f" • Sai số Test RMSE tại điểm tối ưu:  {best_test_rmse:,.2f} MW")
    print(f" • Tỷ lệ sai lệch Train/Test:        {(best_test_rmse - best_train_rmse) / best_train_rmse * 100:.2f}%")
    print(" • Nhận xét: Early stopping phát huy hiệu quả bảo vệ mô hình khỏi hiện tượng")
    print("             quá khớp (Overfitting) trên dữ liệu chuỗi thời gian phụ tải.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="CV02: Huấn luyện mô hình cơ sở XGBoost")
    parser.add_argument("--show", action="store_true", help="Hiển thị biểu đồ sau khi vẽ")
    args = parser.parse_args()

    print("=" * 80)
    print(" [CV02 - NGỌC TRÂM] HUẤN LUYỆN MÔ HÌNH CƠ SỞ XGBOOST REGRESSOR")
    print("=" * 80)

    X_train, y_train, X_test, y_test = step1_load_data()
    model, eval_info = step2_execute_analysis(X_train, y_train, X_test, y_test)
    step3_visualize(eval_info, show=args.show)
    step4_conclude_and_report(eval_info)

    print("\n[THÀNH CÔNG] Nhiệm vụ CV02 hoàn thành 100%! Mô hình đã lưu tại outputs/models/.")


if __name__ == "__main__":
    main()
