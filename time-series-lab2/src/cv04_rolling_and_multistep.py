# -*- coding: utf-8 -*-
"""
===================================================================================
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 3: MACHINE LEARNING & XGBOOST
NHIỆM VỤ CV04: ROLLING WINDOW FEATURES & DỰ BÁO ĐA BƯỚC 24H (BÀI TẬP NÂNG CAO 1, 2)
===================================================================================

[THÔNG TIN DÀNH CHO THÀNH VIÊN & AI AGENT]:
- Mã công việc: CV04
- Thành viên phụ trách: Hoàng Hiệp
- Nhánh Git khuyến nghị: feat/cv04-rolling-multistep
- Tệp thực thi: src/cv04_rolling_and_multistep.py
- Dữ liệu đầu vào:
  + data/processed/train.csv và data/processed/test.csv (hoặc data/raw/AEP_hourly.csv)
- Sản phẩm đầu ra:
  + outputs/figures/cv04_rolling_features_impact.png (Biểu đồ minh họa đặc trưng Rolling Mean/Std)
  + outputs/figures/cv04_multistep_forecast.png (Biểu đồ dự báo đa bước 24 giờ tiếp theo)
  + outputs/tables/cv04_multistep_metrics.csv (Bảng sai số theo từng bước dự báo h=1..24)

[QUY TẮC KỸ THUẬT BẮT BUỘC CHO AI AGENT]:
1. CHỐNG RÒ RỈ DỮ LIỆU KHI TÍNH ROLLING (ANTI-LOOKAHEAD BIAS):
   - Tuyệt đối KHÔNG tính rolling trực tiếp: `df['AEP_MW'].rolling(w)` (gây rò rỉ dữ liệu bước t).
   - Bắt buộc phải tính trên chuỗi quá khứ đã trễ: `df['AEP_MW'].shift(1).rolling(w)`.
2. BÀI TOÁN DỰ BÁO ĐA BƯỚC (MULTI-STEP FORECASTING 24H):
   - Triển khai Chiến lược Đệ quy (Recursive Strategy): Lấy giá trị dự báo y_hat_{t+1} làm 
     đầu vào để cập nhật lại các đặc trưng lag, sau đó dự báo tiếp y_hat_{t+2} cho đến h=24.
   - Phân tích hiện tượng tích lũy sai số (Error Accumulation): Sai số MAE/RMSE sẽ tăng dần 
     khi bước dự báo h càng xa hiện tại.
===================================================================================
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Cấu hình encoding UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG (ĐỒNG BỘ CHO MỌI AGENT & THÀNH VIÊN)
# ---------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DATA = DATA_PROCESSED_DIR / "train.csv"
TEST_DATA = DATA_PROCESSED_DIR / "test.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "target_col": "AEP_MW",
    "rolling_windows": [24, 168],  # 24 giờ (1 ngày) và 168 giờ (1 tuần)
    "forecast_horizon": 24         # Dự báo 24 giờ tiếp theo
}


# ---------------------------------------------------------------------------------
# 2. BÀI TẬP NÂNG CAO 1: TẠO ĐẶC TRƯNG ROLLING MEAN & ROLLING STD
# ---------------------------------------------------------------------------------
def step1_create_rolling_features(
    df: pd.DataFrame, 
    target_col: str = "AEP_MW"
) -> Tuple[pd.DataFrame, Path]:
    """
    [TODO - HOÀNG HIỆP / AI AGENT TRIỂN KHAI]:
    - Tạo các đặc trưng trượt không rò rỉ:
      + rolling_mean_24 = df[target_col].shift(1).rolling(24).mean()
      + rolling_std_24  = df[target_col].shift(1).rolling(24).std()
      + rolling_mean_168 = df[target_col].shift(1).rolling(168).mean()
    - Loại bỏ NaN sinh ra do cửa sổ trượt.
    - Vẽ biểu đồ thể hiện dải dao động trượt outputs/figures/cv04_rolling_features_impact.png.
    """
    print("[CV04 - BƯỚC 1] Đang tạo đặc trưng cửa sổ trượt (Rolling Features)...")
    df_roll = df.copy()
    
    # TODO:
    # df_roll['rolling_mean_24'] = df_roll[target_col].shift(1).rolling(24).mean()
    # df_roll['rolling_std_24'] = df_roll[target_col].shift(1).rolling(24).std()
    # df_roll['rolling_mean_168'] = df_roll[target_col].shift(1).rolling(168).mean()
    # df_roll.dropna(inplace=True)
    
    fig_path = FIGURES_DIR / "cv04_rolling_features_impact.png"
    return df_roll, fig_path


# ---------------------------------------------------------------------------------
# 3. BÀI TẬP NÂNG CAO 2: DỰ BÁO ĐA BƯỚC 24H (MULTI-STEP AHEAD FORECASTING)
# ---------------------------------------------------------------------------------
def step2_multistep_recursive_forecast(
    df: pd.DataFrame, 
    horizon: int = 24, 
    target_col: str = "AEP_MW"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    [TODO - HOÀNG HIỆP / AI AGENT TRIỂN KHAI]:
    - Triển khai chiến lược dự báo đệ quy (Recursive Strategy) cho 24 giờ tiếp theo:
      + Bước h=1: Lấy các lags thực tế, dự báo y_hat_{t+1}.
      + Bước h=2: Cập nhật lag_1 = y_hat_{t+1}, dự báo tiếp y_hat_{t+2}, ..., cho tới h=24.
    - So sánh sai số MAE và RMSE theo từng bước từ 1 đến 24 để chứng minh hiện tượng tích lũy sai số.
    - Lưu kết quả bảng sai số vào outputs/tables/cv04_multistep_metrics.csv.
    - Vẽ biểu đồ đối chứng chuỗi dự báo đa bước outputs/figures/cv04_multistep_forecast.png.
    """
    print(f"[CV04 - BƯỚC 2] Khởi chạy dự báo đa bước (Horizon = {horizon} giờ)...")
    # TODO: Triển khai dự báo đệ quy (Recursive Multi-Step)
    
    df_preds = pd.DataFrame()
    df_metrics = pd.DataFrame()
    return df_preds, df_metrics


# ---------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH (MAIN ENTRYPOINT)
# ---------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("KHỞI CHẠY TIẾN TRÌNH CV04: ROLLING FEATURES & MULTI-STEP FORECAST (HOÀNG HIỆP)")
    print("=" * 80)
    
    if not TRAIN_DATA.exists():
        print(f"[CẢNH BÁO]: Chưa tìm thấy {TRAIN_DATA}. Vui lòng chạy CV01 trước.")
        return
        
    print("Nạp dữ liệu huấn luyện...")
    # TODO: Gọi step 1 và step 2
    
    print("[CV04 HOÀN TẤT: Đã hoàn thành các bài toán nâng cao về Rolling và Multi-step]")


if __name__ == "__main__":
    main()
