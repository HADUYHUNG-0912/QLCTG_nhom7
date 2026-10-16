# -*- coding: utf-8 -*-
"""
===================================================================================
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 3: MACHINE LEARNING & XGBOOST
NHIỆM VỤ CV05: TINH CHỈNH TIMESERIESSPLIT & SO SÁNH RANDOM FOREST (BÀI TẬP NÂNG CAO 3, 4)
===================================================================================

[THÔNG TIN DÀNH CHO THÀNH VIÊN & AI AGENT]:
- Mã công việc: CV05
- Thành viên phụ trách: Hoàng
- Nhánh Git khuyến nghị: feat/cv05-tuning-benchmark
- Tệp thực thi: src/cv05_tuning_random_forest.py
- Dữ liệu đầu vào:
  + data/processed/train.csv và data/processed/test.csv
- Sản phẩm đầu ra:
  + outputs/tables/cv05_best_hyperparameters.csv (Bộ siêu tham số tối ưu từ TimeSeriesSplit)
  + outputs/tables/cv05_benchmark_summary.csv (Bảng so sánh chi tiết XGBoost vs Random Forest)
  + outputs/figures/cv05_benchmark_comparison.png (Đồ thị so sánh MAE, RMSE, Thời gian huấn luyện)
  + outputs/models/rf_benchmark.joblib (Checkpoint mô hình Random Forest đối chứng)

[QUY TẮC KỸ THUẬT BẮT BUỘC CHO AI AGENT]:
1. KIỂM ĐỊNH CHÉO CHUỖI THỜI GIAN (TIME SERIES CROSS-VALIDATION):
   - Tuyệt đối KHÔNG sử dụng KFold ngẫu nhiên (`KFold(shuffle=True)`).
   - BẮT BUỘC dùng `TimeSeriesSplit(n_splits=5)` từ `sklearn.model_selection` để mở rộng 
     cửa sổ huấn luyện tuần tự (Expanding Window Validation).
2. SO SÁNH ĐỐI CHỨNG VỚI RANDOM FOREST:
   - Huấn luyện `RandomForestRegressor` trên cùng không gian đặc trưng và nhãn.
   - Đo lường và lập bảng so sánh đối chứng theo 4 tiêu chí:
     + Thời gian huấn luyện (Training Duration - s).
     + Tốc độ suy luận (Inference Speed - ms).
     + Sai số dự báo (MAE, RMSE, MAPE %, R2 Score).
     + Đánh giá nguy cơ quá khớp (Overfitting Assessment).
===================================================================================
"""

import sys
import os
import time
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
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG (ĐỒNG BỘ CHO MỌI AGENT & THÀNH VIÊN)
# ---------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DATA = DATA_PROCESSED_DIR / "train.csv"
TEST_DATA = DATA_PROCESSED_DIR / "test.csv"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "target_col": "AEP_MW",
    "cv_splits": 5,
    "random_state": 42,
    "param_grid": {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [300, 500],
        "subsample": [0.8, 1.0]
    }
}


# ---------------------------------------------------------------------------------
# 2. BÀI TẬP NÂNG CAO 3: TINH CHỈNH SIÊU THAM SỐ VỚI TIMESERIESSPLIT
# ---------------------------------------------------------------------------------
def step1_tune_timeseriessplit(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    param_grid: Dict[str, Any], 
    n_splits: int = 5
) -> Tuple[Any, Dict[str, Any]]:
    """
    [TODO - HOÀNG / AI AGENT TRIỂN KHAI]:
    - Khởi tạo TimeSeriesSplit(n_splits=n_splits).
    - Thiết lập GridSearchCV hoặc RandomizedSearchCV cho XGBRegressor.
    - Tìm kiếm siêu tham số tối ưu (best_params_) và xuất ra outputs/tables/cv05_best_hyperparameters.csv.
    """
    print(f"[CV05 - BƯỚC 1] Tinh chỉnh siêu tham số bằng TimeSeriesSplit ({n_splits} folds)...")
    # TODO:
    # from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    # from xgboost import XGBRegressor
    # tscv = TimeSeriesSplit(n_splits=n_splits)
    # search = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    best_params = {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 500}
    return None, best_params


# ---------------------------------------------------------------------------------
# 3. BÀI TẬP NÂNG CAO 4: HUẤN LUYỆN VÀ SO SÁNH VỚI RANDOM FOREST
# ---------------------------------------------------------------------------------
def step2_benchmark_random_forest(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> pd.DataFrame:
    """
    [TODO - HOÀNG / AI AGENT TRIỂN KHAI]:
    - Huấn luyện mô hình RandomForestRegressor trên cùng tập train.
    - Đo lường Training Time và Inference Latency.
    - Tính toán MAE, RMSE, MAPE, R2 Score trên tập test.
    - Lưu checkpoint tại outputs/models/rf_benchmark.joblib.
    - Xuất bảng so sánh đối chứng ra outputs/tables/cv05_benchmark_summary.csv.
    - Vẽ biểu đồ cột trực quan outputs/figures/cv05_benchmark_comparison.png.
    """
    print("[CV05 - BƯỚC 2] Huấn luyện mô hình đối chứng RandomForestRegressor...")
    # TODO: Huấn luyện Random Forest và đo lường
    
    df_comparison = pd.DataFrame({
        "Model": ["XGBoost Regressor", "Random Forest Regressor"],
        "MAE (MW)": [0.0, 0.0],
        "RMSE (MW)": [0.0, 0.0],
        "MAPE (%)": [0.0, 0.0],
        "R2 Score": [0.0, 0.0],
        "Train Time (s)": [0.0, 0.0]
    })
    return df_comparison


# ---------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH (MAIN ENTRYPOINT)
# ---------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("KHỞI CHẠY TIẾN TRÌNH CV05: TINH CHỈNH TIMESERIESSPLIT & BENCHMARK RF (HOÀNG)")
    print("=" * 80)
    
    if not TRAIN_DATA.exists():
        print(f"[CẢNH BÁO]: Chưa tìm thấy {TRAIN_DATA}. Vui lòng chạy CV01 trước.")
        return
        
    print("Nạp dữ liệu huấn luyện...")
    # TODO: Gọi step 1 và step 2
    
    print("[CV05 HOÀN TẤT: Đã hoàn thành tinh chỉnh siêu tham số và đối sánh Random Forest]")


if __name__ == "__main__":
    main()
