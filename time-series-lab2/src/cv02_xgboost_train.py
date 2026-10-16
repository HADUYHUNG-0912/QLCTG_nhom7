# -*- coding: utf-8 -*-
"""
===================================================================================
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 3: MACHINE LEARNING & XGBOOST
NHIỆM VỤ CV02: HUẤN LUYỆN MÔ HÌNH CƠ SỞ XGBOOST REGRESSOR (BƯỚC 4, BƯỚC 5)
===================================================================================

[THÔNG TIN DÀNH CHO THÀNH VIÊN & AI AGENT]:
- Mã công việc: CV02
- Thành viên phụ trách: Ngọc Trâm
- Nhánh Git khuyến nghị: feat/cv02-xgb-train
- Tệp thực thi: src/cv02_xgboost_train.py
- Dữ liệu đầu vào:
  + data/processed/train.csv (tập huấn luyện do CV01 cung cấp)
  + data/processed/test.csv  (tập kiểm thử do CV01 cung cấp)
- Sản phẩm đầu ra:
  + outputs/models/xgboost_baseline.json (Checkpoint mô hình XGBoost định dạng chuẩn)
  + outputs/figures/cv02_learning_curve.png (Biểu đồ hàm mất mát RMSE theo từng vòng lặp)
  + outputs/tables/cv02_train_history.csv (Lịch sử hội tụ loss qua các vòng lặp)

[QUY TẮC KỸ THUẬT BẮT BUỘC CHO AI AGENT]:
1. ĐỒNG BỘ ĐƯỜNG DẪN: Quản lý đường dẫn bằng `pathlib.Path` tính từ `PROJECT_ROOT`.
2. CHUẨN MỰC SIÊU THAM SỐ:
   - Mô hình: `xgboost.XGBRegressor`
   - Siêu tham số cơ bản: `n_estimators=1000`, `learning_rate=0.01`, `max_depth=5`,
     `early_stopping_rounds=50`, `objective='reg:squarederror'`, `random_state=42`.
3. THEO DÕI HỘI TỤ (EVALUATION MONITORING):
   - Bắt buộc truyền `eval_set=[(X_train, y_train), (X_test, y_test)]` vào phương thức `.fit()`.
   - Đánh giá trên độ đo `eval_metric='rmse'`.
4. LƯU TRỮ MÔ HÌNH:
   - Lưu model checkpoint định dạng `.json` (khuyến nghị chuẩn mực của XGBoost) để CV03 nạp lại suy luận.
===================================================================================
"""

import sys
import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any

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
TRAIN_DATA = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "target_col": "AEP_MW",
    "random_state": 42,
    "xgb_params": {
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "n_jobs": -1
    }
}


# ---------------------------------------------------------------------------------
# 2. BƯỚC 4: NẠP DỮ LIỆU & TÁCH MA TRẬN ĐẶC TRƯNG X - y
# ---------------------------------------------------------------------------------
def step1_load_train_test(
    train_path: Path, 
    test_path: Path, 
    target_col: str = "AEP_MW"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    [TODO - NGỌC TRÂM / AI AGENT TRIỂN KHAI]:
    - Đọc train.csv và test.csv từ data/processed/.
    - Thiết lập Datetime làm index nếu có cột thời gian.
    - Tách ma trận đặc trưng X (loại bỏ target_col) và nhãn y (target_col).
    """
    print(f"[CV02 - BƯỚC 1] Nạp dữ liệu train từ: {train_path}")
    print(f"[CV02 - BƯỚC 1] Nạp dữ liệu test từ:  {test_path}")
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Chưa tìm thấy train.csv hoặc test.csv! "
            "Vui lòng yêu cầu bạn Hà Duy Hưng (CV01) chạy 'src/cv01_data_and_lags.py' trước."
        )

    # TODO: Đọc dữ liệu và phân tách X, y
    # train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    # test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
    # X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    # X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    print(" -> [TODO] Phân tách ma trận X và vector y thành công.")
    return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()


# ---------------------------------------------------------------------------------
# 3. BƯỚC 5: HUẤN LUYỆN MÔ HÌNH XGBOOST REGRESSOR
# ---------------------------------------------------------------------------------
def step2_train_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    params: Dict[str, Any]
) -> Any:
    """
    [TODO - NGỌC TRÂM / AI AGENT TRIỂN KHAI]:
    - Khởi tạo mô hình `xgboost.XGBRegressor` với tham số cấu hình params.
    - Huấn luyện mô hình và truyền `eval_set=[(X_train, y_train), (X_test, y_test)]`.
    - Trích xuất lịch sử mất mát và lưu vào outputs/tables/cv02_train_history.csv.
    - Vẽ đồ thị Learning Loss Curve lưu vào outputs/figures/cv02_learning_curve.png.
    """
    print("[CV02 - BƯỚC 2] Bắt đầu huấn luyện mô hình XGBoost Regressor...")
    # TODO:
    # from xgboost import XGBRegressor
    # model = XGBRegressor(**params)
    # start_time = time.time()
    # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
    # print(f"Huấn luyện thành công trong {time.time() - start_time:.2f}s. Best iteration: {model.best_iteration}")
    return None


# ---------------------------------------------------------------------------------
# 4. LƯU MÔ HÌNH BÀN GIAO CHO CV03
# ---------------------------------------------------------------------------------
def step3_save_model(model: Any, output_path: Path) -> Path:
    """
    [TODO - NGỌC TRÂM / AI AGENT TRIỂN KHAI]:
    - Lưu checkpoint mô hình ra outputs/models/xgboost_baseline.json.
    """
    print(f"[CV02 - BƯỚC 3] Lưu trữ mô hình XGBoost tại: {output_path}")
    # TODO: model.save_model(output_path)
    return output_path


# ---------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH (MAIN ENTRYPOINT)
# ---------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("KHỞI CHẠY TIẾN TRÌNH CV02: HUẤN LUYỆN XGBOOST REGRESSOR (NGỌC TRÂM)")
    print("=" * 80)
    
    # 1. Nạp dữ liệu
    X_train, X_test, y_train, y_test = step1_load_train_test(
        TRAIN_DATA, TEST_DATA, target_col=CONFIG["target_col"]
    )
    
    # 2. Huấn luyện mô hình
    model = step2_train_xgboost(X_train, y_train, X_test, y_test, CONFIG["xgb_params"])
    
    # 3. Lưu mô hình phục vụ CV03
    model_checkpoint = MODELS_DIR / "xgboost_baseline.json"
    step3_save_model(model, model_checkpoint)
    
    print("[CV02 HOÀN TẤT: Mô hình đã được bàn giao cho CV03 (Thanh Phương) để đánh giá]")


if __name__ == "__main__":
    main()
