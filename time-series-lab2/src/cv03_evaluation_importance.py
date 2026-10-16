# -*- coding: utf-8 -*-
"""
===================================================================================
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 3: MACHINE LEARNING & XGBOOST
NHIỆM VỤ CV03: ĐÁNH GIÁ CHỈ SỐ ĐỊNH LƯỢNG & PHÂN TÍCH FEATURE IMPORTANCE (BƯỚC 6, 7)
===================================================================================

[THÔNG TIN DÀNH CHO THÀNH VIÊN & AI AGENT]:
- Mã công việc: CV03
- Thành viên phụ trách: Thanh Phương
- Nhánh Git khuyến nghị: feat/cv03-evaluation
- Tệp thực thi: src/cv03_evaluation_importance.py
- Dữ liệu đầu vào:
  + data/processed/test.csv (tập kiểm thử do CV01 cung cấp)
  + outputs/models/xgboost_baseline.json (mô hình đã huấn luyện do CV02 cung cấp)
- Sản phẩm đầu ra:
  + outputs/tables/cv03_evaluation_metrics.csv (Bảng chỉ số: MAE, RMSE, MAPE %, R2 Score)
  + outputs/tables/cv03_feature_importance.csv (Bảng xếp hạng độ quan trọng của các đặc trưng)
  + outputs/figures/cv03_actual_vs_predicted.png (Biểu đồ so sánh giá trị Thực tế vs Dự báo)
  + outputs/figures/cv03_feature_importance.png (Biểu đồ thanh trực quan hóa Feature Importance)

[QUY TẮC KỸ THUẬT BẮT BUỘC CHO AI AGENT]:
1. ĐỒNG BỘ ĐƯỜNG DẪN: Quản lý đường dẫn bằng `pathlib.Path` tương đối từ `PROJECT_ROOT`.
2. CHUẨN XÁC 4 ĐỘ ĐO SAI SỐ:
   - MAE  = mean(|y_true - y_pred|)
   - RMSE = sqrt(mean((y_true - y_pred)^2))
   - MAPE = mean(|(y_true - y_pred) / y_true|) * 100%
   - R2   = 1 - sum((y_true - y_pred)^2) / sum((y_true - y_mean)^2)
3. TRỰC QUAN HÓA THỰC TẾ VS DỰ BÁO:
   - Vẽ 2 khung hình (Subplots):
     + Khung 1: Toàn bộ tập kiểm thử để thấy xu hướng dài hạn.
     + Khung 2: Cửa sổ phóng to 1 tuần (168 giờ) để kiểm tra độ bám đỉnh/đáy trong chu kỳ ngày.
4. PHÂN TÍCH ĐẶC TRƯNG ĐỘ TRỄ:
   - Trích xuất `feature_importances_`, vẽ biểu đồ cột ngang giảm dần.
   - Giải thích: Độ trễ nào (lag 1h, lag 24h, hay lag 168h) mang tính quyết định nhất?
===================================================================================
"""

import sys
import os
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
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost_baseline.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "target_col": "AEP_MW",
    "zoom_hours": 168  # Cửa sổ phóng to 1 tuần = 168 giờ
}


# ---------------------------------------------------------------------------------
# 2. BƯỚC 6: SUY LUẬN TRÊN TẬP TEST VÀ TÍNH ĐỘ ĐO SAI SỐ
# ---------------------------------------------------------------------------------
def step1_evaluate_model(
    test_path: Path, 
    model_path: Path, 
    target_col: str = "AEP_MW"
) -> Tuple[pd.DataFrame, Dict[str, float], Any]:
    """
    [TODO - THANH PHƯƠNG / AI AGENT TRIỂN KHAI]:
    - Nạp tập test từ test_path.
    - Nạp mô hình XGBoost đã huấn luyện từ model_path.
    - Dự báo y_pred = model.predict(X_test).
    - Tính toán 4 chỉ số: MAE, RMSE, MAPE (%), R2 Score.
    - Lưu bảng chỉ số vào outputs/tables/cv03_evaluation_metrics.csv.
    """
    print(f"[CV03 - BƯỚC 1] Nạp mô hình từ {model_path} và dự báo tập test...")
    if not test_path.exists():
        raise FileNotFoundError(f"Chưa tìm thấy {test_path}! Cần chạy CV01 trước.")
    
    # TODO: Nạp model, dự báo và tính toán MAE, RMSE, MAPE, R2
    # from xgboost import XGBRegressor
    # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # model = XGBRegressor()
    # model.load_model(model_path)
    
    metrics = {
        "MAE (MW)": 0.0,
        "RMSE (MW)": 0.0,
        "MAPE (%)": 0.0,
        "R2 Score": 0.0
    }
    return pd.DataFrame(), metrics, None


# ---------------------------------------------------------------------------------
# 3. TRỰC QUAN HÓA ACTUAL VS PREDICTED
# ---------------------------------------------------------------------------------
def step2_plot_actual_vs_predicted(
    df_results: pd.DataFrame, 
    output_fig: Path, 
    zoom_hours: int = 168
) -> None:
    """
    [TODO - THANH PHƯƠNG / AI AGENT TRIỂN KHAI]:
    - Vẽ 2 đồ thị con (Subplots):
        + Subplot 1: Toàn bộ tập kiểm thử (Actual vs Predicted)
        + Subplot 2: Phóng to zoom_hours (1 tuần gần nhất)
    - Lưu đồ thị vào outputs/figures/cv03_actual_vs_predicted.png.
    """
    print(f"[CV03 - BƯỚC 2] Vẽ đồ thị đối chứng Actual vs Predicted tại: {output_fig}")
    # TODO: Trực quan hóa kết quả dự báo


# ---------------------------------------------------------------------------------
# 4. BƯỚC 7: TRÍCH XUẤT VÀ PHÂN TÍCH FEATURE IMPORTANCE
# ---------------------------------------------------------------------------------
def step3_feature_importance_analysis(
    model: Any, 
    feature_names: list, 
    output_fig: Path, 
    output_table: Path
) -> pd.DataFrame:
    """
    [TODO - THANH PHƯƠNG / AI AGENT TRIỂN KHAI]:
    - Lấy trọng số đặc trưng: model.feature_importances_.
    - Sắp xếp giảm dần, xuất bảng xếp hạng ra outputs/tables/cv03_feature_importance.csv.
    - Vẽ đồ thị thanh ngang (horizontal bar chart) lưu tại outputs/figures/cv03_feature_importance.png.
    - Phân tích: Lag nào (1h, 24h, 168h) giữ vai trò chủ đạo trong dự báo?
    """
    print(f"[CV03 - BƯỚC 3] Phân tích Feature Importance tại: {output_fig}")
    # TODO: Trích xuất và vẽ đồ thị độ quan trọng
    return pd.DataFrame()


# ---------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH (MAIN ENTRYPOINT)
# ---------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("KHỞI CHẠY TIẾN TRÌNH CV03: ĐÁNH GIÁ CHỈ SỐ & FEATURE IMPORTANCE (THANH PHƯƠNG)")
    print("=" * 80)
    
    # 1. Đánh giá sai số
    df_results, metrics, model = step1_evaluate_model(TEST_DATA, MODEL_PATH, CONFIG["target_col"])
    
    # 2. Vẽ đồ thị Actual vs Predicted
    fig_actual_pred = FIGURES_DIR / "cv03_actual_vs_predicted.png"
    step2_plot_actual_vs_predicted(df_results, fig_actual_pred, zoom_hours=CONFIG["zoom_hours"])
    
    # 3. Phân tích Feature Importance
    fig_importance = FIGURES_DIR / "cv03_feature_importance.png"
    table_importance = TABLES_DIR / "cv03_feature_importance.csv"
    step3_feature_importance_analysis(model, [], fig_importance, table_importance)
    
    print("[CV03 HOÀN TẤT: Đã xuất bảng chỉ số sai số và biểu đồ Feature Importance]")


if __name__ == "__main__":
    main()
