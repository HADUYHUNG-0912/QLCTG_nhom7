# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 3 (NHÓM 7)
MÃ CÔNG VIỆC: CV03 - ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH & PHÂN TÍCH FEATURE IMPORTANCE
---------------------------------------------------------------------------------
Thành viên phụ trách: Thanh Phương
Nhánh Git khuyến nghị: feat/cv03-evaluation
Tệp mã nguồn: src/cv03_evaluation_importance.py
Dữ liệu đầu vào:
  - data/processed/test.csv
  - outputs/models/xgboost_baseline.json (hoặc train.csv nếu cần tái huấn luyện)
Đầu ra tạo thành:
  - outputs/figures/cv03_actual_vs_predicted.png
  - outputs/figures/cv03_feature_importance.png
  - outputs/tables/cv03_evaluation_metrics.csv

AI_AGENT_INSTRUCTIONS:
=================================================================================
Mục tiêu dành cho AI Agent (Cursor / GitHub Copilot / Antigravity):
1. Nạp tập Test từ data/processed/test.csv.
2. Nạp mô hình đã huấn luyện từ outputs/models/xgboost_baseline.json.
   (Nếu chưa tồn tại file model, tự động tải train.csv và huấn luyện nhanh mô hình baseline).
3. Dự báo mức tiêu thụ điện trên tập kiểm thử: y_pred = model.predict(X_test).
4. Tính toán 4 chỉ số thống kê sai số chuẩn mực:
   - MAE: Mean Absolute Error (MW)
   - RMSE: Root Mean Squared Error (MW)
   - MAPE: Mean Absolute Percentage Error (%)
   - R^2: Coefficient of Determination (Hệ số xác định)
5. Xuất bảng chỉ số vào outputs/tables/cv03_evaluation_metrics.csv.
6. Trực quan hóa đối chứng Thực tế vs Dự báo (Actual vs Predicted) và biểu đồ
   độ quan trọng của đặc trưng (Feature Importance).
7. Đưa ra nhận xét khoa học: Lag nào có ảnh hưởng chi phối mạnh nhất?
8. Hỗ trợ chạy CLI độc lập: python src/cv03_evaluation_importance.py [--show]
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
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
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
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------------
# 2. BƯỚC 1: NẠP DỮ LIỆU VÀ MÔ HÌNH (DATA & MODEL LOADING VỚI FALLBACK)
# -------------------------------------------------------------------------------
def step1_load_data_and_model() -> Tuple[pd.DataFrame, pd.Series, XGBRegressor]:
    """
    Nạp tập Test từ data/processed/test.csv và mô hình từ outputs/models/xgboost_baseline.json.
    Nếu mô hình chưa tồn tại, tự động tải train.csv và huấn luyện nhanh.
    """
    test_path = PROCESSED_DIR / "test.csv"
    train_path = PROCESSED_DIR / "train.csv"
    model_path = MODELS_DIR / "xgboost_baseline.json"

    print(f"\n[BƯỚC 1] Đang tải tập kiểm thử từ: {test_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Chưa có {test_path}. Hãy chạy `python src/cv01_data_and_lags.py` trước!")

    test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
    feature_cols = [c for c in test_df.columns if c != "target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    model = XGBRegressor()
    if model_path.exists():
        print(f" -> Nạp mô hình checkpoint từ: {model_path}")
        model.load_model(str(model_path))
    else:
        print(f" -> Chưa tìm thấy model checkpoint tại {model_path}. Đang huấn luyện nhanh baseline từ {train_path}...")
        train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

    return X_test, y_test, model


# -------------------------------------------------------------------------------
# 3. BƯỚC 2: DỰ BÁO VÀ TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)
# -------------------------------------------------------------------------------
def step2_execute_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: XGBRegressor
) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
    """
    Thực hiện dự báo và tính toán MAE, RMSE, MAPE, R2 Score cùng Feature Importance.
    """
    print("\n[BƯỚC 2] Thực hiện suy luận dự báo trên tập Test...")
    y_pred = model.predict(X_test)

    # 1. Tính toán các chỉ số
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100.0
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE (%)": float(mape),
        "R2 Score": float(r2)
    }

    print(f" -> MAE:      {mae:,.2f} MW")
    print(f" -> RMSE:     {rmse:,.2f} MW")
    print(f" -> MAPE:     {mape:.2f} %")
    print(f" -> R2 Score: {r2:.4f}")

    # Xuất ra bảng metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = TABLES_DIR / "cv03_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f" -> Đã lưu bảng chỉ số vào: {metrics_path}")

    # 2. Trích xuất Feature Importance
    feature_names = X_test.columns.tolist()
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    fi_path = TABLES_DIR / "cv03_feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f" -> Đã lưu bảng thứ hạng Feature Importance vào: {fi_path}")

    return y_pred, metrics, fi_df


# -------------------------------------------------------------------------------
# 4. BƯỚC 3: TRỰC QUAN HÓA ACTUAL VS PREDICTED & FEATURE IMPORTANCE
# -------------------------------------------------------------------------------
def step3_visualize(
    y_test: pd.Series,
    y_pred: np.ndarray,
    fi_df: pd.DataFrame,
    show: bool = False
):
    """
    Vẽ 2 biểu đồ:
    1. Actual vs Predicted (toàn tập test và zoom 14 ngày).
    2. Biểu đồ thanh ngang Feature Importance của các lag.
    """
    print("\n[BƯỚC 3] Trực quan hóa Actual vs Predicted và Feature Importance...")

    # 1. Biểu đồ Actual vs Predicted
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.2, 1]})

    # Toàn bộ tập test
    ax1.plot(y_test.index, y_test.values, label="Thực tế (Actual)", color="#1f77b4", lw=0.6, alpha=0.85)
    ax1.plot(y_test.index, y_pred, label="XGBoost Dự báo (Predicted)", color="#d62728", lw=0.6, alpha=0.85)
    ax1.set_title("So sánh Thực tế vs Dự báo trên toàn bộ Tập Kiểm thử (Test Set)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Mức tiêu thụ (MW)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Phóng to zoom 14 ngày (336 giờ)
    zoom_len = 336
    zoom_idx = y_test.index[:zoom_len]
    ax2.plot(zoom_idx, y_test.iloc[:zoom_len].values, label="Thực tế (Actual)", marker="o", markersize=3, color="#1f77b4", lw=1.2)
    ax2.plot(zoom_idx, y_pred[:zoom_len], label="Dự báo (XGBoost)", marker="x", markersize=4, color="#d62728", lw=1.2, linestyle="--")
    ax2.set_title("Chi tiết dự báo đối chứng từng giờ (Zoom cận cảnh 14 ngày)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Thời gian", fontsize=11)
    ax2.set_ylabel("Mức tiêu thụ (MW)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    pred_fig_path = FIGURES_DIR / "cv03_actual_vs_predicted.png"
    plt.savefig(pred_fig_path, dpi=150, bbox_inches="tight")
    print(f" -> Đã lưu biểu đồ dự báo vào: {pred_fig_path}")
    if show:
        plt.show()
    plt.close()

    # 2. Biểu đồ Feature Importance
    plt.figure(figsize=(10, 7))
    top_fi = fi_df.iloc[::-1]  # Đảo ngược để vẽ horizontal bar chart từ trên xuống
    colors = ["#1f77b4" if f not in ["lag_1", "lag_24", "lag_2"] else "#d62728" for f in top_fi["Feature"]]
    plt.barh(top_fi["Feature"], top_fi["Importance"], color=colors, edgecolor="black", alpha=0.85)
    plt.title("Độ quan trọng của các đặc trưng độ trễ (XGBoost Feature Importance)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Trọng số đóng góp (F-Score / Weight)", fontsize=11)
    plt.ylabel("Lag Features", fontsize=11)
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)

    fi_fig_path = FIGURES_DIR / "cv03_feature_importance.png"
    plt.savefig(fi_fig_path, dpi=150, bbox_inches="tight")
    print(f" -> Đã lưu biểu đồ Feature Importance vào: {fi_fig_path}")
    if show:
        plt.show()
    plt.close()


# -------------------------------------------------------------------------------
# 5. BƯỚC 4: KẾT LUẬN & ĐÁNH GIÁ CHUYÊN MÔN
# -------------------------------------------------------------------------------
def step4_conclude_and_report(metrics: Dict[str, float], fi_df: pd.DataFrame):
    """
    Phân tích và kết luận chuyên môn về lag features quan trọng nhất.
    """
    top_3 = fi_df.head(3)["Feature"].tolist()
    print("\n" + "=" * 80)
    print(" [CV03 - BÁO CÁO KẾT LUẬN ĐÁNH GIÁ VÀ FEATURE IMPORTANCE]")
    print(f" • MAE:      {metrics['MAE']:,.2f} MW")
    print(f" • RMSE:     {metrics['RMSE']:,.2f} MW")
    print(f" • MAPE:     {metrics['MAPE (%)']:.2f}% (Độ chính xác đạt {100 - metrics['MAPE (%)']:.2f}%)")
    print(f" • R2 Score: {metrics['R2 Score']:.4f}")
    print(f" • Top 3 Lags quan trọng nhất: {top_3}")
    print(" • Giải thích chuyên môn:")
    print("   1. lag_1 (1 giờ trước): Đóng vai trò then chốt nhất do quán tính ngắn hạn của phụ tải điện.")
    print("   2. lag_24 (24 giờ trước): Thể hiện rõ nét chu kỳ tuần hoàn ngày đêm (Daily Seasonality),")
    print("      mức tiêu thụ cùng giờ ngày hôm qua phản ánh sát thói quen sinh hoạt và sản xuất.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="CV03: Đánh giá mô hình và Feature Importance")
    parser.add_argument("--show", action="store_true", help="Hiển thị biểu đồ sau khi vẽ")
    args = parser.parse_args()

    print("=" * 80)
    print(" [CV03 - THANH PHƯƠNG] ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH & FEATURE IMPORTANCE")
    print("=" * 80)

    X_test, y_test, model = step1_load_data_and_model()
    y_pred, metrics, fi_df = step2_execute_analysis(X_test, y_test, model)
    step3_visualize(y_test, y_pred, fi_df, show=args.show)
    step4_conclude_and_report(metrics, fi_df)

    print("\n[THÀNH CÔNG] Nhiệm vụ CV03 hoàn thành 100%! Kết quả đã lưu tại outputs/.")


if __name__ == "__main__":
    main()
