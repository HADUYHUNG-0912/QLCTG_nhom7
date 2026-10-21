# -*- coding: utf-8 -*-
"""CV03: đánh giá XGBoost baseline và phân tích độ quan trọng đặc trưng.

Nhận ``test.csv`` từ CV01 và ``xgboost_baseline.json`` từ CV02; xuất hai
bảng CSV cùng hai biểu đồ theo hợp đồng bàn giao của nhóm.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost_baseline.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

CONFIG = {"target_col": "AEP_MW", "zoom_hours": 168}


def _load_test_data(test_path: Path, target_col: str) -> pd.DataFrame:
    """Đọc tập test của CV01 và kiểm tra contract thời gian/schema."""
    if not test_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {test_path}. Cần chạy src/cv01_data_and_lags.py trước."
        )
    test_df = pd.read_csv(test_path)
    if test_df.empty:
        raise ValueError(f"Tập kiểm thử {test_path.name} đang rỗng.")
    if "Datetime" not in test_df.columns:
        raise KeyError(f"{test_path.name} thiếu cột Datetime do CV01 bàn giao.")

    datetime_values = pd.to_datetime(test_df.pop("Datetime"), errors="coerce")
    if datetime_values.isna().any():
        invalid_count = int(datetime_values.isna().sum())
        raise ValueError(f"{test_path.name} có {invalid_count} mốc thời gian không hợp lệ.")
    test_df.index = pd.DatetimeIndex(datetime_values, name="Datetime")
    if test_df.index.has_duplicates:
        raise ValueError(f"{test_path.name} có mốc thời gian bị trùng lặp.")
    if not test_df.index.is_monotonic_increasing:
        raise ValueError(f"{test_path.name} chưa được sắp xếp tăng dần theo thời gian.")
    if target_col not in test_df.columns:
        raise KeyError(f"Không tìm thấy cột mục tiêu '{target_col}' trong {test_path.name}.")

    non_numeric = list(test_df.select_dtypes(exclude=[np.number]).columns)
    if non_numeric:
        raise TypeError(f"Tập test còn cột không phải số: {non_numeric}.")
    if not np.isfinite(test_df.to_numpy(dtype=float)).all():
        raise ValueError("Tập test chứa NaN hoặc giá trị vô hạn.")
    return test_df


def _align_features_with_model(model: Any, X_test: pd.DataFrame) -> pd.DataFrame:
    """Khớp chính xác tên và thứ tự feature với checkpoint CV02."""
    model_features = model.get_booster().feature_names
    if not model_features:
        raise ValueError("Checkpoint không lưu tên đặc trưng; không thể kiểm tra schema.")
    missing = sorted(set(model_features) - set(X_test.columns))
    extra = sorted(set(X_test.columns) - set(model_features))
    if missing or extra:
        raise ValueError(
            "Schema tập test không khớp checkpoint CV02. "
            f"Thiếu: {missing}; dư: {extra}."
        )
    return X_test.loc[:, model_features]


def step1_evaluate_model(
    test_path: Path,
    model_path: Path,
    target_col: str = "AEP_MW",
) -> Tuple[pd.DataFrame, Dict[str, float], Any]:
    """Nạp checkpoint, dự báo test và xuất MAE/RMSE/MAPE/R2."""
    print(f"[CV03 - BƯỚC 1] Nạp mô hình từ {model_path} và dự báo tập test...")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {model_path}. Cần chạy src/cv02_xgboost_train.py trước."
        )
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "Chưa cài đặt xgboost. Hãy chạy 'pip install -r requirements.txt'."
        ) from exc

    test_df = _load_test_data(test_path, target_col)
    y_true = test_df[target_col].astype(float)
    X_test = test_df.drop(columns=[target_col])

    model = XGBRegressor()
    model.load_model(str(model_path))
    X_test = _align_features_with_model(model, X_test)
    y_pred = np.asarray(model.predict(X_test), dtype=float)
    if y_pred.shape != (len(y_true),):
        raise ValueError(f"Kích thước dự báo {y_pred.shape} không khớp target.")
    if not np.isfinite(y_pred).all():
        raise ValueError("Kết quả dự báo chứa NaN hoặc giá trị vô hạn.")

    actual = y_true.to_numpy(dtype=float)
    if (actual == 0).any():
        raise ValueError("MAPE không xác định vì y_true chứa giá trị 0.")
    errors = actual - y_pred
    residual_sum_squares = float(np.sum(np.square(errors)))
    total_sum_squares = float(np.sum(np.square(actual - float(actual.mean()))))
    if total_sum_squares == 0:
        raise ValueError("R2 không xác định vì y_true không có phương sai.")

    metrics = {
        "MAE (MW)": float(np.mean(np.abs(errors))),
        "RMSE (MW)": float(np.sqrt(np.mean(np.square(errors)))),
        "MAPE (%)": float(np.mean(np.abs(errors / actual)) * 100),
        "R2 Score": float(1 - residual_sum_squares / total_sum_squares),
    }
    metrics_table = pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "MAPE", "R2"],
            "value": list(metrics.values()),
            "unit": ["MW", "MW", "%", "score"],
        }
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = TABLES_DIR / "cv03_evaluation_metrics.csv"
    metrics_table.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    df_results = pd.DataFrame(
        {"Actual": actual, "Predicted": y_pred}, index=y_true.index
    )
    print(f" -> Đã dự báo {len(df_results):,} quan sát và lưu: {metrics_path}")
    for metric_name, metric_value in metrics.items():
        print(f" -> {metric_name}: {metric_value:,.4f}")
    return df_results, metrics, model


def step2_plot_actual_vs_predicted(
    df_results: pd.DataFrame,
    output_fig: Path,
    zoom_hours: int = 168,
) -> None:
    """Vẽ toàn bộ tập test và cửa sổ một tuần gần nhất."""
    print(f"[CV03 - BƯỚC 2] Vẽ Actual vs Predicted tại: {output_fig}")
    if df_results.empty or not {"Actual", "Predicted"}.issubset(df_results.columns):
        raise ValueError("Kết quả dự báo rỗng hoặc thiếu cột Actual/Predicted.")
    if zoom_hours <= 0:
        raise ValueError("zoom_hours phải là số nguyên dương.")
    output_fig.parent.mkdir(parents=True, exist_ok=True)

    colors = {"actual": "#2563a9", "predicted": "#e56b2f"}
    surface, grid = "#fcfcfb", "#deddd7"
    with plt.rc_context({"font.family": ["Segoe UI", "Arial", "DejaVu Sans"]}):
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor=surface)
        for ax in axes:
            ax.set_facecolor(surface)
            ax.grid(axis="y", color=grid, linewidth=0.8)
            ax.set_ylabel("Công suất (MW)")

        axes[0].plot(
            df_results.index, df_results["Actual"], color=colors["actual"],
            linewidth=0.8, label="Thực tế"
        )
        axes[0].plot(
            df_results.index, df_results["Predicted"], color=colors["predicted"],
            linewidth=0.8, alpha=0.85, label="Dự báo"
        )
        axes[0].set_title("XGBoost baseline: toàn bộ tập kiểm thử", fontweight="semibold")
        axes[0].legend(frameon=False, ncol=2)

        zoom = df_results.iloc[-min(zoom_hours, len(df_results)):]
        axes[1].plot(
            zoom.index, zoom["Actual"], color=colors["actual"], linewidth=1.8,
            marker="o", markersize=2, label="Thực tế"
        )
        axes[1].plot(
            zoom.index, zoom["Predicted"], color=colors["predicted"], linewidth=1.6,
            linestyle="--", label="Dự báo"
        )
        axes[1].set_title(
            f"Phóng to {len(zoom)} giờ cuối (chu kỳ tuần)", fontweight="semibold"
        )
        axes[1].set_xlabel("Thời gian")
        axes[1].legend(frameon=False, ncol=2)
        fig.suptitle(
            "So sánh mức tiêu thụ điện thực tế và dự báo",
            fontsize=16, fontweight="bold"
        )
        fig.tight_layout()
        fig.savefig(output_fig, dpi=150, bbox_inches="tight", facecolor=surface)
        plt.close(fig)
    print(f" -> Đã lưu biểu đồ: {output_fig}")


def step3_feature_importance_analysis(
    model: Any,
    feature_names: Sequence[str],
    output_fig: Path,
    output_table: Path,
) -> pd.DataFrame:
    """Xuất xếp hạng importance và xác định lag chủ đạo."""
    print(f"[CV03 - BƯỚC 3] Phân tích Feature Importance tại: {output_fig}")
    if model is None or not hasattr(model, "feature_importances_"):
        raise TypeError("Mô hình không hỗ trợ feature_importances_.")
    importances = np.asarray(model.feature_importances_, dtype=float)
    feature_names = list(feature_names) or list(model.get_booster().feature_names or [])
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Số tên đặc trưng ({len(feature_names)}) không khớp "
            f"số importance ({len(importances)})."
        )
    if not np.isfinite(importances).all() or (importances < 0).any():
        raise ValueError("Feature importance chứa giá trị không hợp lệ.")

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(["importance", "feature"], ascending=[False, True], ignore_index=True)
    importance_df.insert(0, "rank", np.arange(1, len(importance_df) + 1))
    importance_df["feature_type"] = np.where(
        importance_df["feature"].str.startswith("lag_"), "lag", "calendar"
    )
    output_table.parent.mkdir(parents=True, exist_ok=True)
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_table, index=False, encoding="utf-8-sig")

    plot_df = importance_df.sort_values("importance", ascending=True)
    bar_colors = np.where(plot_df["feature_type"].eq("lag"), "#2563a9", "#e6a04b")
    with plt.rc_context({"font.family": ["Segoe UI", "Arial", "DejaVu Sans"]}):
        fig_height = max(5.5, 0.42 * len(plot_df) + 1.8)
        fig, ax = plt.subplots(figsize=(10.5, fig_height), facecolor="#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        bars = ax.barh(plot_df["feature"], plot_df["importance"], color=bar_colors)
        ax.set_title(
            "Độ quan trọng đặc trưng của XGBoost baseline",
            fontsize=15, fontweight="semibold", pad=14
        )
        ax.set_xlabel("Feature importance (gain chuẩn hóa)")
        ax.set_ylabel("Đặc trưng")
        ax.grid(axis="x", color="#deddd7", linewidth=0.8)
        ax.set_axisbelow(True)
        offset = max(float(importances.max()), 1e-9) * 0.008
        for bar, value in zip(bars, plot_df["importance"]):
            ax.text(
                bar.get_width() + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", fontsize=8
            )
        fig.tight_layout()
        fig.savefig(output_fig, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(fig)

    lag_ranking = importance_df[importance_df["feature_type"].eq("lag")]
    if lag_ranking.empty:
        print(" -> Checkpoint không có đặc trưng lag để xác định lag chủ đạo.")
    else:
        dominant = lag_ranking.iloc[0]
        print(
            f" -> Lag chủ đạo: {dominant['feature']} "
            f"(importance={dominant['importance']:.4f})."
        )
    print(f" -> Đã lưu bảng xếp hạng: {output_table}")
    print(f" -> Đã lưu biểu đồ:       {output_fig}")
    return importance_df


def main() -> None:
    print("=" * 80)
    print("KHỞI CHẠY CV03: ĐÁNH GIÁ & FEATURE IMPORTANCE (HOÀNG)")
    print("=" * 80)
    df_results, _, model = step1_evaluate_model(
        TEST_DATA, MODEL_PATH, CONFIG["target_col"]
    )
    step2_plot_actual_vs_predicted(
        df_results,
        FIGURES_DIR / "cv03_actual_vs_predicted.png",
        zoom_hours=CONFIG["zoom_hours"],
    )
    step3_feature_importance_analysis(
        model,
        model.get_booster().feature_names or [],
        FIGURES_DIR / "cv03_feature_importance.png",
        TABLES_DIR / "cv03_feature_importance.csv",
    )
    print("[CV03 HOÀN TẤT: Đã xuất đủ 2 bảng và 2 biểu đồ]")


if __name__ == "__main__":
    main()
