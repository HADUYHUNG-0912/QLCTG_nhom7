# -*- coding: utf-8 -*-
"""
CV05 — Hiệp: Tinh chỉnh siêu tham số bằng TimeSeriesSplit & so sánh với Random Forest.

Đầu vào : data/processed/train.csv, test.csv (từ CV01)
Đầu ra  : outputs/tables/cv05_best_hyperparameters.csv
          outputs/tables/cv05_benchmark_summary.csv
          outputs/figures/cv05_benchmark_comparison.png
          outputs/models/rf_benchmark.joblib
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Cấu hình encoding UTF-8 cho console Windows
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from xgboost import XGBRegressor

# ---------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG
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
        "subsample": [0.8, 1.0],
    },
    "rf_params": {
        "n_estimators": 300,
        "max_depth": None,
        "n_jobs": -1,
        "random_state": 42,
    },
}


# ---------------------------------------------------------------------------------
# 1b. NẠP DỮ LIỆU & TÁCH X - y
# ---------------------------------------------------------------------------------
def load_train_test(
    train_path: Path,
    test_path: Path,
    target_col: str = "AEP_MW",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Nạp train/test từ CV01, kiểm tra hợp đồng dữ liệu rồi tách X và y."""
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {train_path}. Cần chạy src/cv01_data_and_lags.py trước."
        )
    if not test_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {test_path}. Cần chạy src/cv01_data_and_lags.py trước."
        )

    frames = {}
    for name, path in (("train", train_path), ("test", test_path)):
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Tập {name} đang rỗng.")
        if "Datetime" not in df.columns:
            raise KeyError(f"{path.name} thiếu cột Datetime do CV01 bàn giao.")
        parsed = pd.to_datetime(df.pop("Datetime"), errors="coerce")
        if parsed.isna().any():
            invalid_count = int(parsed.isna().sum())
            raise ValueError(f"{path.name} có {invalid_count} mốc thời gian không hợp lệ.")
        df.index = pd.DatetimeIndex(parsed, name="Datetime")
        if df.index.has_duplicates:
            raise ValueError(f"{path.name} có mốc thời gian bị trùng lặp.")
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"{path.name} chưa được sắp xếp tăng dần theo thời gian.")
        if target_col not in df.columns:
            raise KeyError(f"Không tìm thấy cột mục tiêu '{target_col}' trong {path.name}.")
        non_numeric = list(df.select_dtypes(exclude=[np.number]).columns)
        if non_numeric:
            raise TypeError(f"{path.name} còn cột không phải số: {non_numeric}.")
        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name} chứa NaN hoặc giá trị vô hạn.")
        frames[name] = df

    train_df, test_df = frames["train"], frames["test"]
    train_features = [col for col in train_df.columns if col != target_col]
    test_features = [col for col in test_df.columns if col != target_col]
    if not train_features:
        raise ValueError("Tập dữ liệu không có đặc trưng để huấn luyện.")
    if train_features != test_features:
        raise ValueError(
            "Schema đặc trưng train/test không khớp; không thể bảo toàn schema model."
        )
    if train_df.index.max() >= test_df.index.min():
        raise ValueError(
            "Train và test bị chồng lấn hoặc không theo đúng thứ tự thời gian."
        )

    X_train = train_df[train_features]
    y_train = train_df[target_col].astype(float)
    X_test = test_df[test_features]
    y_test = test_df[target_col].astype(float)
    print(
        f" -> Nạp thành công: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"số đặc trưng={len(train_features)}."
    )
    return X_train, X_test, y_train, y_test


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính MAE, RMSE, MAPE (%), R2 từ vector dự báo."""
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Kích thước dự báo {y_pred.shape} không khớp target {y_true.shape}."
        )
    if not np.isfinite(y_pred).all():
        raise ValueError("Kết quả dự báo chứa NaN hoặc giá trị vô hạn.")
    if (y_true == 0).any():
        raise ValueError("MAPE không xác định vì y_true chứa giá trị 0.")
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE (MW)": mae, "RMSE (MW)": rmse, "MAPE (%)": mape, "R2 Score": r2}


# ---------------------------------------------------------------------------------
# 2. BÀI TẬP NÂNG CAO 3: TINH CHỈNH SIÊU THAM SỐ VỚI TIMESERIESSPLIT
# ---------------------------------------------------------------------------------
def step1_tune_timeseriessplit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, Any],
    n_splits: int = 5,
) -> Tuple[XGBRegressor, Dict[str, Any]]:
    """
    Tìm kiếm siêu tham số tối ưu cho XGBRegressor bằng GridSearchCV với
    TimeSeriesSplit (expanding window) - tuyệt đối không dùng KFold(shuffle=True).
    """
    print(f"[CV05 - BƯỚC 1] Tinh chỉnh siêu tham số bằng TimeSeriesSplit ({n_splits} folds)...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=CONFIG["random_state"],
        n_jobs=-1,
        tree_method="hist",
    )
    n_candidates = int(np.prod([len(v) for v in param_grid.values()]))
    print(
        f" -> GridSearchCV: {n_candidates} tổ hợp x {n_splits} folds "
        f"(scoring=neg_mean_absolute_error)."
    )

    grid_start = time.perf_counter()
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=False,
        verbose=1,
    )
    search.fit(X_train, y_train)
    grid_duration = time.perf_counter() - grid_start

    best_params = dict(search.best_params_)
    best_cv_mae = float(-search.best_score_)
    print(f" -> Hoàn tất tìm kiếm sau {grid_duration:.1f}s.")
    print(f" -> Bộ tham số tối ưu: {best_params}")
    print(f" -> CV MAE tốt nhất: {best_cv_mae:.3f} MW")

    # Xuất bảng siêu tham số tối ưu kèm điểm CV của từng tổ hợp.
    results = pd.DataFrame(search.cv_results_)
    results = results.sort_values("rank_test_score")
    hyper_table = pd.DataFrame(
        {
            "hyperparameter": list(best_params.keys()) + ["_search_meta"],
            "best_value": [best_params[k] for k in best_params.keys()]
            + [
                (
                    f"GridSearchCV(TimeSeriesSplit n_splits={n_splits}); "
                    f"best_CV_MAE_MW={best_cv_mae:.4f}; "
                    f"search_time_s={grid_duration:.1f}"
                )
            ],
        }
    )
    hyper_path = TABLES_DIR / "cv05_best_hyperparameters.csv"
    hyper_table.to_csv(hyper_path, index=False, encoding="utf-8")
    print(f" -> Đã xuất bảng siêu tham số: {hyper_path}")

    full_path = TABLES_DIR / "cv05_cv_results_all.csv"
    cols = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    results[cols].to_csv(full_path, index=False, encoding="utf-8")
    print(f" -> Đã xuất kết quả CV đầy đủ ({n_candidates} tổ hợp): {full_path}")

    return best_params


# ---------------------------------------------------------------------------------
# 3. BÀI TẬP NÂNG CAO 4: HUẤN LUYỆN VÀ SO SÁNH VỚI RANDOM FOREST
# ---------------------------------------------------------------------------------
def step2_benchmark_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tuned_xgb: XGBRegressor,
) -> pd.DataFrame:
    """
    Huấn luyện RandomForestRegressor trên cùng không gian đặc trưng/nhãn,
    đo Training Time, Inference Latency, sai số MAE/RMSE/MAPE/R2 và
    nguy cơ quá khớp của cả hai mô hình; lưu checkpoint RF và bảng so sánh.
    """
    print("[CV05 - BƯỚC 2] Huấn luyện mô hình đối chứng RandomForestRegressor...")

    # ---- Random Forest: huấn luyện + đo thời gian ----
    rf_model = RandomForestRegressor(**CONFIG["rf_params"])
    rf_start = time.perf_counter()
    rf_model.fit(X_train, y_train)
    rf_train_time = time.perf_counter() - rf_start

    # Inference: đo bằng perf_counter trên toàn tập test, lặp 3 lần lấy trung bình
    rf_pred = None
    rf_inference_ms = None
    for _ in range(3):
        inf_start = time.perf_counter()
        rf_pred = np.asarray(rf_model.predict(X_test), dtype=float)
        rf_inference_ms = (time.perf_counter() - inf_start) * 1000.0
    print(
        f" -> RF hoàn tất: train {rf_train_time:.1f}s, "
        f"inference ~{rf_inference_ms:.0f} ms cho {len(X_test)} mẫu test."
    )

    # ---- XGBoost đã tinh chỉnh: đo inference (đã fit ở bước 1) ----
    xgb_pred = None
    xgb_inference_ms = None
    for _ in range(3):
        inf_start = time.perf_counter()
        xgb_pred = np.asarray(tuned_xgb.predict(X_test), dtype=float)
        xgb_inference_ms = (time.perf_counter() - inf_start) * 1000.0

    # ---- Sai số trên tập test ----
    y_test_arr = y_test.to_numpy(dtype=float)
    rf_metrics = _regression_metrics(y_test_arr, rf_pred)
    xgb_metrics = _regression_metrics(y_test_arr, xgb_pred)

    # ---- Đánh giá quá khớp: sai số trên tập train vs tập test ----
    rf_train_pred = np.asarray(rf_model.predict(X_train), dtype=float)
    xgb_train_pred = np.asarray(tuned_xgb.predict(X_train), dtype=float)
    y_train_arr = y_train.to_numpy(dtype=float)
    rf_train_mae = float(mean_absolute_error(y_train_arr, rf_train_pred))
    xgb_train_mae = float(mean_absolute_error(y_train_arr, xgb_train_pred))
    rf_overfit_gap = rf_metrics["MAE (MW)"] - rf_train_mae
    xgb_overfit_gap = xgb_metrics["MAE (MW)"] - xgb_train_mae

    # ---- Lưu checkpoint RF ----
    import joblib

    rf_path = MODELS_DIR / "rf_benchmark.joblib"
    joblib.dump(rf_model, rf_path)
    print(f" -> Đã lưu checkpoint RF: {rf_path}")

    # ---- Bảng so sánh benchmark ----
    df_comparison = pd.DataFrame(
        {
            "Model": ["XGBoost (tuned)", "Random Forest"],
            "MAE (MW)": [xgb_metrics["MAE (MW)"], rf_metrics["MAE (MW)"]],
            "RMSE (MW)": [xgb_metrics["RMSE (MW)"], rf_metrics["RMSE (MW)"]],
            "MAPE (%)": [xgb_metrics["MAPE (%)"], rf_metrics["MAPE (%)"]],
            "R2 Score": [xgb_metrics["R2 Score"], rf_metrics["R2 Score"]],
            "Train Time (s)": [CONFIG["xgb_tuned_train_time"], rf_train_time],
            "Inference (ms)": [xgb_inference_ms, rf_inference_ms],
            "Train MAE (MW)": [xgb_train_mae, rf_train_mae],
            "Overfit Gap MAE (MW)": [xgb_overfit_gap, rf_overfit_gap],
        }
    )
    summary_path = TABLES_DIR / "cv05_benchmark_summary.csv"
    df_comparison.to_csv(summary_path, index=False, encoding="utf-8")
    print(f" -> Đã xuất bảng so sánh: {summary_path}")

    # ---- Biểu đồ so sánh ----
    _plot_benchmark(df_comparison)

    print("\n[Bảng so sánh đối chứng]")
    print(df_comparison.round(4).to_string(index=False))
    return df_comparison


def _plot_benchmark(df_comparison: pd.DataFrame) -> None:
    """Vẽ biểu đồ cột so sánh MAE, RMSE, thời gian huấn luyện và suy luận."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("MAE (MW)", "So sánh sai số MAE (MW)"),
        ("RMSE (MW)", "So sánh sai số RMSE (MW)"),
        ("Train Time (s)", "So sánh thời gian huấn luyện (s)"),
    ]
    colors = ["#4C78A8", "#F58518"]  # hai màu đầu categorical palette đã thẩm định

    for ax, (col, title), color in zip(axes, metrics, colors):
        bars = ax.bar(df_comparison["Model"], df_comparison[col], color=color)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:,.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    axes[0].set_ylabel("MW")
    axes[1].set_ylabel("MW")
    axes[2].set_ylabel("giây")
    fig.suptitle("CV05: XGBoost (tuned) vs Random Forest - Benchmark đối chứng", fontsize=13)
    fig.tight_layout()

    fig_path = FIGURES_DIR / "cv05_benchmark_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f" -> Đã lưu biểu đồ so sánh: {fig_path}")


# ---------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH (MAIN ENTRYPOINT)
# ---------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("KHỞI CHẠY TIẾN TRÌNH CV05: TINH CHỈNH TIMESERIESSPLIT & BENCHMARK RF (HOÀNG)")
    print("=" * 80)

    X_train, X_test, y_train, y_test = load_train_test(TRAIN_DATA, TEST_DATA, CONFIG["target_col"])

    print("-" * 80)
    best_params = step1_tune_timeseriessplit(
        X_train, y_train, CONFIG["param_grid"], CONFIG["cv_splits"]
    )

    # Huấn luyện lại mô hình tốt nhất trên toàn bộ tập train và đo thời gian.
    xgb_fit_start = time.perf_counter()
    tuned_xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=CONFIG["random_state"],
        n_jobs=-1,
        tree_method="hist",
        **best_params,
    )
    tuned_xgb.fit(X_train, y_train)
    CONFIG["xgb_tuned_train_time"] = time.perf_counter() - xgb_fit_start
    print(
        f" -> Thời gian huấn luyện XGBoost (tuned) trên toàn tập train: "
        f"{CONFIG['xgb_tuned_train_time']:.1f}s"
    )

    print("-" * 80)
    step2_benchmark_random_forest(X_train, y_train, X_test, y_test, tuned_xgb)

    print("-" * 80)
    print("[CV05 HOÀN TẤT: Đã hoàn thành tinh chỉnh siêu tham số và đối sánh Random Forest]")


if __name__ == "__main__":
    main()
