# -*- coding: utf-8 -*-
"""
===================================================================================
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 3: MACHINE LEARNING & XGBOOST
NHIỆM VỤ CV02: HUẤN LUYỆN MÔ HÌNH CƠ SỞ XGBOOST REGRESSOR (BƯỚC 4, BƯỚC 5)
===================================================================================

[THÔNG TIN DÀNH CHO THÀNH VIÊN & AI AGENT]:
- Mã công việc: CV02
- Thành viên phụ trách: Khải
- Nhánh Git: khai_lab2_CV02
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
   - Lưu model checkpoint định dạng `.json` để CV03 nạp lại suy luận.
===================================================================================
"""

import csv
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

CONFIG = {
    "target_col": "AEP_MW",
    "random_state": 42,
    "xgb_params": {
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "max_depth": 5,
        "early_stopping_rounds": 50,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "n_jobs": -1,
    },
}


def _load_dataset(data_path: Path, target_col: str) -> pd.DataFrame:
    """Đọc và kiểm tra một tập dữ liệu đã xử lý của CV01."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {data_path}. Cần chạy src/cv01_data_and_lags.py trước."
        )
    if not data_path.is_file():
        raise ValueError(f"Đường dẫn đầu vào không phải tệp: {data_path}")

    with data_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        header = next(csv.reader(csv_file), None)
    if not header:
        raise ValueError(f"Tập dữ liệu {data_path.name} không có header.")
    duplicate_headers = sorted({col for col in header if header.count(col) > 1})
    if duplicate_headers:
        raise ValueError(
            f"{data_path.name} có tên cột bị trùng: {duplicate_headers}."
        )

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError(f"Tập dữ liệu {data_path.name} đang rỗng.")

    datetime_col = "Datetime" if "Datetime" in df.columns else None
    if datetime_col is None and str(df.columns[0]).startswith("Unnamed:"):
        candidate_values = df.iloc[:, 0]
        if not pd.api.types.is_numeric_dtype(candidate_values):
            candidate = pd.to_datetime(candidate_values, errors="coerce")
            if candidate.notna().all():
                datetime_col = str(df.columns[0])

    if datetime_col is not None:
        parsed_datetime = pd.to_datetime(df.pop(datetime_col), errors="coerce")
        if parsed_datetime.isna().any():
            invalid_count = int(parsed_datetime.isna().sum())
            raise ValueError(
                f"{data_path.name} có {invalid_count} mốc thời gian không hợp lệ."
            )
        df.index = pd.DatetimeIndex(parsed_datetime, name="Datetime")
        if df.index.has_duplicates:
            raise ValueError(f"{data_path.name} có mốc thời gian bị trùng lặp.")
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                f"{data_path.name} chưa được sắp xếp tăng dần theo thời gian."
            )

    if target_col not in df.columns:
        raise KeyError(
            f"Không tìm thấy cột mục tiêu '{target_col}' trong {data_path.name}."
        )

    non_numeric_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    if non_numeric_cols:
        raise TypeError(
            f"{data_path.name} còn cột không phải số: {non_numeric_cols}."
        )

    values = df.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{data_path.name} chứa NaN hoặc giá trị vô hạn.")

    return df


# ---------------------------------------------------------------------------------
# 2. BƯỚC 4: NẠP DỮ LIỆU & TÁCH MA TRẬN ĐẶC TRƯNG X - y
# ---------------------------------------------------------------------------------
def step1_load_train_test(
    train_path: Path,
    test_path: Path,
    target_col: str = "AEP_MW",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Nạp train/test, kiểm tra hợp đồng dữ liệu rồi tách X và y."""
    print(f"[CV02 - BƯỚC 1] Nạp dữ liệu train từ: {train_path}")
    print(f"[CV02 - BƯỚC 1] Nạp dữ liệu test từ:  {test_path}")

    train_df = _load_dataset(train_path, target_col)
    test_df = _load_dataset(test_path, target_col)

    train_features = [col for col in train_df.columns if col != target_col]
    test_features = [col for col in test_df.columns if col != target_col]
    if not train_features:
        raise ValueError("Tập dữ liệu không có đặc trưng để huấn luyện.")
    train_feature_set = set(train_features)
    test_feature_set = set(test_features)
    if train_feature_set != test_feature_set:
        missing_in_test = sorted(train_feature_set - test_feature_set)
        extra_in_test = sorted(test_feature_set - train_feature_set)
        raise ValueError(
            "Schema đặc trưng train/test không khớp. "
            f"Thiếu trong test: {missing_in_test}; dư trong test: {extra_in_test}."
        )
    if train_features != test_features:
        raise ValueError(
            "Thứ tự đặc trưng train/test không khớp; không thể bảo toàn schema model."
        )

    train_has_datetime = isinstance(train_df.index, pd.DatetimeIndex)
    test_has_datetime = isinstance(test_df.index, pd.DatetimeIndex)
    if train_has_datetime != test_has_datetime:
        raise ValueError(
            "Contract thời gian không nhất quán: chỉ một trong hai tập có Datetime."
        )
    if train_has_datetime and test_has_datetime:
        if train_df.index.max() >= test_df.index.min():
            raise ValueError(
                "Train và test bị chồng lấn hoặc không theo đúng thứ tự thời gian."
            )
    else:
        raise ValueError(
            "Train/test phải có cột Datetime để kiểm chứng chia theo thời gian."
        )

    X_train = train_df[train_features]
    y_train = train_df[target_col]
    X_test = test_df[test_features]
    y_test = test_df[target_col]

    print(
        " -> Phân tách thành công: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"số đặc trưng={len(train_features)}."
    )
    return X_train, X_test, y_train, y_test


def _save_training_history_and_curve(model: Any) -> Tuple[Path, Path]:
    """Xuất lịch sử RMSE và biểu đồ learning curve từ mô hình đã fit."""
    evals_result = model.evals_result()
    try:
        train_rmse = evals_result["validation_0"]["rmse"]
        test_rmse = evals_result["validation_1"]["rmse"]
    except KeyError as exc:
        raise RuntimeError(
            "Không tìm thấy lịch sử RMSE của cả tập train và test."
        ) from exc

    if len(train_rmse) != len(test_rmse) or not train_rmse:
        raise RuntimeError("Lịch sử RMSE không hợp lệ hoặc không đồng nhất.")

    history = pd.DataFrame(
        {
            "iteration": np.arange(len(train_rmse)),
            "train_rmse": np.asarray(train_rmse, dtype=float),
            "test_rmse": np.asarray(test_rmse, dtype=float),
        }
    )
    rmse_values = history[["train_rmse", "test_rmse"]].to_numpy()
    if not np.isfinite(rmse_values).all() or (rmse_values < 0).any():
        raise RuntimeError("Lịch sử RMSE chứa giá trị không hợp lệ.")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    history_path = TABLES_DIR / "cv02_train_history.csv"
    history.to_csv(history_path, index=False, encoding="utf-8")

    best_iteration = int(getattr(model, "best_iteration", history["test_rmse"].idxmin()))
    expected_best_iteration = int(history["test_rmse"].idxmin())
    if not 0 <= best_iteration < len(history):
        raise RuntimeError("Best iteration nằm ngoài phạm vi lịch sử huấn luyện.")
    if best_iteration != expected_best_iteration:
        raise RuntimeError(
            "Best iteration của model không khớp với test RMSE nhỏ nhất."
        )
    best_score = float(history.loc[best_iteration, "test_rmse"])

    # Hai màu đầu của categorical palette đã được thẩm định cho khả năng phân biệt.
    train_color = "#2a78d6"
    test_color = "#eb6834"
    surface_color = "#fcfcfb"
    grid_color = "#e1e0d9"
    text_color = "#0b0b0b"

    with plt.rc_context(
        {
            "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": text_color,
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
        }
    ):
        fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=surface_color)
        ax.set_facecolor(surface_color)
        ax.plot(
            history["iteration"] + 1,
            history["train_rmse"],
            color=train_color,
            linewidth=2,
            linestyle="-",
            label="Train RMSE",
        )
        ax.plot(
            history["iteration"] + 1,
            history["test_rmse"],
            color=test_color,
            linewidth=2,
            linestyle="--",
            label="Test RMSE",
        )
        best_round = best_iteration + 1
        ax.axvline(
            best_round,
            color="#898781",
            linewidth=1.2,
            linestyle=":",
            label=f"Best iteration: {best_round}",
        )
        ax.scatter(
            [best_round],
            [best_score],
            s=52,
            color=test_color,
            edgecolor=surface_color,
            linewidth=2,
            zorder=3,
        )
        ax.annotate(
            f"RMSE = {best_score:,.2f}",
            xy=(best_round, best_score),
            xytext=(10, 12),
            textcoords="offset points",
            color=text_color,
            fontsize=9,
        )
        ax.set_title(
            "Đường cong mất mát của XGBoost Baseline",
            color=text_color,
            fontsize=15,
            fontweight="semibold",
            pad=14,
        )
        ax.set_xlabel("Vòng lặp boosting")
        ax.set_ylabel("RMSE (MW)")
        ax.grid(axis="y", color=grid_color, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()

        figure_path = FIGURES_DIR / "cv02_learning_curve.png"
        fig.savefig(
            figure_path,
            dpi=150,
            bbox_inches="tight",
            facecolor=surface_color,
        )
        plt.close(fig)

    print(f" -> Đã lưu lịch sử huấn luyện: {history_path}")
    print(f" -> Đã lưu learning curve:      {figure_path}")
    return history_path, figure_path


# ---------------------------------------------------------------------------------
# 3. BƯỚC 5: HUẤN LUYỆN MÔ HÌNH XGBOOST REGRESSOR
# ---------------------------------------------------------------------------------
def step2_train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any],
) -> Any:
    """Huấn luyện XGBoost baseline, theo dõi RMSE và xuất learning curve."""
    print("[CV02 - BƯỚC 2] Bắt đầu huấn luyện mô hình XGBoost Regressor...")
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "Chưa cài đặt xgboost. Hãy chạy 'pip install -r requirements.txt'."
        ) from exc

    model = XGBRegressor(**params)
    start_time = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100,
    )
    elapsed_seconds = time.perf_counter() - start_time

    fallback_iteration = int(params.get("n_estimators", 100)) - 1
    best_iteration = int(getattr(model, "best_iteration", fallback_iteration))
    best_score = float(getattr(model, "best_score", np.nan))
    print(
        f" -> Huấn luyện thành công trong {elapsed_seconds:.2f} giây. "
        f"Best iteration: {best_iteration + 1}; "
        f"Test RMSE tốt nhất: {best_score:,.4f}."
    )

    _save_training_history_and_curve(model)
    return model


# ---------------------------------------------------------------------------------
# 4. LƯU MÔ HÌNH BÀN GIAO CHO CV03
# ---------------------------------------------------------------------------------
def step3_save_model(model: Any, output_path: Path) -> Path:
    """Lưu checkpoint XGBoost JSON để CV03 nạp lại và suy luận."""
    if model is None or not hasattr(model, "save_model"):
        raise TypeError("Mô hình chưa hợp lệ hoặc chưa được huấn luyện.")

    print(f"[CV02 - BƯỚC 3] Lưu trữ mô hình XGBoost tại: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise OSError(f"Không thể tạo checkpoint mô hình tại {output_path}.")
    print(f" -> Checkpoint đã lưu ({output_path.stat().st_size:,} bytes).")
    return output_path


# ---------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH (MAIN ENTRYPOINT)
# ---------------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("KHỞI CHẠY TIẾN TRÌNH CV02: HUẤN LUYỆN XGBOOST REGRESSOR (KHẢI)")
    print("=" * 80)

    X_train, X_test, y_train, y_test = step1_load_train_test(
        TRAIN_DATA, TEST_DATA, target_col=CONFIG["target_col"]
    )

    model = step2_train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        CONFIG["xgb_params"],
    )

    model_checkpoint = MODELS_DIR / "xgboost_baseline.json"
    step3_save_model(model, model_checkpoint)

    print("[CV02 HOÀN TẤT: Mô hình và lịch sử huấn luyện đã sẵn sàng cho CV03]")


if __name__ == "__main__":
    main()
