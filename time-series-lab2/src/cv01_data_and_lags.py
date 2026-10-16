# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 3 (NHÓM 7)
MÃ CÔNG VIỆC: CV01 - TIỀN XỬ LÝ DỮ LIỆU & TẠO ĐẶC TRƯNG ĐỘ TRỄ (LAG FEATURES)
---------------------------------------------------------------------------------
Thành viên phụ trách: Hà Duy Hưng (Nhóm trưởng)
Nhánh Git khuyến nghị: main / feat/cv01-data-preprocessing
Tệp mã nguồn: src/cv01_data_and_lags.py
Dữ liệu đầu vào: data/raw/AEP_hourly.csv
Đầu ra tạo thành:
  - data/processed/train.csv, data/processed/test.csv
  - outputs/tables/cv01_data_summary.csv
  - outputs/figures/cv01_raw_and_lags.png

AI_AGENT_INSTRUCTIONS:
=================================================================================
Mục tiêu dành cho AI Agent (Cursor / GitHub Copilot / Antigravity):
1. Đọc và chuẩn hóa dữ liệu phụ tải điện AEP_hourly.csv (chuyển Datetime thành DatetimeIndex,
   sắp xếp theo thời gian tăng dần, kiểm tra trùng lặp và khuyết thiếu).
2. Tạo các độ trễ (Lag features) từ y_{t-1} đến y_{t-24} bằng phương thức .shift(p).
3. Tách ma trận đặc trưng X và vector mục tiêu y = y_t.
4. Chia tập Train/Test theo tỷ lệ 80% - 20% THEO THỨ TỰ THỜI GIAN.
   [NGHIÊM CẤM]: Tuyệt đối KHÔNG dùng train_test_split(shuffle=True) vì sẽ làm rò rỉ dữ liệu (Data Leakage).
5. Lưu kết quả xử lý vào thư mục data/processed/ để các thành viên khác (CV02 - CV06) tái sử dụng.
6. Hỗ trợ chạy dòng lệnh CLI: python src/cv01_data_and_lags.py [--show]
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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "datetime_col": "Datetime",
    "target_col": "AEP_MW",
    "num_lags": 24,           # Tạo 24 lag tương ứng chu kỳ 24 giờ trong ngày
    "train_ratio": 0.80,      # 80% Train, 20% Test theo thời gian
}


# -------------------------------------------------------------------------------
# 2. BƯỚC 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU GỐC (DATA INGESTION)
# -------------------------------------------------------------------------------
def step1_load_data(data_path: Path) -> pd.DataFrame:
    """
    Tải file AEP_hourly.csv, chuyển cột Datetime sang DatetimeIndex,
    sắp xếp tăng dần và loại bỏ trùng lặp nếu có.
    """
    print(f"\n[BƯỚC 1] Đang nạp dữ liệu gốc từ: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file {data_path}. Vui lòng kiểm tra lại data/raw/!")

    df = pd.read_csv(data_path)
    print(f" -> Kích thước ban đầu: {df.shape[0]:,} dòng, {df.shape[1]} cột.")

    # Chuyển đổi Datetime
    df[CONFIG["datetime_col"]] = pd.to_datetime(df[CONFIG["datetime_col"]])
    df.set_index(CONFIG["datetime_col"], inplace=True)
    df.sort_index(inplace=True)

    # Kiểm tra và xử lý trùng lặp chỉ mục
    if df.index.has_duplicates:
        dup_count = df.index.duplicated().sum()
        print(f" [!] Phát hiện {dup_count} mốc thời gian bị trùng lặp. Đang giữ bản ghi đầu tiên...")
        df = df[~df.index.duplicated(keep="first")]

    # Kiểm tra khuyết thiếu
    missing_count = df[CONFIG["target_col"]].isna().sum()
    if missing_count > 0:
        print(f" [!] Phát hiện {missing_count} giá trị NaN. Đang nội suy tuyến tính...")
        df[CONFIG["target_col"]] = df[CONFIG["target_col"]].interpolate(method="time")

    print(f" -> Dữ liệu sau chuẩn hóa: {len(df):,} quan sát từ {df.index.min()} đến {df.index.max()}.")
    return df


# -------------------------------------------------------------------------------
# 3. BƯỚC 2: TẠO LAG FEATURES (SUPERVISED LEARNING TRANSFORMATION)
# -------------------------------------------------------------------------------
def step2_execute_analysis(df: pd.DataFrame, num_lags: int = 24) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tạo các đặc trưng độ trễ (lags) y_{t-1} ... y_{t-p} và tách X, y.
    """
    print(f"\n[BƯỚC 2] Biến đổi bài toán chuỗi thời gian thành học có giám sát...")
    print(f" -> Tạo {num_lags} lag features từ Lag 1 (1 giờ trước) đến Lag {num_lags} (24 giờ trước)...")

    df_lags = pd.DataFrame(index=df.index)
    target = df[CONFIG["target_col"]]
    df_lags[CONFIG["target_col"]] = target

    for p in range(1, num_lags + 1):
        df_lags[f"lag_{p}"] = target.shift(p)

    # Loại bỏ các dòng NaN ở đầu chuỗi do shift
    rows_before = len(df_lags)
    df_lags.dropna(inplace=True)
    rows_after = len(df_lags)
    print(f" -> Đã loại bỏ {rows_before - rows_after} dòng NaN ban đầu. Số mẫu khả dụng: {rows_after:,}")

    # Tách X và y
    feature_cols = [f"lag_{p}" for p in range(1, num_lags + 1)]
    X = df_lags[feature_cols]
    y = df_lags[CONFIG["target_col"]]

    return X, y


# -------------------------------------------------------------------------------
# 4. BƯỚC 3: PHÂN CHIA TẬP TRAIN / TEST THEO THỜI GIAN & XUẤT FILE
# -------------------------------------------------------------------------------
def step3_split_and_save(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.80) -> Dict[str, Any]:
    """
    Chia Train / Test theo thời gian (Time-based split, không shuffle)
    và lưu ra thư mục data/processed/ để các module khác tái sử dụng.
    """
    print(f"\n[BƯỚC 3] Phân chia tập Train/Test theo thời gian ({train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%)...")
    split_idx = int(len(X) * train_ratio)
    split_date = X.index[split_idx]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f" -> Điểm cắt thời gian (Split Point): {split_date}")
    print(f" -> Tập Train: {len(X_train):,} mẫu ({X_train.index.min()} đến {X_train.index.max()})")
    print(f" -> Tập Test:  {len(X_test):,} mẫu ({X_test.index.min()} đến {X_test.index.max()})")

    # Ghép lại thành DataFrames đầy đủ để lưu trữ
    train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
    test_df = pd.concat([X_test, y_test.rename("target")], axis=1)

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
    print(f" -> Đã lưu dữ liệu train vào: {train_path}")
    print(f" -> Đã lưu dữ liệu test vào:  {test_path}")

    # Bảng tổng hợp thống kê
    summary_df = pd.DataFrame({
        "Thông số": [
            "Tổng số quan sát khả dụng",
            "Số đặc trưng Lag (Features)",
            "Tỷ lệ phân chia Train/Test",
            "Số lượng mẫu Train",
            "Thời gian bắt đầu Train",
            "Thời gian kết thúc Train",
            "Số lượng mẫu Test",
            "Thời gian bắt đầu Test",
            "Thời gian kết thúc Test",
            "Giá trị Target Min (MW)",
            "Giá trị Target Max (MW)",
            "Giá trị Target Mean (MW)"
        ],
        "Giá trị": [
            f"{len(X):,}",
            f"{X.shape[1]} (lag_1 đến lag_{CONFIG['num_lags']})",
            f"{train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%",
            f"{len(X_train):,}",
            str(X_train.index.min()),
            str(X_train.index.max()),
            f"{len(X_test):,}",
            str(X_test.index.min()),
            str(X_test.index.max()),
            f"{y.min():,.2f}",
            f"{y.max():,.2f}",
            f"{y.mean():,.2f}"
        ]
    })
    summary_path = TABLES_DIR / "cv01_data_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f" -> Đã lưu bảng tóm tắt vào: {summary_path}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "split_date": split_date,
        "summary_df": summary_df
    }


# -------------------------------------------------------------------------------
# 5. BƯỚC 4: TRỰC QUAN HÓA & KẾT LUẬN (VISUALIZATION)
# -------------------------------------------------------------------------------
def step4_visualize(y: pd.Series, split_date: pd.Timestamp, show: bool = False):
    """
    Vẽ đồ thị chuỗi dữ liệu gốc cùng đường phân chia Train/Test
    và biểu đồ tương quan phân tán (Scatter) giữa y_t với y_{t-1} và y_{t-24}.
    """
    print(f"\n[BƯỚC 4] Trực quan hóa dữ liệu và xuất biểu đồ...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1.5, 1]})

    # 1. Biểu đồ chuỗi thời gian phân đoạn Train/Test
    ax1 = axes[0]
    train_part = y.loc[:split_date]
    test_part = y.loc[split_date:]
    ax1.plot(train_part.index, train_part.values, label="Tập Huấn luyện (Train - 80%)", color="#1f77b4", lw=0.6, alpha=0.85)
    ax1.plot(test_part.index, test_part.values, label="Tập Kiểm thử (Test - 20%)", color="#ff7f0e", lw=0.6, alpha=0.85)
    ax1.axvline(split_date, color="#d62728", linestyle="--", lw=2, label=f"Điểm cắt Train/Test ({split_date.strftime('%Y-%m-%d')})")
    ax1.set_title("Chuỗi tiêu thụ điện AEP (Hourly Energy Consumption) - Phân chia Train / Test theo thời gian", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Mức tiêu thụ (MW)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # 2. Phóng to một khoảng thời gian 2 tuần minh họa nhịp điệu ngày đêm
    ax2 = axes[1]
    sample_zoom = test_part.iloc[:336] # 14 ngày x 24 giờ = 336 giờ
    ax2.plot(sample_zoom.index, sample_zoom.values, marker="o", markersize=3, color="#2ca02c", lw=1.2, label="Phụ tải thực tế 14 ngày đầu tập Test")
    ax2.set_title("Trực quan hóa chi tiết nhịp điệu phụ tải điện theo giờ (Zoom 14 ngày)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Thời gian", fontsize=11)
    ax2.set_ylabel("Mức tiêu thụ (MW)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "cv01_raw_and_lags.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f" -> Đã lưu biểu đồ trực quan vào: {fig_path}")

    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="CV01: Tiền xử lý dữ liệu và tạo Lag Features")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ biểu đồ khi chạy")
    args = parser.parse_args()

    print("=" * 80)
    print(" [CV01 - HÀ DUY HƯNG] TIỀN XỬ LÝ DỮ LIỆU & TẠO ĐẶC TRƯNG ĐỘ TRỄ (LAG FEATURES)")
    print("=" * 80)

    # Thực thi tuần tự 4 bước
    df = step1_load_data(RAW_DATA_PATH)
    X, y = step2_execute_analysis(df, num_lags=CONFIG["num_lags"])
    split_res = step3_split_and_save(X, y, train_ratio=CONFIG["train_ratio"])
    step4_visualize(y, split_res["split_date"], show=args.show)

    print("\n[THÀNH CÔNG] Nhiệm vụ CV01 hoàn thành 100%. Dữ liệu đã sẵn sàng cho CV02 - CV06!")


if __name__ == "__main__":
    main()
