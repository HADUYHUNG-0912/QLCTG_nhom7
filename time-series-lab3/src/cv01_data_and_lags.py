# -*- coding: utf-8 -*-
"""
CV01 — Hà Duy Hưng: Tiền xử lý dữ liệu & sinh đặc trưng độ trễ (Lag Features).

Đầu vào : data/raw/AEP_hourly.csv
Đầu ra  : data/processed/train.csv, test.csv, AEP_features.csv
          outputs/tables/cv01_data_summary.csv, train_test_split_info.csv
          outputs/figures/cv01_raw_and_lags.png, eda_aep_hourly.png
"""

import sys
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any

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
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG
# ---------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "datetime_col": "Datetime",
    "target_col": "AEP_MW",
    "lags": [1, 2, 3, 24, 48, 168],  # 1-3h (ngắn hạn), 24h/48h (chu kỳ ngày), 168h (chu kỳ tuần)
    "train_ratio": 0.80,
    "random_state": 42
}


# ---------------------------------------------------------------------------------
# 2. BƯỚC 1: NẠP VÀ KIỂM TRA TÍNH TOÀN VẸN CỦA DỮ LIỆU GỐC
# ---------------------------------------------------------------------------------
def step1_load_and_audit(data_path: Path) -> pd.DataFrame:
    """
    Nhiệm vụ CV01 - Bước 1:
    - Đọc file CSV từ data_path.
    - Chuyển đổi cột Datetime thành pd.DatetimeIndex, sắp xếp tăng dần theo thời gian.
    - Kiểm tra và xử lý khuyết thiếu (missing values) và trùng lặp thời gian (duplicates).
    - Xuất bảng thống kê mô tả ra outputs/tables/cv01_data_summary.csv và data_summary.csv.
    """
    print(f"[CV01 - BƯỚC 1] Đang tải dữ liệu từ: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp {data_path}! Hãy tải file về thư mục data/raw/.")

    # 1. Đọc dữ liệu thô
    df = pd.read_csv(data_path)
    raw_count = len(df)
    print(f" -> Tải thành công {raw_count:,} dòng dữ liệu.")

    # 2. Chuyển đổi và thiết lập DatetimeIndex
    df[CONFIG["datetime_col"]] = pd.to_datetime(df[CONFIG["datetime_col"]])
    df.set_index(CONFIG["datetime_col"], inplace=True)
    df.sort_index(inplace=True)

    # 3. Kiểm toán chất lượng dữ liệu
    missing_count = int(df[CONFIG["target_col"]].isna().sum())
    duplicate_count = int(df.index.duplicated().sum())

    print(f" -> Kiểm tra khuyết thiếu (NaN): {missing_count} dòng.")
    print(f" -> Kiểm tra trùng lặp chỉ mục: {duplicate_count} dòng.")

    # 4. Xử lý trùng lặp chỉ mục (do Daylight Saving Time)
    if duplicate_count > 0:
        print(f" -> Xử lý {duplicate_count} dòng trùng lặp bằng cách tính trung bình theo mốc thời gian...")
        df = df.groupby(df.index).mean()

    # 5. Thống kê mô tả
    stats = df[CONFIG["target_col"]].describe()

    summary_rows = [
        {"Chỉ số": "Tên tệp dữ liệu gốc", "Giá trị": data_path.name},
        {"Chỉ số": "Mốc thời gian bắt đầu", "Giá trị": str(df.index.min())},
        {"Chỉ số": "Mốc thời gian kết thúc", "Giá trị": str(df.index.max())},
        {"Chỉ số": "Tổng số quan sát thô", "Giá trị": f"{raw_count:,}"},
        {"Chỉ số": "Số bản ghi trùng lặp (DST)", "Giá trị": f"{duplicate_count:,}"},
        {"Chỉ số": "Số quan sát sau làm sạch", "Giá trị": f"{len(df):,}"},
        {"Chỉ số": "Số lượng giá trị khuyết thiếu (NaN)", "Giá trị": f"{missing_count:,}"},
        {"Chỉ số": "Mức tiêu thụ trung bình (Mean MW)", "Giá trị": f"{stats['mean']:.2f}"},
        {"Chỉ số": "Độ lệch chuẩn (Std MW)", "Giá trị": f"{stats['std']:.2f}"},
        {"Chỉ số": "Mức tiêu thụ nhỏ nhất (Min MW)", "Giá trị": f"{stats['min']:.2f}"},
        {"Chỉ số": "Phân vị 25% (Q1 MW)", "Giá trị": f"{stats['25%']:.2f}"},
        {"Chỉ số": "Trung vị 50% (Median MW)", "Giá trị": f"{stats['50%']:.2f}"},
        {"Chỉ số": "Phân vị 75% (Q3 MW)", "Giá trị": f"{stats['75%']:.2f}"},
        {"Chỉ số": "Mức tiêu thụ lớn nhất (Max MW)", "Giá trị": f"{stats['max']:.2f}"}
    ]
    summary_df = pd.DataFrame(summary_rows)

    # Xuất bảng tổng hợp ra outputs/tables/
    summary_df.to_csv(TABLES_DIR / "cv01_data_summary.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(TABLES_DIR / "data_summary.csv", index=False, encoding="utf-8-sig")
    print(f" -> Đã lưu bảng thống kê kiểm toán: {TABLES_DIR / 'cv01_data_summary.csv'}")

    return df


# ---------------------------------------------------------------------------------
# 3. BƯỚC 2: TẠO LAG FEATURES & CALENDAR FEATURES
# ---------------------------------------------------------------------------------
def step2_create_features(df: pd.DataFrame, lags: List[int], target_col: str = "AEP_MW") -> pd.DataFrame:
    """
    Nhiệm vụ CV01 - Bước 2:
    - Biến đổi bài toán chuỗi thời gian thành học có giám sát:
      y_{t-1}, y_{t-2}, ..., y_{t-p} sử dụng `.shift(lag)`.
    - Tạo các đặc trưng lịch (Calendar features):
      + hour (0 - 23)
      + dayofweek (0 - 6)
      + quarter (1 - 4)
      + month (1 - 12)
      + year
      + dayofyear
      + is_weekend (0 hoặc 1)
    - Loại bỏ các giá trị NaN phát sinh do độ trễ lớn nhất (`dropna()`).
    """
    print(f"[CV01 - BƯỚC 2] Tạo các Lag Features: {lags} và Calendar Features...")
    df_feat = df.copy()

    # 1. Tạo các biến độ trễ (Lag Features)
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)

    # 2. Tạo các biến thời gian lịch (Calendar Features)
    df_feat["hour"] = df_feat.index.hour
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["quarter"] = df_feat.index.quarter
    df_feat["month"] = df_feat.index.month
    df_feat["year"] = df_feat.index.year
    df_feat["dayofyear"] = df_feat.index.dayofyear
    df_feat["is_weekend"] = (df_feat.index.dayofweek >= 5).astype(int)

    # 3. Xử lý các giá trị NaN do lag
    initial_rows = len(df_feat)
    df_feat.dropna(inplace=True)
    dropped_rows = initial_rows - len(df_feat)
    print(f" -> Đã tạo {len(lags)} lag features và 7 calendar features.")
    print(f" -> Loại bỏ {dropped_rows} dòng NaN ở đầu chuỗi (tương ứng lag tối đa = {max(lags)}h).")
    print(f" -> Kích thước dữ liệu sau khi tạo đặc trưng: {df_feat.shape}")

    return df_feat


# ---------------------------------------------------------------------------------
# 4. BƯỚC 3: CHIA TRAIN/TEST THEO THỜI GIAN VÀ XUẤT TẬP DỮ LIỆU
# ---------------------------------------------------------------------------------
def step3_split_and_save(
    df_features: pd.DataFrame, 
    train_ratio: float = 0.80
) -> Tuple[Path, Path]:
    """
    Nhiệm vụ CV01 - Bước 3 & Bước 4:
    - Cắt chuỗi theo thứ tự thời gian tuần tự (không shuffle để tránh data leakage):
      + train_size = int(len(df_features) * train_ratio)
      + train_df = df_features.iloc[:train_size]
      + test_df  = df_features.iloc[train_size:]
    - Lưu train.csv, test.csv và AEP_features.csv vào data/processed/.
    - Xuất bảng thông tin phân chia vào outputs/tables/train_test_split_info.csv.
    - Vẽ đồ thị tổng quan chuỗi và lưu vào outputs/figures/cv01_raw_and_lags.png.
    """
    train_size = int(len(df_features) * train_ratio)
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:]

    print(f"[CV01 - BƯỚC 3] Phân chia Train/Test theo thời gian ({train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%):")
    print(f" -> Tập Train: {len(train_df):,} dòng ({train_df.index.min()} đến {train_df.index.max()})")
    print(f" -> Tập Test:  {len(test_df):,} dòng ({test_df.index.min()} đến {test_df.index.max()})")

    # Lưu dữ liệu đã xử lý
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    all_path = DATA_PROCESSED / "AEP_features.csv"

    train_df.to_csv(train_path, index=True)
    test_df.to_csv(test_path, index=True)
    df_features.to_csv(all_path, index=True)
    print(f" -> Đã lưu: {train_path}")
    print(f" -> Đã lưu: {test_path}")
    print(f" -> Đã lưu: {all_path}")

    # Xuất bảng thông tin chia tập
    split_info = pd.DataFrame([
        {
            "Tập dữ liệu": "Tập huấn luyện (Train)",
            "Số quan sát": len(train_df),
            "Tỷ lệ": f"{train_ratio * 100:.1f}%",
            "Mốc bắt đầu": str(train_df.index.min()),
            "Mốc kết thúc": str(train_df.index.max())
        },
        {
            "Tập dữ liệu": "Tập kiểm thử (Test)",
            "Số quan sát": len(test_df),
            "Tỷ lệ": f"{(1 - train_ratio) * 100:.1f}%",
            "Mốc bắt đầu": str(test_df.index.min()),
            "Mốc kết thúc": str(test_df.index.max())
        },
        {
            "Tập dữ liệu": "Toàn bộ chuỗi",
            "Số quan sát": len(df_features),
            "Tỷ lệ": "100.0%",
            "Mốc bắt đầu": str(df_features.index.min()),
            "Mốc kết thúc": str(df_features.index.max())
        }
    ])
    split_info.to_csv(TABLES_DIR / "train_test_split_info.csv", index=False, encoding="utf-8-sig")

    # 4. Trực quan hóa dữ liệu và lưu ảnh
    print("[CV01] Đang vẽ biểu đồ phân chia chuỗi thời gian & tương quan các độ trễ...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    plt.subplots_adjust(hspace=0.35)

    # Subplot 1: Biểu đồ chuỗi thời gian Train vs Test
    ax1 = axes[0]
    ax1.plot(train_df.index, train_df[CONFIG["target_col"]], label="Tập Train (80%)", color="#1f77b4", linewidth=0.6, alpha=0.85)
    ax1.plot(test_df.index, test_df[CONFIG["target_col"]], label="Tập Test (20%)", color="#ff7f0e", linewidth=0.6, alpha=0.85)
    split_time = test_df.index.min()
    ax1.axvline(split_time, color="red", linestyle="--", linewidth=1.5, label=f"Điểm chia ({split_time.strftime('%Y-%m-%d')})")
    ax1.set_title("Phân chia Chuỗi Thời Gian Tiêu Thụ Điện Năng AEP (Train / Test Split 80:20 - Không Shuffle)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Mức tiêu thụ điện (MW)", fontsize=11)
    ax1.set_xlabel("Mốc thời gian", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Hệ số tương quan Pearson giữa Target và các Lag Features
    ax2 = axes[1]
    lag_cols = [f"lag_{lag}" for lag in CONFIG["lags"]]
    correlations = [df_features[CONFIG["target_col"]].corr(df_features[col]) for col in lag_cols]
    
    bars = ax2.bar(lag_cols, correlations, color="#2ca02c", edgecolor="#1e7120", alpha=0.85, width=0.5)
    ax2.set_title("Hệ Số Tương Quan Pearson Giữa AEP_MW Hiện Tại và Các Biến Độ Trễ (Lag Features)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Hệ số tương quan (r)", fontsize=11)
    ax2.set_xlabel("Đặc trưng độ trễ (Lags)", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Hiển thị giá trị cụ thể trên từng cột
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    figure_path = FIGURES_DIR / "cv01_raw_and_lags.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    # Lưu thêm tệp eda_aep_hourly.png để khớp hợp đồng bàn giao
    plt.savefig(FIGURES_DIR / "eda_aep_hourly.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f" -> Đã lưu biểu đồ: {figure_path}")

    return train_path, test_path


# ---------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH (MAIN ENTRYPOINT)
# ---------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("KHỞI CHẠY TIẾN TRÌNH CV01: QUẢN TRỊ DỮ LIỆU & TẠO LAG FEATURES (HÀ DUY HƯNG)")
    print("=" * 80)
    
    df_raw = step1_load_and_audit(DATA_RAW)
    df_feat = step2_create_features(df_raw, lags=CONFIG["lags"], target_col=CONFIG["target_col"])
    step3_split_and_save(df_feat, train_ratio=CONFIG["train_ratio"])
    
    print("=" * 80)
    print("[CV01 HOÀN TẤT THÀNH CÔNG: Dữ liệu đã sẵn sàng cho CV02 tiếp nhận huấn luyện model]")
    print("=" * 80)


if __name__ == "__main__":
    main()
