# -*- coding: utf-8 -*-
"""
===================================================================================
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 3: MACHINE LEARNING & XGBOOST
NHIỆM VỤ CV01: QUẢN TRỊ PIPELINE, TIỀN XỬ LÝ & SINH ĐẶC TRƯNG ĐỘ TRỄ (LAG FEATURES)
===================================================================================

[THÔNG TIN DÀNH CHO THÀNH VIÊN & AI AGENT]:
- Mã công việc: CV01 (Bước 1, Bước 2, Bước 3 của bài lab)
- Thành viên phụ trách: Hà Duy Hưng (Nhóm trưởng điều phối)
- Nhánh Git khuyến nghị: main (hoặc feat/cv01-data-prep)
- Tệp thực thi: src/cv01_data_and_lags.py
- Dữ liệu đầu vào:
  + data/raw/AEP_hourly.csv (121,273 quan sát mức tiêu thụ điện năng)
- Sản phẩm đầu ra:
  + data/processed/train.csv (80% quan sát đầu tiên theo thời gian)
  + data/processed/test.csv  (20% quan sát sau cùng theo thời gian)
  + outputs/tables/cv01_data_summary.csv (Bảng thống kê mô tả chuỗi thời gian)
  + outputs/figures/cv01_raw_and_lags.png (Biểu đồ EDA chuỗi gốc và ma trận tương quan Lags)

[QUY TẮC KỸ THUẬT BẮT BUỘC CHO AI AGENT]:
1. ĐỒNG BỘ ĐƯỜNG DẪN: Luôn sử dụng `pathlib.Path` tính từ `PROJECT_ROOT`, tuyệt đối 
   không sử dụng đường dẫn tuyệt đối (hard-coded path).
2. THỨ TỰ THỜI GIAN: Chuỗi thời gian phải được sắp xếp tăng dần đơn điệu (`sort_index()`).
3. NGĂN NGỪA DATA LEAKAGE:
   - Khi chia tập Train/Test theo tỷ lệ 80/20, TUYỆT ĐỐI KHÔNG dùng `shuffle=True`.
   - Cắt tuần tự: 80% thời gian đầu làm Train, 20% thời gian sau làm Test.
4. XỬ LÝ ĐỘ TRỄ (LAGS):
   - Tạo các lag features y_{t-1} đến y_{t-24} (hoặc các lag chu kỳ tuần y_{t-168}).
   - Xử lý các giá trị NaN phát sinh do độ trễ bằng `dropna()` trước khi chia tập.
5. TƯƠNG THÍCH MÔI TRƯỜNG WINDOWS:
   - Cấu hình UTF-8 encoding cho stdout/stderr để tránh lỗi UnicodeEncodeError.
===================================================================================
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
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG (ĐỒNG BỘ CHO MỌI AGENT & THÀNH VIÊN)
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
    [TODO - HÀ DUY HƯNG / AI AGENT TRIỂN KHAI]:
    - Đọc file CSV từ data_path.
    - Chuyển đổi cột Datetime thành pd.DatetimeIndex, sắp xếp tăng dần theo thời gian.
    - Kiểm tra và xử lý khuyết thiếu (missing values) và trùng lặp thời gian (duplicates).
    - Xuất bảng thống kê mô tả ra outputs/tables/cv01_data_summary.csv.
    """
    print(f"[CV01 - BƯỚC 1] Đang tải dữ liệu từ: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp {data_path}! Hãy tải file về thư mục data/raw/.")

    # TODO: Nạp dữ liệu và kiểm toán chất lượng
    # df = pd.read_csv(data_path)
    # df[CONFIG["datetime_col"]] = pd.to_datetime(df[CONFIG["datetime_col"]])
    # df.set_index(CONFIG["datetime_col"], inplace=True)
    # df.sort_index(inplace=True)
    
    print(" -> [TODO] Nạp và xử lý DatetimeIndex thành công.")
    return pd.DataFrame()


# ---------------------------------------------------------------------------------
# 3. BƯỚC 2: TẠO LAG FEATURES & CALENDAR FEATURES
# ---------------------------------------------------------------------------------
def step2_create_features(df: pd.DataFrame, lags: List[int], target_col: str = "AEP_MW") -> pd.DataFrame:
    """
    [TODO - HÀ DUY HƯNG / AI AGENT TRIỂN KHAI]:
    - Biến đổi bài toán chuỗi thời gian thành học có giám sát bằng các biến độ trễ (Lags):
      y_{t-1}, y_{t-2}, ..., y_{t-p} sử dụng `.shift(lag)`.
    - Tạo các đặc trưng lịch (Calendar features):
      + hour (0 - 23)
      + dayofweek (0 - 6)
      + quarter (1 - 4)
      + month (1 - 12)
      + is_weekend (0 hoặc 1)
    - Loại bỏ các giá trị NaN phát sinh do độ trễ lớn nhất (`dropna()`).
    """
    print(f"[CV01 - BƯỚC 2] Tạo các Lag Features: {lags} và Calendar Features...")
    df_feat = df.copy()

    # TODO: Tạo các cột lag features và calendar features
    # for lag in lags:
    #     df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)
    # df_feat["hour"] = df_feat.index.hour
    # df_feat["dayofweek"] = df_feat.index.dayofweek
    # df_feat["month"] = df_feat.index.month
    # df_feat.dropna(inplace=True)

    print(" -> [TODO] Tạo đặc trưng hoàn tất.")
    return df_feat


# ---------------------------------------------------------------------------------
# 4. BƯỚC 3: CHIA TRAIN/TEST THEO THỜI GIAN VÀ XUẤT TẬP DỮ LIỆU
# ---------------------------------------------------------------------------------
def step3_split_and_save(
    df_features: pd.DataFrame, 
    train_ratio: float = 0.80
) -> Tuple[Path, Path]:
    """
    [TODO - HÀ DUY HƯNG / AI AGENT TRIỂN KHAI]:
    - Cắt chuỗi theo thứ tự thời gian tuần tự:
      + train_size = int(len(df_features) * train_ratio)
      + train_df = df_features.iloc[:train_size]
      + test_df  = df_features.iloc[train_size:]
    - Lưu train.csv và test.csv vào data/processed/.
    - Vẽ đồ thị tổng quan chuỗi và lưu vào outputs/figures/cv01_raw_and_lags.png.
    """
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    print(f"[CV01 - BƯỚC 3] Lưu tập Train ({train_ratio*100:.0f}%) và Test ({(1-train_ratio)*100:.0f}%)...")
    
    # TODO: Lưu file CSV
    # train_df.to_csv(train_path)
    # test_df.to_csv(test_path)
    
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
    
    print("[CV01 HOÀN TẤT: Dữ liệu đã sẵn sàng cho CV02 tiếp nhận huấn luyện model]")


if __name__ == "__main__":
    main()
