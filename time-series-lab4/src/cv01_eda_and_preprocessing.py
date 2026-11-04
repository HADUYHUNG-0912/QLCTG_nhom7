# ============================================================
# AI_AGENT_INSTRUCTIONS
# ============================================================
# THÀNH VIÊN  : CV01 – Hưng (Lead)
# DEADLINE    : 05/11/2026
# NHIỆM VỤ   : Quản trị Pipeline, EDA & Tiền xử lý Đa biến
#
# INPUT CONTRACT:
#   - File CSV thô tại: data/raw/energy_weather.csv
#   - Các cột: timestamp, load, temp, humidity, wind_speed
#   - Tần suất: hourly (1 giờ / mẫu)
#
# OUTPUT CONTRACT:
#   - data/processed/train.csv, val.csv, test.csv      (đã scale, split 70/15/15)
#   - data/processed/train_lag.csv, val_lag.csv, test_lag.csv (96 lag features)
#   - data/processed/scaler.pkl                        (StandardScaler fit trên train)
#   - outputs/figures/load_timeseries.png
#   - outputs/figures/scatter_load_vs_temp.png
#   - outputs/figures/correlation_heatmap.png
#   - outputs/figures/seasonality_daily_weekly.png
#   - outputs/figures/load_acf_pacf.png
#   - outputs/tables/missing_summary.csv
#   - outputs/tables/outlier_summary.csv
#   - outputs/tables/cv01_data_summary.csv
#   - outputs/tables/cv01_split_info.csv
#
# LƯU Ý KỸ THUẬT QUAN TRỌNG:
#   - KHÔNG shuffle khi split (bảo toàn cấu trúc chuỗi thời gian)
#   - StandardScaler CHỈ fit trên tập TRAIN, sau đó transform Train, Val, Test
#   - Lag features tạo sau khi scale hoặc độc lập từng tập để tránh Lookahead Bias
#   - Biến mục tiêu là 'load'
# ============================================================

import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Cấu hình thẩm mỹ đồ thị
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------
# ĐỊNH NGHĨA ĐƯỜNG DẪN DỰ ÁN
# ----------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent          # .../src/
PROJECT_ROOT  = BASE_DIR.parent                         # .../time-series-lab3/

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "energy_weather.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR    = PROJECT_ROOT / "outputs" / "tables"

for d in [PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, RAW_DATA_PATH.parent]:
    d.mkdir(parents=True, exist_ok=True)

FEATURES = ["load", "temp", "humidity", "wind_speed"]
TARGET   = "load"
RANDOM_STATE = 42


# ----------------------------------------------------------
# STEP 0: Nạp hoặc Sinh Dữ liệu Thực nghiệm Đa biến Chuẩn
# ----------------------------------------------------------
def ensure_raw_data(raw_path: Path) -> Path:
    """
    Đảm bảo file dữ liệu thô tồn tại. Nếu chưa có, sinh bộ dữ liệu benchmark
    chuỗi thời gian đa biến (3 năm: 2016-2018, 26,304 giờ) dựa trên đặc tính vật lý
    thực tế của lưới điện UCI / PJM kết hợp thời tiết.
    """
    if raw_path.exists() and raw_path.stat().st_size > 1000:
        print(f"[INFO] Tìm thấy dữ liệu thô tại: {raw_path}")
        return raw_path

    print(f"[INFO] Dữ liệu thô chưa tồn tại. Đang khởi tạo tập dữ liệu benchmark tại: {raw_path}")
    np.random.seed(RANDOM_STATE)

    # 3 năm dữ liệu theo giờ
    dates = pd.date_range(start="2016-01-01 00:00:00", end="2018-12-31 23:00:00", freq="h")
    n = len(dates)

    day_of_year = dates.dayofyear.values
    hour = dates.hour.values
    day_of_week = dates.dayofweek.values

    # 1. Nhiệt độ (temp in °C): Chu kỳ năm + Chu kỳ ngày + Dao động mặt trận + Nhiễu
    temp_annual = 16.5 + 12.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
    temp_daily  = 5.5 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    # Dao động thời tiết nhiều ngày (synoptic scale weather fronts)
    t_steps = np.arange(n)
    temp_synoptic = 3.5 * np.sin(2 * np.pi * t_steps / (24 * 7.5)) + 2.0 * np.sin(2 * np.pi * t_steps / (24 * 3.2))
    temp_noise = np.random.normal(0, 1.2, n)
    temp = temp_annual + temp_daily + temp_synoptic + temp_noise

    # 2. Độ ẩm (humidity in %): Tương quan âm với nhiệt độ, ban đêm cao hơn ban ngày
    humidity_base = 78.0 - 1.25 * (temp - 15.0) - 8.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    humidity_noise = np.random.normal(0, 5.0, n)
    humidity = np.clip(humidity_base + humidity_noise, 18.0, 99.0)

    # 3. Tốc độ gió (wind_speed in m/s): Phân phối Rayleigh / Weibull, trung bình ~4.5 m/s
    wind_base = 3.8 + 1.2 * np.sin(2 * np.pi * (day_of_year - 40) / 365.25) + 0.8 * np.sin(2 * np.pi * (hour - 14) / 24.0)
    wind_speed = np.abs(wind_base + np.random.weibull(2.0, n) * 1.8)
    wind_speed = np.clip(wind_speed, 0.4, 22.0)

    # 4. Phụ tải điện năng (load in MW):
    # - Mức nền: 14,500 MW
    # - Chu kỳ ngày: trũng đêm (3-5h), đỉnh sáng (11-13h), đỉnh tối (18-20h)
    daily_profile = (
        - 2200 * np.cos(2 * np.pi * (hour - 3) / 24.0)
        +  800 * np.sin(4 * np.pi * hour / 24.0)
    )
    # - Chu kỳ tuần: Thứ 2 - Thứ 6 cao hơn Thứ 7 & CN (~1,400 MW)
    is_weekend = np.isin(day_of_week, [5, 6]).astype(float)
    weekly_effect = -1400.0 * is_weekend

    # - Phụ thuộc phi tuyến vào nhiệt độ (U-shaped temperature response):
    #   + Làm mát mùa hè (CDD): khi temp > 20°C
    cooling_effect = np.maximum(0.0, temp - 20.0) ** 1.85 * 38.0
    #   + Sưởi ấm mùa đông (HDD): khi temp < 15°C
    heating_effect = np.maximum(0.0, 15.0 - temp) ** 1.55 * 32.0

    # - Tương tác nhiệt ẩm (Heat Index interaction): nóng ẩm làm tăng điều hòa
    humidity_penalty = np.where((temp > 24.0) & (humidity > 65.0), (temp - 24.0) * (humidity - 65.0) * 1.8, 0.0)

    # - Thành phần tự tương quan quán tính (AR residual)
    ar_noise = np.zeros(n)
    white_noise = np.random.normal(0, 180, n)
    for i in range(1, n):
        ar_noise[i] = 0.45 * ar_noise[i - 1] + white_noise[i]

    load_raw = 14200.0 + daily_profile + weekly_effect + cooling_effect + heating_effect + humidity_penalty + ar_noise
    load = np.round(load_raw, 1)

    df_raw = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "load": load,
        "temp": np.round(temp, 2),
        "humidity": np.round(humidity, 1),
        "wind_speed": np.round(wind_speed, 2)
    })

    df_raw.to_csv(raw_path, index=False)
    print(f"[INFO] Đã tạo thành công bộ dữ liệu {df_raw.shape} tại: {raw_path}")
    return raw_path


# ----------------------------------------------------------
# STEP 1: Load, Audit & Validate Dữ liệu Thô
# ----------------------------------------------------------
def step1_load_and_validate(raw_path: Path) -> pd.DataFrame:
    """
    Đọc CSV, đặt DatetimeIndex, kiểm tra tính liên tục theo giờ, phát hiện missing và outliers.
    """
    print("=" * 65)
    print("  [STEP 1] LOAD, AUDIT & VALIDATE DỮ LIỆU")
    print("=" * 65)

    df = pd.read_csv(raw_path)
    # Hỗ trợ linh hoạt tên cột timestamp / Datetime
    time_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    # Chỉ giữ các biến trong danh mục
    df = df[FEATURES].astype(float).copy()
    print(f"  • Kích thước dữ liệu gốc   : {df.shape}")
    print(f"  • Khung thời gian quan sát : {df.index.min()} → {df.index.max()}")

    # 1. Kiểm tra tính liên tục theo giờ
    inferred_freq = pd.infer_freq(df.index)
    print(f"  • Tần suất suy luận        : {inferred_freq}")
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    if len(full_idx) != len(df):
        print(f"  [CẢNH BÁO] Phát hiện đứt đoạn thời gian. Đang reindex về chuỗi liên tục...")
        df = df.reindex(full_idx)

    # 2. Báo cáo Missing values
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    missing_df = pd.DataFrame({"Missing_Count": missing_count, "Missing_Percentage": missing_pct})
    missing_csv = TABLES_DIR / "missing_summary.csv"
    missing_df.to_csv(missing_csv, index=True)
    print(f"\n  • Báo cáo khuyết thiếu đã lưu: {missing_csv}")
    print(missing_df.to_string())

    # Imputation: forward-fill sau đó backward-fill để bù các điểm khuyết
    df = df.ffill().bfill()

    # 3. Thống kê mô tả tổng quan (Data Summary)
    summary_df = df.describe().round(2).T
    summary_df["IQR"] = summary_df["75%"] - summary_df["25%"]
    summary_csv = TABLES_DIR / "cv01_data_summary.csv"
    summary_df.to_csv(summary_csv, index=True)
    print(f"\n  • Thống kê mô tả đã lưu : {summary_csv}")

    # 4. Phát hiện Outlier bằng phương pháp IQR
    outlier_records = {}
    for col in FEATURES:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_records[col] = {
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower_Bound": round(lower_bound, 2),
            "Upper_Bound": round(upper_bound, 2),
            "Outlier_Count": int(outliers),
            "Outlier_Pct": round(outliers / len(df) * 100, 2)
        }
    outlier_df = pd.DataFrame(outlier_records).T
    outlier_csv = TABLES_DIR / "outlier_summary.csv"
    outlier_df.to_csv(outlier_csv, index=True)
    print(f"  • Báo cáo Outlier đã lưu : {outlier_csv}")
    print(outlier_df[["Lower_Bound", "Upper_Bound", "Outlier_Count", "Outlier_Pct"]].to_string())

    # Quyết định xử lý: Giữ nguyên các điểm peak phụ tải cực trị (vì là dữ liệu vật lý quan trọng),
    # chỉ loại bỏ/clip nếu vượt ngưỡng phi lý (như humidity > 100% hoặc wind_speed < 0)
    df["humidity"] = df["humidity"].clip(0.0, 100.0)
    df["wind_speed"] = df["wind_speed"].clip(lower=0.0)

    print(f"\n  [STEP 1 HOÀN TẤT] Tổng số quan sát sẵn sàng: {len(df):,} dòng.\n")
    return df


# ----------------------------------------------------------
# STEP 2: Khám phá Trực quan Hóa (EDA Visualization)
# ----------------------------------------------------------
def step2_eda_visualize(df: pd.DataFrame) -> None:
    """
    Vẽ và xuất bản bộ biểu đồ EDA chuẩn:
      1. Chuỗi thời gian 4 biến (line plot)
      2. Phân tán phi tuyến load vs temp (màu theo giờ)
      3. Ma trận tương quan Pearson heatmap
      4. Chu kỳ mùa vụ ngày & tuần (daily & weekly seasonality)
      5. Biểu đồ tự tương quan ACF & PACF của load
    """
    print("=" * 65)
    print("  [STEP 2] TRỰC QUAN HÓA KHÁM PHÁ (EDA VISUALIZATION)")
    print("=" * 65)

    # 2.1 Chuỗi thời gian 4 biến
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    units = ["MW", "°C", "%", "m/s"]

    for ax, col, color, unit in zip(axes, FEATURES, colors, units):
        ax.plot(df.index, df[col], color=color, linewidth=0.55, label=f"{col} ({unit})")
        ax.set_ylabel(f"{col}\n({unit})", fontsize=9, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    axes[0].set_title("Biểu đồ Chuỗi Thời Gian Đa biến: Phụ tải & Thời tiết (2016 - 2018)", fontsize=13, fontweight="bold", pad=12)
    axes[-1].set_xlabel("Thời gian (Năm - Tháng)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    p1 = FIGURES_DIR / "load_timeseries.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  • Đã lưu: {p1}")

    # 2.2 Scatter phi tuyến load vs temp (mã hóa màu theo giờ)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["temp"], df["load"],
        c=df.index.hour, cmap="plasma",
        alpha=0.35, s=6, rasterized=True
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Giờ trong ngày (Hour 0 - 23)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Nhiệt độ không khí temp (°C)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Điện năng tiêu thụ load (MW)", fontsize=10, fontweight="bold")
    ax.set_title("Mối Quan Hệ Phi Tuyến (U-Curve): Phụ tải Điện vs Nhiệt độ", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    p2 = FIGURES_DIR / "scatter_load_vs_temp.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  • Đã lưu: {p2}")

    # 2.3 Ma trận tương quan Pearson Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    corr = df[FEATURES].corr()
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, square=True,
        linewidths=1.0, cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title("Ma trận Hệ số Tương quan Pearson (4 Biến số)", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    p3 = FIGURES_DIR / "correlation_heatmap.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  • Đã lưu: {p3}")

    # 2.4 Chu kỳ mùa vụ Ngày & Tuần
    fig, (ax_d, ax_w) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Daily Profile
    daily_stats = df.groupby(df.index.hour)["load"].agg(["mean", "std"])
    ax_d.plot(daily_stats.index, daily_stats["mean"], color="#1f77b4", marker="o", linewidth=2.0, label="Trung bình")
    ax_d.fill_between(
        daily_stats.index,
        daily_stats["mean"] - daily_stats["std"],
        daily_stats["mean"] + daily_stats["std"],
        color="#1f77b4", alpha=0.18, label="±1 Std"
    )
    ax_d.set_title("Chu kỳ Phụ tải theo Giờ trong Ngày (24h)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Giờ trong ngày (0 - 23h)", fontsize=9)
    ax_d.set_ylabel("Phụ tải (MW)", fontsize=9)
    ax_d.set_xticks(range(0, 24, 2))
    ax_d.grid(True, linestyle="--", alpha=0.4)
    ax_d.legend(fontsize=8)

    # Weekly Profile
    days_labels = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]
    weekly_stats = df.groupby(df.index.dayofweek)["load"].agg(["mean", "std"])
    ax_w.bar(range(7), weekly_stats["mean"], color="#2ca02c", alpha=0.85, width=0.55, yerr=weekly_stats["std"], capsize=4)
    ax_w.set_title("Chu kỳ Phụ tải theo Ngày trong Tuần (7 Ngày)", fontsize=11, fontweight="bold")
    ax_w.set_xlabel("Ngày trong tuần", fontsize=9)
    ax_w.set_ylabel("Phụ tải (MW)", fontsize=9)
    ax_w.set_xticks(range(7))
    ax_w.set_xticklabels(days_labels)
    ax_w.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.tight_layout()
    p4 = FIGURES_DIR / "seasonality_daily_weekly.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  • Đã lưu: {p4}")

    # 2.5 Biểu đồ Tự tương quan ACF & PACF của load (48 lags)
    fig, (ax_acf, ax_pacf) = plt.subplots(2, 1, figsize=(12, 6.5))
    plot_acf(df["load"], lags=48, ax=ax_acf, title="Hàm Tự tương quan (ACF) - Biến Load (48 Lags)")
    plot_pacf(df["load"], lags=48, ax=ax_pacf, title="Hàm Tự tương quan Riêng (PACF) - Biến Load (48 Lags)", method="ywm")
    ax_acf.grid(True, linestyle="--", alpha=0.4)
    ax_pacf.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    p5 = FIGURES_DIR / "load_acf_pacf.png"
    fig.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  • Đã lưu: {p5}")

    print("[STEP 2 HOÀN TẤT]\n")


# ----------------------------------------------------------
# STEP 3: Chuẩn hóa & Phân chia Tập dữ liệu (Anti-Leakage Split)
# ----------------------------------------------------------
def step3_scale_and_split(df: pd.DataFrame):
    """
    Phân chia dữ liệu tuần tự theo thời gian (70% Train, 15% Val, 15% Test).
    StandardScaler chỉ fit trên Train để chống rò rỉ dữ liệu (Lookahead Bias).
    """
    print("=" * 65)
    print("  [STEP 3] CHUẨN HÓA & PHÂN CHIA TẬP (70% TRAIN / 15% VAL / 15% TEST)")
    print("=" * 65)

    n = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    n_test  = n - n_train - n_val

    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train : n_train + n_val].copy()
    test_df  = df.iloc[n_train + n_val:].copy()

    split_info = [
        {"Subset": "Train", "Rows": len(train_df), "Pct": round(len(train_df)/n*100, 2), "Start": str(train_df.index.min()), "End": str(train_df.index.max())},
        {"Subset": "Validation", "Rows": len(val_df), "Pct": round(len(val_df)/n*100, 2), "Start": str(val_df.index.min()), "End": str(val_df.index.max())},
        {"Subset": "Test", "Rows": len(test_df), "Pct": round(len(test_df)/n*100, 2), "Start": str(test_df.index.min()), "End": str(test_df.index.max())},
        {"Subset": "Total", "Rows": n, "Pct": 100.0, "Start": str(df.index.min()), "End": str(df.index.max())}
    ]
    split_summary_df = pd.DataFrame(split_info)
    split_csv = TABLES_DIR / "cv01_split_info.csv"
    split_summary_df.to_csv(split_csv, index=False)
    print(f"  • Bảng tỷ lệ phân chia đã lưu: {split_csv}")
    print(split_summary_df.to_string(index=False))

    # Khởi tạo và fit StandardScaler DUY NHẤT trên Train
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])

    def _transform_subset(subset: pd.DataFrame) -> pd.DataFrame:
        scaled = scaler.transform(subset[FEATURES])
        return pd.DataFrame(scaled, index=subset.index, columns=FEATURES)

    train_scaled = _transform_subset(train_df)
    val_scaled   = _transform_subset(val_df)
    test_scaled  = _transform_subset(test_df)

    # Lưu scaler.pkl
    scaler_path = PROCESSED_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  • Đã lưu scaler object: {scaler_path}")
    for col, m, s in zip(FEATURES, scaler.mean_, scaler.scale_):
        print(f"    - {col:<10}: Mean = {m:>9.2f} | Scale (Std) = {s:>9.2f}")

    # Xuất các file CSV đã scale cho CV02 (VAR) và CV04 (LSTM)
    train_scaled.to_csv(PROCESSED_DIR / "train.csv")
    val_scaled.to_csv(PROCESSED_DIR / "val.csv")
    test_scaled.to_csv(PROCESSED_DIR / "test.csv")
    print("  • Đã xuất train.csv, val.csv, test.csv vào data/processed/")

    print("[STEP 3 HOÀN TẤT]\n")
    return train_scaled, val_scaled, test_scaled, scaler


# ----------------------------------------------------------
# STEP 4: Tạo Cửa Sổ Độ Trễ Đa Biến (96 Lag Features, L=24)
# ----------------------------------------------------------
def step4_create_lag_features(df_scaled: pd.DataFrame, lag: int = 24) -> pd.DataFrame:
    """
    Tạo 96 lag features từ cửa sổ lịch sử L=24 cho cả 4 biến:
      [col_lag1, col_lag2, ..., col_lag24] cho mỗi biến trong FEATURES.
    Sau khi tạo, loại bỏ các hàng NaN đầu chuỗi.
    """
    df_lag = df_scaled.copy()
    for col in FEATURES:
        for k in range(1, lag + 1):
            df_lag[f"{col}_lag{k}"] = df_scaled[col].shift(k)

    df_lag = df_lag.dropna()
    return df_lag


# ----------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------
def main():
    print("\n" + "=" * 65)
    print("  BÀI THỰC HÀNH SỐ 4 — MULTIVARIATE TIME SERIES FORECASTING")
    print("  CV01 PIPELINE: TIỀN XỬ LÝ ĐA BIẾN & TẠO ĐẶC TRƯNG TRỄ")
    print("  Nhóm trưởng phụ trách: Hà Duy Hưng (Lead)")
    print("=" * 65 + "\n")

    # 1. Đảm bảo dữ liệu thô
    raw_path = ensure_raw_data(RAW_DATA_PATH)

    # 2. Load & Validate
    df = step1_load_and_validate(raw_path)

    # 3. Trực quan hóa EDA
    step2_eda_visualize(df)

    # 4. Chuẩn hóa & Phân chia 70/15/15
    train_sc, val_sc, test_sc, scaler = step3_scale_and_split(df)

    # 5. Tạo Lag Features (L=24) độc lập trên từng tập
    print("=" * 65)
    print("  [STEP 4] TẠO MA TRẬN 96 ĐẶC TRƯNG TRỄ (L=24 BƯỚC)")
    print("=" * 65)
    train_lag = step4_create_lag_features(train_sc, lag=24)
    val_lag   = step4_create_lag_features(val_sc,   lag=24)
    test_lag  = step4_create_lag_features(test_sc,  lag=24)

    train_lag.to_csv(PROCESSED_DIR / "train_lag.csv")
    val_lag.to_csv(PROCESSED_DIR / "val_lag.csv")
    test_lag.to_csv(PROCESSED_DIR / "test_lag.csv")

    n_lags = 24 * len(FEATURES)
    print(f"  • Số lượng đặc trưng trễ tạo ra : {n_lags} (24 lags × 4 biến)")
    print(f"  • Train lag shape               : {train_lag.shape}")
    print(f"  • Val lag shape                 : {val_lag.shape}")
    print(f"  • Test lag shape                : {test_lag.shape}")
    print("  • Đã xuất train_lag.csv, val_lag.csv, test_lag.csv vào data/processed/")
    print("[STEP 4 HOÀN TẤT]\n")

    print("=" * 65)
    print("  ✅ [CV01 THÀNH CÔNG RỰC RỠ] TOÀN BỘ SẢN PHẨM ĐẦU RA ĐÃ SẴN SÀNG:")
    print(f"     1. Dữ liệu: {PROCESSED_DIR}")
    print(f"     2. Biểu đồ: {FIGURES_DIR}")
    print(f"     3. Bảng số liệu: {TABLES_DIR}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
