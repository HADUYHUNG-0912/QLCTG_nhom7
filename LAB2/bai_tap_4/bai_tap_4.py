# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 2
BÀI TẬP 4: KIỂM TRA VÀ XỬ LÝ CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY CLEANING)
---------------------------------------------------------------------------------
Thành viên phụ trách: Hoàng (CV05 - nhánh hoang-bt4)
Thư mục: LAB2/bai_tap_4/
Tệp thực thi: bai_tap_4.py
Dataset đầu vào: LAB2/data/raw/AirPassengers.csv

=================================================================================
🤖 CHỈ THỊ DÀNH CHO AI AGENT (AI AGENT PROMPT & GUARDRAILS SPECIFICATION)
=================================================================================
Nếu bạn là một AI Agent (Cursor, GitHub Copilot, Gemini, ChatGPT, Claude...):
1. TUÂN THỦ KIẾN TRÚC ĐỒNG NHẤT:
   - Không được đổi tên 4 hàm chuẩn: `step1_load_data`, `step2_execute_analysis`, 
     `step3_visualize`, `step4_conclude_and_report`.
   - Giữ nguyên cấu trúc trả về `Dict[str, Any]` và các type hints.
2. NGUYÊN TẮC KỸ THUẬT:
   - BẮT BUỘC tạo bản sao `series.copy()`, tuyệt đối không ghi đè làm hỏng chuỗi dữ liệu gốc!
   - Tỷ lệ thiếu giả lập: 10% ngẫu nhiên với seed=101 (`np.random.seed(101)`).
   - Thêm đúng 3 điểm ngoại lai (Outliers) có giá trị phóng đại gấp 3.5 lần.
   - So sánh 2 phương pháp xử lý missing: Forward-fill (`ffill()`) vs Time Interpolation (`interpolate(method='time')`).
   - Xử lý Outlier bằng thuật toán Hampel Filter (Rolling Median +/- 3 * MAD).
3. NGUYÊN TẮC TRỰC QUAN HÓA & LƯU TRỮ:
   - KHÔNG tự ý lưu tệp hình ảnh (.png/.jpg) ra ổ đĩa để tránh làm nặng repository.
   - Hàm `step3_visualize` tạo biểu đồ so sánh Trước - Sau khi làm sạch, giải phóng bộ nhớ hoặc hiển thị nếu cần.
   - Báo cáo đánh giá sai số MAE/RMSE giữa chuỗi phục hồi và chuỗi gốc tại `LAB2/bai_tap_4/ket_qua_bai_4.txt`.
4. CÂU HỎI NGHIỆM THU:
   - Câu 1: Forward-fill hay Time Interpolation cho kết quả phục hồi tốt hơn trên chuỗi có xu hướng tăng?
   - Câu 2: Hampel Filter có ưu điểm gì so với Z-score khi phát hiện ngoại lai cục bộ?
   - Câu 3: Sai số tái tạo (MAE, RMSE) sau khi làm sạch là bao nhiêu?
=================================================================================
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple

# Cấu hình encoding UTF-8 để không bị lỗi console trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ TOÀN CỤC
# -------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AirPassengers.csv"

CONFIG = {
    "missing_rate": 0.10,
    "seed": 101,
    "outlier_indices": [25, 75, 110],
    "outlier_multiplier": 3.5,
    "target_col": "Passengers",
    "datetime_col": "Month",
    # 13 tháng bao quát trọn một chu kỳ mùa vụ 12 tháng, giúp Hampel Filter
    # không nhầm các đỉnh mùa vụ tự nhiên với spike bất thường.
    "hampel_window": 13,
    "hampel_n_sigma": 3.0,
}


def _error_metrics(actual: pd.Series, predicted: pd.Series) -> Tuple[float, float]:
    """Trả về (MAE, RMSE) trên hai chuỗi đã căn chỉnh chỉ mục."""
    error = predicted.astype(float) - actual.astype(float)
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    return mae, rmse


# -------------------------------------------------------------------------------
# 2. BƯỚC 1: TẢI DỮ LIỆU GỐC & TẠO DỮ LIỆU KHUYẾT TẬT MÔ PHỎNG
# -------------------------------------------------------------------------------
def step1_load_data(data_path: Path) -> Tuple[pd.Series, pd.Series]:
    """
    Tải chuỗi gốc và tạo bản sao chuỗi bị khuyết tật (10% Missing + 3 Outliers).
    """
    print(f"[BƯỚC 1] Tải dữ liệu và giả lập lỗi chất lượng dữ liệu...")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tại {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {CONFIG["datetime_col"], CONFIG["target_col"]}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Tệp dữ liệu thiếu cột bắt buộc: {sorted(missing_columns)}")

    df[CONFIG["datetime_col"]] = pd.to_datetime(df[CONFIG["datetime_col"]])
    df.set_index(CONFIG["datetime_col"], inplace=True)
    df.sort_index(inplace=True)
    if df.index.has_duplicates:
        raise ValueError("Chỉ mục thời gian có giá trị trùng lặp.")

    clean_series = df[CONFIG["target_col"]].astype(float)
    if clean_series.isna().any():
        raise ValueError("Chuỗi gốc chứa NaN nên không thể dùng làm chuẩn đánh giá.")

    # Tạo bản sao khuyết tật
    np.random.seed(CONFIG["seed"])
    corrupted_series = clean_series.copy()

    # 1. Tạo Missing ngẫu nhiên
    missing_mask = np.random.rand(len(corrupted_series)) < CONFIG["missing_rate"]
    corrupted_series.loc[missing_mask] = np.nan

    # 2. Tạo 3 Outliers đột biến
    for idx in CONFIG["outlier_indices"]:
        if idx < len(corrupted_series):
            corrupted_series.iloc[idx] = clean_series.iloc[idx] * CONFIG["outlier_multiplier"]

    print(f" -> Chuỗi gốc: {len(clean_series)} điểm.")
    print(f" -> Số điểm bị khuyết thiếu (NaN): {corrupted_series.isna().sum()} điểm ({corrupted_series.isna().mean()*100:.1f}%).")
    print(f" -> Các vị trí ngoại lai được tạo: {CONFIG['outlier_indices']}")
    return clean_series, corrupted_series


# -------------------------------------------------------------------------------
# 2. BƯỚC 2: SO SÁNH PHƯƠNG PHÁP XỬ LÝ & LÀM SẠCH BẰNG HAMPEL FILTER
# -------------------------------------------------------------------------------
def step2_execute_analysis(clean_series: pd.Series, corrupted_series: pd.Series) -> Dict[str, Any]:
    """
    So sánh Forward-fill vs Time Interpolation, sau đó áp dụng Hampel Filter lọc Outlier.
    """
    print(f"[BƯỚC 2] Thực hiện làm sạch dữ liệu và đánh giá sai số...")

    # 1. Xử lý Missing bằng Forward-fill
    filled_ffill = corrupted_series.ffill().bfill()

    # 2. Xử lý Missing bằng Time-based Interpolation
    filled_interpolated = corrupted_series.interpolate(method="time").bfill()

    # Tính sai số phục hồi chỉ trên đúng các vị trí Missing.
    missing_mask = corrupted_series.isna()
    mae_ffill, rmse_ffill = _error_metrics(
        clean_series.loc[missing_mask], filled_ffill.loc[missing_mask]
    )
    mae_interp, rmse_interp = _error_metrics(
        clean_series.loc[missing_mask], filled_interpolated.loc[missing_mask]
    )

    print(
        f" -> Forward-fill trên điểm Missing:      "
        f"MAE={mae_ffill:.2f}, RMSE={rmse_ffill:.2f}"
    )
    print(
        f" -> Time Interpolation trên điểm Missing: "
        f"MAE={mae_interp:.2f}, RMSE={rmse_interp:.2f}"
    )

    # 3. Lọc ngoại lai bằng Hampel Filter (Rolling Median & MAD)
    window_size = CONFIG["hampel_window"]
    min_periods = window_size // 2 + 1
    rolling_median = filled_interpolated.rolling(
        window=window_size, center=True, min_periods=min_periods
    ).median()
    diff = np.abs(filled_interpolated - rolling_median)
    mad = diff.rolling(
        window=window_size, center=True, min_periods=min_periods
    ).median()
    threshold = CONFIG["hampel_n_sigma"] * 1.4826 * mad
    is_outlier = (diff > threshold).fillna(False)

    # Khắc phục Outlier bằng cách thay bằng Rolling Median
    restored_series = filled_interpolated.copy()
    restored_series[is_outlier] = rolling_median[is_outlier]

    detected_positions = np.flatnonzero(is_outlier.to_numpy()).tolist()
    expected_positions = CONFIG["outlier_indices"]
    true_positives = sorted(set(detected_positions).intersection(expected_positions))
    false_positives = sorted(set(detected_positions).difference(expected_positions))
    missed_outliers = sorted(set(expected_positions).difference(detected_positions))

    before_mae, before_rmse = _error_metrics(clean_series, filled_interpolated)
    overall_mae, overall_rmse = _error_metrics(clean_series, restored_series)

    results = {
        "filled_ffill": filled_ffill,
        "filled_interp": filled_interpolated,
        "restored_series": restored_series,
        "mae_ffill": mae_ffill,
        "rmse_ffill": rmse_ffill,
        "mae_interp": mae_interp,
        "rmse_interp": rmse_interp,
        "missing_count": int(missing_mask.sum()),
        "missing_rate_actual": float(missing_mask.mean()),
        "detected_outliers": len(detected_positions),
        "detected_positions": detected_positions,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "missed_outliers": missed_outliers,
        "before_hampel_mae": before_mae,
        "before_hampel_rmse": before_rmse,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
    }

    print(
        f" -> Hampel Filter (window={window_size}) phát hiện các vị trí: "
        f"{detected_positions}"
    )
    print(
        f" -> Đúng {len(true_positives)}/3 spike; phát hiện nhầm "
        f"{len(false_positives)}; bỏ sót {len(missed_outliers)}."
    )
    print(
        f" -> Trước Hampel: MAE={before_mae:.2f}, RMSE={before_rmse:.2f}; "
        f"sau Hampel: MAE={overall_mae:.2f}, RMSE={overall_rmse:.2f}."
    )
    return results


# -------------------------------------------------------------------------------
# 3. BƯỚC 3: TRỰC QUAN HÓA SO SÁNH BEFORE VS AFTER
# -------------------------------------------------------------------------------
def step3_visualize(clean_series: pd.Series, corrupted_series: pd.Series, analysis_results: Dict[str, Any], show_plot: bool = False) -> None:
    """
    Khởi tạo đồ thị so sánh chuỗi gốc, chuỗi khuyết tật và chuỗi sau khi phục hồi.
    Lưu ý: Không tự ý xuất file ảnh ra ổ đĩa theo quy định tinh gọn của dự án.
    """
    print(f"[BƯỚC 3] Đang khởi tạo đồ thị so sánh Trước - Sau khi làm sạch...")
    restored = analysis_results["restored_series"]
    detected_positions = analysis_results["detected_positions"]
    detected_index = restored.index[detected_positions]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # Đồ thị 1: Chuỗi lỗi vs Chuỗi gốc
    axes[0].plot(clean_series.index, clean_series.values, label="Chuỗi Gốc Chuẩn", color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[0].plot(corrupted_series.index, corrupted_series.values, label="Chuỗi Bị Lỗi (10% NaN & 3 Spikes)", color="crimson", linewidth=1.5, marker="o", markersize=3)
    missing_index = corrupted_series.index[corrupted_series.isna()]
    axes[0].scatter(
        missing_index,
        clean_series.loc[missing_index],
        label="Vị trí bị khuyết (giá trị gốc để đối chiếu)",
        color="darkorange",
        marker="x",
        s=45,
        zorder=4,
    )
    axes[0].set_title("1. Dữ Liệu Khuyết Tật (Trước Khi Làm Sạch)", fontsize=13)
    axes[0].set_ylabel("Lượng khách")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Đồ thị 2: Chuỗi sau khi phục hồi vs Chuỗi gốc
    axes[1].plot(clean_series.index, clean_series.values, label="Chuỗi Gốc Chuẩn", color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[1].plot(restored.index, restored.values, label="Chuỗi Phục Hồi (Time Interpolation + Hampel Filter)", color="royalblue", linewidth=2.0)
    axes[1].scatter(
        detected_index,
        restored.loc[detected_index],
        label="Spike được Hampel phát hiện và thay thế",
        color="seagreen",
        marker="D",
        s=38,
        zorder=4,
    )
    axes[1].set_title(f"2. Dữ Liệu Đã Phục Hồi (Sau Khi Làm Sạch - RMSE: {analysis_results['overall_rmse']:.2f})", fontsize=13)
    axes[1].set_xlabel("Thời gian")
    axes[1].set_ylabel("Lượng khách")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(" -> Đã hoàn thành trực quan hóa làm sạch dữ liệu (không xuất tệp ảnh ra đĩa).")
    return None


# -------------------------------------------------------------------------------
# 4. BƯỚC 4: TỔNG KẾT VÀ BÁO CÁO KẾT QUẢ BÀI 4
# -------------------------------------------------------------------------------
def step4_conclude_and_report(analysis_results: Dict[str, Any]) -> str:
    """
    Tổng hợp nhận xét học thuật và trả lời các câu hỏi nghiệm thu Bài 4.
    """
    print("\n[BƯỚC 4] TỔNG HỢP KẾT QUẢ VÀ TRẢ LỜI CÂU HỎI NGHIỆM THU BÀI 4:")
    report_text = f"""
=================================================================================
BÁO CÁO PHÂN TÍCH KẾT QUẢ BÀI TẬP 4 (KIỂM TRA & XỬ LÝ CHẤT LƯỢNG DỮ LIỆU)
=================================================================================
1. KIỂM TOÁN DỮ LIỆU KHUYẾT TẬT:
   - Chuỗi gốc được giữ nguyên; mọi lỗi chỉ được tạo trên bản sao series.copy().
   - Seed = {CONFIG['seed']}; số NaN = {analysis_results['missing_count']}/144
     ({analysis_results['missing_rate_actual'] * 100:.2f}%); 3 spike nhân {CONFIG['outlier_multiplier']} tại vị trí {CONFIG['outlier_indices']}.

2. BẢNG SO SÁNH XỬ LÝ MISSING (CHỈ TÍNH TRÊN CÁC ĐIỂM NaN):
   +--------------------------+----------+----------+
   | Phương pháp              | MAE      | RMSE     |
   +--------------------------+----------+----------+
   | Forward-fill             | {analysis_results['mae_ffill']:8.2f} | {analysis_results['rmse_ffill']:8.2f} |
   | Time-based Interpolation | {analysis_results['mae_interp']:8.2f} | {analysis_results['rmse_interp']:8.2f} |
   +--------------------------+----------+----------+
   - Time-based Interpolation tốt hơn theo cả MAE và RMSE. Phương pháp này bám theo độ dốc
     giữa hai mốc thời gian, trong khi ffill tạo các đoạn nằm ngang và trễ so với xu hướng tăng.

3. HIỆU QUẢ CỦA HAMPEL FILTER:
   - Cửa sổ {CONFIG['hampel_window']} tháng bao quát một chu kỳ mùa vụ; ngưỡng = {CONFIG['hampel_n_sigma']:.0f} x 1.4826 x MAD.
   - Vị trí phát hiện: {analysis_results['detected_positions']}.
   - Đúng {len(analysis_results['true_positives'])}/3 spike; phát hiện nhầm {len(analysis_results['false_positives'])};
     bỏ sót {len(analysis_results['missed_outliers'])}.
   - Hampel dùng trung vị trượt và MAD nên ít bị chính spike kéo lệch ngưỡng hơn Z-score,
     vốn dựa vào mean và standard deviation nhạy với giá trị cực đoan.

4. BẢNG SAI SỐ TỔNG THỂ TRƯỚC VÀ SAU LÀM SẠCH NGOẠI LAI:
   +--------------------------------------+----------+----------+
   | Trạng thái                           | MAE      | RMSE     |
   +--------------------------------------+----------+----------+
   | Sau nội suy, trước Hampel             | {analysis_results['before_hampel_mae']:8.2f} | {analysis_results['before_hampel_rmse']:8.2f} |
   | Sau nội suy và Hampel                 | {analysis_results['overall_mae']:8.2f} | {analysis_results['overall_rmse']:8.2f} |
   +--------------------------------------+----------+----------+

5. TRẢ LỜI CÂU HỎI LÝ THUYẾT 7 VÀ 8:
   - Câu 7: Ba lỗi phổ biến là missing values (mất quan sát), outliers (giá trị bất thường)
     và non-stationarity (mean/variance thay đổi do xu hướng hoặc mùa vụ).
   - Câu 8: Có thể dùng ffill, bfill, nội suy tuyến tính/spline theo thời gian, giá trị cùng kỳ
     mùa vụ, hoặc mô hình bù khuyết như KNN, Kalman Filter và Prophet. Cần chọn phương pháp
     theo cơ chế sinh dữ liệu và chỉ đánh giá trên các điểm bị khuyết để tránh kết luận sai.
=================================================================================
"""
    print(report_text)

    report_file = BASE_DIR / "ket_qua_bai_4.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f" -> Đã lưu báo cáo phân tích tại: {report_file}")
    return report_text


# -------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH
# -------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("KHỞI CHẠY BÀI TẬP 4: KIỂM TRA & XỬ LÝ CHẤT LƯỢNG DỮ LIỆU (DATA CLEANING)")
    print("=" * 80)
    clean_series, corrupted_series = step1_load_data(DATA_PATH)
    analysis_results = step2_execute_analysis(clean_series, corrupted_series)
    step3_visualize(clean_series, corrupted_series, analysis_results)
    step4_conclude_and_report(analysis_results)
    print("\n[HOÀN THÀNH BÀI TẬP 4 THÀNH CÔNG]")


if __name__ == "__main__":
    main()
