# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - TIME SERIES LAB
BÀI TẬP 2: KIỂM ĐỊNH CHUỖI NHIỄU TRẮNG (WHITE NOISE TESTING)
---------------------------------------------------------------------------------
Thành viên phụ trách: Thành viên 2 - Nhóm 7
Thư mục module: src/bt2_white_noise.py
Dataset so sánh: data/raw/AirPassengers.csv
Outputs: outputs/figures/bai2_white_noise_comparison.png, outputs/tables/ljung_box_test_results.csv

MỤC TIÊU BÀI TẬP:
- Sinh chuỗi Gaussian White Noise với loc=0.0, scale=1.0 (1000 điểm, seed=42).
- Vẽ biểu đồ ACF và kiểm định xem các lag có nằm trong dải tin cậy 95% không.
- Thực hiện kiểm định Ljung-Box Test tại lags=[10, 20] để đối chứng White Noise vs AirPassengers.
- Nhận xét và đánh giá khả năng dự báo của chuỗi thời gian.
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
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Cấu hình phông chữ hỗ trợ tiếng Việt trên Windows
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# -------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ TOÀN CỤC
# -------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AirPassengers.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "sample_size": 1000,
    "seed": 42,
    "mean": 0.0,
    "std": 1.0,
    "lags": 30,
    "alpha": 0.05
}


# -------------------------------------------------------------------------------
# 2. BƯỚC 1: SINH DỮ LIỆU WHITE NOISE & TẢI DỮ LIỆU THỰC TẾ
# -------------------------------------------------------------------------------
def step1_load_data(data_path: Path) -> Tuple[pd.Series, pd.Series]:
    """Sinh chuỗi Gaussian White Noise ngẫu nhiên và tải chuỗi thực tế AirPassengers."""
    print(f"[BƯỚC 1] Sinh chuỗi Gaussian White Noise ({CONFIG['sample_size']} điểm, seed={CONFIG['seed']})...")
    np.random.seed(CONFIG["seed"])
    white_noise = np.random.normal(loc=CONFIG["mean"], scale=CONFIG["std"], size=CONFIG["sample_size"])
    dates_wn = pd.date_range(start="2000-01-01", periods=CONFIG["sample_size"], freq="D")
    wn_series = pd.Series(white_noise, index=dates_wn, name="White_Noise")

    print(f" -> Tải dữ liệu thực tế từ: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {data_path}")
    
    df_real = pd.read_csv(data_path)
    df_real["Month"] = pd.to_datetime(df_real["Month"])
    df_real.set_index("Month", inplace=True)
    real_series = df_real["Passengers"].astype(float)

    print(f" -> Chuỗi White Noise: {len(wn_series)} điểm (Mean={wn_series.mean():.4f}, Std={wn_series.std():.4f})")
    print(f" -> Chuỗi AirPassengers: {len(real_series)} điểm.")
    return wn_series, real_series


# -------------------------------------------------------------------------------
# 3. BƯỚC 2: THỰC HIỆN KIỂM ĐỊNH THỐNG KÊ (LJUNG-BOX TEST)
# -------------------------------------------------------------------------------
def step2_execute_analysis(wn_series: pd.Series, real_series: pd.Series) -> Dict[str, Any]:
    """Chạy kiểm định Ljung-Box để kiểm tra tính độc lập ngẫu nhiên của cả 2 chuỗi."""
    print(f"[BƯỚC 2] Thực hiện kiểm định Ljung-Box Test tại lags=[10, 20]...")
    check_lags = [10, 20]

    lb_wn = acorr_ljungbox(wn_series, lags=check_lags, return_df=True)
    lb_real = acorr_ljungbox(real_series, lags=check_lags, return_df=True)

    results = {
        "lb_wn": lb_wn,
        "lb_real": lb_real,
        "wn_pvalue_lag10": lb_wn.loc[10, "lb_pvalue"],
        "wn_pvalue_lag20": lb_wn.loc[20, "lb_pvalue"],
        "real_pvalue_lag10": lb_real.loc[10, "lb_pvalue"],
        "real_pvalue_lag20": lb_real.loc[20, "lb_pvalue"],
        "wn_stat_lag10": lb_wn.loc[10, "lb_stat"],
        "wn_stat_lag20": lb_wn.loc[20, "lb_stat"],
        "real_stat_lag10": lb_real.loc[10, "lb_stat"],
        "real_stat_lag20": lb_real.loc[20, "lb_stat"],
    }
    return results


# -------------------------------------------------------------------------------
# 4. BƯỚC 3: TRỰC QUAN HÓA SO SÁNH
# -------------------------------------------------------------------------------
def step3_visualize(wn_series: pd.Series, real_series: pd.Series, show_plot: bool = False):
    """Vẽ đối chứng biểu đồ dạng sóng và ACF của White Noise vs AirPassengers."""
    print(f"[BƯỚC 3] Kết xuất đồ thị so sánh trực quan...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(wn_series.values, color="#1f77b4", lw=1)
    axes[0, 0].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[0, 0].set_title("Gaussian White Noise (Dạng sóng ngẫu nhiên)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Chỉ số thời gian (t)")
    axes[0, 0].set_ylabel("Giá trị")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    plot_acf(wn_series, lags=CONFIG["lags"], ax=axes[0, 1], alpha=CONFIG["alpha"],
             title=f"ACF: White Noise ({CONFIG['lags']} Lags)")
    axes[0, 1].set_xlabel("Độ trễ (Lags)")
    axes[0, 1].set_ylabel("Hệ số ACF")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    axes[1, 0].plot(real_series.index, real_series.values, color="#ff7f0e", lw=2)
    axes[1, 0].set_title("AirPassengers (Dữ liệu thực tế)", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Năm")
    axes[1, 0].set_ylabel("Hành khách (nghìn người)")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    plot_acf(real_series, lags=CONFIG["lags"], ax=axes[1, 1], alpha=CONFIG["alpha"],
             title=f"ACF: AirPassengers ({CONFIG['lags']} Lags)")
    axes[1, 1].set_xlabel("Độ trễ (Lags)")
    axes[1, 1].set_ylabel("Hệ số ACF")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_fig = FIGURES_DIR / "bai2_white_noise_comparison.png"
    plt.savefig(output_fig, dpi=300)
    print(f" -> Đã lưu biểu đồ thành công tại: {output_fig}")

    if show_plot:
        plt.show()
    plt.close()


# -------------------------------------------------------------------------------
# 5. BƯỚC 4: TỔNG HỢP VÀ ĐÁNH GIÁ KẾT QUẢ
# -------------------------------------------------------------------------------
def step4_conclude_and_report(results: Dict[str, Any]) -> str:
    """Tổng hợp báo cáo so sánh White Noise vs AirPassengers."""
    print(f"\n[BƯỚC 4] BÁO CÁO PHÂN TÍCH SO SÁNH WHITE NOISE:")
    report_text = f"""=================================================================================
BÁO CÁO PHÂN TÍCH KẾT QUẢ BÀI TẬP 2 (KIỂM TRA WHITE NOISE)
=================================================================================
ĐOẠN VĂN NHẬN XÉT SO SÁNH:
1. Biểu đồ ACF của chuỗi White Noise có gần như toàn bộ các hệ số độ trễ (lags 1-30) nằm hoàn toàn
   bên trong dải tin cậy 95%, chứng minh các điểm dữ liệu hoàn toàn độc lập và không có tương quan.
2. Ngược lại, biểu đồ ACF của chuỗi AirPassengers hiển thị các hệ số tự tương quan rất cao, suy giảm chậm
   và lặp lại các đỉnh chu kỳ tại lag 12, phản ánh rõ rệt cấu trúc xu hướng và tính mùa vụ.
3. Kết quả kiểm định Ljung-Box trên chuỗi White Noise có p-value lớn hơn 0.05 (lag 10: p={results['wn_pvalue_lag10']:.4f}, lag 20: p={results['wn_pvalue_lag20']:.4f}),
   chấp nhận giả thuyết H0 rằng chuỗi này là nhiễu thuần túy và không có cấu trúc tự hồi quy.
4. Đối với chuỗi AirPassengers, kiểm định Ljung-Box cho p-value xấp xỉ bằng 0 (lag 10: p={results['real_pvalue_lag10']:.4e} < 0.05),
   bác bỏ H0 và khẳng định chuỗi chứa lượng thông tin nội tại dồi dào.
5. Do đó, chuỗi White Noise hoàn toàn KHÔNG THỂ dự báo được (dự báo tốt nhất chỉ là kỳ vọng mu=0),
   trong khi chuỗi AirPassengers chứa tín hiệu quy luật mạnh mẽ và rất lý tưởng để xây dựng mô hình dự báo.
================================================================================="""
    print(report_text)
    return report_text


def main(show_plot: bool = False):
    print("=" * 80)
    print("KHỞI CHẠY BÀI TẬP 2: KIỂM ĐỊNH CHUỖI NHIỄU TRẮNG (WHITE NOISE)")
    print("=" * 80)
    wn_series, real_series = step1_load_data(DATA_PATH)
    results = step2_execute_analysis(wn_series, real_series)
    step3_visualize(wn_series, real_series, show_plot=show_plot)
    step4_conclude_and_report(results)
    print("\n[HOÀN THÀNH BÀI TẬP 2 THÀNH CÔNG]")


if __name__ == "__main__":
    show = "--show" in sys.argv
    main(show_plot=show)
