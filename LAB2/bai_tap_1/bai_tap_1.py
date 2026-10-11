# -*- coding: utf-8 -*-
"""LAB2 - Bài tập 1: ACF/PACF, AirPassengers. CV02 - Ngọc Trâm.

AI_AGENT_INSTRUCTIONS / GUARDRAILS:
- Giữ nguyên 4 hàm step1_load_data, step2_execute_analysis,
  step3_visualize, step4_conclude_and_report.
- Không sửa dữ liệu nguồn, không lưu ảnh PNG/JPG vào repository.
- Chuỗi phải có DatetimeIndex, tần suất MS; báo cáo được tạo từ kết quả tính thật.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import matplotlib
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "raw" / "AirPassengers.csv"
OUTPUT_PATH = BASE_DIR / "ket_qua_bai_1.txt"
NLAGS = 40
ALPHA = 0.05


def step1_load_data(data_path: Path) -> pd.Series:
    """Đọc dữ liệu theo tháng, xác nhận chuỗi 144 tháng liên tục, không thiếu."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {data_path}")
    df = pd.read_csv(data_path)
    if not {"Month", "Passengers"}.issubset(df.columns):
        raise ValueError("CSV phải có cột Month và Passengers")
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="raise")
    if df["Month"].duplicated().any():
        raise ValueError("Tháng trùng lặp trong dữ liệu")
    df["Passengers"] = pd.to_numeric(df["Passengers"], errors="raise")
    df = df.set_index("Month").sort_index()
    expected = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    if len(expected) != len(df) or not df.index.equals(expected):
        raise ValueError("Dữ liệu có tháng bị thiếu hoặc không khớp tần suất MS")
    if df["Passengers"].isna().any():
        raise ValueError("Cột Passengers có NaN")
    series = df["Passengers"].astype(float).asfreq("MS")
    if len(series) != 144 or series.index.min() != pd.Timestamp("1949-01-01") or series.index.max() != pd.Timestamp("1960-12-01"):
        raise ValueError("Dataset AirPassengers chuẩn phải có 144 tháng từ 1949-01 đến 1960-12")
    print(f"[STEP 1] {len(series)} tháng, {series.index.min():%Y-%m} đến {series.index.max():%Y-%m}; freq={series.index.freqstr}; NaN={series.isna().sum()}")
    return series


def step2_execute_analysis(series: pd.Series, nlags: int = NLAGS) -> dict[str, Any]:
    """Tính ACF/PACF theo mẫu; ngưỡng xấp xỉ 95% ±1.96/sqrt(N)."""
    if nlags < 36 or nlags >= len(series) // 2:
        raise ValueError("nlags cần từ 36 tới nhỏ hơn N/2 cho phương pháp PACF")
    a = acf(series, nlags=nlags, fft=False)
    p = pacf(series, nlags=nlags, method="ywm")
    threshold = float(1.96 / np.sqrt(len(series)))
    sig_a = [i for i in range(1, nlags + 1) if abs(a[i]) > threshold]
    sig_p = [i for i in range(1, nlags + 1) if abs(p[i]) > threshold]
    results: dict[str, Any] = {
        "acf": a, "pacf": p, "threshold": threshold,
        "significant_acf": sig_a, "significant_pacf": sig_p,
        "n": len(series), "nlags": nlags,
        "start": str(series.index.min().date()), "end": str(series.index.max().date()),
    }
    print(f"[STEP 2] Ngưỡng 95% xấp xỉ = ±{threshold:.4f}; ACF(1)={a[1]:.4f}; ACF(12)={a[12]:.4f}")
    print(f"ACF(24)={a[24]:.4f}; ACF(36)={a[36]:.4f}; PACF(1)={p[1]:.4f}")
    return results


def step3_visualize(series: pd.Series, analysis_results: dict[str, Any], show_plot: bool = False) -> None:
    """Tạo biểu đồ ACF/PACF trong bộ nhớ, không lưu hình ra đĩa."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(series, lags=analysis_results["nlags"], ax=axes[0], alpha=ALPHA, zero=False)
    plot_pacf(series, lags=analysis_results["nlags"], ax=axes[1], alpha=ALPHA, method="ywm", zero=False)
    axes[0].set_title("ACF - AirPassengers (raw)")
    axes[1].set_title("PACF - AirPassengers (raw)")
    for ax in axes:
        ax.set_xlabel("Lag (tháng)")
        ax.axhline(analysis_results["threshold"], ls="--", lw=1, color="gray", label="±1.96/√N (tham khảo)")
        ax.axhline(-analysis_results["threshold"], ls="--", lw=1, color="gray")
        for k in (12, 24, 36):
            ax.axvline(k, ls=":", lw=0.8, color="teal")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    if show_plot:
        plt.show()
    plt.close(fig)
    print("[STEP 3] Đã tạo ACF/PACF, không lưu ảnh.")


def step4_conclude_and_report(analysis_results: dict[str, Any]) -> str:
    """Viết báo cáo số liệu thực, 3 câu nghiệm thu và câu lý thuyết 1,2,10."""
    r = analysis_results
    a, p, c = r["acf"], r["pacf"], r["threshold"]
    def lag_summary(values: np.ndarray, ks: tuple[int, ...]) -> str:
        return ", ".join(f"{k}: {values[k]:.4f} ({'vượt' if abs(values[k]) > c else 'không vượt'} ngưỡng)" for k in ks)
    def lag_list(v: list[int]) -> str:
        return ", ".join(map(str, v)) if v else "không có"
    report = f"""BÀI TẬP 1 - PHÂN TÍCH ACF/PACF (CV02 - NGỌC TRÂM)
{'=' * 74}
DATA & PHƯƠNG PHÁP
- AirPassengers: {r['n']} quan sát tháng, {r['start']} đến {r['end']}; DatetimeIndex freq='MS'.
- ACF: statsmodels.acf(fft=False); PACF: statsmodels.pacf(method='ywm').
- Độ trễ 1..{r['nlags']}. Ngưỡng kiểm tra xấp xỉ 95%: ±1.96/√N = ±{c:.4f}.
- Lưu ý: ngưỡng cố định này là quy tắc gần đúng; dải tin cậy mặc định
  trong plot_acf/plot_pacf có thể khác theo phương pháp ước lượng.

KẾT QUẢ ĐỊNH LƯỢNG
- ACF tại lag 1/12/24/36: {lag_summary(a, (1, 12, 24, 36))}.
- PACF tại lag 1/2/12/13/24/36: {lag_summary(p, (1, 2, 12, 13, 24, 36))}.
- Các lag ACF vượt ngưỡng xấp xỉ: {lag_list(r['significant_acf'])}.
- Các lag PACF vượt ngưỡng xấp xỉ: {lag_list(r['significant_pacf'])}.

3 CÂU HỎI NGHIỆM THU
1) Chuỗi có tự tương quan mạnh không?
   Có. ACF(1)={a[1]:.4f} lớn và ACF giữ tương quan dương qua nhiều độ trễ;
   sự suy giảm chậm là dấu hiệu chuỗi có thành phần xu hướng/không dừng.
2) Có trend và seasonality không?
   Có dấu hiệu trend và mùa vụ chu kỳ 12 tháng. ACF tại 12, 24, 36
   lần lượt {a[12]:.4f}, {a[24]:.4f}, {a[36]:.4f}; đối chiếu mức ±{c:.4f}.
   ACF/PACF chỉ hỗ trợ chẩn đoán; kiểm định dừng và sai phân cần xem xét thêm.
3) Lag nào quan trọng nhất?
   Lag 1 biểu thị phụ thuộc ngắn hạn; lag 12, 24, 36 gợi ý mùa vụ theo năm.
   Về mặt xấp xỉ thống kê, danh sách lag PACF vượt ngưỡng đã nêu phía trên.
   Không nên khẳng định PACF hoàn toàn cắt cụt sau lag 1 trên chuỗi gốc
   vì có thể tồn tại xung ở các lag khác và chuỗi chưa dừng.

GIẢI THÍCH CUT-OFF VÀ HƯỚNG MÔ HÌNH
- PACF đo tương quan trực tiếp tại lag k sau khi loại ảnh hưởng các lag xen giữa.
- PACF chỉ có một xung đáng kể tại lag 1 (trên một chuỗi ĐÃ dừng) có thể
  gợi ý thành phần AR(1), nhưng KHÔNG chứng minh ngay p=1 trên chuỗi gốc.
- Mô hình ứng viên thử nghiệm: SARIMA(1,1,0)x(1,1,0,12), không phải mô hình
  tối ưu đã được xác nhận. Cần kiểm định ADF/KPSS, xem ACF/PACF sau sai phân,
  so sánh AIC/BIC, đánh giá dự báo ngoài mẫu và kiểm tra phần dư.

CÂU HỎI LÝ THUYẾT CUỐI CHƯƠNG
1) Tự tương quan là mối liên hệ giữa y_t và y_(t-k) của cùng chuỗi;
   giúp phát hiện phụ thuộc thời gian, xu hướng và mùa vụ. ACF(k) là hệ số
   tương quan theo độ trễ k; k=0 có ACF=1.
2) ACF bao gồm liên hệ trực tiếp và gián tiếp; PACF loại ảnh hưởng của các
   độ trễ trung gian. Lag quan trọng là lag có tín hiệu đáng chú ý so với
   biên tin cậy, có ý nghĩa thực tế hoặc lặp lại theo mùa; vượt ngưỡng riêng
   lẻ không tự động chứng minh quan hệ nhân quả.
10) ACF giúp thấy mùa vụ qua các đỉnh lặp lại gần các bội số chu kỳ s;
    với số liệu theo tháng, đỉnh quanh k=12,24,36 gợi ý mùa vụ 12 tháng.

KẾT LUẬN
- Chuỗi gốc có tự tương quan mạnh và mẫu mùa vụ 12 tháng; cần xử lý
  không dừng trước khi quyết định bậc SARIMA và đánh giá mô hình.
"""
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"[STEP 4] Đã xuất {OUTPUT_PATH}")
    print(report)
    return report


def main() -> None:
    series = step1_load_data(DATA_PATH)
    result = step2_execute_analysis(series)
    step3_visualize(series, result, show_plot="--show" in sys.argv)
    step4_conclude_and_report(result)


if __name__ == "__main__":
    main()
