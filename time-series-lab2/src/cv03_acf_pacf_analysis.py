# =============================================================================
# AI_AGENT_INSTRUCTIONS — ĐỌC TRƯỚC KHI THỰC HIỆN BẤT KỲ THAY ĐỔI NÀO
# =============================================================================
# MODULE     : CV03 — Phân tích tương quan động ACF / PACF & Đề xuất bậc (p, q)
# PHỤ TRÁCH  : Thành viên 3
# MÔ TẢ TASK : Tính toán và vẽ hàm tự tương quan (ACF) và hàm tự tương quan
#              riêng phần (PACF) cho chuỗi dừng (Δy_t) đến 40 lags. Phân tích
#              hiện tượng cắt cụt (cut-off) hoặc suy giảm dần (tail-off) để
#              xác định bậc p (AR) và q (MA). Nhận diện các gai nhọn (spikes)
#              tại lag bội số 12 (chu kỳ mùa vụ s=12) báo hiệu thành phần
#              SARIMA. Lập danh sách các bộ tham số (p, d, q) ứng viên.
#
# DELIVERABLES (ĐẦU RA):
#   • outputs/figures/cv03_acf_pacf_plots.png
#   • outputs/figures/cv03_lag_correlation_analysis.png
#   • outputs/tables/cv03_candidate_orders.csv
#
# RÀNG BUỘC BẮT BUỘC (HARD CONSTRAINTS):
#   [1] CHỈ sử dụng data/processed/train.csv (tính diff(1) tại chỗ).
#       KHÔNG đọc test.csv.
#   [2] KHÔNG fit mô hình — bước này chỉ phân tích tương quan thuần túy.
#   [3] Sử dụng dải tin cậy 95%: ±1.96 / √N khi vẽ ACF/PACF.
#   [4] Vẽ ít nhất 40 lags để bao quát > 3 chu kỳ mùa vụ 12 tháng.
#   [5] Lưu hình ảnh với dpi=150 và bbox_inches='tight'.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# ---------------------------------------------------------------------------
# 0. THIẾT LẬP ĐƯỜNG DẪN GỐC
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(ROOT, "data", "processed")
FIG_DIR   = os.path.join(ROOT, "outputs", "figures")
TABLE_DIR = os.path.join(ROOT, "outputs", "tables")

TRAIN_CSV = os.path.join(PROC_DIR, "train.csv")

for d in [FIG_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. NẠP & CHUẨN BỊ CHUỖI DỪNG
# ---------------------------------------------------------------------------
def load_differenced_train(path: str) -> pd.Series:
    """
    Đọc train.csv, lấy sai phân bậc 1 (d=1) tại chỗ.
    Trả về pd.Series Δy_t (đã dropna).

    Lưu ý: CV03 tự tính diff() — không phụ thuộc vào output của CV02.
    """
    # TODO: df = pd.read_csv(path, index_col='date', parse_dates=True)
    #       return df['value'].diff(1).dropna()
    raise NotImplementedError("Thành viên 3 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 2. VẼ ĐỒ THỊ ACF & PACF (40 LAGS)
# ---------------------------------------------------------------------------
def plot_acf_pacf(series: pd.Series, lags: int = 40) -> None:
    """
    Vẽ ACF và PACF trên 2 subplot theo chiều dọc.

    QUY TẮC ĐỌC ACF / PACF (Box-Jenkins):
      • PACF cắt cụt tại lag p → Mô hình AR(p)
      • ACF  cắt cụt tại lag q → Mô hình MA(q)
      • Cả hai suy giảm dần   → Mô hình ARMA(p, q)
      • Gai nhọn tại lag 12, 24, 36 → Có thành phần mùa vụ → SARIMA

    Lưu: outputs/figures/cv03_acf_pacf_plots.png
    """
    # TODO: fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    #       plot_acf(series,  lags=lags, ax=ax1, alpha=0.05, title='ACF — Hàm tự tương quan (40 lags)')
    #       plot_pacf(series, lags=lags, ax=ax2, alpha=0.05, method='ywm', title='PACF — Hàm tự tương quan riêng phần (40 lags)')
    raise NotImplementedError("Thành viên 3 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 3. PHÂN TÍCH GIA NHỌN MÙA VỤ (SEASONAL SPIKE DETECTION)
# ---------------------------------------------------------------------------
def plot_lag_analysis(series: pd.Series, seasonal_lags=None) -> None:
    """
    Vẽ biểu đồ lag plot nhấn mạnh các lag mùa vụ 12, 24, 36.
    Đánh dấu trực quan các cột vượt ngưỡng tin cậy ±1.96/√N.

    Lưu: outputs/figures/cv03_lag_correlation_analysis.png
    """
    if seasonal_lags is None:
        seasonal_lags = [12, 24, 36]
    # TODO: Dùng autocorr() hoặc acf() từ statsmodels
    #       Tính dải tin cậy: conf_interval = 1.96 / (len(series) ** 0.5)
    #       Tô màu đỏ các cột tại lag 12, 24, 36 để nhận diện spikes mùa vụ
    raise NotImplementedError("Thành viên 3 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 4. LẬP DANH SÁCH BỘ THAM SỐ ỨNG VIÊN (p, d, q)
# ---------------------------------------------------------------------------
def generate_candidate_orders(p_max: int = 2, d: int = 1, q_max: int = 2) -> pd.DataFrame:
    """
    Sinh danh sách các bộ tham số (p, d, q) ứng viên theo quy tắc Box-Jenkins.
    Khuyến nghị bắt đầu với p ≤ 2 và q ≤ 2.

    Args:
        p_max : Bậc AR tối đa (từ phân tích PACF).
        d     : Bậc sai phân đã xác định = 1.
        q_max : Bậc MA tối đa (từ phân tích ACF).

    Returns:
        pd.DataFrame với các cột: [p, d, q, Ghi chú]
    Lưu: outputs/tables/cv03_candidate_orders.csv
    """
    # TODO: Tạo DataFrame với itertools.product(range(p_max+1), [d], range(q_max+1))
    #       Ghi chú: (1,1,1) là mô hình khởi điểm kinh điển cho a10.csv
    raise NotImplementedError("Thành viên 3 cài đặt tại đây")


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("CV03 — Phân tích ACF / PACF & Đề xuất bậc (p, q) (Thành viên 3)")
    print("=" * 70)

    # Bước 1: Nạp chuỗi dừng
    diff_series = load_differenced_train(TRAIN_CSV)
    print(f"[OK] Chuỗi sau sai phân: {len(diff_series)} quan sát")

    # Bước 2: Vẽ ACF/PACF 40 lags
    plot_acf_pacf(diff_series, lags=40)
    print("[OK] Đã lưu biểu đồ ACF/PACF 40 lags.")

    # Bước 3: Phân tích lag mùa vụ
    plot_lag_analysis(diff_series, seasonal_lags=[12, 24, 36])
    print("[OK] Đã lưu biểu đồ phân tích gai nhọn mùa vụ.")

    # Bước 4: Lập danh sách bộ tham số ứng viên
    candidates = generate_candidate_orders(p_max=2, d=1, q_max=2)
    print(f"[OK] Danh sách ứng viên:\n{candidates.to_string()}")

    print("[DONE] CV03 hoàn tất. Chuyển cv03_candidate_orders.csv cho CV04 & CV05.")
