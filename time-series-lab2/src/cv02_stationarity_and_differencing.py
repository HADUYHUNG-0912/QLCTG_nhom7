# =============================================================================
# AI_AGENT_INSTRUCTIONS — ĐỌC TRƯỚC KHI THỰC HIỆN BẤT KỲ THAY ĐỔI NÀO
# =============================================================================
# MODULE     : CV02 — Kiểm định tính dừng ADF & Lấy sai phân xác định bậc d
# PHỤ TRÁCH  : Thành viên 2
# MÔ TẢ TASK : Kiểm định Augmented Dickey-Fuller (ADF) trên chuỗi Train gốc
#              để xác định tính không dừng (p-value ≥ 0.05). Thực hiện sai
#              phân bậc 1 (d=1: Δy_t = y_t − y_{t-1}). Kiểm định ADF lần 2
#              trên chuỗi đã sai phân để xác nhận tính dừng (p-value < 0.05).
#              Vẽ biểu đồ so sánh chuỗi trước và sau sai phân.
#
# DELIVERABLES (ĐẦU RA):
#   • data/processed/train_differenced.csv  – chuỗi đã lấy sai phân bậc 1
#   • outputs/tables/cv02_adf_test_summary.csv
#   • outputs/figures/cv02_adf_before_after_differencing.png
#
# RÀNG BUỘC BẮT BUỘC (HARD CONSTRAINTS):
#   [1] CHỈ sử dụng data/processed/train.csv (KHÔNG đọc test.csv).
#   [2] Kiểm định ADF chỉ trên tập TRAIN — không rò rỉ thông tin Test.
#   [3] Mức ý nghĩa chuẩn: alpha = 0.05.
#   [4] Lưu hình ảnh với dpi=150 và bbox_inches='tight'.
#   [5] Xuất bảng ADF đầy đủ: ADF Statistic, p-value, Critical Values (1%, 5%, 10%).
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os

# ---------------------------------------------------------------------------
# 0. THIẾT LẬP ĐƯỜNG DẪN GỐC
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(ROOT, "data", "processed")
FIG_DIR   = os.path.join(ROOT, "outputs", "figures")
TABLE_DIR = os.path.join(ROOT, "outputs", "tables")

TRAIN_CSV = os.path.join(PROC_DIR, "train.csv")
OUT_DIFF  = os.path.join(PROC_DIR, "train_differenced.csv")

for d in [FIG_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. NẠP TẬP TRAIN
# ---------------------------------------------------------------------------
def load_train(path: str) -> pd.Series:
    """Đọc train.csv và trả về pd.Series với DatetimeIndex."""
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    series = pd.to_numeric(df["value"], errors="raise")
    series.index = pd.DatetimeIndex(series.index)
    series = series.sort_index()
    series.name = "value"
    return series


# ---------------------------------------------------------------------------
# 2. KIỂM ĐỊNH ADF
# ---------------------------------------------------------------------------
def run_adf_test(series: pd.Series, label: str = "Chuỗi") -> dict:
    """
    Chạy kiểm định Augmented Dickey-Fuller.

    H₀: Chuỗi có nghiệm đơn vị (Unit Root) → Chuỗi KHÔNG dừng.
    H₁: Chuỗi không có nghiệm đơn vị       → Chuỗi DỪng.

    Kết luận:
      p-value < 0.05  → Bác bỏ H₀ → Chuỗi dừng ✓
      p-value ≥ 0.05  → Không đủ cơ sở bác bỏ H₀ → Chuỗi không dừng ✗

    Returns:
        dict: {'adf_stat', 'p_value', 'n_lags', 'crit_1pct', 'crit_5pct', 'crit_10pct', 'is_stationary'}
    """
    result = adfuller(series.dropna(), autolag="AIC")
    crit = result[4]
    return {
        "adf_stat": float(result[0]),
        "p_value": float(result[1]),
        "n_lags": int(result[2]),
        "crit_1pct": float(crit["1%"]),
        "crit_5pct": float(crit["5%"]),
        "crit_10pct": float(crit["10%"]),
        "is_stationary": bool(result[1] < 0.05)
    }


# ---------------------------------------------------------------------------
# 3. SẠI PHÂN BẬC 1 (DIFFERENCING d=1)
# ---------------------------------------------------------------------------
def apply_differencing(series: pd.Series, d: int = 1) -> pd.Series:
    """
    Áp dụng sai phân bậc d:
      Δy_t = y_t − y_{t−1}  (với d=1)
    Lưu train_differenced.csv.

    Args:
        series : pd.Series gốc (Train).
        d      : Bậc sai phân (mặc định = 1).

    Returns:
        pd.Series đã sai phân (dropna() đã được áp dụng).
    """
    if d < 1:
        raise ValueError("Bậc sai phân d phải >= 1")
    diff_series = series.copy()
    for _ in range(d):
        diff_series = diff_series.diff().dropna()
    diff_series.name = "value"
    diff_series.to_csv(OUT_DIFF, index_label="date")
    return diff_series


# ---------------------------------------------------------------------------
# 4. VẼ BIỂU ĐỒ SO SÁNH TRƯỚC & SAU SAI PHÂN
# ---------------------------------------------------------------------------
def plot_before_after(original: pd.Series, differenced: pd.Series) -> None:
    """
    Vẽ 2 subplot:
      (a) Chuỗi Train gốc — hiển thị trend tăng & phương sai mở rộng.
      (b) Chuỗi sau sai phân bậc 1 — dao động ổn định quanh 0.
    Lưu: outputs/figures/cv02_adf_before_after_differencing.png
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=False
    )
    ax1.plot(original.index, original, color="steelblue")
    ax1.set_title("Chuỗi Train gốc")
    ax1.set_ylabel("Giá trị")
    ax1.grid(alpha=0.3)

    ax2.plot(
        differenced.index, differenced,
        color="darkorange"
    )
    ax2.axhline(
        0, linestyle="--", color="gray", alpha=0.7
    )
    ax2.set_title("Chuỗi sau sai phân bậc 1 (d=1)")
    ax2.set_xlabel("Thời gian")
    ax2.set_ylabel("Sai phân")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            FIG_DIR,
            "cv02_adf_before_after_differencing.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. LƯU BẢNG KẾT QUẢ ADF
# ---------------------------------------------------------------------------
def save_adf_summary(result_raw: dict, result_diff: dict) -> None:
    """
    Lưu bảng so sánh kết quả ADF trước và sau sai phân.
    outputs/tables/cv02_adf_test_summary.csv
    """
    df = pd.DataFrame(
        [result_raw, result_diff],
        index=["Chuỗi gốc", "Chuỗi sai phân d=1"]
    )
    df.index.name = "series"
    df.to_csv(
        os.path.join(
            TABLE_DIR, "cv02_adf_test_summary.csv"
        ),
        encoding="utf-8-sig"
    )


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("CV02 — Kiểm định ADF & Sai phân (Thành viên 2)")
    print("=" * 70)

    # Bước 1: Nạp Train
    train = load_train(TRAIN_CSV)
    print(f"[OK] Đã nạp Train: {len(train)} tháng")

    # Bước 2: ADF lần 1 — chuỗi gốc
    res_raw = run_adf_test(train, label="Chuỗi Train gốc")
    print(f"[ADF-1] ADF={res_raw['adf_stat']:.4f} | p-value={res_raw['p_value']:.4f} | Dừng: {res_raw['is_stationary']}")

    # Bước 3: Sai phân bậc 1
    train_diff = apply_differencing(train, d=1)
    print(f"[OK] Đã lấy sai phân bậc 1: {len(train_diff)} quan sát còn lại")

    # Bước 4: ADF lần 2 — chuỗi sau sai phân
    res_diff = run_adf_test(train_diff, label="Chuỗi sau sai phân d=1")
    print(f"[ADF-2] ADF={res_diff['adf_stat']:.4f} | p-value={res_diff['p_value']:.4f} | Dừng: {res_diff['is_stationary']}")

    # Bước 5: Lưu kết quả
    plot_before_after(train, train_diff)
    save_adf_summary(res_raw, res_diff)
    print("[DONE] CV02 hoàn tất. Chuyển train_differenced.csv cho CV03.")

