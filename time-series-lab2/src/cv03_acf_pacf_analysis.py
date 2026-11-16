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

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
SEASONAL_LAGS = (12, 24, 36)


def load_differenced_train(path: Path) -> pd.Series:
    """Đọc tập train theo tháng và lấy sai phân bậc một."""
    train = pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()
    expected = pd.date_range(train.index.min(), periods=len(train), freq="MS")
    if not train.index.equals(expected) or train["value"].isna().any():
        raise ValueError("Tập train phải có một giá trị hợp lệ cho mỗi tháng liên tiếp.")
    return train["value"].diff().dropna()


def lag_correlations(series: pd.Series, lags: int = 40) -> pd.DataFrame:
    """Tính ACF và PACF (Yule–Walker) cho các lag từ 1 đến lags."""
    return pd.DataFrame(
        {
            "ACF": acf(series, nlags=lags, fft=True)[1:],
            "PACF": pacf(series, nlags=lags, method="ywm")[1:],
        },
        index=pd.RangeIndex(1, lags + 1, name="Lag"),
    )


def plot_acf_pacf(series: pd.Series, lags: int = 40) -> Path:
    correlations = lag_correlations(series, lags)
    bound = 1.96 / len(series) ** 0.5
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, label in zip(axes, correlations.columns):
        ax.axhspan(-bound, bound, color="#dbeafe", label="Ngưỡng 95%")
        ax.axhline(0, color="#64748b", linewidth=0.8)
        ax.stem(correlations.index, correlations[label], basefmt=" ")
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Độ trễ (tháng)")
    axes[-1].set_xticks(range(0, lags + 1, 4))
    fig.suptitle("ACF và PACF của chuỗi a10 sau sai phân bậc một")
    return save_figure(fig, "cv03_acf_pacf_plots.png")


def plot_lag_analysis(series: pd.Series, seasonal_lags=SEASONAL_LAGS) -> Path:
    correlations = lag_correlations(series)["ACF"]
    bound = 1.96 / len(series) ** 0.5
    colors = ["#b45309" if lag in seasonal_lags else "#2563eb" for lag in correlations.index]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhspan(-bound, bound, color="#dbeafe", label="Ngưỡng 95%")
    ax.axhline(0, color="#64748b", linewidth=0.8)
    ax.bar(correlations.index, correlations, color=colors)
    for lag in seasonal_lags:
        value = correlations.loc[lag]
        ax.annotate(f"{value:.3f}", (lag, value), xytext=(0, 5),
                    textcoords="offset points", ha="center")
    ax.set(title="Tương quan mùa vụ tại các lag 12, 24 và 36",
           xlabel="Độ trễ (tháng)", ylabel="ACF", ylim=(-0.4, 1.0))
    ax.legend(loc="upper right")
    return save_figure(fig, "cv03_lag_correlation_analysis.png")


def save_figure(fig: plt.Figure, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_candidate_orders(p_max: int = 2, d: int = 1, q_max: int = 2) -> pd.DataFrame:
    """Liệt kê ứng viên nhỏ để CV04 kiểm tra, ưu tiên hai cấu hình phù hợp."""
    orders = list(product(range(p_max + 1), [d], range(q_max + 1)))
    preferred = {
        (1, d, 1): "Khởi điểm gọn; ACF và PACF đều vượt ngưỡng tại lag 1.",
        (2, d, 1): "Thử thêm một bậc AR; PACF vượt ngưỡng tại lag 2.",
    }
    orders.sort(key=lambda order: order not in preferred)
    candidates = pd.DataFrame(
        [(*order, preferred.get(order, "Ứng viên đối chiếu")) for order in orders],
        columns=["p", "d", "q", "Ghi chú"],
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(TABLE_DIR / "cv03_candidate_orders.csv", index=False)
    return candidates


def main() -> None:
    if not TRAIN_CSV.is_file():
        raise SystemExit(
            "Thiếu data/processed/train.csv. Chạy CV01 hoặc mở notebook CV03 "
            "để tạo tập train tạm từ a10.csv."
        )
    series = load_differenced_train(TRAIN_CSV)
    plot_acf_pacf(series)
    plot_lag_analysis(series)
    print(f"Chuỗi sai phân: {len(series)} quan sát.")
    print(generate_candidate_orders().to_string(index=False))


if __name__ == "__main__":
    main()
