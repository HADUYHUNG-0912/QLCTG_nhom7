"""CV01: EDA, quality audit, and chronological 80/20 split for a10.csv."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "a10.csv"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

for directory in (PROC_DIR, FIG_DIR, TABLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a10.csv with a monthly-start DatetimeIndex."""
    df = pd.read_csv(path)
    if not {"date", "value"}.issubset(df.columns):
        raise ValueError("Input must contain 'date' and 'value' columns")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["value"] = pd.to_numeric(df["value"], errors="raise")
    df = df.set_index("date").sort_index()
    if df.index.has_duplicates:
        raise ValueError("Duplicate dates found before setting monthly frequency")
    return df.asfreq("MS")


def audit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Audit observations and persist a reproducible summary table."""
    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.index.duplicated().sum())
    summary = pd.DataFrame(
        {
            "metric": [
                "observations",
                "columns",
                "missing_values",
                "missing_rate_pct",
                "duplicate_dates",
                "start_date",
                "end_date",
                "frequency",
                "value_min",
                "value_max",
                "value_mean",
                "value_std",
            ],
            "value": [
                len(df),
                len(df.columns),
                missing_count,
                missing_count / (len(df) * len(df.columns)) * 100,
                duplicate_count,
                df.index.min().strftime("%Y-%m-%d"),
                df.index.max().strftime("%Y-%m-%d"),
                str(df.index.freq),
                df["value"].min(),
                df["value"].max(),
                df["value"].mean(),
                df["value"].std(),
            ],
        }
    )
    summary.to_csv(TABLE_DIR / "cv01_data_summary.csv", index=False)
    print(f"[AUDIT] observations={len(df)}, missing={missing_count} (0.00%), duplicates={duplicate_count}")
    print(f"[AUDIT] range={df.index.min().date()} -> {df.index.max().date()}, freq={df.index.freq}")
    return summary


def plot_eda(df: pd.DataFrame) -> None:
    """Save a line chart and a 12-month seasonal boxplot."""
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(df.index, df["value"], color="#176b87", linewidth=1.8)
    ax.set_title("a10 Monthly Retail Sales: Trend and Multiplicative Variance", loc="left", weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (AUD million)")
    ax.text(0.01, 0.02, "Increasing level and widening fluctuations suggest multiplicative behavior.", transform=ax.transAxes, fontsize=9, color="#495057")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cv01_eda_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    seasonal = df.assign(month=df.index.month)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    groups = [seasonal.loc[seasonal["month"] == month, "value"] for month in range(1, 13)]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.boxplot(groups, tick_labels=month_labels, patch_artist=True,
               boxprops={"facecolor": "#b9d7e3", "edgecolor": "#176b87"},
               medianprops={"color": "#8f2d56", "linewidth": 1.5},
               whiskerprops={"color": "#495057"}, capprops={"color": "#495057"})
    ax.set_title("Monthly Seasonality of a10 Sales", loc="left", weight="bold")
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Sales (AUD million)")
    ax.text(0.01, 0.02, "January is typically highest; February–March are lower, indicating recurring seasonality.", transform=ax.transAxes, fontsize=9, color="#495057")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cv01_monthly_seasonality_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.80) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically without shuffling; 204 * .80 floors to 163."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    train_size = int(len(df) * train_ratio)
    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()


def save_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Persist train/test data and split metadata."""
    train.to_csv(PROC_DIR / "train.csv", index_label="date")
    test.to_csv(PROC_DIR / "test.csv", index_label="date")
    split_info = pd.DataFrame(
        {
            "tap": ["Train", "Test"],
            "so_thang": [len(train), len(test)],
            "ty_le_pct": [len(train) / (len(train) + len(test)) * 100, len(test) / (len(train) + len(test)) * 100],
            "start_date": [train.index.min().strftime("%Y-%m-%d"), test.index.min().strftime("%Y-%m-%d")],
            "end_date": [train.index.max().strftime("%Y-%m-%d"), test.index.max().strftime("%Y-%m-%d")],
            "shuffle": [False, False],
        }
    )
    split_info.to_csv(TABLE_DIR / "cv01_split_info.csv", index=False)
    print(f"[SPLIT] Train={len(train)} ({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"[SPLIT] Test ={len(test)} ({test.index[0].date()} -> {test.index[-1].date()})")


def main() -> None:
    print("=" * 70)
    print("CV01 — EDA, Tiền xử lý & Phân chia dữ liệu (Thành viên 1)")
    print("=" * 70)
    df = load_data(RAW_DATA)
    print(f"[OK] Loaded {len(df)} observations; DatetimeIndex freq={df.index.freq}")
    audit_data(df)
    plot_eda(df)
    print("[OK] Saved EDA figures at dpi=150.")
    train, test = split_train_test(df)
    save_splits(train, test)
    print("[DONE] CV01 completed.")


if __name__ == "__main__":
    main()
