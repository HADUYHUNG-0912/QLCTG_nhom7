"""Bài tập 3: Phân rã chuỗi thời gian (Decomposition) - AirPassengers.

1. Chọn chuỗi dữ liệu theo tháng (1949-01 đến 1960-12, 144 tháng).
2. Dùng STL (chính) và classical decomposition (đối chiếu).
3. Vẽ 4 thành phần: Observed, Trend, Seasonal, Residual.
4. Mô tả: xu hướng, mùa vụ, residual có giống white noise không.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "AirPassengers.csv"
OUT_FILE = BASE_DIR / "ket_qua_bai_3.txt"
FIG_DIR = BASE_DIR / "bai_3_figures"

PERIOD = 12  # chu kỳ mùa vụ theo tháng
LINES = []


def emit(line=""):
    print(line)
    LINES.append(line)


def rule(char="=", n=78):
    emit(char * n)


def section(title):
    emit()
    rule("=")
    emit(title)
    rule("=")


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def decomp_strength(trend, seasonal, resid):
    """Strength theo Cleveland (1990): 0 = thành phần không có, 1 = thành phần trội hẳn."""
    r = resid.dropna()
    var_r = r.var()
    f_trend = max(0.0, 1.0 - var_r / (trend + r).dropna().var())
    f_seas = max(0.0, 1.0 - var_r / (seasonal + r).dropna().var())
    return f_trend, f_seas


def plot_components(y, res, title_prefix, fname, seasonal_title, resid_title):
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    series = [
        (axes[0], y, title_prefix, "#1f77b4"),
        (axes[1], res.trend, "Trend (xu hướng)", "#d62728"),
        (axes[2], res.seasonal, seasonal_title, "#2ca02c"),
        (axes[3], res.resid, resid_title, "#9467bd"),
    ]
    for ax, data, label, color in series:
        ax.plot(data.index, data.values, color=color, linewidth=1.3)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{title_prefix} - Phân rã chuỗi thời gian (chu kỳ = {PERIOD} tháng)",
                      fontsize=13, fontweight="bold")
    axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def main():
    # ------------------------------------------------------------------
    section("BAI TAP 3 - PHAN RA CHUOI THOI GIAN (DECOMPOSITION)")
    emit(f"Nguon du lieu : {DATA_FILE.name}")
    emit("Chuoi du lieu : Theo THANG (1949-01 den 1960-12)")
    emit("Phuong phap   : STL (chinh) + classical decomposition (doi chieu)")
    emit(f"Chu ky mua vu : {PERIOD} thang")

    # 1. Nap va chuan bi du lieu
    df = pd.read_csv(DATA_FILE, parse_dates=["Month"]).set_index("Month").asfreq("MS")
    y = df["Passengers"]

    section("1. THONG KE MO TA DU LIEU")
    emit(f"So quan sat     : {len(y)} (tu {y.index[0]:%Y-%m} den {y.index[-1]:%Y-%m})")
    emit(f"Trung binh     : {y.mean():.2f}   Trung vị : {y.median():.2f}")
    emit(f"Min / Max      : {y.min()} ({y.index[y.argmin()]:%Y-%m}) / {y.max()} ({y.index[y.argmax()]:%Y-%m})")
    emit(f"Do lech chuan   : {y.std():.2f}   CV       : {y.std() / y.mean():.4f}")
    emit(f"Ket qua nghiea  : mean = {y.mean():.1f}, std = {y.std():.1f}, min = {y.min()}, max = {y.max()}")
    emit(f"Khong co gia tri thieu : {y.isna().sum() == 0}")

    rule("-")
    emit("Trung binh theo nam:")
    for year, val in y.groupby(y.index.year).mean().round(1).items():
        emit(f"  {year}: {val:6.1f}")
    emit("Trung binh va do lech chuan theo thang (trong 12 nam):")
    by_month = y.groupby(y.index.month).agg(["mean", "std"])
    for m in range(1, 13):
        emit(f"  Thang {m:2d}: mean = {by_month.loc[m, 'mean']:6.1f}   std = {by_month.loc[m, 'std']:6.1f}")

    # 2a. STL tren thang duoc tu
    stl = STL(y, period=PERIOD, robust=True).fit()
    f_trend, f_seas = decomp_strength(stl.trend, stl.seasonal, stl.resid)

    # 2b. STL tren thang log - bien do mua vu tang dan
    stl_log = STL(np.log(y), period=PERIOD, robust=True).fit()
    f_trend_l, f_seas_l = decomp_strength(stl_log.trend, stl_log.seasonal, stl_log.resid)

    # 2c. Classical decomposition: cong va nhan
    cls_add = seasonal_decompose(y, model="additive", period=PERIOD, extrapolate_trend="period")
    cls_mul = seasonal_decompose(y, model="multiplicative", period=PERIOD, extrapolate_trend="period")

    plot_components(y, stl, "STL - AirPassengers (thang)", "stl_phan_rava.png",
                    "Seasonal (mua vu)", "Residual (phan du)")
    plot_components(np.log(y), stl_log, "STL tren thang log - AirPassengers", "stl_phan_rava_log.png",
                    "Seasonal (mua vu, thang log)", "Residual (phan du, thang log)")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(cls_add.resid.index, cls_add.resid.values, color="#9467bd", linewidth=1.2)
    axes[0].set_ylabel("Residual (cong)", fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Classical decomposition - Cong so voi Nhan so",
                      fontsize=13, fontweight="bold")
    axes[1].plot(cls_mul.resid.index, cls_mul.resid.values, color="#8c564b", linewidth=1.2)
    axes[1].axhline(1, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Residual (nhan)", fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "classical_phan_rava.png", dpi=150)
    plt.close(fig)

    # 3. Ket qua phan ra
    section("2. KET QUA PHAN RA (STL, robust=True, chu ky = 12 thang)")
    emit(f"{'Thoi diem':<10}{'Observed':>10}{'Trend':>10}{'Seasonal':>11}{'Residual':>11}")
    rule("-")
    for i in range(0, len(y), 6):
        emit(f"{y.index[i]:%Y-%m}  {y.iloc[i]:>9.0f}  {stl.trend.iloc[i]:>10.2f}"
             f"{stl.seasonal.iloc[i]:>11.2f}{stl.resid.iloc[i]:>11.2f}")
    emit(f"{y.index[-1]:%Y-%m}  {y.iloc[-1]:>9.0f}  {stl.trend.iloc[-1]:>10.2f}"
         f"{stl.seasonal.iloc[-1]:>11.2f}{stl.resid.iloc[-1]:>11.2f}")

    rule("-")
    emit("Do lon cac thanh phan:")
    emit(f"  Observed  : mean = {y.mean():.2f}   std = {y.std():.2f}")
    emit(f"  Trend     : min  = {stl.trend.min():.2f}   max  = {stl.trend.max():.2f}"
         f"   mean = {stl.trend.mean():.2f}")
    emit(f"  Seasonal  : min  = {stl.seasonal.min():.2f}   max  = {stl.seasonal.max():.2f}"
         f"   std  = {stl.seasonal.std():.2f}")
    emit(f"  Residual  : min  = {stl.resid.min():.2f}   max  = {stl.resid.max():.2f}"
         f"   std  = {stl.resid.std():.2f}")
    emit(f"  Ty le std Residual / std Observed = {stl.resid.std() / y.std():.4f}")

    rule("-")
    emit("Su manh thanh phan (Cleveland 1990, gia tri 0-1):")
    emit(f"  STL tren thang    : F_trend = {fmt(f_trend)}   F_seasonal = {fmt(f_seas)}")
    emit(f"  STL tren thang log: F_trend = {fmt(f_trend_l)}   F_seasonal = {fmt(f_seas_l)}")

    section("3. MO TA THANH PHAN")
    emit("3.1. XU HUONG (Trend)")
    rule("-")
    emit(f"  Xhuong: TANG manh va tuong duong tuyet doi.")
    emit(f"  Gia tri Trend dau chuoi ({y.index[0]:%Y-%m}) : {stl.trend.iloc[0]:.2f}")
    emit(f"  Gia tri Trend cuoi chuoi ({y.index[-1]:%Y-%m}): {stl.trend.iloc[-1]:.2f}")
    emit(f"  Tang gap            : {stl.trend.iloc[-1] - stl.trend.iloc[0]:.2f} luat khach")
    emit(f"  He so tang theo nam  : {(stl.trend.iloc[-1] / stl.trend.iloc[0]) ** (1 / 11.9) - 1:.2%} /nam")
    emit(f"  Trend tang dan deu khong co bien dong chu ky nam (nhac lan khach nang khong bi pha).")

    emit()
    emit("3.2. MUA VU (Seasonal)")
    rule("-")
    seasonal_by_month = stl.seasonal.groupby(stl.seasonal.index.month).mean().round(2)
    emit("  He so mua vu trung binh theo thang (do lech so voi Trend):")
    for m in range(1, 13):
        marker = " <-- DINH" if seasonal_by_month[m] == seasonal_by_month.max() else (
            " <-- DINH" if seasonal_by_month[m] == seasonal_by_month.min() else "")
        emit(f"    Thang {m:2d}: {seasonal_by_month[m]:+8.2f}{marker}")
    emit(f"  Bien do mua vu (max - min) = {seasonal_by_month.max() - seasonal_by_month.min():.2f} luat khach")
    emit(f"  Ty le bien do / gia tri Trend trung binh = "
         f"{(seasonal_by_month.max() - seasonal_by_month.min()) / stl.trend.mean():.2%} -> MUA VU RAT MANH")

    emit()
    emit("  Bien do mua vu co tang dan theo thoi gian (chu ky nhan):")
    emit(f"    {'Giai doan':<18}{'TB T7-T8':>10}{'TB T11-T2':>11}{'Bien do':>9}")
    periods = [(1949, 1952), (1953, 1956), (1957, 1960)]
    amplitudes = []
    for start, end in periods:
        sub = y[f"{start}":f"{end}"]
        hi = sub[sub.index.month.isin([7, 8])].mean()
        lo = sub[sub.index.month.isin([11, 12, 1, 2])].mean()
        amplitudes.append(hi - lo)
        emit(f"    {start}-{end}{hi:>13.1f}{lo:>13.1f}{hi - lo:>11.1f}")
    emit(f"    Bien do tang tu {amplitudes[0]:.0f} len {amplitudes[-1]:.0f} "
         f"({amplitudes[-1] / amplitudes[0]:.1f} lan) trong khi Trend chi gap {stl.trend.iloc[-1] / stl.trend.iloc[0]:.1f} lan.")
    emit("    => Bien do mua vu KHONG cang bang voi xu huong => phai dung mo hinh NHAN,")
    emit("       hoac phan ra tren thang LOG de co bien do mua vu ổn định.")
    emit(f"  Tren thang log: bien do mua vu chi con "
         f"{stl_log.seasonal.max() - stl_log.seasonal.min():.4f} log-don vi (~"
         f"{100 * (np.exp(stl_log.seasonal.max()) - np.exp(stl_log.seasonal.min())):.0f}%).")

    emit()
    emit("  So sanh classical additive vs multiplicative (chat luong phan du):")
    emit(f"    Cong   : std(Residual) = {cls_add.resid.std():.4f}   "
         f"F_trend = {fmt(decomp_strength(cls_add.trend, cls_add.seasonal, cls_add.resid)[0])}   "
         f"F_seas = {fmt(decomp_strength(cls_add.trend, cls_add.seasonal, cls_add.resid)[1])}")
    emit(f"    Nhan   : std(Residual) = {cls_mul.resid.std():.4f}   "
         f"F_trend = {fmt(decomp_strength(cls_mul.trend, cls_mul.seasonal, cls_mul.resid)[0])}   "
         f"F_seas = {fmt(decomp_strength(cls_mul.trend, cls_mul.seasonal, cls_mul.resid)[1])}")
    emit(f"    => Nhan tot hon {cls_add.resid.std() / cls_mul.resid.std():.2f} lan, xac nhan mua vu nhan.")

    section("4. RESIDUAL CO GIONG WHITE NOISE KHONG?")
    rule("-")
    emit("Kiem dinh cac gia dy cua white noise: mean = 0, phuong sai hang, tuong quan = 0, phan phoi chuan.")

    for label, resid in [("STL (thang)", stl.resid.dropna()),
                         ("STL (thang log)", stl_log.resid.dropna()),
                         ("Classical Nhan", cls_mul.resid.dropna())]:
        emit()
        emit(f"  --- {label} ---")
        emit(f"    n               = {len(resid)}")
        emit(f"    Mean            = {resid.mean():.4f} (std loi cua mean = {resid.std() / np.sqrt(len(resid)):.4f})")
        emit(f"    Std             = {resid.std():.4f}")
        emit(f"    Min / Max       = {resid.min():.2f} / {resid.max():.2f}")
        emit(f"    Skewness        = {stats.skew(resid):.4f}   Kurtosis = {stats.kurtosis(resid):.4f}")

        lb = acorr_ljungbox(resid, lags=[1, 6, 12, 24], return_df=True)
        emit("    Kiem dinh Ljung-Box (H0: residual la white noise):")
        for lag in lb.index:
            emit(f"      lag = {lag:2d}: Q = {lb.loc[lag, 'lb_stat']:8.3f}   "
                 f"p = {lb.loc[lag, 'lb_pvalue']:.6f}   "
                 f"{'BAC BO' if lb.loc[lag, 'lb_pvalue'] < 0.05 else 'khong bac bo'}")

        a = acf(resid, nlags=24, fft=False)
        acf_str = ", ".join(f"{k}:{a[k]:+.3f}" for k in [1, 2, 3, 12, 24])
        emit(f"    ACF             : {acf_str}")

        if label == "STL (thang log)":
            sw_p = stats.shapiro(resid).pvalue
            jb_p = stats.jarque_bera(resid).pvalue
        else:
            sw_p = stats.shapiro(resid).pvalue
            jb_p = stats.jarque_bera(resid).pvalue
        emit(f"    Shapiro-Wilk    : p = {sw_p:.4f}   "
             f"{'duong bang chuan' if sw_p >= 0.05 else 'KHONG duong bang chuan'}")
        emit(f"    Jarque-Bera     : p = {jb_p:.4f}   "
             f"{'duong bang chuan' if jb_p >= 0.05 else 'KHONG duong bang chuan'}")

    emit()
    emit("  Kiem dinh phuong sai khac nghiet (dich lech chuan cua 2 nua):")
    e, f_ = stats.levene(stl.resid[:72], stl.resid[72:])
    emit(f"    Levene (1949-54 vs 1955-60): stat = {e:.4f}, p = {f_:.6f}   "
         f"{'BAC BO (hetero)' if f_ < 0.05 else 'khong bac bo'}")
    first_half, second_half = stl.resid[:72], stl.resid[72:]
    emit(f"    std nua dau = {first_half.std():.3f}, std nua sau = {second_half.std():.3f} "
         f"(tang {second_half.std() / first_half.std():.2f} lan)")

    section("5. KET LUAN VA NHAN XET")
    rule("-")
    emit("Cau 1 - Chon chuoi du lieu theo thang hay theo quy:")
    emit("  Chon CHUOI THEO THANG. Ly do:")
    emit("    - Du lieu co 144 quan sat (12 nam x 12 thang), du de xac dinh chu ky mua vu 12 thang")
    emit("      va uoc luong Trend + Seasonal + Residual deu o do phan giai cao nhat.")
    emit("    - Chu ky mua vu 12 thang khong chia het cho 4 nen phan ra theo quy (chu ky 4)")
    emit("      se lam mat phan thong tin mua vu (chi con 4 diem mot vong).")
    emit("    - Neu tach theo quy, bien do mua vu van tang dan theo nam, khong giai quyet het")
    emit("      van de khac phuc cua phan ra.")

    emit()
    emit("Cau 2 - STL hay classical decomposition:")
    emit("  Dung STL (Seasonal-Trend decomposition using LOESS) lam phuong phap chinh,")
    emit("  ket hop classical decomposition de doi chieu.")
    emit("  Ly do chon STL:")
    emit("    - STL giu nguyen phan Seasonal o bien goc dui dang cong, khong ep bien do mua vu")
    emit("      khong doi (classical decomposition ep nhieu mua vu khac nhau).")
    emit("    - Xu huong duoc uoc luong boi LOESS: khong ep da bac da thu truong nhan voi tu")
    emit("      lai khu du lich, phu hop hon moving average co dinh.")
    emit("    - Bien so 'robust=True' chong lai nhieu du lieu bat thuong, thich hop voi du lieu")
    emit("      thuc te co dinh bien dong lon.")
    emit("  Classical decomposition: phuong phap cong bi lech do bien do mua vu tang dan;")
    emit("  phuong phap nhan cho ket qua tot hon va la luon chon cho du lieu nay.")

    emit()
    emit("Cau 3 - Bon thanh phan:")
    emit("  Da ve 4 thanh phan va luu vao thu muc 'bai_3_figures':")
    emit("    stl_phan_rava.png      - Observed / Trend / Seasonal / Residual (thang duoc tu)")
    emit("    stl_phan_rava_log.png  - 4 thanh phan tren thang log")
    emit("    classical_phan_rava.png - Residual cua cong so voi nhan")

    emit()
    emit("Cau 4 - Mo ta:")
    emit("  a) Xu huong TANG hay GIAM?")
    emit("     => TANG manh va on dinh. Trend tang tu %.1f (1949-01) len %.1f (1960-12),"
         % (stl.trend.iloc[0], stl.trend.iloc[-1]))
    emit("        gap %.0f luat khach, tuong duong ~%.2f%%/nam. Khong co dau hieu suy giam."
         % (stl.trend.iloc[-1] - stl.trend.iloc[0],
            100 * ((stl.trend.iloc[-1] / stl.trend.iloc[0]) ** (1 / 11.9) - 1)))
    emit("     Luu y: xu huong khong phai la duong thang thuan ma nhoi dan o phia cuoi")
    emit("     (dot bien nganh khach hang hang khong co du chieu tang). Do la ly do dung STL")
    emit("     voi ban kinh cua nua doi thay vi moving average co dinh.")

    emit()
    emit("  b) Mua vu MANH hay YEU?")
    emit("     => MANH. F_seasonal = %s (STL) va %s (thang log), gan sat 1." % (fmt(f_seas), fmt(f_seas_l)))
    emit("     Dac diem: dinh o T7-T8 (mua he), thap nhat o T11 (mua dong), T1-T2 cung thap.")
    emit("     Bien do tuyet doi TANG theo nam (46 -> 100 -> 179 luat khach giuaa cac giai doan),")
    emit("     nen day la mua vu TYC DO (nhieu) chu khong phai mua vu cong. Phai log-transform")
    emit("     hoac mo hinh nhan de mo ta dung.")

    emit()
    emit("  c) Residual co giong white noise khong?")
    emit("     => KHONG hoan toan. Bang chung:")
    emit("       - Gia tri mean gan 0 (%s) va std nho (%s), nen phan du da 'sach' phan lon bien dong."
         % (fmt(stl.resid.mean()), fmt(stl.resid.std(), 2)))
    emit("       - NHUNG Ljung-Box bac bo H0 o moi do tre (lag 1: p = %s): con tuong quan."
         % fmt(acorr_ljungbox(stl.resid.dropna(), lags=[1], return_df=True).loc[1, 'lb_pvalue']))
    emit("       - ACF(1) = %s > 0, cac lag 1-3 deu duong: phan du co xu huong ngay sau"
         % fmt(acf(stl.resid.dropna(), nlags=1, fft=False)[1]))
    emit("       " + "tinh trang 'nhieu o cap' (over-smoothing) dac trung cua STL.")
    emit("       - Shapiro-Wilk p = %s va Jarque-Bera p = %s: phan du KHONG phan phoi chuan"
         % (fmt(stats.shapiro(stl.resid.dropna()).pvalue),
            fmt(stats.jarque_bera(stl.resid.dropna()).pvalue)))
    emit("         (nhieu lon nhat nam o 2 dau chuoi - noi Trend chua on dinh).")
    emit("       - Phuong sai khac nghiet: std phan du tang tu %s len %s giua 2 nua chuoi."
         % (fmt(first_half.std(), 2), fmt(second_half.std(), 2)))
    emit("     => Vi vay Residual CHUA phai white noise. Khi uoc luong mo hinh du bao nen dung")
    emit("        phan du nay lam input vao ARIMA/SARIMA hoac mo hinh bien doi trang thai,")
    emit("        va nen dung ban phan ra tren thang log de phan du on dinh hon.")

    emit()
    emit("  Ket luon chung:")
    emit("    Chuoi co XU HUONG TANG manh + MUA VU MANH bien dong, phan ra bang STL tren")
    emit("    thang log cho ket qua tot nhat. Residual con tuong quan va hetero, nen mo hinh")
    emit("    phai tiep tuc mo hinh hoa phan du.")

    section("6. FILE DA TAO")
    rule("-")
    emit("  bai_tap_3.py           - script phan tich nay")
    emit("  ket_qua_bai_3.txt      - ket qua xuat ban nay")
    emit("  bai_3_figures/stl_phan_rava.png")
    emit("  bai_3_figures/stl_phan_rava_log.png")
    emit("  bai_3_figures/classical_phan_rava.png")

    OUT_FILE.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\n[OK] Da luu ket qua vao {OUT_FILE}")


if __name__ == "__main__":
    main()