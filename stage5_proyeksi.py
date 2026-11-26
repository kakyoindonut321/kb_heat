"""
stage5_proyeksi.py - Proyeksi tren Tmax rata-rata tahunan per kota (Fase 5, Track C).

Yang dilakukan:
  1. Fit regresi linear Tmax tahunan ~ tahun per kota (OLS) + kemiringan Sen sebagai pembanding.
  2. Proyeksi tahun depan sampai --akhir (default 2035) dengan dua rentang:
       - pita keyakinan 95% untuk GARIS TREN (seberapa pasti kemiringannya)
       - pita prediksi 95% untuk NILAI TAHUNAN (termasuk variasi antartahun, jauh lebih lebar)
  3. Backtest: fit hanya sampai --backtest-akhir (default 2015), lalu cek berapa persen
     tahun 2016-2025 yang jatuh di dalam pita prediksi 95%.

Yang TIDAK dilakukan (sengaja):
  - Tidak memproyeksikan jumlah hari ekstrem (tidak linear, didominasi tahun El Nino).
  - Tidak memakai ENSO. Ini ekstrapolasi tren, bukan ramalan cuaca atau iklim.
  - Tidak menyentuh database dan tidak mengubah schema.sql. Output hanya CSV dan PNG.

Pemakaian:
  python stage5_proyeksi.py                              # baca data/derived/yearly.csv
  python stage5_proyeksi.py --csv path/ke/yearly.csv
  python stage5_proyeksi.py --source db                  # hitung dari PostgreSQL (.env), BELUM DIUJI
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from dotenv import load_dotenv
load_dotenv()

OUT_DIR = Path("hasil/proyeksi")
BASELINE = (1991, 2020)


def load_yearly(args) -> pd.DataFrame:
    if args.source == "csv":
        df = pd.read_csv(args.csv, encoding="utf-8")
    else:
        import psycopg2
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parent / ".env")
        con = psycopg2.connect(
            host=os.environ["PGHOST"], port=os.environ["PGPORT"],
            dbname=os.environ["PGDATABASE"], user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
        )
        sql = """
            SELECT c.city_code, c.city_name, d.year,
                   ROUND(AVG(f.tmax_c)::numeric, 2) AS tmax_mean_c
            FROM fact_weather_daily f
            JOIN dim_date d ON d.date_id = f.date_id
            JOIN dim_city c ON c.city_id = f.city_id
            GROUP BY c.city_code, c.city_name, d.year
            ORDER BY c.city_code, d.year
        """
        df = pd.read_sql(sql, con)
        con.close()
    df["tmax_mean_c"] = df["tmax_mean_c"].astype(float)
    return df


def ols_fit(x: np.ndarray, y: np.ndarray):
    n = len(x)
    xbar, ybar = x.mean(), y.mean()
    sxx = ((x - xbar) ** 2).sum()
    b = ((x - xbar) * (y - ybar)).sum() / sxx
    a = ybar - b * xbar
    resid = y - (a + b * x)
    s = np.sqrt((resid ** 2).sum() / (n - 2))
    se_b = s / np.sqrt(sxx)
    t_b = b / se_b
    p_b = 2 * (1 - stats.t.cdf(abs(t_b), n - 2))
    r2 = 1 - (resid ** 2).sum() / ((y - ybar) ** 2).sum()
    ar1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    return dict(n=n, a=a, b=b, s=s, se_b=se_b, p=p_b, r2=r2, xbar=xbar, sxx=sxx, ar1=ar1)


def predict(fit: dict, years: np.ndarray, level: float = 0.95) -> pd.DataFrame:
    tcrit = stats.t.ppf(0.5 + level / 2, fit["n"] - 2)
    yhat = fit["a"] + fit["b"] * years
    lev = 1.0 / fit["n"] + (years - fit["xbar"]) ** 2 / fit["sxx"]
    ci = tcrit * fit["s"] * np.sqrt(lev)
    pi = tcrit * fit["s"] * np.sqrt(1 + lev)
    return pd.DataFrame({
        "year": years, "yhat_c": yhat,
        "tren_lo_c": yhat - ci, "tren_hi_c": yhat + ci,
        "pred_lo_c": yhat - pi, "pred_hi_c": yhat + pi,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csv", "db"], default="csv")
    ap.add_argument("--csv", default="data/derived/yearly.csv")
    ap.add_argument("--akhir", type=int, default=2035)
    ap.add_argument("--backtest-akhir", type=int, default=2015)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_yearly(args)
    last_obs = int(df["year"].max())

    ringkas, proy, bt_rows = [], [], []
    for code, g in df.groupby("city_code"):
        g = g.sort_values("year")
        name = g["city_name"].iloc[0]
        x = g["year"].to_numpy(float)
        y = g["tmax_mean_c"].to_numpy(float)

        fit = ols_fit(x, y)
        sen = stats.theilslopes(y, x, alpha=0.95)  # slope, intercept, lo, hi
        base = g[(g.year >= BASELINE[0]) & (g.year <= BASELINE[1])]["tmax_mean_c"].mean()

        years_f = np.arange(last_obs + 1, args.akhir + 1, dtype=float)
        pr = predict(fit, years_f)
        pr.insert(0, "city_code", code)
        pr.insert(1, "city_name", name)
        pr["anomali_vs_baseline_c"] = pr["yhat_c"] - base
        proy.append(pr)

        t_end = predict(fit, np.array([float(args.akhir)]))
        ringkas.append({
            "city_code": code, "city_name": name, "n_tahun": fit["n"],
            "ols_slope_c_per_dekade": fit["b"] * 10,
            "ols_slope_ci95_lo": (fit["b"] - stats.t.ppf(0.975, fit["n"] - 2) * fit["se_b"]) * 10,
            "ols_slope_ci95_hi": (fit["b"] + stats.t.ppf(0.975, fit["n"] - 2) * fit["se_b"]) * 10,
            "ols_p_mentah": fit["p"], "r2": fit["r2"],
            "sen_slope_c_per_dekade": sen[0] * 10,
            "sen_ci95_lo": sen[2] * 10, "sen_ci95_hi": sen[3] * 10,
            "sd_residu_c": fit["s"], "ar1_residu": fit["ar1"],
            "tmax_baseline_1991_2020_c": base,
            f"proyeksi_{args.akhir}_c": float(t_end["yhat_c"][0]),
            f"pred_lo_{args.akhir}_c": float(t_end["pred_lo_c"][0]),
            f"pred_hi_{args.akhir}_c": float(t_end["pred_hi_c"][0]),
        })

        # Backtest: fit sampai backtest_akhir, uji pada tahun sesudahnya.
        tr = g[g.year <= args.backtest_akhir]
        te = g[g.year > args.backtest_akhir]
        if len(tr) >= 10 and len(te) > 0:
            f2 = ols_fit(tr["year"].to_numpy(float), tr["tmax_mean_c"].to_numpy(float))
            p2 = predict(f2, te["year"].to_numpy(float))
            inside = ((te["tmax_mean_c"].to_numpy() >= p2["pred_lo_c"].to_numpy()) &
                      (te["tmax_mean_c"].to_numpy() <= p2["pred_hi_c"].to_numpy()))
            mae = np.abs(te["tmax_mean_c"].to_numpy() - p2["yhat_c"].to_numpy()).mean()
            # pembanding naif: rata-rata periode latih (tanpa tren)
            mae_naif = np.abs(te["tmax_mean_c"].to_numpy() - tr["tmax_mean_c"].mean()).mean()
            bt_rows.append({
                "city_code": code, "city_name": name,
                "tahun_latih": f"{int(tr.year.min())}-{int(tr.year.max())}",
                "tahun_uji": f"{int(te.year.min())}-{int(te.year.max())}",
                "n_uji": len(te), "n_dalam_pita_95": int(inside.sum()),
                "cakupan_pita_95": inside.mean(),
                "mae_tren_c": mae, "mae_rata2_latih_c": mae_naif,
                "slope_latih_c_per_dekade": f2["b"] * 10,
            })

    ringkas = pd.DataFrame(ringkas)
    proy = pd.concat(proy, ignore_index=True)
    bt = pd.DataFrame(bt_rows)

    ringkas.round(4).to_csv(OUT_DIR / "tren_ringkas.csv", index=False, encoding="utf-8")
    proy.round(3).to_csv(OUT_DIR / "proyeksi_tmax.csv", index=False, encoding="utf-8")
    bt.round(4).to_csv(OUT_DIR / "backtest_proyeksi.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("== Ringkasan tren (OLS dan Sen, per dekade) ==")
    print(ringkas[["city_name", "ols_slope_c_per_dekade", "ols_slope_ci95_lo", "ols_slope_ci95_hi",
                   "ols_p_mentah", "sen_slope_c_per_dekade", "sd_residu_c", "ar1_residu"]].round(3).to_string(index=False))
    print(f"\n== Proyeksi tahun {args.akhir} ==")
    print(ringkas[["city_name", "tmax_baseline_1991_2020_c", f"proyeksi_{args.akhir}_c",
                   f"pred_lo_{args.akhir}_c", f"pred_hi_{args.akhir}_c"]].round(2).to_string(index=False))
    print(f"\n== Backtest (latih sampai {args.backtest_akhir}) ==")
    print(bt.round(3).to_string(index=False))
    print(f"\nTersimpan di {OUT_DIR}/")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        codes = list(df["city_code"].unique())
        fig, axes = plt.subplots(len(codes), 1, figsize=(8, 2.6 * len(codes)), sharex=True)
        for ax, code in zip(np.atleast_1d(axes), codes):
            g = df[df.city_code == code].sort_values("year")
            p = proy[proy.city_code == code]
            ax.plot(g["year"], g["tmax_mean_c"], "o-", ms=3, lw=1, color="#444", label="Observasi")
            ax.plot(p["year"], p["yhat_c"], color="#c0392b", label="Proyeksi tren")
            ax.fill_between(p["year"], p["tren_lo_c"], p["tren_hi_c"], color="#c0392b", alpha=0.35,
                            label="Pita keyakinan garis tren 95%")
            ax.fill_between(p["year"], p["pred_lo_c"], p["pred_hi_c"], color="#c0392b", alpha=0.12,
                            label="Pita prediksi nilai tahunan 95%")
            ax.set_ylabel("Tmax rata-rata (°C)")
            ax.set_title(g["city_name"].iloc[0], loc="left", fontsize=10)
        np.atleast_1d(axes)[0].legend(fontsize=7, ncol=2, loc="upper left")
        np.atleast_1d(axes)[-1].set_xlabel("Tahun")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "proyeksi_tmax.png", dpi=150)
        print("Grafik:", OUT_DIR / "proyeksi_tmax.png")


if __name__ == "__main__":
    main()
