#!/usr/bin/env python
"""
stats_tests.py  |  Fase 4 / Track B: statistik deskriptif dan diagnostik (kb_heat)

Menjawab BQ1 (tren per kota) dan BQ2 (efek ENSO) dengan uji non-parametrik.
Satuan analisis SELALU tahun (atau kota x tahun x bulan), BUKAN hari,
karena hari berurutan tidak independen.

Cara pakai (dari folder kb_heat, environment variable PG* sudah di-set):
    python stats_tests.py                 # ambil dari PostgreSQL, simpan input ke data/derived/
    python stats_tests.py --source csv    # pakai data/derived/*.csv (tanpa database)

Keluaran: hasil/stats/*.csv

Dependensi: pandas, numpy, scipy (+ psycopg2 untuk --source db).
Definisi hari ekstrem tidak dihitung ulang: memakai is_extreme_day dari database.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

load_dotenv()

PHASES = ["El Nino", "Netral", "La Nina"]
MIN_N = 3          # minimal tahun per fase agar uji dijalankan
ALPHA = 0.05
SEASONS = {
    # nama: (bulan, bulan pusat untuk menentukan fase ENSO, catatan)
    "SON (Sep-Nov)": ([9, 10, 11], 10, "ditetapkan sebelum melihat hasil uji (musim ENSO-sensitif)"),
    "Mar-Jul": ([3, 4, 5, 6, 7], 5, "EKSPLORATIF: dipilih setelah melihat deskriptif (puncak Medan)"),
}

SQL_YEARLY = """
SELECT c.city_code, c.city_name, y.year::int AS year,
       y.tmax_mean_c::float AS tmax_mean_c,
       y.n_extreme_days::int AS n_extreme_days,
       y.pct_extreme::float AS pct_extreme
FROM mv_extreme_yearly y JOIN dim_city c USING (city_id)
ORDER BY c.city_code, y.year
"""
SQL_MONTHLY = """
SELECT city_code, city_name, year::int AS year, month::int AS month, enso_phase,
       COUNT(*)::int AS n_days,
       SUM(is_extreme_day)::int AS n_extreme,
       SUM(tmax_anom_c)::float AS anom_sum
FROM v_tmax_anomaly_daily
GROUP BY city_code, city_name, year, month, enso_phase
ORDER BY city_code, year, month
"""


# ----------------------------------------------------------------- statistik
def mann_kendall(y):
    """Uji Mann-Kendall (dua sisi, koreksi ties) + kemiringan Sen + AR(1) residu."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    s, slopes = 0.0, []
    for i in range(n - 1):
        d = y[i + 1:] - y[i]
        s += np.sign(d).sum()
        slopes.extend(d / np.arange(1, n - i))
    _, cnt = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(cnt * (cnt - 1) * (2 * cnt + 5))) / 18.0
    if var <= 0 or s == 0:
        z = 0.0
    else:
        z = (s - 1) / np.sqrt(var) if s > 0 else (s + 1) / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    sen = float(np.median(slopes))
    resid = y - sen * np.arange(n)
    resid = resid - np.median(resid)
    ar1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if np.std(resid) > 0 else np.nan
    return {"S": s, "z": z, "p": p, "sen": sen, "ar1_residu": ar1}


def holm(p):
    """Koreksi Holm-Bonferroni; NaN dibiarkan NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    idx = [i for i in np.argsort(p) if not np.isnan(p[i])]
    m, run = len(idx), 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def cliffs_delta(a, b):
    """Cliff's delta (-1..1): peluang nilai a > b dikurangi peluang a < b."""
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2.0 * u / (len(a) * len(b)) - 1.0


def kruskal_safe(groups):
    """Kruskal-Wallis + epsilon^2; NaN bila data tidak memungkinkan."""
    allv = np.concatenate(groups)
    if min(len(g) for g in groups) < MIN_N or np.ptp(allv) == 0:
        return np.nan, np.nan, np.nan
    h, p = stats.kruskal(*groups)
    n = len(allv)
    return float(h), float(p), float(h * (n + 1) / (n * n - 1))


def mw_safe(a, b):
    if len(a) < MIN_N or len(b) < MIN_N or np.ptp(np.concatenate([a, b])) == 0:
        return np.nan, np.nan
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue), float(cliffs_delta(a, b))


def phase_compare(g, value_col):
    """Bandingkan 3 fase ENSO pada satu kelompok (satu nilai per tahun)."""
    grp = {ph: g.loc[g.enso_phase == ph, value_col].to_numpy(float) for ph in PHASES}
    row = {f"n_{ph}": len(v) for ph, v in grp.items()}
    row.update({f"rerata_{ph}": float(v.mean()) if len(v) else np.nan for ph, v in grp.items()})
    row["H_kw"], row["p_kw"], row["eps2_kw"] = kruskal_safe([grp[p] for p in PHASES])
    row["p_ElNino_vs_Netral"], row["delta_ElNino_vs_Netral"] = mw_safe(grp["El Nino"], grp["Netral"])
    row["p_ElNino_vs_LaNina"], row["delta_ElNino_vs_LaNina"] = mw_safe(grp["El Nino"], grp["La Nina"])
    return row


# ----------------------------------------------------------------- analisis
def trend_tests(yearly):
    rows = []
    for var in ["tmax_mean_c", "n_extreme_days"]:
        block = []
        for city, g in yearly.groupby("city_name"):
            g = g.sort_values("year")
            r = mann_kendall(g[var])
            block.append({"variabel": var, "kota": city, "n_tahun": len(g),
                          "sen_per_dekade": r["sen"] * 10, "S": r["S"], "z": r["z"],
                          "p_mentah": r["p"], "ar1_residu": r["ar1_residu"]})
        df = pd.DataFrame(block)
        df["p_holm"] = holm(df["p_mentah"])
        df["signifikan_holm"] = df["p_holm"] < ALPHA
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def city_difference(yearly):
    """Beda antar kota pada % hari ekstrem per tahun. Tahun yang sama dipakai semua kota,
    jadi Friedman (blok = tahun) adalah desain yang tepat; Kruskal-Wallis disertakan sesuai spesifikasi."""
    wide = yearly.pivot(index="year", columns="city_name", values="pct_extreme")
    cols = [wide[c].to_numpy() for c in wide.columns]
    chi, pf = stats.friedmanchisquare(*cols)
    h, pk = stats.kruskal(*cols)
    return pd.DataFrame([
        {"uji": "Friedman (blok=tahun)", "statistik": chi, "p": pf, "n_tahun": len(wide)},
        {"uji": "Kruskal-Wallis (abaikan pasangan tahun)", "statistik": h, "p": pk, "n_tahun": len(wide)},
    ])


def baseline_vs_uji(yearly):
    rows = []
    yearly = yearly.assign(periode=np.where(yearly.year >= 2021, "uji", "baseline"))
    for var in ["pct_extreme", "tmax_mean_c"]:
        block = []
        for city, g in yearly.groupby("city_name"):
            b = g.loc[g.periode == "baseline", var].to_numpy(float)
            u = g.loc[g.periode == "uji", var].to_numpy(float)
            p, d = mw_safe(u, b)
            block.append({"variabel": var, "kota": city, "n_baseline": len(b), "n_uji": len(u),
                          "median_baseline": float(np.median(b)), "median_uji": float(np.median(u)),
                          "p_mentah": p, "cliff_delta_uji_vs_baseline": d})
        df = pd.DataFrame(block)
        df["p_holm"] = holm(df["p_mentah"])
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def enso_city_month(monthly, value_col):
    rows = []
    for (city, month), g in monthly.groupby(["city_name", "month"]):
        rows.append({"kota": city, "bulan": month, **phase_compare(g, value_col)})
    df = pd.DataFrame(rows)
    for col in ["p_kw", "p_ElNino_vs_Netral", "p_ElNino_vs_LaNina"]:
        df[col.replace("p_", "p_holm_")] = holm(df[col])
    return df


def season_table(monthly, months, center):
    g = monthly[monthly.month.isin(months)]
    agg = g.groupby(["city_name", "year"], as_index=False).agg(
        n_days=("n_days", "sum"), n_extreme=("n_extreme", "sum"), anom_sum=("anom_sum", "sum"))
    phase = monthly.loc[monthly.month == center, ["city_name", "year", "enso_phase"]]
    agg = agg.merge(phase, on=["city_name", "year"], how="left")
    agg["pct_extreme"] = 100.0 * agg.n_extreme / agg.n_days
    agg["anom_mean"] = agg.anom_sum / agg.n_days
    return agg


def enso_season(monthly):
    out = []
    for name, (months, center, note) in SEASONS.items():
        st = season_table(monthly, months, center)
        for var in ["pct_extreme", "anom_mean"]:
            for city, g in st.groupby("city_name"):
                out.append({"musim": name, "catatan": note, "variabel": var, "kota": city,
                            **phase_compare(g, var)})
    df = pd.DataFrame(out)
    # keluarga uji: semua uji KW utama (pct_extreme) dikoreksi bersama; anomali = sensitivitas, dikoreksi terpisah
    df["p_holm_kw"] = np.nan
    for var in ["pct_extreme", "anom_mean"]:
        m = df.variabel == var
        df.loc[m, "p_holm_kw"] = holm(df.loc[m, "p_kw"])
    return df


def adjusted_model(monthly):
    """OLS per kota: pct_extreme_musim ~ tahun + El Nino + La Nina (acuan: Netral), SE HC3.
    Tujuan: memisahkan efek tren dari efek ENSO. Eksploratif (n kecil, residu bisa berautokorelasi)."""
    rows = []
    for name, (months, center, _) in SEASONS.items():
        st = season_table(monthly, months, center)
        for city, g in st.groupby("city_name"):
            g = g.dropna(subset=["enso_phase"]).sort_values("year")
            X = np.column_stack([np.ones(len(g)), g.year - 2008,
                                 (g.enso_phase == "El Nino").astype(float),
                                 (g.enso_phase == "La Nina").astype(float)])
            y = g.pct_extreme.to_numpy(float)
            xtx_inv = np.linalg.pinv(X.T @ X)
            beta = xtx_inv @ X.T @ y
            e = y - X @ beta
            h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
            z = X * (e / (1 - h))[:, None]
            meat = z.T @ z
            se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
            dfree = len(y) - X.shape[1]
            for k, term in enumerate(["intersep", "tren_per_tahun", "El Nino vs Netral", "La Nina vs Netral"]):
                if term == "intersep":
                    continue
                t = beta[k] / se[k] if se[k] > 0 else np.nan
                p = float(2 * stats.t.sf(abs(t), dfree)) if not np.isnan(t) else np.nan
                rows.append({"musim": name, "kota": city, "suku": term, "koef_poin_persen": beta[k],
                             "se_hc3": se[k], "p": p, "n_tahun": len(y)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- I/O
def load_db():
    import psycopg2
    try:
        conn = psycopg2.connect(host=os.environ.get("PGHOST", "localhost"),
                                port=os.environ.get("PGPORT", "5432"),
                                dbname=os.environ.get("PGDATABASE", "kb_heat"),
                                user=os.environ["PGUSER"], password=os.environ["PGPASSWORD"])
    except KeyError as e:
        sys.exit(f"Environment variable {e} belum di-set (lihat .env.example).")
    out = []
    with conn, conn.cursor() as cur:
        for sql in (SQL_YEARLY, SQL_MONTHLY):
            cur.execute(sql)
            out.append(pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description]))
    conn.close()
    return out


def validate(yearly, monthly):
    """Gagal cepat bila input tidak sesuai harapan."""
    assert yearly.city_name.nunique() == 5, "harus ada 5 kota"
    assert (yearly.groupby("city_name").year.nunique() == 35).all(), "tiap kota harus 35 tahun"
    assert len(monthly) == 5 * 35 * 12, f"monthly harus 2100 baris, ditemukan {len(monthly)}"
    assert monthly.enso_phase.isin(PHASES).all(), "ada fase ENSO di luar El Nino/Netral/La Nina"
    assert monthly.groupby(["city_name", "year", "month"]).size().max() == 1, \
        "fase ENSO tidak konstan dalam satu bulan"
    assert monthly.n_extreme.le(monthly.n_days).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["db", "csv"], default="db")
    ap.add_argument("--derived-dir", default=os.path.join("data", "derived"))
    ap.add_argument("--out-dir", default=os.path.join("hasil", "stats"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.derived_dir, exist_ok=True)
    yp, mp = os.path.join(a.derived_dir, "yearly.csv"), os.path.join(a.derived_dir, "monthly.csv")

    if a.source == "db":
        yearly, monthly = load_db()
        yearly.to_csv(yp, index=False, encoding="utf-8")
        monthly.to_csv(mp, index=False, encoding="utf-8")
    else:
        yearly = pd.read_csv(yp, encoding="utf-8")
        monthly = pd.read_csv(mp, encoding="utf-8")
    validate(yearly, monthly)
    print(f"Input OK: yearly={len(yearly)} baris, monthly={len(monthly)} baris")

    outputs = {
        "trend_mk.csv": trend_tests(yearly),
        "beda_antar_kota.csv": city_difference(yearly),
        "baseline_vs_uji.csv": baseline_vs_uji(yearly),
        "enso_kota_bulan_pct_ekstrem.csv": enso_city_month(
            monthly.assign(pct_extreme=100.0 * monthly.n_extreme / monthly.n_days), "pct_extreme"),
        "enso_kota_bulan_anomali.csv": enso_city_month(
            monthly.assign(anom_mean=monthly.anom_sum / monthly.n_days), "anom_mean"),
        "enso_musim.csv": enso_season(monthly),
        "enso_model_terkoreksi_tren.csv": adjusted_model(monthly),
    }
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for name, df in outputs.items():
        df.to_csv(os.path.join(a.out_dir, name), index=False, encoding="utf-8")
        print(f"\n== {name} ({len(df)} baris)")
    print("\n-- Tren (ringkas)")
    print(outputs["trend_mk.csv"].round(3).to_string(index=False))
    print("\n-- ENSO per musim, pct_extreme (ringkas)")
    s = outputs["enso_musim.csv"]
    cols = ["musim", "kota", "n_El Nino", "n_Netral", "n_La Nina", "rerata_El Nino", "rerata_Netral",
            "rerata_La Nina", "p_kw", "p_holm_kw", "eps2_kw"]
    print(s[s.variabel == "pct_extreme"][cols].round(3).to_string(index=False))
    print(f"\nSelesai. Tabel lengkap di {a.out_dir}/")


if __name__ == "__main__":
    main()
