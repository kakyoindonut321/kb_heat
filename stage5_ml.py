#!/usr/bin/env python3
"""stage5_ml.py - Track C: prediksi hari ekstrem H+1 per kota.

Target : is_extreme_day pada hari t+1 (definisi tunggal dari fact_weather_daily).
Split  : berdasarkan TANGGAL TARGET (t+1) agar tidak ada kebocoran di batas periode.
         latih <= 2017, validasi 2018-2020 (hanya untuk memilih ambang probabilitas),
         uji 2021-2025 (tidak dipakai untuk tuning apa pun).
Model  : persistence (baseline), Logistic Regression, Random Forest, masing-masing
         juga tanpa fitur ONI (ablasi: apakah ENSO membantu?). Satu model per kota.
Output : hasil/prediksi/*.csv dan isi fact_heat_prediction (periode uji saja).

Pakai:
  python stage5_ml.py --dry-run     # baca DB, latih, simpan CSV, TIDAK menulis DB
  python stage5_ml.py               # sama, lalu tulis fact_heat_prediction (idempoten)
Koneksi lewat env var PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv
load_dotenv()

TRAIN_END = pd.Timestamp("2017-12-31")
VAL_END = pd.Timestamp("2020-12-31")
ONI_LAG_MONTHS = 2          # ONI bulan m-2 dipakai untuk hari di bulan m
SEED = 42

BASE_COLS = ["tmax_c", "tmin_c", "apparent_tmax_c", "wet_bulb_tmax_c",
             "rh_mean_pct", "precip_mm", "radiation_mj", "is_extreme_day"]
MODEL_NAMES = ["persistence", "logreg", "rf", "logreg_no_oni", "rf_no_oni"]

SQL_LOAD = """
SELECT c.city_id, c.city_code, d.date_id, d.full_date, d.year, d.month, d.oni_value,
       f.tmax_c, f.tmin_c, f.apparent_tmax_c, f.wet_bulb_tmax_c,
       f.rh_mean_pct, f.precip_mm, f.radiation_mj, f.is_extreme_day
FROM fact_weather_daily f
JOIN dim_date d ON d.date_id = f.date_id
JOIN dim_city c ON c.city_id = f.city_id
ORDER BY c.city_id, d.full_date
"""


# ----------------------------------------------------------------------------- data
def load_from_db():
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "kb_heat"),
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )
    with conn.cursor() as cur:
        cur.execute(SQL_LOAD)
        cols = [c[0] for c in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    conn.close()
    df["full_date"] = pd.to_datetime(df["full_date"])
    for c in BASE_COLS + ["oni_value"]:
        df[c] = df[c].astype(float)
    return df


def add_oni_lag(df):
    """ONI bulanan digeser ONI_LAG_MONTHS bulan ke belakang (sama untuk semua kota)."""
    m = df.groupby(["year", "month"])["oni_value"].first().sort_index()
    mi = np.array([y * 12 + mo for y, mo in m.index])
    if not (np.diff(mi) == 1).all():
        raise ValueError("Deret ONI bulanan tidak kontinu")
    lagged = m.shift(ONI_LAG_MONTHS).rename("oni_lag")
    df = df.merge(lagged.reset_index(), on=["year", "month"], how="left")
    return df


# ------------------------------------------------------------------------ features
def build_features(g):
    """Fitur untuk satu kota. Semua fitur pada baris t hanya memakai data <= t."""
    g = g.sort_values("full_date").reset_index(drop=True)
    gaps = g["full_date"].diff().dropna()
    if not (gaps == pd.Timedelta(days=1)).all():
        raise ValueError(f"Tanggal tidak kontinu untuk kota {g['city_id'].iloc[0]}")
    o = g[["city_id", "date_id", "full_date"]].copy()
    for c in BASE_COLS:
        o[c] = g[c]
    for k in (1, 2, 3, 7):
        o[f"tmax_lag{k}"] = g["tmax_c"].shift(k)
        o[f"ext_lag{k}"] = g["is_extreme_day"].shift(k)
    o["tmax_roll3"] = g["tmax_c"].rolling(3).mean()
    o["tmax_roll7"] = g["tmax_c"].rolling(7).mean()
    o["ext_roll7"] = g["is_extreme_day"].rolling(7).sum()
    o["precip_roll7"] = g["precip_mm"].rolling(7).sum()
    o["rad_roll3"] = g["radiation_mj"].rolling(3).mean()
    o["month_sin"] = np.sin(2 * np.pi * g["month"] / 12)
    o["month_cos"] = np.cos(2 * np.pi * g["month"] / 12)
    o["oni_lag"] = g["oni_lag"]
    o["y"] = g["is_extreme_day"].shift(-1)                 # target: hari t+1
    o["target_date"] = g["full_date"] + pd.Timedelta(days=1)
    return o


FEATURES_ALL = None  # diisi setelah fitur dibangun


def feature_lists(cols):
    skip = {"city_id", "date_id", "full_date", "y", "target_date"}
    allf = [c for c in cols if c not in skip]
    return allf, [c for c in allf if c != "oni_lag"]


def check_causality(g):
    """Cek kebocoran: fitur baris t harus identik bila data setelah t dibuang."""
    full = build_features(g)
    i = int(len(g) * 0.7)
    part = build_features(g.sort_values("full_date").iloc[: i + 1])
    fcols, _ = feature_lists(full.columns)
    a = full.loc[i, fcols].to_numpy(dtype=float)
    b = part.loc[i, fcols].to_numpy(dtype=float)
    if not np.allclose(a, b, equal_nan=True):
        raise AssertionError("Fitur memakai informasi masa depan (kebocoran)!")


# ------------------------------------------------------------------------- metrics
def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1))])


def calc_metrics(y, p, label):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    yhat = np.asarray(label, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    npos = int(y.sum())
    return {
        "n": int(len(y)), "n_pos": npos, "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "recall": tp / npos if npos else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if npos else np.nan,
        "pr_auc": average_precision_score(y, p) if npos else np.nan,
        "prevalensi": npos / len(y),
    }


def make_model(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=3000))
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                  class_weight="balanced_subsample", random_state=SEED)


# ----------------------------------------------------------------------------- run
def run(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = add_oni_lag(df)
    preds, rows_metric, imps = [], [], []

    for city_id, g in df.groupby("city_id"):
        code = g["city_code"].iloc[0]
        check_causality(g)
        feats = build_features(g)
        f_all, f_noni = feature_lists(feats.columns)
        feats = feats.dropna(subset=f_all + ["y"]).reset_index(drop=True)
        tr = feats[feats["target_date"] <= TRAIN_END]
        va = feats[(feats["target_date"] > TRAIN_END) & (feats["target_date"] <= VAL_END)]
        te = feats[feats["target_date"] > VAL_END]
        print(f"[{code}] latih={len(tr)} (pos {int(tr.y.sum())}), "
              f"val={len(va)} (pos {int(va.y.sum())}), uji={len(te)} (pos {int(te.y.sum())})")

        for name in MODEL_NAMES:
            if name == "persistence":
                p_va = va["is_extreme_day"].to_numpy()
                p_te = te["is_extreme_day"].to_numpy()
                thr = 0.5
            else:
                kind = "logreg" if name.startswith("logreg") else "rf"
                cols = f_noni if name.endswith("no_oni") else f_all
                m = make_model(kind).fit(tr[cols], tr["y"].astype(int))
                p_va = m.predict_proba(va[cols])[:, 1]
                p_te = m.predict_proba(te[cols])[:, 1]
                thr = best_threshold(va["y"].astype(int), p_va)   # tuning hanya di validasi
                if name == "rf":
                    for c, v in zip(cols, m.feature_importances_):
                        imps.append({"city": code, "fitur": c, "importance": v})

            for split, part, p in (("val", va, p_va), ("uji", te, p_te)):
                lab = (p >= thr).astype(int)
                met = calc_metrics(part["y"], p, lab)
                met.update(city=code, city_id=int(city_id), model=name, split=split,
                           ambang=thr, tahun="semua")
                rows_metric.append(met)
                if split == "uji":
                    for yr in sorted(part["target_date"].dt.year.unique()):
                        mk = (part["target_date"].dt.year == yr).to_numpy()
                        m2 = calc_metrics(part["y"].to_numpy()[mk], p[mk], lab[mk])
                        m2.update(city=code, city_id=int(city_id), model=name,
                                  split="uji", ambang=thr, tahun=int(yr))
                        rows_metric.append(m2)
                    preds.append(pd.DataFrame({
                        "city_id": int(city_id), "date_id": part["date_id"].astype(int).to_numpy(),
                        "full_date": part["full_date"].to_numpy(), "model_name": name,
                        "threshold_used": thr, "prob_extreme": p, "pred_label": lab,
                        "actual_label": part["y"].astype(int).to_numpy()}))

    pred = pd.concat(preds, ignore_index=True)
    met = pd.DataFrame(rows_metric)

    # agregat semua kota (tahun = tahun hari target t+1) pada periode uji, per tahun dan total
    pooled = []
    for (model, yr), s in pred.assign(
            tahun=(pred["full_date"] + pd.Timedelta(days=1)).dt.year).groupby(["model_name", "tahun"]):
        r = calc_metrics(s["actual_label"], s["prob_extreme"], s["pred_label"])
        r.update(city="SEMUA", model=model, split="uji", tahun=int(yr), ambang=np.nan)
        pooled.append(r)
    for model, s in pred.groupby("model_name"):
        r = calc_metrics(s["actual_label"], s["prob_extreme"], s["pred_label"])
        r.update(city="SEMUA", model=model, split="uji", tahun="semua", ambang=np.nan)
        pooled.append(r)
    met = pd.concat([met, pd.DataFrame(pooled)], ignore_index=True)

    met.to_csv(out_dir / "metrik_prediksi.csv", index=False, encoding="utf-8")
    pred.drop(columns="full_date").to_csv(out_dir / "prediksi_uji.csv", index=False,
                                          encoding="utf-8")
    if imps:
        (pd.DataFrame(imps).groupby(["city", "fitur"], as_index=False)["importance"].mean()
         .sort_values(["city", "importance"], ascending=[True, False])
         .to_csv(out_dir / "rf_importance.csv", index=False, encoding="utf-8"))
    return pred, met


def summary(met):
    t = met[(met.split == "uji") & (met.tahun == "semua")]
    cols = ["city", "model", "n_pos", "precision", "recall", "f1", "pr_auc", "ambang"]
    print("\n=== Periode uji 2021-2025 (semua tahun) ===")
    print(t[cols].round(3).to_string(index=False))
    piv = t.pivot(index="city", columns="model", values="pr_auc")
    if {"rf", "rf_no_oni"} <= set(piv.columns):
        print("\nSelisih PR-AUC RF dengan ONI - tanpa ONI (per kota):")
        print((piv["rf"] - piv["rf_no_oni"]).round(3).to_string())


def write_db(pred):
    import psycopg2
    from psycopg2.extras import execute_values
    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"), port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "kb_heat"),
        user=os.environ["PGUSER"], password=os.environ["PGPASSWORD"])
    rows = [(int(r.city_id), int(r.date_id), str(r.model_name), float(r.threshold_used),
             float(r.prob_extreme), int(r.pred_label), int(r.actual_label))
            for r in pred.itertuples(index=False)]
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM fact_heat_prediction")
            before = cur.fetchone()[0]
            cur.execute("DELETE FROM fact_heat_prediction WHERE model_name = ANY(%s)",
                        (MODEL_NAMES,))
            execute_values(cur, """INSERT INTO fact_heat_prediction
                (city_id, date_id, model_name, threshold_used, prob_extreme,
                 pred_label, actual_label) VALUES %s""", rows, page_size=5000)
            cur.execute("SELECT COUNT(*) FROM fact_heat_prediction")
            after = cur.fetchone()[0]
        print(f"fact_heat_prediction: sebelum={before}, sesudah={after}")
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="jangan tulis ke database")
    ap.add_argument("--out", default="hasil/prediksi")
    a = ap.parse_args()
    df = load_from_db()
    print(f"Dimuat {len(df)} baris, {df.city_id.nunique()} kota")
    pred, met = run(df, Path(a.out))
    summary(met)
    if a.dry_run:
        print("\n--dry-run: database tidak diubah.")
    else:
        write_db(pred)


if __name__ == "__main__":
    main()
