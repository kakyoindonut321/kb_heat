"""
etl_load_pg.py -- TRANSFORM + VALIDATE + LOAD ke PostgreSQL (database kb_heat)

Input  (default folder data/raw/):
  weather_daily_all.csv   <- hasil etl_fetch.py
  city_meta.json          <- hasil etl_fetch.py
  onyv6.csv               <- tabel ONI v6 NOAA CPC (format lebar: Year, DJF..NDJ)

Yang dilakukan:
  1. dim_city            : UPSERT dari city_meta.json
  2. dim_date (ENSO)     : UPDATE oni_value / enso_phase per bulan
  3. ref_heat_threshold  : p95 Tmax per kota dari periode baseline (default 1991-2020)
  4. fact_weather_daily  : UPSERT (ON CONFLICT) + flag is_extreme_day
  5. REFRESH materialized view

Idempoten: aman dijalankan berulang. Semua dalam SATU transaksi (gagal = rollback).

Koneksi lewat environment variable (jangan hardcode password):
  PGHOST (localhost) PGPORT (5432) PGDATABASE (kb_heat) PGUSER (postgres) PGPASSWORD

Jalankan:  python etl_load_pg.py
           python etl_load_pg.py --dry-run     (extract+transform+validate saja, tanpa load)
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("etl_load")

SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
# Musim 3-bulanan dipetakan ke BULAN TENGAHNYA: DJF -> Januari (1), ..., NDJ -> Desember (12)
SEASON_TO_MONTH = {s: i + 1 for i, s in enumerate(SEASONS)}
ENSO_SOURCE = "CPC ONI v6"

FACT_COLS = {  # kolom CSV -> kolom tabel fakta
    "temperature_2m_max": "tmax_c",
    "temperature_2m_min": "tmin_c",
    "temperature_2m_mean": "tmean_c",
    "apparent_temperature_max": "apparent_tmax_c",
    "apparent_temperature_mean": "apparent_tmean_c",
    "wet_bulb_temperature_2m_max": "wet_bulb_tmax_c",
    "dew_point_2m_mean": "dew_point_c",
    "relative_humidity_2m_mean": "rh_mean_pct",
    "precipitation_sum": "precip_mm",
    "shortwave_radiation_sum": "radiation_mj",
}
FACT_INSERT_ORDER = ["city_id", "date_id", "tmax_c", "tmin_c", "tmean_c", "apparent_tmax_c",
                     "apparent_tmean_c", "wet_bulb_tmax_c", "dew_point_c", "rh_mean_pct",
                     "precip_mm", "radiation_mj", "is_extreme_day"]


# ----------------------------------------------------------------- EXTRACT
def read_json(path: Path) -> dict:
    """Baca JSON; toleran terhadap file lama yang tersimpan cp1252 (Windows)."""
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Encoding {path} tidak dikenali")


def extract(raw_dir: Path, oni_path: Path):
    weather = pd.read_csv(raw_dir / "weather_daily_all.csv", parse_dates=["date"])
    meta = read_json(raw_dir / "city_meta.json")
    oni_wide = pd.read_csv(oni_path)
    log.info("Extract: weather=%d baris, kota=%d, ONI=%d tahun", len(weather), len(meta), len(oni_wide))
    return weather, meta, oni_wide


# --------------------------------------------------------------- TRANSFORM
def build_city_dim(meta: dict) -> pd.DataFrame:
    rows = [dict(city_code=k, city_name=v["name"], province=v["province"], geo_type=v["geo_type"],
                 req_latitude=v["requested_lat"], req_longitude=v["requested_lon"],
                 grid_latitude=v["grid_lat"], grid_longitude=v["grid_lon"],
                 elevation_m=v["elevation_m"], timezone=v["timezone"]) for k, v in meta.items()]
    return pd.DataFrame(rows)


def build_enso_long(oni_wide: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    """Wide (Year x musim) -> long (year, month, oni_value, enso_phase)."""
    long = oni_wide.melt(id_vars="Year", value_vars=SEASONS, var_name="season", value_name="oni_value")
    long["month"] = long["season"].map(SEASON_TO_MONTH)
    long = long.rename(columns={"Year": "year"})
    long = long[long.year.between(year_min, year_max)].dropna(subset=["oni_value"]).copy()
    long["enso_phase"] = np.select(
        [long.oni_value >= 0.5, long.oni_value <= -0.5], ["El Nino", "La Nina"], default="Netral")
    long["enso_source"] = ENSO_SOURCE
    return long[["year", "month", "oni_value", "enso_phase", "enso_source"]].sort_values(["year", "month"])


def compute_thresholds(w: pd.DataFrame, baseline_start: str, baseline_end: str, pct: float) -> pd.DataFrame:
    base = w[(w.date >= baseline_start) & (w.date <= baseline_end)]
    th = (base.groupby("city_slug")["temperature_2m_max"].quantile(pct)
          .round(2).rename("threshold_value").reset_index())
    th["metric"], th["percentile"] = "temperature_2m_max", pct
    th["baseline_start"], th["baseline_end"] = pd.Timestamp(baseline_start).date(), pd.Timestamp(baseline_end).date()
    return th


def transform_fact(w: pd.DataFrame, th: pd.DataFrame) -> pd.DataFrame:
    f = w.rename(columns=FACT_COLS).copy()
    f["date_id"] = f["date"].dt.strftime("%Y%m%d").astype(int)
    f = f.merge(th[["city_slug", "threshold_value"]], on="city_slug", how="left")
    f["is_extreme_day"] = (f["tmax_c"] >= f["threshold_value"]).astype(int)
    return f


# ---------------------------------------------------------------- VALIDATE
def validate(f: pd.DataFrame, enso: pd.DataFrame, n_cities: int, exp_start: str, exp_end: str):
    """Fail-fast. Setiap blok = satu dimensi kualitas data (minggu 4)."""
    errs, warns = [], []
    core = list(FACT_COLS.values())

    # 1. Completeness
    n_null = int(f[core].isnull().sum().sum())
    if n_null:
        errs.append(f"Completeness: {n_null} sel null di kolom inti")
    exp_days = (pd.Timestamp(exp_end) - pd.Timestamp(exp_start)).days + 1
    per_city = f.groupby("city_slug").size()
    bad = per_city[per_city != exp_days]
    if len(bad) or f.city_slug.nunique() != n_cities:
        errs.append(f"Completeness: baris per kota harus {exp_days}; ditemukan {per_city.to_dict()}")
    if len(enso) != 12 * (pd.Timestamp(exp_end).year - pd.Timestamp(exp_start).year + 1):
        errs.append(f"Completeness: ONI harus punya 12 bulan/tahun; ditemukan {len(enso)} baris")

    # 2. Uniqueness
    n_dup = int(f.duplicated(["city_slug", "date_id"]).sum())
    if n_dup:
        errs.append(f"Uniqueness: {n_dup} duplikat (kota, tanggal)")

    # 3. Validity (aturan rentang)
    if ((f.rh_mean_pct < 0) | (f.rh_mean_pct > 100)).any():
        errs.append("Validity: kelembapan di luar 0-100")
    if (f.precip_mm < 0).any() or (f.radiation_mj < 0).any():
        errs.append("Validity: hujan/radiasi negatif")

    # 4. Accuracy (kewajaran nilai fisik untuk kota tropis)
    out = int(((f.tmax_c < 15) | (f.tmax_c > 45)).sum())
    if out:
        errs.append(f"Accuracy: {out} Tmax di luar 15-45 C")

    # 5. Consistency (antar kolom)
    if (f.tmax_c < f.tmin_c).any():
        errs.append("Consistency: ada Tmax < Tmin")
    n_mean = int(((f.tmean_c < f.tmin_c - 0.5) | (f.tmean_c > f.tmax_c + 0.5)).sum())
    if n_mean:
        warns.append(f"Consistency: {n_mean} baris Tmean di luar [Tmin, Tmax] (toleransi 0,5 C)")

    # 6. Timeliness (rentang tanggal sesuai ekspektasi)
    if f.date.min() != pd.Timestamp(exp_start) or f.date.max() != pd.Timestamp(exp_end):
        errs.append(f"Timeliness: rentang {f.date.min().date()}..{f.date.max().date()} != {exp_start}..{exp_end}")

    for w_ in warns:
        log.warning(w_)
    if errs:
        raise ValueError("Validasi gagal, ETL dihentikan:\n - " + "\n - ".join(errs))
    log.info("Validasi lolos (6 dimensi kualitas)")


# -------------------------------------------------------------------- LOAD
def to_rows(df: pd.DataFrame, cols):
    """DataFrame -> list tuple tipe Python murni (psycopg2 tidak bisa adaptasi numpy.int64)."""
    sub = df[cols].astype(object).where(df[cols].notna(), None)
    return [tuple(r) for r in sub.values.tolist()]


def load(conn, city_dim, enso, th, fact):
    from psycopg2.extras import execute_values
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM fact_weather_daily")
    before = cur.fetchone()[0]

    # 1. dim_city
    cols = list(city_dim.columns)
    upd = ",".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "city_code")
    execute_values(cur, f"INSERT INTO dim_city ({','.join(cols)}) VALUES %s "
                        f"ON CONFLICT (city_code) DO UPDATE SET {upd}", to_rows(city_dim, cols))
    cur.execute("SELECT city_code, city_id FROM dim_city")
    city_id = dict(cur.fetchall())

    # 2. dim_date: ENSO per bulan (SCD Type 1: timpa)
    execute_values(cur,
        "UPDATE dim_date d SET oni_value = v.oni, enso_phase = v.phase, enso_source = v.src "
        "FROM (VALUES %s) AS v(yr, mo, oni, phase, src) WHERE d.year = v.yr AND d.month = v.mo",
        to_rows(enso, ["year", "month", "oni_value", "enso_phase", "enso_source"]),
        template="(%s::smallint, %s::smallint, %s::numeric, %s::text, %s::text)")

    # 3. ref_heat_threshold
    th = th.assign(city_id=th.city_slug.map(city_id))
    tcols = ["city_id", "metric", "percentile", "baseline_start", "baseline_end", "threshold_value"]
    execute_values(cur, f"INSERT INTO ref_heat_threshold ({','.join(tcols)}) VALUES %s "
                        "ON CONFLICT (city_id, metric, percentile) DO UPDATE SET "
                        "baseline_start=EXCLUDED.baseline_start, baseline_end=EXCLUDED.baseline_end, "
                        "threshold_value=EXCLUDED.threshold_value", to_rows(th, tcols))

    # 4. fact_weather_daily (cek FK dulu supaya pesan error ramah)
    fact = fact.assign(city_id=fact.city_slug.map(city_id))
    cur.execute("SELECT date_id FROM dim_date")
    known = {r[0] for r in cur.fetchall()}
    missing = set(fact.date_id) - known
    if missing:
        raise ValueError(f"{len(missing)} date_id tidak ada di dim_date (contoh {sorted(missing)[:3]})")
    set_cols = [c for c in FACT_INSERT_ORDER if c not in ("city_id", "date_id")]
    execute_values(cur, f"INSERT INTO fact_weather_daily ({','.join(FACT_INSERT_ORDER)}) VALUES %s "
                        f"ON CONFLICT (city_id, date_id) DO UPDATE SET "
                        + ",".join(f"{c}=EXCLUDED.{c}" for c in set_cols),
                   to_rows(fact, FACT_INSERT_ORDER), page_size=5000)

    # 5. materialized view
    cur.execute("REFRESH MATERIALIZED VIEW mv_extreme_yearly")
    cur.execute("REFRESH MATERIALIZED VIEW mv_warm_spell")

    cur.execute("SELECT COUNT(*) FROM fact_weather_daily")
    after = cur.fetchone()[0]
    log.info("fact_weather_daily: sebelum=%d, sesudah=%d", before, after)
    return before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--oni", default=None, help="default: <raw-dir>/onyv6.csv")
    ap.add_argument("--start", default="1991-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--baseline-start", default="1991-01-01")
    ap.add_argument("--baseline-end", default="2020-12-31")
    ap.add_argument("--percentile", type=float, default=0.95)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw = Path(a.raw_dir)
    weather, meta, oni_wide = extract(raw, Path(a.oni) if a.oni else raw / "onyv6.csv")

    y0, y1 = pd.Timestamp(a.start).year, pd.Timestamp(a.end).year
    city_dim = build_city_dim(meta)
    enso = build_enso_long(oni_wide, y0, y1)
    th = compute_thresholds(weather, a.baseline_start, a.baseline_end, a.percentile)
    fact = transform_fact(weather, th)

    validate(fact, enso, len(meta), a.start, a.end)

    summ = fact.groupby("city_slug").agg(hari=("is_extreme_day", "size"), ekstrem=("is_extreme_day", "sum"))
    summ["pct"] = (100 * summ.ekstrem / summ.hari).round(2)
    log.info("Ambang p95 per kota:\n%s", th.set_index("city_slug").threshold_value.to_string())
    log.info("Ringkasan hari ekstrem (seluruh periode):\n%s", summ.to_string())

    if a.dry_run:
        log.info("Dry-run selesai, tidak ada yang dimuat.")
        return

    import psycopg2
    conn = psycopg2.connect(host=os.getenv("PGHOST", "localhost"), port=os.getenv("PGPORT", "5432"),
                            dbname=os.getenv("PGDATABASE", "kb_heat"), user=os.getenv("PGUSER", "postgres"),
                            password=os.getenv("PGPASSWORD"))
    try:
        load(conn, city_dim, enso, th, fact)
        conn.commit()
        log.info("ETL selesai, transaksi di-commit.")
    except Exception:
        conn.rollback()
        log.exception("ETL gagal, rollback.")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
