"""
etl_fetch.py  --  EXTRACT (Open-Meteo Historical Weather API, resolusi harian)

Output (folder data/raw/):
  - weather_daily_<slug>.csv   : data mentah per kota
  - weather_daily_all.csv      : gabungan 5 kota (long format, grain = kota x hari)
  - city_meta.json             : koordinat diminta vs grid cell yang dipakai API + elevasi
  - extract_report.csv         : ringkasan kualitas data per kota

Jalankan:  python etl_fetch.py            (semua kota)
           python etl_fetch.py jakarta     (uji satu kota dulu)
"""
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = "1991-01-01", "2025-12-31"   # 1991-2020 = baseline WMO, 2021-2025 = periode uji
MODEL = "era5_seamless"                   # reanalisis konsisten (bukan best_match) -> aman untuk tren
YEARS_PER_CALL = 5                        # pecah permintaan agar ringan
PAUSE_SEC = 1.0                           # jeda antar permintaan (sopan terhadap API gratis)
RAW_DIR = Path("data/raw")

CITIES = [
    dict(slug="jakarta",   name="Jakarta",   province="DKI Jakarta",      lat=-6.2088, lon=106.8456, tz="Asia/Jakarta",   geo_type="pesisir"),
    dict(slug="bandung",   name="Bandung",   province="Jawa Barat",       lat=-6.9175, lon=107.6191, tz="Asia/Jakarta",   geo_type="dataran tinggi"),
    dict(slug="surabaya",  name="Surabaya",  province="Jawa Timur",       lat=-7.2575, lon=112.7521, tz="Asia/Jakarta",   geo_type="pesisir"),
    dict(slug="makassar",  name="Makassar",  province="Sulawesi Selatan", lat=-5.1477, lon=119.4327, tz="Asia/Makassar",  geo_type="pesisir"),
    dict(slug="medan",     name="Medan",     province="Sumatera Utara",   lat=3.5952,  lon=98.6722,  tz="Asia/Jakarta",   geo_type="dataran rendah"),
]

# Variabel inti (wajib). Nama sesuai dokumentasi Open-Meteo.
DAILY_CORE = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum", "shortwave_radiation_sum",
]
# Variabel tambahan (opsional). Kalau API menolak salah satunya, skrip mencoba tanpa ini.
DAILY_OPTIONAL = ["wet_bulb_temperature_2m_max", "dew_point_2m_mean"]


def call_api(city, start, end, daily_vars, retries=5):
    params = dict(
        latitude=city["lat"], longitude=city["lon"],
        start_date=start, end_date=end,
        daily=",".join(daily_vars),
        timezone=city["tz"], models=MODEL,
    )
    for attempt in range(1, retries + 1):
        r = requests.get(URL, params=params, timeout=90)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:                      # kena rate limit
            wait = 60 * attempt
            print(f"   429 rate limit, tunggu {wait}s ...")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(5 * attempt)
            continue
        # 4xx lain (mis. nama variabel salah) -> tampilkan alasan dari API
        reason = r.json().get("reason", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise ValueError(f"HTTP {r.status_code}: {reason}")
    raise RuntimeError("Gagal setelah beberapa percobaan")


def year_chunks():
    y0, y1 = int(START[:4]), int(END[:4])
    for y in range(y0, y1 + 1, YEARS_PER_CALL):
        yield f"{y}-01-01", f"{min(y + YEARS_PER_CALL - 1, y1)}-12-31"


def fetch_city(city, daily_vars):
    frames, meta = [], None
    for s, e in year_chunks():
        print(f"   {city['name']}: {s} .. {e}")
        js = call_api(city, s, e, daily_vars)
        if meta is None:
            meta = dict(
                requested_lat=city["lat"], requested_lon=city["lon"],
                grid_lat=js.get("latitude"), grid_lon=js.get("longitude"),
                elevation_m=js.get("elevation"), timezone=js.get("timezone"),
                units=js.get("daily_units"),
            )
        frames.append(pd.DataFrame(js["daily"]))
        time.sleep(PAUSE_SEC)
    df = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df.insert(0, "city_slug", city["slug"])
    df["source"] = "open-meteo-archive"
    df["model"] = MODEL
    df["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df, meta


def quality_report(df, slug):
    """Cek ringan 6 dimensi kualitas (versi lengkap dilakukan di tahap transform/validate)."""
    expected = (date.fromisoformat(END) - date.fromisoformat(START)).days + 1
    num = df.select_dtypes("number")
    return dict(
        city_slug=slug,
        rows=len(df), rows_expected=expected,                         # completeness
        completeness_ok=len(df) == expected,
        dup_dates=int(df.duplicated(["city_slug", "date"]).sum()),    # uniqueness
        null_cells=int(num.isnull().sum().sum()),                     # completeness
        tmax_lt_tmin=int((df["temperature_2m_max"] < df["temperature_2m_min"]).sum()),  # consistency
        tmax_out_of_range=int(((df["temperature_2m_max"] < 15) | (df["temperature_2m_max"] > 45)).sum()),  # validity
        tmax_min=df["temperature_2m_max"].min(), tmax_max=df["temperature_2m_max"].max(),
        date_min=str(df["date"].min()), date_max=str(df["date"].max()),  # timeliness
    )


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cities = [c for c in CITIES if only in (None, c["slug"])]

    # Uji dulu apakah variabel opsional diterima model yang dipilih.
    daily_vars = DAILY_CORE + DAILY_OPTIONAL
    try:
        call_api(cities[0], "2024-01-01", "2024-01-03", daily_vars)
    except ValueError as err:
        print(f"Variabel opsional ditolak ({err}). Lanjut hanya variabel inti.")
        daily_vars = DAILY_CORE

    all_frames, metas, reports = [], {}, []
    for c in cities:
        print(f"== {c['name']}")
        df, meta = fetch_city(c, daily_vars)
        df.to_csv(RAW_DIR / f"weather_daily_{c['slug']}.csv", index=False)
        all_frames.append(df)
        metas[c["slug"]] = {**c, **meta}
        reports.append(quality_report(df, c["slug"]))

    pd.concat(all_frames, ignore_index=True).to_csv(RAW_DIR / "weather_daily_all.csv", index=False)
    (RAW_DIR / "city_meta.json").write_text(json.dumps(metas, indent=2, ensure_ascii=False))
    rep = pd.DataFrame(reports)
    rep.to_csv(RAW_DIR / "extract_report.csv", index=False)
    print("\n", rep.to_string(index=False))

    # Peringatan: apakah beberapa kota jatuh di grid cell yang sama?
    cells = pd.DataFrame(metas).T[["grid_lat", "grid_lon"]]
    if cells.duplicated().any():
        print("\nPERINGATAN: ada kota dengan grid cell yang sama:\n", cells)


if __name__ == "__main__":
    main()
