# kb_heat: Hari Panas Ekstrem 5 Kota Indonesia dan ENSO (1991-2025)

Proyek akhir Kecerdasan Bisnis (124210593), S1 Sistem Informasi, UPN Veteran Yogyakarta.
Alur: Open-Meteo API -> ETL Python -> PostgreSQL (star schema) -> SQL/statistik/ML -> Power BI.

## Prasyarat
- Python 3.11, PostgreSQL + pgAdmin
- `pip install -r requirements.txt`
- Salin `.env.example` ke `.env`, isi, lalu muat sebagai environment variable
  (PowerShell: `$env:PGPASSWORD="..."` untuk tiap variabel).

## Urutan menjalankan
1. Buat database kosong `kb_heat` di PostgreSQL.
2. `psql -d kb_heat -f schema.sql` (membuat tabel dan mengisi `dim_date`).
3. `python etl_fetch.py` (opsional bila `data/raw/` sudah ada).
4. `python etl_load_pg.py --dry-run`, lalu `python etl_load_pg.py`.
   Bisa dijalankan ulang (idempoten). Harapan: `fact_weather_daily` = 63.920 baris.
5. Jalankan blok di `sql/analytics.sql` (view dan query analitik).
6. `python stats_tests.py` (tahap statistik).
7. `python stage5_ml.py` (tahap prediktif; mengisi `fact_heat_prediction`).

## Keputusan kunci
- Hari ekstrem: Tmax >= p95 baseline 1991-2020 per kota (`ref_heat_threshold`, `is_extreme_day`).
- Baseline 1991-2020, uji 2021-2025. Warm spell: >= 3 hari ekstrem berturut-turut.
- Data reanalisis ERA5 via Open-Meteo, satu titik per kota, bukan data stasiun BMKG.

## Struktur
| Path | Isi |
|---|---|
| `schema.sql` | DDL + `dim_date` |
| `etl_fetch.py`, `etl_load_pg.py` | Extract; transform, validasi, load |
| `sql/analytics.sql` | View semantic layer dan query analitik |
| `data/raw/` | Data mentah dan ONI v6 |
| `dokumentasi/` | Bukti dan tangkapan layar |
