-- =====================================================================
-- schema.sql  --  Data Warehouse: Hari Panas Ekstrem 5 Kota Indonesia
-- Kimball four-step:
--   1. Proses bisnis : pengamatan cuaca harian per kota
--   2. Grain         : 1 baris fact_weather_daily = 1 kota x 1 hari
--   3. Dimensi       : dim_date, dim_city
--   4. Fakta         : suhu, kelembapan, hujan, radiasi, flag hari ekstrem
-- Jalankan sekali pada database kosong (mis. kb_heat). Aman dijalankan ulang.
-- =====================================================================

DROP MATERIALIZED VIEW IF EXISTS mv_warm_spell;
DROP MATERIALIZED VIEW IF EXISTS mv_extreme_yearly;
DROP TABLE IF EXISTS fact_heat_prediction;
DROP TABLE IF EXISTS fact_weather_daily;
DROP TABLE IF EXISTS ref_heat_threshold;
DROP TABLE IF EXISTS dim_city;
DROP TABLE IF EXISTS dim_date;

-- ---------------------------------------------------------------------
-- DIMENSI
-- ---------------------------------------------------------------------
-- dim_date: smart key YYYYMMDD (praktik umum untuk dimensi tanggal).
-- Kolom ENSO diisi ETL ONI (satu nilai per bulan, sama untuk semua kota).
-- Nilai ONI terbaru bisa direvisi NOAA -> perlakuan SCD Type 1 (timpa).
CREATE TABLE dim_date (
    date_id          INTEGER      PRIMARY KEY,           -- 19910101
    full_date        DATE         NOT NULL UNIQUE,
    year             SMALLINT     NOT NULL,
    quarter          SMALLINT     NOT NULL,
    month            SMALLINT     NOT NULL,
    month_name       VARCHAR(10)  NOT NULL,
    day_of_month     SMALLINT     NOT NULL,
    day_of_year      SMALLINT     NOT NULL,
    day_name         VARCHAR(10)  NOT NULL,
    is_weekend       BOOLEAN      NOT NULL,
    analysis_period  VARCHAR(10)  NOT NULL
                     CHECK (analysis_period IN ('baseline','uji')),  -- 1991-2020 | 2021-2025
    oni_value        NUMERIC(4,2),                        -- NULL sampai ETL ONI jalan
    enso_phase       VARCHAR(10)
                     CHECK (enso_phase IN ('El Nino','La Nina','Netral')),
    enso_source      VARCHAR(40)                          -- mis. 'CPC ONI v6' (catat versi!)
);

-- dim_city: diisi dari city_meta.json (hasil etl_fetch.py)
CREATE TABLE dim_city (
    city_id        SERIAL        PRIMARY KEY,            -- surrogate key
    city_code      VARCHAR(20)   NOT NULL UNIQUE,        -- natural key (slug)
    city_name      VARCHAR(50)   NOT NULL,
    province       VARCHAR(50)   NOT NULL,
    geo_type       VARCHAR(20),                          -- pesisir / dataran tinggi / dataran rendah
    req_latitude   NUMERIC(9,6)  NOT NULL,
    req_longitude  NUMERIC(9,6)  NOT NULL,
    grid_latitude  NUMERIC(9,6),                         -- grid cell yang dipakai API
    grid_longitude NUMERIC(9,6),
    elevation_m    NUMERIC(6,1),
    timezone       VARCHAR(30)   NOT NULL
);

-- Ambang hari ekstrem per kota (transparan & bisa diaudit).
-- Dihitung dari baseline 1991-2020, bukan dari seluruh data.
CREATE TABLE ref_heat_threshold (
    city_id         INTEGER       NOT NULL REFERENCES dim_city(city_id),
    metric          VARCHAR(30)   NOT NULL,              -- 'temperature_2m_max', dst.
    percentile      NUMERIC(4,3)  NOT NULL,              -- 0.950
    baseline_start  DATE          NOT NULL,
    baseline_end    DATE          NOT NULL,
    threshold_value NUMERIC(5,2)  NOT NULL,
    PRIMARY KEY (city_id, metric, percentile)
);

-- ---------------------------------------------------------------------
-- FAKTA
-- ---------------------------------------------------------------------
-- Catatan additivity: precip_mm, radiation_mj, flag 0/1 = additive (boleh SUM).
-- Suhu & kelembapan = non-additive -> agregasi pakai AVG/MAX (di SQL/DAX), bukan SUM.
CREATE TABLE fact_weather_daily (
    city_id          INTEGER      NOT NULL REFERENCES dim_city(city_id),
    date_id          INTEGER      NOT NULL REFERENCES dim_date(date_id),
    tmax_c           NUMERIC(4,1) NOT NULL,
    tmin_c           NUMERIC(4,1) NOT NULL,
    tmean_c          NUMERIC(4,1),
    apparent_tmax_c  NUMERIC(4,1),
    apparent_tmean_c NUMERIC(4,1),
    wet_bulb_tmax_c  NUMERIC(4,1),
    dew_point_c      NUMERIC(4,1),
    rh_mean_pct      NUMERIC(5,1) CHECK (rh_mean_pct BETWEEN 0 AND 100),
    precip_mm        NUMERIC(6,1) CHECK (precip_mm >= 0),
    radiation_mj     NUMERIC(5,2) CHECK (radiation_mj >= 0),
    is_extreme_day   SMALLINT     NOT NULL CHECK (is_extreme_day IN (0,1)),  -- tmax >= p95 kota
    PRIMARY KEY (city_id, date_id),
    CHECK (tmax_c >= tmin_c)
);
CREATE INDEX idx_fwd_date ON fact_weather_daily(date_id);

-- Hasil model H+1: baris = 1 kota x 1 hari asal prediksi x 1 model.
-- Memprediksi apakah HARI BERIKUTNYA (date_id + 1 hari) ekstrem.
CREATE TABLE fact_heat_prediction (
    city_id        INTEGER      NOT NULL REFERENCES dim_city(city_id),
    date_id        INTEGER      NOT NULL REFERENCES dim_date(date_id),  -- hari t (asal prediksi)
    model_name     VARCHAR(40)  NOT NULL,                               -- 'persistence','logreg','rf'
    threshold_used NUMERIC(4,3) NOT NULL,
    prob_extreme   NUMERIC(5,4),
    pred_label     SMALLINT     NOT NULL CHECK (pred_label IN (0,1)),
    actual_label   SMALLINT     NOT NULL CHECK (actual_label IN (0,1)), -- kondisi hari t+1
    PRIMARY KEY (city_id, date_id, model_name)
);
CREATE INDEX idx_fhp_date ON fact_heat_prediction(date_id);

-- ---------------------------------------------------------------------
-- POPULATE dim_date (1991-2025)
-- ---------------------------------------------------------------------
INSERT INTO dim_date (date_id, full_date, year, quarter, month, month_name,
                      day_of_month, day_of_year, day_name, is_weekend, analysis_period)
SELECT to_char(d, 'YYYYMMDD')::int,
       d::date,
       EXTRACT(YEAR FROM d)::smallint,
       EXTRACT(QUARTER FROM d)::smallint,
       EXTRACT(MONTH FROM d)::smallint,
       (ARRAY['Januari','Februari','Maret','April','Mei','Juni','Juli',
              'Agustus','September','Oktober','November','Desember'])[EXTRACT(MONTH FROM d)::int],
       EXTRACT(DAY FROM d)::smallint,
       EXTRACT(DOY FROM d)::smallint,
       (ARRAY['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'])[EXTRACT(ISODOW FROM d)::int],
       EXTRACT(ISODOW FROM d) IN (6,7),
       CASE WHEN EXTRACT(YEAR FROM d) <= 2020 THEN 'baseline' ELSE 'uji' END
FROM generate_series('1991-01-01'::date, '2025-12-31'::date, interval '1 day') AS g(d);

-- ---------------------------------------------------------------------
-- MATERIALIZED VIEW (jalankan REFRESH setelah ETL / setelah flag dihitung)
-- ---------------------------------------------------------------------
-- Ringkasan tahunan per kota -> sumber utama halaman tren di Power BI
CREATE MATERIALIZED VIEW mv_extreme_yearly AS
SELECT f.city_id, d.year,
       COUNT(*)                         AS n_days,
       SUM(f.is_extreme_day)            AS n_extreme_days,
       ROUND(100.0 * SUM(f.is_extreme_day) / COUNT(*), 2) AS pct_extreme,
       ROUND(AVG(f.tmax_c), 2)          AS tmax_mean_c,
       MAX(f.tmax_c)                    AS tmax_peak_c
FROM fact_weather_daily f
JOIN dim_date d USING (date_id)
GROUP BY f.city_id, d.year;

-- Warm spell: >= 3 hari ekstrem berturut-turut (teknik gaps-and-islands, minggu 5)
CREATE MATERIALIZED VIEW mv_warm_spell AS
WITH ext AS (
    SELECT f.city_id, d.full_date,
           d.full_date - (ROW_NUMBER() OVER (PARTITION BY f.city_id ORDER BY d.full_date))::int AS grp
    FROM fact_weather_daily f
    JOIN dim_date d USING (date_id)
    WHERE f.is_extreme_day = 1
),
runs AS (
    SELECT city_id, grp,
           MIN(full_date) AS start_date,
           MAX(full_date) AS end_date,
           COUNT(*)       AS duration_days
    FROM ext
    GROUP BY city_id, grp
)
SELECT city_id, start_date, end_date, duration_days
FROM runs
WHERE duration_days >= 3;
