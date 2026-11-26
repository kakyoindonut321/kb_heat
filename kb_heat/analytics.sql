-- =====================================================================
-- sql/analytics.sql  |  Track A: Query analitik SQL (database kb_heat)
-- Urutan: jalankan blok per blok di pgAdmin. Bagian 1-2 membuat VIEW
-- (semantic layer), bagian 3-9 adalah query analitik.
-- Definisi hari ekstrem SELALU memakai fact_weather_daily.is_extreme_day
-- (berasal dari ref_heat_threshold). Jangan hitung ulang ambang di sini.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 0. Cek cepat sebelum mulai (hasil yang diharapkan di komentar)
-- ---------------------------------------------------------------------
SELECT COUNT(*) AS n_fact FROM fact_weather_daily;         -- 63920
SELECT COUNT(*) AS n_mv   FROM mv_extreme_yearly;          -- 175 (5 x 35)
SELECT city_id, city_code, city_name FROM dim_city ORDER BY city_id;  -- peta city_id -> kota

-- ---------------------------------------------------------------------
-- 1. Semantic layer: satu view gabungan fakta + dimensi
--    Semua query lain membaca dari sini agar definisi seragam.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_weather_enriched AS
SELECT f.city_id, c.city_code, c.city_name,
       f.date_id, d.full_date, d.year, d.month, d.analysis_period,
       d.enso_phase, d.oni_value,
       f.tmax_c, f.tmean_c, f.is_extreme_day
FROM fact_weather_daily f
JOIN dim_city c USING (city_id)
JOIN dim_date d USING (date_id);

-- ---------------------------------------------------------------------
-- 2. Klimatologi bulanan (baseline 1991-2020) dan anomali Tmax harian
--    Dipakai untuk analisis sensitivitas (anomali, bukan ambang p95).
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_climatology_monthly AS
SELECT city_id, month,
       AVG(tmax_c)::numeric(5,2) AS tmax_clim_c,
       COUNT(*)                  AS n_days
FROM v_weather_enriched
WHERE analysis_period = 'baseline'
GROUP BY city_id, month;

CREATE OR REPLACE VIEW v_tmax_anomaly_daily AS
SELECT w.city_id, w.city_code, w.city_name, w.date_id, w.full_date,
       w.year, w.month, w.analysis_period, w.enso_phase,
       w.tmax_c, k.tmax_clim_c,
       (w.tmax_c - k.tmax_clim_c)::numeric(5,2) AS tmax_anom_c,
       w.is_extreme_day
FROM v_weather_enriched w
JOIN v_climatology_monthly k USING (city_id, month);

-- Cek: klimatologi 5 kota x 12 bulan = 60 baris; rata-rata anomali baseline ~ 0
SELECT COUNT(*) AS n_rows FROM v_climatology_monthly;      -- 60
SELECT city_name, ROUND(AVG(tmax_anom_c),3) AS rata_anom_baseline
FROM v_tmax_anomaly_daily
WHERE analysis_period = 'baseline'
GROUP BY city_name ORDER BY city_name;                     -- mendekati 0.000

-- ---------------------------------------------------------------------
-- 3. Rata-rata bergulir 30 hari (window function + CTE)
--    Ganti 'jakarta' dengan slug kota lain. 29 hari pertama dibuang
--    karena jendelanya belum penuh.
-- ---------------------------------------------------------------------
WITH harian AS (
    SELECT city_name, full_date, tmax_c,
           ROW_NUMBER() OVER (PARTITION BY city_id ORDER BY full_date) AS n,
           AVG(tmax_c) OVER (PARTITION BY city_id ORDER BY full_date
                             ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma30
    FROM v_weather_enriched
    WHERE city_code = 'jakarta'
)
SELECT city_name, full_date, tmax_c, ROUND(ma30::numeric, 2) AS tmax_ma30
FROM harian
WHERE n >= 30
ORDER BY full_date;

-- ---------------------------------------------------------------------
-- 4. Tmax tahunan: selisih vs tahun lalu (LAG) dan rata-rata 5 tahun
-- ---------------------------------------------------------------------
SELECT c.city_name, y.year, y.tmax_mean_c,
       ROUND((y.tmax_mean_c - LAG(y.tmax_mean_c) OVER w)::numeric, 2) AS delta_vs_tahun_lalu,
       ROUND(AVG(y.tmax_mean_c) OVER (PARTITION BY y.city_id ORDER BY y.year
                                      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)::numeric, 2) AS ma5_tahun
FROM mv_extreme_yearly y
JOIN dim_city c USING (city_id)
WINDOW w AS (PARTITION BY y.city_id ORDER BY y.year)
ORDER BY c.city_name, y.year;

-- ---------------------------------------------------------------------
-- 5. Peringkat kota per tahun (RANK): suhu rata-rata dan % hari ekstrem
--    Catatan: RANK memberi nilai sama bila seri (bisa ada celah nomor).
-- ---------------------------------------------------------------------
SELECT y.year, c.city_name,
       y.tmax_mean_c,
       RANK() OVER (PARTITION BY y.year ORDER BY y.tmax_mean_c DESC)   AS rank_tmax,
       y.pct_extreme,
       RANK() OVER (PARTITION BY y.year ORDER BY y.pct_extreme DESC)   AS rank_ekstrem
FROM mv_extreme_yearly y
JOIN dim_city c USING (city_id)
ORDER BY y.year, rank_tmax;

-- ---------------------------------------------------------------------
-- 6. Warm spell (>= 3 hari ekstrem berturut-turut), dengan nama kota
-- ---------------------------------------------------------------------
-- 6a. 10 warm spell terpanjang (memperbaiki ambiguitas city_id di konteks)
SELECT c.city_name, w.start_date, w.end_date, w.duration_days
FROM mv_warm_spell w
JOIN dim_city c USING (city_id)
ORDER BY w.duration_days DESC, w.start_date
LIMIT 10;
-- Diharapkan: teratas 51 hari (2023-09-02 s/d 2023-10-22)

-- 6b. Ringkasan per kota dan tahun (tahun = tahun mulai spell)
SELECT c.city_name,
       EXTRACT(YEAR FROM w.start_date)::int AS tahun,
       COUNT(*)                AS n_spell,
       MAX(w.duration_days)    AS spell_terpanjang,
       SUM(w.duration_days)    AS total_hari_dalam_spell
FROM mv_warm_spell w
JOIN dim_city c USING (city_id)
GROUP BY c.city_name, EXTRACT(YEAR FROM w.start_date)
ORDER BY c.city_name, tahun;

-- 6c. Bulan mulai spell (dugaan: semua Agu-Nov)
SELECT EXTRACT(MONTH FROM start_date)::int AS bulan_mulai, COUNT(*) AS n_spell
FROM mv_warm_spell GROUP BY 1 ORDER BY 1;

-- ---------------------------------------------------------------------
-- 7. % hari ekstrem per kota x fase ENSO x bulan
--    Selalu tampilkan n_hari DAN n_tahun: sampel per sel bisa kecil.
-- ---------------------------------------------------------------------
-- 7a. Per kota x fase (seluruh periode)
-- Diharapkan: Bandung El Nino 11,06 | Netral 4,03 | La Nina 1,84
SELECT city_name, enso_phase,
       COUNT(*) AS n_hari,
       ROUND(100.0 * SUM(is_extreme_day) / COUNT(*), 2) AS pct_ekstrem
FROM v_weather_enriched
GROUP BY city_name, enso_phase
ORDER BY city_name, enso_phase;

-- 7b. Per bulan x fase, gabungan 5 kota (format lebar)
-- Diharapkan: Okt El Nino 48,2 | Sep El Nino 30,9 | Okt Netral 11,3
SELECT month,
  ROUND(100.0*SUM(is_extreme_day) FILTER (WHERE enso_phase='El Nino')
        / NULLIF(COUNT(*) FILTER (WHERE enso_phase='El Nino'),0), 1) AS el_nino,
  ROUND(100.0*SUM(is_extreme_day) FILTER (WHERE enso_phase='Netral')
        / NULLIF(COUNT(*) FILTER (WHERE enso_phase='Netral'),0), 1)  AS netral,
  ROUND(100.0*SUM(is_extreme_day) FILTER (WHERE enso_phase='La Nina')
        / NULLIF(COUNT(*) FILTER (WHERE enso_phase='La Nina'),0), 1) AS la_nina
FROM v_weather_enriched
GROUP BY month ORDER BY month;

-- 7c. Jumlah tahun per bulan x fase (untuk batasan di laporan)
-- Diharapkan: Oktober El Nino = 10 tahun, September = 7 tahun
SELECT month, enso_phase, COUNT(DISTINCT year) AS n_tahun
FROM v_weather_enriched
GROUP BY month, enso_phase
ORDER BY month, enso_phase;

-- 7d. Per kota x bulan x fase (untuk heatmap dashboard)
SELECT city_name, month, enso_phase,
       COUNT(*) AS n_hari, COUNT(DISTINCT year) AS n_tahun,
       ROUND(100.0 * SUM(is_extreme_day) / COUNT(*), 2) AS pct_ekstrem
FROM v_weather_enriched
GROUP BY city_name, month, enso_phase
ORDER BY city_name, month, enso_phase;

-- ---------------------------------------------------------------------
-- 8. Baseline (1991-2020) vs uji (2021-2025)
-- ---------------------------------------------------------------------
SELECT city_name, analysis_period,
       COUNT(*)                                        AS n_hari,
       ROUND(AVG(tmax_c)::numeric, 2)                  AS tmax_rata2,
       SUM(is_extreme_day)                             AS n_ekstrem,
       ROUND(100.0 * SUM(is_extreme_day) / COUNT(*), 2) AS pct_ekstrem
FROM v_weather_enriched
GROUP BY city_name, analysis_period
ORDER BY city_name, analysis_period;
-- Diharapkan: baseline pct_ekstrem 5,3-5,6% (sedikit >5% karena pembulatan 0,1 C dan '>=')

-- 8b. Subtotal bertingkat dengan ROLLUP (kota -> periode -> total)
SELECT COALESCE(city_name,'SEMUA KOTA')       AS kota,
       COALESCE(analysis_period,'semua periode') AS periode,
       SUM(is_extreme_day) AS n_ekstrem
FROM v_weather_enriched
GROUP BY ROLLUP (city_name, analysis_period)
ORDER BY city_name NULLS LAST, analysis_period NULLS LAST;

-- 8c. Kontribusi tahun 2023-2024 terhadap hari ekstrem 2021-2025 (diharapkan ~87%)
WITH uji AS (
    SELECT year, SUM(is_extreme_day) AS n FROM v_weather_enriched
    WHERE analysis_period = 'uji' GROUP BY year
)
SELECT year, n,
       ROUND(100.0 * n / SUM(n) OVER (), 1) AS pct_dari_total_uji
FROM uji ORDER BY year;

-- ---------------------------------------------------------------------
-- 9. EXPLAIN ANALYZE sebelum dan sesudah index
--    Query uji: filter rentang tanggal TANPA kota. PK (city_id, date_id)
--    tidak membantu karena city_id bukan kolom pertama yang difilter.
--    Catat 'Execution Time' dan jenis scan di tiap langkah (untuk LK/laporan).
--    Cek dulu index yang sudah ada: SELECT indexname FROM pg_indexes WHERE tablename='fact_weather_daily';
-- ---------------------------------------------------------------------
DROP INDEX IF EXISTS idx_fact_weather_date;
ANALYZE fact_weather_daily;

-- 9a. SEBELUM
EXPLAIN (ANALYZE, BUFFERS)
SELECT city_id, SUM(is_extreme_day) AS n_ekstrem
FROM fact_weather_daily
WHERE date_id BETWEEN 20230901 AND 20231031
GROUP BY city_id;

-- 9b. Buat index, perbarui statistik
CREATE INDEX idx_fact_weather_date ON fact_weather_daily (date_id);
ANALYZE fact_weather_daily;

-- 9c. SESUDAH
EXPLAIN (ANALYZE, BUFFERS)
SELECT city_id, SUM(is_extreme_day) AS n_ekstrem
FROM fact_weather_daily
WHERE date_id BETWEEN 20230901 AND 20231031
GROUP BY city_id;
-- Catatan jujur: tabel hanya ~64 ribu baris, jadi planner bisa tetap memilih
-- Seq Scan. Jika begitu, laporkan apa adanya (dan jelaskan alasannya) -- jangan
-- memaksa hasil. Jika index dipertahankan, tambahkan ke schema.sql.
