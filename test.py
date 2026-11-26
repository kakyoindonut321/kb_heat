import pandas as pd
import pymannkendall as mk

df = pd.read_csv("data/raw/weather_daily_all.csv", parse_dates=["date"])
base = df[df.date.dt.year <= 2020]
p95 = base.groupby("city_slug")["temperature_2m_max"].quantile(0.95).rename("p95")
df = df.join(p95, on="city_slug")
df["extreme"] = df.temperature_2m_max >= df.p95
# print(p95)
# print(df.groupby([df.date.dt.year, "city_slug"])["extreme"].sum().unstack().tail(12))

ann = df.groupby([df.date.dt.year.rename("year"), "city_slug"]).agg(tmax_mean=("temperature_2m_max","mean"), n_ext=("extreme","sum")).reset_index()
for c, g in ann.groupby("city_slug"):
    t, e = mk.original_test(g.tmax_mean), mk.original_test(g.n_ext)
    print(c, f"Tmax {t.slope*10:+.2f} °C/dekade (p={t.p:.3f}) | hari ekstrem {e.slope*10:+.1f}/dekade (p={e.p:.3f})")