"""Fase 6 (versi aman): tabel aturan rekomendasi per kota -> hasil/aturan/aturan_rekomendasi.csv

Hanya membaca CSV hasil yang sudah ada. Tidak menyentuh database.
Ambang siaga = ambang logreg yang dipilih di data VALIDASI (kolom 'ambang' pada split val),
bukan dari data uji. Metrik uji hanya dilaporkan untuk mengecek.
Kalender musim memakai hasil Bab III (SON; Medan Mar-Agu), dicantumkan sebagai konstanta.
"""
import argparse
from pathlib import Path
import pandas as pd

KOTA = {"jakarta": "Jakarta", "bandung": "Bandung", "surabaya": "Surabaya",
        "makassar": "Makassar", "medan": "Medan"}
# Dari hasil Bab III (enso_musim.csv dan pola bulanan). Asosiasi, bukan sebab-akibat.
KALENDER = {
    "Jakarta": ("Agu-Nov", "ya (SON: El Nino 41,9% vs Netral 11,0% vs La Nina 6,8%)"),
    "Bandung": ("Agu-Nov", "ya (SON: 34,9% vs 6,5% vs 5,5%)"),
    "Surabaya": ("Agu-Nov", "ya (SON: 40,2% vs 17,2% vs 7,9%)"),
    "Makassar": ("Agu-Nov", "ya (SON: 41,7% vs 8,7% vs 4,8%)"),
    "Medan": ("Mar-Agu", "tidak tampak (SON: p Holm 0,85)"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrik", default="hasil/prediksi/metrik_prediksi.csv")
    ap.add_argument("--tren", default="hasil/stats/trend_mk.csv")
    ap.add_argument("--spell", default="analytics_results/013_SELECT_c_city_name__EXTRACT_YEAR_FROM_w.csv")
    ap.add_argument("--out", default="hasil/aturan/aturan_rekomendasi.csv")
    a = ap.parse_args()

    m = pd.read_csv(a.metrik)
    m = m[(m.tahun == "semua") & (m.city != "SEMUA")].copy()
    m["kota"] = m.city.map(KOTA)

    def pick(model, split, col):
        s = m[(m.model == model) & (m.split == split)].set_index("kota")[col]
        return s

    t = pd.read_csv(a.tren)
    tmax = t[t.variabel == "tmax_mean_c"].set_index("kota")
    hx = t[t.variabel == "n_extreme_days"].set_index("kota")

    w = pd.read_csv(a.spell).groupby("city_name").agg(
        n_spell=("n_spell", "sum"), spell_terpanjang_hari=("spell_terpanjang", "max"),
        total_hari_dalam_spell=("total_hari_dalam_spell", "sum"))

    rows = []
    for k in KOTA.values():
        f1_p, f1_l = pick("persistence", "uji", "f1")[k], pick("logreg", "uji", "f1")[k]
        npos = int(pick("logreg", "uji", "n_pos")[k])
        lemah = f1_l < f1_p - 0.02  # selisih kecil dianggap setara
        rows.append({
            "kota": k,
            "ambang_siaga_logreg_dari_validasi": round(pick("logreg", "val", "ambang")[k], 3),
            "uji_recall_logreg": round(pick("logreg", "uji", "recall")[k], 3),
            "uji_precision_logreg": round(pick("logreg", "uji", "precision")[k], 3),
            "uji_f1_logreg": round(f1_l, 3),
            "uji_f1_persistence": round(f1_p, 3),
            "uji_n_hari_ekstrem": npos,
            "catatan_model": ("model lebih lemah dari persistence (F1); pakai persistence sebagai dasar, "
                              "logreg hanya sebagai penanda recall tinggi" if lemah
                              else "setara persistence pada F1; nilai tambah = probabilitas dan recall"),
            "kalender_waspada": KALENDER[k][0],
            "naik_saat_el_nino": KALENDER[k][1],
            "tmax_sen_c_per_dekade": round(tmax.loc[k, "sen_per_dekade"], 3),
            "tmax_p_holm": round(tmax.loc[k, "p_holm"], 3),
            "hari_ekstrem_sen_per_dekade": round(hx.loc[k, "sen_per_dekade"], 1),
            "hari_ekstrem_p_holm": round(hx.loc[k, "p_holm"], 3),
            "bukti_tren_setelah_holm": "kuat" if (tmax.loc[k, "signifikan_holm"] or hx.loc[k, "signifikan_holm"]) else "tidak cukup",
            "n_warm_spell_1991_2025": int(w.loc[k, "n_spell"]),
            "warm_spell_terpanjang_hari": int(w.loc[k, "spell_terpanjang_hari"]),
            "total_hari_dalam_warm_spell": int(w.loc[k, "total_hari_dalam_spell"]),
        })
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, encoding="utf-8")
    print(out.T.to_string())
    print("\nTersimpan di", a.out)

if __name__ == "__main__":
    main()
