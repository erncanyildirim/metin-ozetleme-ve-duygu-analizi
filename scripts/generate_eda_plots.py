"""
EDA Grafiklerini Lokal Olarak Üret
====================================
Notebook 1'in oluşturduğu 4 grafiği (Şekil 4.1, 4.2, 4.3, 4.3b) yeniden üretir.
rapor/ klasörüne PNG olarak kaydeder.

Kullanım:
    python3 scripts/generate_eda_plots.py
"""

import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Proje kök dizinini path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import clean_data, add_sentiment_labels, load_data
from src.preprocessing import preprocess_dataframe, get_word_freq


def main():
    sns.set_style("whitegrid")

    # Çıkış klasörü
    rapor_dir = os.path.join(PROJECT_ROOT, "rapor")
    os.makedirs(rapor_dir, exist_ok=True)

    # Veriyi yükle (v2 versiyonu varsa onu kullan)
    data_dir = os.path.join(PROJECT_ROOT, "data")
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    csv_path = os.path.join(
        data_dir,
        next((f for f in csv_files if "v2" in f and "clean" not in f), csv_files[0])
    )
    print(f"📖 Okunuyor: {csv_path}")

    df = load_data(csv_path)
    df = clean_data(df)
    df = add_sentiment_labels(df)

    text_col = "review_body" if "review_body" in df.columns else df.select_dtypes("object").columns[0]
    df["word_count"] = df[text_col].astype(str).str.split().str.len()

    # Renk paleti
    colors_map = {"pozitif": "#28a745", "nötr": "#ffc107", "negatif": "#dc3545"}

    # ============================================================
    # Şekil 4.1 — Sınıf Dağılımı (pasta + bar)
    # ============================================================
    print("\n🎨 Şekil 4.1 üretiliyor...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sentiment_counts = df["sentiment"].value_counts()
    color_list = [colors_map.get(s, "#888") for s in sentiment_counts.index]

    axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index,
                colors=color_list, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Duygu Sınıfı Dağılımı", fontsize=14, fontweight="bold")

    axes[1].bar(sentiment_counts.index, sentiment_counts.values, color=color_list)
    axes[1].set_title("Yorum Sayıları", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Yorum sayısı")
    for i, v in enumerate(sentiment_counts.values):
        axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(rapor_dir, "sekil_4_1_sinif_dagilimi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {out}")

    # ============================================================
    # Şekil 4.2 — Yorum Uzunluk Dağılımı
    # ============================================================
    print("\n🎨 Şekil 4.2 üretiliyor...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["word_count"], bins=50, color="#4a90e2", edgecolor="black")
    axes[0].set_title("Yorum Uzunluğu Dağılımı", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Kelime sayısı")
    axes[0].set_ylabel("Yorum sayısı")
    axes[0].axvline(df["word_count"].mean(), color="red", linestyle="--",
                    label=f"Ortalama: {df['word_count'].mean():.1f}")
    axes[0].legend()

    df_plot = df[df["word_count"] < df["word_count"].quantile(0.95)]
    sns.boxplot(data=df_plot, x="sentiment", y="word_count",
                hue="sentiment", palette=colors_map, legend=False, ax=axes[1])
    axes[1].set_title("Sınıf Başına Uzunluk", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Kelime sayısı")

    plt.tight_layout()
    out = os.path.join(rapor_dir, "sekil_4_2_uzunluk_dagilimi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {out}")

    # ============================================================
    # ÖN İŞLEME (Şekil 4.3 ve 4.3b için)
    # ============================================================
    print("\n🧹 NLP ön işleme uygulanıyor (146K yorum, ~1-2 dakika)...")
    df = preprocess_dataframe(df)
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"   ✅ Temizleme sonrası: {len(df):,} yorum")

    # Temizlenmiş veriyi kaydet (faydalı olur)
    clean_path = os.path.join(data_dir, "hepsiburada_clean.csv")
    if not os.path.exists(clean_path):
        df.to_csv(clean_path, index=False, encoding="utf-8-sig")
        print(f"   💾 Kaydedildi: {clean_path}")

    # ============================================================
    # Şekil 4.3 — En Sık 20 Kelime
    # ============================================================
    print("\n🎨 Şekil 4.3 üretiliyor...")
    all_words = " ".join(df["clean_text"].astype(str)).split()
    freq = pd.Series(all_words).value_counts().head(20)

    plt.figure(figsize=(10, 6))
    freq.plot(kind="barh", color="#667eea")
    plt.gca().invert_yaxis()
    plt.title("En Sık 20 Kelime (Ön İşleme Sonrası)", fontsize=14, fontweight="bold")
    plt.xlabel("Frekans")
    plt.tight_layout()
    out = os.path.join(rapor_dir, "sekil_4_3_en_sik_kelimeler.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {out}")

    # ============================================================
    # Şekil 4.3b — Sınıf Bazlı En Sık Kelimeler
    # ============================================================
    print("\n🎨 Şekil 4.3b üretiliyor...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, sentiment in enumerate(["pozitif", "nötr", "negatif"]):
        subset = df[df["sentiment"] == sentiment]["clean_text"]
        words = " ".join(subset.astype(str)).split()
        common = Counter(words).most_common(15)
        if common:
            w, c = zip(*common)
            axes[i].barh(w, c, color=colors_map.get(sentiment, "#888"))
            axes[i].invert_yaxis()
            axes[i].set_title(f"{sentiment.upper()} sınıfında en sık kelimeler",
                              fontsize=12, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(rapor_dir, "sekil_4_3b_sinif_bazli_kelimeler.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {out}")

    # ============================================================
    # Özet
    # ============================================================
    print("\n" + "=" * 60)
    print("  ✅ TÜM GRAFİKLER ÜRETİLDİ")
    print("=" * 60)
    print(f"\nÜretilenler ({rapor_dir}):")
    for f in sorted(os.listdir(rapor_dir)):
        if f.startswith("sekil_4_"):
            size = os.path.getsize(os.path.join(rapor_dir, f)) // 1024
            print(f"  • {f} ({size} KB)")


if __name__ == "__main__":
    main()
