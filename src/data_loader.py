"""
📦 Veri Seti İndirme ve Yükleme Modülü
Sorumlu: Eren Can

Bu modül Kaggle'dan Hepsiburada ürün yorumları veri setini indirir,
Pandas ile okur ve eksik verileri temizler.

Kullanım:
    python -m src.data_loader
"""

import os
import subprocess
import sys
import pandas as pd


# ============================================================
#  AYARLAR
# ============================================================
KAGGLE_DATASET = "yusufkesmenn/hepsiburada-product-reviews-and-ratings-dataset"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_FILE_NAME = "hepsiburada.csv"   # İndirilen dosyanın adı (Kaggle'dan gelen)


# ============================================================
#  1) Kaggle'dan Veri Setini İndirme
# ============================================================
def download_dataset() -> str:
    """
    Kaggle API aracılığıyla veri setini data/ klasörüne indirir.

    Ön koşul:
        - pip install kaggle
        - ~/.kaggle/kaggle.json dosyasının mevcut olması
          (Kaggle > Account > Create New API Token)

    Returns:
        str: İndirilen CSV dosyasının tam yolu.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"📥 Kaggle'dan veri seti indiriliyor: {KAGGLE_DATASET}")
    print(f"   Hedef klasör: {DATA_DIR}")

    try:
        subprocess.run(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", DATA_DIR,
                "--unzip",
            ],
            check=True,
        )
        print("✅ İndirme tamamlandı!")
    except FileNotFoundError:
        print("❌ Kaggle CLI bulunamadı. Lütfen şu komutu çalıştırın:")
        print("   pip install kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ İndirme hatası: {e}")
        print("   ~/.kaggle/kaggle.json dosyasını kontrol edin.")
        sys.exit(1)

    # İndirilen CSV dosyasını bul
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("❌ data/ klasöründe CSV dosyası bulunamadı!")
        sys.exit(1)

    csv_path = os.path.join(DATA_DIR, csv_files[0])
    print(f"📄 Bulunan dosya: {csv_path}")
    return csv_path


# ============================================================
#  2) Veri Setini Pandas ile Okuma
# ============================================================
def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    CSV dosyasını Pandas DataFrame olarak okur.

    Args:
        csv_path: CSV dosyasının yolu. None ise data/ altındaki
                  ilk CSV dosyasını arar.

    Returns:
        pd.DataFrame: Ham veri seti.
    """
    if csv_path is None:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(
                f"data/ klasöründe CSV dosyası bulunamadı. "
                f"Önce download_dataset() çalıştırın."
            )
        csv_path = os.path.join(DATA_DIR, csv_files[0])

    print(f"📖 Veri seti okunuyor: {csv_path}")

    # Farklı encoding'ler dene
    for encoding in ["utf-8", "latin-1", "iso-8859-9"]:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"   Encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV dosyası okunamadı — desteklenen encoding bulunamadı.")

    print(f"   Satır sayısı : {len(df):,}")
    print(f"   Sütun sayısı : {len(df.columns)}")
    print(f"   Sütunlar     : {list(df.columns)}")
    return df


# ============================================================
#  3) Eksik Verileri Temizleme
# ============================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksik ve hatalı verileri temizler.

    İşlemler:
        1. Yorum metni boş olan satırları siler
        2. Puan (rating) sütunundaki NaN değerleri siler
        3. Yinelenen (duplicate) yorumları kaldırır
        4. Indeks'i sıfırlar

    Args:
        df: Ham DataFrame.

    Returns:
        pd.DataFrame: Temizlenmiş DataFrame.
    """
    print("\n🧹 Veri temizleme başlıyor...")
    initial_rows = len(df)

    # ---- Sütun isimlerini standartlaştır ----
    df.columns = df.columns.str.strip().str.lower()
    print(f"   Standartlaştırılmış sütunlar: {list(df.columns)}")

    # ---- Eksik değer raporu ----
    missing = df.isnull().sum()
    if missing.any():
        print("\n   📊 Eksik değer raporu:")
        for col, count in missing[missing > 0].items():
            pct = count / len(df) * 100
            print(f"      {col}: {count:,} ({pct:.1f}%)")
    else:
        print("   ✅ Eksik değer bulunamadı.")

    # ---- Olası yorum sütunu isimlerini belirle ----
    text_col_candidates = ["review", "yorum", "comment", "text", "review_text",
                           "metin", "description", "content"]
    rating_col_candidates = ["rating", "puan", "star", "stars", "score",
                             "rate", "yıldız"]

    text_col = None
    rating_col = None

    for col in df.columns:
        if col in text_col_candidates:
            text_col = col
            break
    for col in df.columns:
        if col in rating_col_candidates:
            rating_col = col
            break

    # Bulunamazsa en uzun string sütununu yorum olarak al
    if text_col is None:
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) > 0:
            avg_len = {c: df[c].astype(str).str.len().mean() for c in str_cols}
            text_col = max(avg_len, key=avg_len.get)
            print(f"   ⚠️  Yorum sütunu otomatik tespit edildi: '{text_col}'")

    if rating_col is None:
        num_cols = df.select_dtypes(include=["number"]).columns
        for c in num_cols:
            if df[c].dropna().between(1, 5).all():
                rating_col = c
                print(f"   ⚠️  Puan sütunu otomatik tespit edildi: '{rating_col}'")
                break

    print(f"\n   📌 Yorum sütunu : {text_col}")
    print(f"   📌 Puan sütunu  : {rating_col}")

    # ---- Temizlik ----
    if text_col:
        df = df.dropna(subset=[text_col])
        df = df[df[text_col].astype(str).str.strip().str.len() > 0]

    if rating_col:
        df = df.dropna(subset=[rating_col])

    # Duplicate'ları kaldır
    df = df.drop_duplicates()

    # Indeks sıfırla
    df = df.reset_index(drop=True)

    removed = initial_rows - len(df)
    print(f"\n   🗑️  Silinen satır sayısı: {removed:,}")
    print(f"   ✅ Kalan satır sayısı  : {len(df):,}")

    return df


# ============================================================
#  4) Duygu Etiketi Oluşturma (Puana göre)
# ============================================================
def add_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Puan (rating) değerine göre duygu etiketi ekler.

    Etiketleme kuralı:
        - 1-2 yıldız  →  'negatif'
        - 3 yıldız     →  'nötr'
        - 4-5 yıldız   →  'pozitif'

    Args:
        df: Temizlenmiş DataFrame.

    Returns:
        pd.DataFrame: 'sentiment' sütunu eklenmiş DataFrame.
    """
    print("\n🏷️  Duygu etiketleri oluşturuluyor...")

    # Puan sütununu bul
    rating_col_candidates = ["rating", "puan", "star", "stars", "score",
                             "rate", "yıldız"]
    rating_col = None
    for col in df.columns:
        if col in rating_col_candidates:
            rating_col = col
            break

    if rating_col is None:
        num_cols = df.select_dtypes(include=["number"]).columns
        for c in num_cols:
            if df[c].dropna().between(1, 5).all():
                rating_col = c
                break

    if rating_col is None:
        print("   ⚠️  Puan sütunu bulunamadı, etiketleme atlanıyor.")
        return df

    def map_sentiment(rating):
        if rating <= 2:
            return "negatif"
        elif rating == 3:
            return "nötr"
        else:
            return "pozitif"

    df["sentiment"] = df[rating_col].apply(map_sentiment)

    # İstatistikler
    print("   📊 Duygu dağılımı:")
    dist = df["sentiment"].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        emoji = {"pozitif": "😊", "nötr": "😐", "negatif": "😠"}.get(label, "")
        print(f"      {emoji} {label}: {count:,} ({pct:.1f}%)")

    return df


# ============================================================
#  ANA ÇALIŞTIRMA
# ============================================================
def main():
    """Tam veri pipeline'ını çalıştırır."""
    print("=" * 60)
    print("  🚀 Hepsiburada Veri Seti — İndirme & Temizleme Pipeline")
    print("=" * 60)

    # 1. İndir
    csv_path = download_dataset()

    # 2. Oku
    df = load_data(csv_path)

    # 3. Temizle
    df = clean_data(df)

    # 4. Duygu etiketi ekle
    df = add_sentiment_labels(df)

    # 5. İlk 5 satırı göster
    print("\n📋 İlk 5 satır:")
    print(df.head().to_string())

    # 6. Temizlenmiş veriyi kaydet
    clean_csv_path = os.path.join(DATA_DIR, "hepsiburada_clean.csv")
    df.to_csv(clean_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Temizlenmiş veri kaydedildi: {clean_csv_path}")

    return df


if __name__ == "__main__":
    main()
