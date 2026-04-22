"""
🧹 Türkçe NLP Ön İşleme Modülü
Sorumlu: Hamza

Bu modül Türkçe e-ticaret ürün yorumlarını derin öğrenme ve
makine öğrenmesi modelleri için hazırlar.

İşlemler:
    1. Küçük harfe çevirme
    2. Noktalama işaretlerini temizleme
    3. Emojileri kaldırma
    4. Türkçe stop-words (anlamsız bağlaçlar) temizleme
    5. Fazla boşlukları kaldırma
    6. Sayıları temizleme (opsiyonel)
    7. URL ve e-posta temizleme

Kullanım:
    from src.preprocessing import preprocess_text, preprocess_dataframe
    clean_text = preprocess_text("Harika bir ürün!!! 😍😍")
"""

import re
import string
import pandas as pd


# ============================================================
#  TÜRKÇE STOP-WORDS LİSTESİ
# ============================================================
# Türkçe'de anlam taşımayan bağlaç, edat ve yardımcı kelimeler.
# Duygu analizi için önemli olan kelimeler (değil, yok, hiç, ama)
# bilinçli olarak LİSTEYE DAHİL EDİLMEMİŞTİR.
TURKISH_STOP_WORDS = {
    # Bağlaçlar ve edatlar
    "ve", "ile", "için", "bir", "bu", "da", "de", "mi", "mu", "mı", "mü",
    "ki", "ya", "hem", "ne", "o", "şu", "her", "en", "daha",
    # Zamirler ve ekler
    "ben", "sen", "biz", "siz", "onlar", "benim", "senin", "onun",
    "bizim", "sizin", "onların",
    # Yardımcı fiiller
    "olan", "olarak", "olup", "oldu", "olmuş", "olan",
    "ise", "iken", "gibi", "kadar", "göre", "rağmen",
    "sonra", "önce", "beri", "diye", "üzere",
    # Sık geçen anlamsız kelimeler
    "var", "çok", "daha", "bile", "sadece", "ancak",
    "ayrıca", "yani", "zaten", "şey", "falan", "filan",
    "etc", "tl", "adet", "ay",
    # Ek edatlar
    "üzerinde", "altında", "yanında", "içinde", "arasında",
    "tarafından", "karşı", "doğru",
}


# ============================================================
#  EMOJİ TEMİZLEME DESENI
# ============================================================
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Yüz ifadeleri
    "\U0001F300-\U0001F5FF"  # Semboller & piktogramlar
    "\U0001F680-\U0001F6FF"  # Ulaşım & harita sembolleri
    "\U0001F1E0-\U0001F1FF"  # Bayraklar
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Çeşitli semboller
    "\U0001f926-\U0001f937"  # Ek yüz ifadeleri
    "\U00010000-\U0010ffff"  # Diğer Unicode sembolleri
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)

# URL deseni
URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
    flags=re.IGNORECASE,
)

# E-posta deseni
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")

# Tekrarlayan karakterler (3+ aynı harf → 2'ye düşür)
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}")


# ============================================================
#  ANA ÖN İŞLEME FONKSİYONU
# ============================================================
def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    remove_numbers: bool = False,
    remove_emojis: bool = True,
    remove_punctuation: bool = True,
    fix_repeated_chars: bool = True,
    min_word_length: int = 2,
) -> str:
    """
    Tek bir Türkçe metni ön işleme adımlarından geçirir.

    Args:
        text: Ham yorum metni.
        remove_stopwords: Türkçe stop-words'leri kaldır.
        remove_numbers: Sayıları kaldır.
        remove_emojis: Emojileri kaldır.
        remove_punctuation: Noktalama işaretlerini kaldır.
        fix_repeated_chars: "çoooook" → "çook" gibi düzeltmeler.
        min_word_length: Minimum kelime uzunluğu (daha kısa olanlar silinir).

    Returns:
        str: Temizlenmiş metin.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Küçük harfe çevir (Türkçe karakterlerle uyumlu)
    text = text.lower()

    # 2. URL'leri kaldır
    text = URL_PATTERN.sub(" ", text)

    # 3. E-postaları kaldır
    text = EMAIL_PATTERN.sub(" ", text)

    # 4. HTML tag'lerini kaldır
    text = re.sub(r"<[^>]+>", " ", text)

    # 5. Emojileri kaldır
    if remove_emojis:
        text = EMOJI_PATTERN.sub(" ", text)

    # 6. Noktalama işaretlerini kaldır
    if remove_punctuation:
        # Türkçe karakterleri koru, noktalama sil
        text = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", " ", text)

    # 7. Sayıları kaldır
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)

    # 8. Tekrarlayan karakterleri düzelt
    if fix_repeated_chars:
        text = REPEATED_CHARS_PATTERN.sub(r"\1\1", text)

    # 9. Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()

    # 10. Stop-words temizliği
    if remove_stopwords:
        words = text.split()
        words = [
            w for w in words
            if w not in TURKISH_STOP_WORDS and len(w) >= min_word_length
        ]
        text = " ".join(words)

    return text


# ============================================================
#  DATAFRAME ÖN İŞLEME
# ============================================================
def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = None,
    new_column: str = "clean_text",
    **kwargs,
) -> pd.DataFrame:
    """
    DataFrame'deki bir metin sütununu toplu olarak ön işlemden geçirir.

    Args:
        df: Pandas DataFrame.
        text_column: Ön işlenecek sütun adı. None ise otomatik tespit eder.
        new_column: Temizlenmiş metnin kaydedileceği yeni sütun adı.
        **kwargs: preprocess_text fonksiyonuna geçirilecek parametreler.

    Returns:
        pd.DataFrame: Yeni 'clean_text' sütunu eklenmiş DataFrame.
    """
    # Metin sütununu otomatik tespit et
    if text_column is None:
        text_col_candidates = [
            "review", "yorum", "comment", "text", "review_text",
            "metin", "description", "content",
        ]
        for col in df.columns:
            if col.lower() in text_col_candidates:
                text_column = col
                break

        if text_column is None:
            str_cols = df.select_dtypes(include=["object"]).columns
            if len(str_cols) > 0:
                avg_len = {c: df[c].astype(str).str.len().mean() for c in str_cols}
                text_column = max(avg_len, key=avg_len.get)

    if text_column is None:
        raise ValueError("Metin sütunu bulunamadı!")

    print(f"🔄 Ön işleme başlıyor — Sütun: '{text_column}'")
    print(f"   Toplam satır: {len(df):,}")

    # Ön işleme uygula
    df[new_column] = df[text_column].astype(str).apply(
        lambda x: preprocess_text(x, **kwargs)
    )

    # Boş olanları kaldır
    empty_count = (df[new_column].str.strip() == "").sum()
    if empty_count > 0:
        print(f"   ⚠️  Ön işleme sonrası boş kalan satır: {empty_count:,}")
        df = df[df[new_column].str.strip() != ""].reset_index(drop=True)

    print(f"   ✅ Ön işleme tamamlandı! Kalan satır: {len(df):,}")

    # Örnek çıktı göster
    print("\n   📋 Örnek ön işleme sonuçları:")
    for i in range(min(3, len(df))):
        original = str(df[text_column].iloc[i])[:80]
        cleaned = str(df[new_column].iloc[i])[:80]
        print(f"      Orijinal : {original}...")
        print(f"      Temiz    : {cleaned}...")
        print()

    return df


# ============================================================
#  YARDIMCI FONKSİYONLAR
# ============================================================
def get_word_freq(df: pd.DataFrame, column: str = "clean_text", top_n: int = 20):
    """
    Temizlenmiş metinlerdeki en sık geçen kelimeleri döndürür.

    Args:
        df: DataFrame.
        column: Metin sütunu.
        top_n: Kaç kelime gösterilsin.

    Returns:
        pd.Series: Kelime frekansları.
    """
    all_words = " ".join(df[column].astype(str)).split()
    freq = pd.Series(all_words).value_counts().head(top_n)

    print(f"\n📊 En sık {top_n} kelime:")
    for word, count in freq.items():
        bar = "█" * min(int(count / freq.max() * 30), 30)
        print(f"   {word:20s} {count:6,}  {bar}")

    return freq


def get_text_stats(df: pd.DataFrame, column: str = "clean_text"):
    """
    Temizlenmiş metinlerin istatistiklerini yazdırır.

    Args:
        df: DataFrame.
        column: Metin sütunu.
    """
    lengths = df[column].astype(str).str.split().str.len()

    print("\n📏 Metin İstatistikleri:")
    print(f"   Ortalama kelime sayısı : {lengths.mean():.1f}")
    print(f"   Medyan kelime sayısı   : {lengths.median():.1f}")
    print(f"   Min kelime sayısı      : {lengths.min()}")
    print(f"   Max kelime sayısı      : {lengths.max()}")
    print(f"   Standart sapma         : {lengths.std():.1f}")


# ============================================================
#  TEST / DEMO
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  🧹 Türkçe NLP Ön İşleme — Demo")
    print("=" * 60)

    test_texts = [
        "Ürün çoooook güzel!!! 😍😍😍 Herkese tavsiye ederim <3 <3",
        "Kargo 5 günde geldi, PAKET HASARLI idi!!! http://example.com/photo.jpg",
        "Bu ürünü  aldım ve   çok memnun kaldım...   Teşekkürler 🙏🙏",
        "idare eder bir ürün, fiyatına göre iyi, 3 yıldız verdim.",
        "Kesinlikle almayın!!! Paramı çöpee attım 😡😡 berbat kalite",
        "",
        None,
    ]

    print("\n--- Tekli Metin Ön İşleme ---")
    for text in test_texts:
        cleaned = preprocess_text(str(text) if text else "")
        print(f"  Girdi  : {text}")
        print(f"  Çıktı  : '{cleaned}'")
        print()

    print("✅ Demo tamamlandı!")
