"""
🤖 Transformer Modelleri Modülü (BERTurk + mT5)
Sorumlu: Hamza

Bu modül:
    - BERTurk (dbmdz/bert-base-turkish-cased) ile Türkçe duygu analizi
    - mT5-small ile çoklu metin özetleme
işlerini gerçekleştirir.

Eğitim için GPU önerilir (Google Colab T4 ücretsizdir).

Kullanım:
    # Duygu analizi eğit:
    python -m src.transformer_model train_sentiment

    # Çıkarım (inference):
    python -m src.transformer_model predict "Bu ürün harika!"

    # Çoklu özetleme:
    python -m src.transformer_model summarize
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Proje modüllerini import et
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ============================================================
#  AYARLAR
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

SENTIMENT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
SUMMARIZATION_MODEL_NAME = "google/mt5-small"

SENTIMENT_OUTPUT_DIR = os.path.join(MODELS_DIR, "berturk-sentiment")
SUMMARIZATION_OUTPUT_DIR = os.path.join(MODELS_DIR, "mt5-summarization")

# Eğitim hiperparametreleri
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
RANDOM_STATE = 42

# Etiket eşleştirmesi
LABEL2ID = {"negatif": 0, "nötr": 1, "pozitif": 2}
ID2LABEL = {0: "negatif", 1: "nötr", 2: "pozitif"}


# ============================================================
#  1) BERTURK DUYGU ANALİZİ — EĞİTİM
# ============================================================
def train_sentiment_model(
    csv_path: str = None,
    sample_size: int = None,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> str:
    """
    BERTurk modelini Hepsiburada veri setinde ince ayar yapar (fine-tune).

    Args:
        csv_path: Temizlenmiş CSV yolu. None ise data/ altında arar.
        sample_size: Küçük testler için örnek sayısını sınırla (örn. 10000).
                     None ise tüm veri kullanılır.
        num_epochs: Eğitim epoch sayısı.
        batch_size: Mini-batch boyutu.

    Returns:
        str: Eğitilmiş modelin kaydedildiği klasör yolu.
    """
    print("=" * 60)
    print("  🤖 BERTurk Duygu Analizi — Fine-Tuning")
    print("=" * 60)

    # --- Lazy import (transformers ağır) ---
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    # --- GPU kontrolü ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   🖥️  Cihaz: {device}")
    if device == "cpu":
        print("   ⚠️  GPU bulunamadı. Eğitim yavaş olacak.")
        print("       Google Colab → Çalışma Zamanı → Çalışma Zamanı Türü → T4 GPU")

    # --- Veri yükle ---
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "hepsiburada_clean.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Temizlenmiş veri bulunamadı: {csv_path}\n"
                "Önce 'python -m src.data_loader' çalıştırın."
            )

    print(f"\n📖 Veri okunuyor: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Yorum sütununu otomatik tespit et
    text_col = None
    for c in ["clean_text", "review", "yorum", "comment", "text"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError("Yorum sütunu bulunamadı!")

    if "sentiment" not in df.columns:
        raise ValueError(
            "'sentiment' sütunu bulunamadı! "
            "Önce data_loader.add_sentiment_labels() çalıştırın."
        )

    # Boş yorumları çıkar
    df = df[df[text_col].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    print(f"   Toplam örnek: {len(df):,}")

    # Örnekleme (hızlı test için)
    if sample_size and sample_size < len(df):
        # Sınıf dengesini koruyarak örnekle
        df = df.groupby("sentiment", group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), sample_size // 3),
                random_state=RANDOM_STATE,
            )
        ).reset_index(drop=True)
        print(f"   🔬 Örnekleme uygulandı: {len(df):,} örnek")

    # Etiketleri sayıya çevir
    df["label"] = df["sentiment"].map(LABEL2ID)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # --- Train/Validation/Test bölme ---
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df["label"]
    )

    print(f"\n📊 Veri bölme:")
    print(f"   Eğitim     : {len(train_df):,}")
    print(f"   Doğrulama  : {len(val_df):,}")
    print(f"   Test       : {len(test_df):,}")

    # --- Tokenizer & Model ---
    print(f"\n📥 Model indiriliyor: {SENTIMENT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # --- Dataset oluştur ---
    def tokenize_function(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,  # Dynamic padding kullanacağız
        )

    train_ds = Dataset.from_pandas(train_df[[text_col, "label"]])
    val_ds = Dataset.from_pandas(val_df[[text_col, "label"]])
    test_ds = Dataset.from_pandas(test_df[[text_col, "label"]])

    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Metrik fonksiyonu ---
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=SENTIMENT_OUTPUT_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=os.path.join(SENTIMENT_OUTPUT_DIR, "logs"),
        logging_steps=50,
        save_total_limit=2,
        report_to="none",  # Wandb vs. devre dışı
        fp16=(device == "cuda"),  # GPU varsa mixed-precision
        seed=RANDOM_STATE,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Eğit ---
    print("\n🏋️ Eğitim başlıyor...")
    trainer.train()

    # --- Test seti değerlendirmesi ---
    print("\n📊 Test seti değerlendirmesi...")
    test_results = trainer.evaluate(test_ds)
    print(f"   Test Accuracy : {test_results['eval_accuracy']:.4f}")
    print(f"   Test F1-Macro : {test_results['eval_f1_macro']:.4f}")

    # --- Kaydet ---
    trainer.save_model(SENTIMENT_OUTPUT_DIR)
    tokenizer.save_pretrained(SENTIMENT_OUTPUT_DIR)

    # Sonuçları JSON olarak kaydet
    import json
    with open(os.path.join(SENTIMENT_OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump({k: float(v) for k, v in test_results.items() if isinstance(v, (int, float))}, f, indent=2)

    print(f"\n✅ Model kaydedildi: {SENTIMENT_OUTPUT_DIR}")
    return SENTIMENT_OUTPUT_DIR


# ============================================================
#  2) BERTURK DUYGU ANALİZİ — ÇIKARIM
# ============================================================
def load_sentiment_model(model_path: str = None):
    """
    Eğitilmiş BERTurk duygu analizi modelini pipeline olarak yükler.

    Args:
        model_path: Modelin yolu. None ise SENTIMENT_OUTPUT_DIR kullanılır.

    Returns:
        transformers.Pipeline: Hazır çıkarım pipeline'ı.
    """
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    if model_path is None:
        model_path = SENTIMENT_OUTPUT_DIR

    # Eğer fine-tuned model yoksa, savasy/bert-base-turkish-sentiment-cased kullan
    if not os.path.exists(model_path):
        print(f"⚠️  Fine-tuned model bulunamadı: {model_path}")
        print("   Hazır 'savasy/bert-base-turkish-sentiment-cased' modeli kullanılıyor...")
        model_path = "savasy/bert-base-turkish-sentiment-cased"

    print(f"📥 Duygu modeli yükleniyor: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if _has_gpu() else -1,
        truncation=True,
        max_length=MAX_LENGTH,
    )


def predict_sentiment_transformer(text: str, model_pipeline=None) -> dict:
    """
    Tek bir metin için transformer ile duygu tahmini yapar.
    """
    if model_pipeline is None:
        model_pipeline = load_sentiment_model()

    result = model_pipeline(text)[0]

    label_map = {
        "POSITIVE": "pozitif", "NEGATIVE": "negatif", "NEUTRAL": "nötr",
        "positive": "pozitif", "negative": "negatif", "neutral": "nötr",
        "LABEL_0": "negatif", "LABEL_1": "nötr", "LABEL_2": "pozitif",
        "pozitif": "pozitif", "negatif": "negatif", "nötr": "nötr",
    }
    sentiment = label_map.get(result["label"], result["label"].lower())

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": result["score"],
        "emoji": {"pozitif": "😊", "nötr": "😐", "negatif": "😠"}.get(sentiment, ""),
    }


# ============================================================
#  3) mT5 ÇOKLU METİN ÖZETLEME
# ============================================================
def load_summarization_model(model_name: str = None):
    """
    mT5 özetleme pipeline'ını yükler.

    Türkçe-spesifik fine-tune yapılmadığı için zero-shot kullanılır.

    Returns:
        transformers.Pipeline: summarization pipeline.
    """
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    if model_name is None:
        # Eğitilmiş varsa kullan, yoksa hazır mT5-small
        if os.path.exists(SUMMARIZATION_OUTPUT_DIR):
            model_name = SUMMARIZATION_OUTPUT_DIR
        else:
            # Türkçe için fine-tune edilmiş alternatif
            model_name = "ozcangundes/mt5-small-turkish-summarization"

    print(f"📥 Özetleme modeli yükleniyor: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        return pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if _has_gpu() else -1,
        )
    except Exception as e:
        print(f"   ⚠️  Model yüklenemedi: {e}")
        print(f"   Yedek: google/mt5-small kullanılıyor")
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
        return pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if _has_gpu() else -1,
        )


def summarize_multiple_reviews(
    reviews: list,
    model_pipeline=None,
    max_length: int = 150,
    min_length: int = 30,
) -> str:
    """
    Birden fazla yorumu birleştirip tek bir özet üretir.

    Args:
        reviews: Yorum metni listesi.
        model_pipeline: Yüklenmiş özetleme pipeline'ı.
        max_length: Üretilen özetin maks. token uzunluğu.
        min_length: Üretilen özetin min. token uzunluğu.

    Returns:
        str: Üretilen özet.
    """
    if model_pipeline is None:
        model_pipeline = load_summarization_model()

    # Yorumları birleştir (özetleme için)
    combined = " ".join(reviews)

    # mT5'in input limiti 512 token civarında, kelime bazında ~400'lü
    # Truncation transformers tarafından otomatik yapılacak
    try:
        summary = model_pipeline(
            combined,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            num_beams=4,
            truncation=True,
        )
        return summary[0]["summary_text"]
    except Exception as e:
        print(f"⚠️  Özetleme hatası: {e}")
        return _fallback_extractive_summary(reviews)


def _fallback_extractive_summary(reviews: list, top_n: int = 3) -> str:
    """
    Transformer çalışmazsa basit extractive özetleme.
    """
    all_sentences = []
    for r in reviews:
        sentences = [s.strip() for s in r.replace("!", ".").replace("?", ".").split(".")]
        all_sentences.extend([s for s in sentences if len(s.split()) >= 3])

    if not all_sentences:
        return "Yeterli içerik bulunamadı."

    word_freq = {}
    for s in all_sentences:
        for w in s.lower().split():
            word_freq[w] = word_freq.get(w, 0) + 1

    scored = sorted(
        [(sum(word_freq.get(w, 0) for w in s.lower().split()) / max(len(s.split()), 1), s)
         for s in all_sentences],
        reverse=True,
    )

    selected = [s for _, s in scored[:top_n]]
    return ". ".join(selected) + "."


# ============================================================
#  4) YARDIMCI FONKSİYONLAR
# ============================================================
def _has_gpu() -> bool:
    """Sistemde CUDA destekli GPU var mı kontrol eder."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================
#  CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="BERTurk + mT5 Transformer Modülü")
    parser.add_argument(
        "command",
        choices=["train_sentiment", "predict", "summarize", "demo"],
        help="Çalıştırılacak komut.",
    )
    parser.add_argument("--text", type=str, default=None, help="Tahmin için metin.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Eğitim için örnek sayısı (test amaçlı).")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Eğitim epoch sayısı.")

    args = parser.parse_args()

    if args.command == "train_sentiment":
        train_sentiment_model(sample_size=args.sample, num_epochs=args.epochs)

    elif args.command == "predict":
        if not args.text:
            print("Lütfen --text parametresi ile bir cümle girin.")
            return
        pipeline_obj = load_sentiment_model()
        result = predict_sentiment_transformer(args.text, pipeline_obj)
        print(f"\n{result['emoji']} {args.text}")
        print(f"   → {result['sentiment']} (skor: {result['confidence']:.3f})")

    elif args.command == "summarize":
        # Demo özetleme
        demo_reviews = [
            "Ürün gerçekten kaliteli, çok memnun kaldım. Kargolama hızlıydı.",
            "Fiyatına göre güzel ama beklediğim kadar değil. Renk biraz farklı çıktı.",
            "Mükemmel bir ürün! Tavsiye ederim. Hediye olarak aldım çok beğenildi.",
            "Kargo geç geldi ve ürün hasarlıydı, iade etmek zorunda kaldım.",
        ]
        pipeline_obj = load_summarization_model()
        summary = summarize_multiple_reviews(demo_reviews, pipeline_obj)
        print(f"\n📝 Üretilen Özet:\n   {summary}")

    elif args.command == "demo":
        # Hem duygu hem özet demo
        print("=" * 60)
        print("  🎯 Transformer Modüller — Demo")
        print("=" * 60)

        reviews = [
            "Bu ürün gerçekten harika, çok memnun kaldım!",
            "Kargo geç geldi, ürün hasarlıydı, kötü deneyim.",
            "İdare eder, fiyatına göre fena değil.",
        ]

        # Duygu
        print("\n--- DUYGU ANALİZİ ---")
        s_pipe = load_sentiment_model()
        for r in reviews:
            res = predict_sentiment_transformer(r, s_pipe)
            print(f"   {res['emoji']} {r}")
            print(f"      → {res['sentiment']} ({res['confidence']:.2f})")

        # Özetleme
        print("\n--- ÇOKLU ÖZETLEME ---")
        sum_pipe = load_summarization_model()
        summary = summarize_multiple_reviews(reviews, sum_pipe)
        print(f"   📝 Özet: {summary}")


if __name__ == "__main__":
    main()
