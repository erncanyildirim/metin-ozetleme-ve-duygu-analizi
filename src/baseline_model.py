"""
🤖 Baseline Duygu Sınıflandırma Modeli (SVM & Naive Bayes)
Sorumlu: Eren Can

Bu modül, TF-IDF vektörizasyonu ile SVM ve Naive Bayes
modellerini eğitip karşılaştırır.

Kullanım:
    python -m src.baseline_model
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Proje modüllerini import et
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import preprocess_dataframe


# ============================================================
#  AYARLAR
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ============================================================
#  1) Veri Yükleme ve Hazırlama
# ============================================================
def load_and_prepare_data(csv_path: str = None) -> tuple:
    """
    Veri setini yükler, ön işlemden geçirir ve eğitim/test olarak böler.

    Args:
        csv_path: CSV dosya yolu. None ise data/ altında arar.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_names)
    """
    # CSV dosyasını bul
    if csv_path is None:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(
                "data/ klasöründe CSV dosyası bulunamadı. "
                "Önce 'python -m src.data_loader' çalıştırın."
            )
        # Temizlenmiş versiyonu tercih et
        if "hepsiburada_clean.csv" in csv_files:
            csv_path = os.path.join(DATA_DIR, "hepsiburada_clean.csv")
        else:
            csv_path = os.path.join(DATA_DIR, csv_files[0])

    print(f"📖 Veri okunuyor: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"   Satır: {len(df):,} | Sütunlar: {list(df.columns)}")

    # --- Ön işleme ---
    df = preprocess_dataframe(df)

    # --- Duygu etiketlerini kontrol et ---
    if "sentiment" not in df.columns:
        # Puan sütununu bul ve etiketle
        rating_col = None
        for col in df.columns:
            if col.lower() in ["rating", "puan", "star", "stars", "score"]:
                rating_col = col
                break
        if rating_col is None:
            num_cols = df.select_dtypes(include=["number"]).columns
            for c in num_cols:
                if df[c].dropna().between(1, 5).all():
                    rating_col = c
                    break

        if rating_col:
            df["sentiment"] = df[rating_col].apply(
                lambda r: "negatif" if r <= 2 else ("nötr" if r == 3 else "pozitif")
            )
        else:
            raise ValueError("Puan sütunu bulunamadı, duygu etiketi oluşturulamıyor!")

    # --- Boş clean_text satırlarını kaldır ---
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)

    # --- Eğitim / Test bölme ---
    X = df["clean_text"]
    y = df["sentiment"]
    label_names = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\n📊 Veri Bölme:")
    print(f"   Eğitim seti : {len(X_train):,} örnek")
    print(f"   Test seti   : {len(X_test):,} örnek")
    print(f"   Etiketler   : {label_names}")

    return X_train, X_test, y_train, y_test, label_names


# ============================================================
#  2) Model Pipeline Oluşturma
# ============================================================
def create_svm_pipeline() -> Pipeline:
    """TF-IDF + Linear SVM pipeline'ı oluşturur."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),       # Unigram + Bigram
            sublinear_tf=True,        # Logaritmik TF
            min_df=2,                 # En az 2 dokümanda geçen kelimeler
            max_df=0.95,              # %95'ten fazla dokümanda geçenleri at
        )),
        ("clf", LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=RANDOM_STATE,
            class_weight="balanced",  # Dengesiz sınıflar için
        )),
    ])


def create_nb_pipeline() -> Pipeline:
    """TF-IDF + Multinomial Naive Bayes pipeline'ı oluşturur."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


# ============================================================
#  3) Model Eğitimi
# ============================================================
def train_model(pipeline: Pipeline, X_train, y_train, model_name: str) -> Pipeline:
    """
    Modeli eğitir ve çapraz doğrulama skoru hesaplar.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Eğitim metinleri.
        y_train: Eğitim etiketleri.
        model_name: Modelin adı (loglama için).

    Returns:
        Pipeline: Eğitilmiş pipeline.
    """
    print(f"\n{'='*50}")
    print(f"  🏋️ {model_name} Eğitimi Başlıyor...")
    print(f"{'='*50}")

    # 5-Fold Cross Validation
    print(f"   📐 5-Fold Cross Validation yapılıyor...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"   CV F1-Macro Skorları: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"   CV F1-Macro Ortalama: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Tam eğitim
    pipeline.fit(X_train, y_train)
    print(f"   ✅ {model_name} eğitimi tamamlandı!")

    return pipeline


# ============================================================
#  4) Model Değerlendirme
# ============================================================
def evaluate_model(pipeline: Pipeline, X_test, y_test, label_names: list, model_name: str) -> dict:
    """
    Modeli test seti üzerinde değerlendirir.

    Returns:
        dict: Metrik sonuçları.
    """
    print(f"\n{'='*50}")
    print(f"  📊 {model_name} — Test Seti Değerlendirmesi")
    print(f"{'='*50}")

    y_pred = pipeline.predict(X_test)

    # Classification Report
    report = classification_report(
        y_test, y_pred,
        target_names=label_names,
        output_dict=True,
    )
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=label_names)
    print("   Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(f"   {cm_df.to_string()}")

    results = {
        "model_name": model_name,
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }

    return results


# ============================================================
#  5) Model Kaydetme
# ============================================================
def save_model(pipeline: Pipeline, model_name: str) -> str:
    """Eğitilmiş modeli models/ klasörüne kaydeder."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filename = f"{model_name.lower().replace(' ', '_')}.joblib"
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(pipeline, filepath)
    print(f"   💾 Model kaydedildi: {filepath}")
    return filepath


def load_model(model_name: str) -> Pipeline:
    """Kaydedilmiş modeli yükler."""
    filename = f"{model_name.lower().replace(' ', '_')}.joblib"
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model bulunamadı: {filepath}")
    return joblib.load(filepath)


# ============================================================
#  6) Tekli Tahmin
# ============================================================
def predict_sentiment(text: str, pipeline: Pipeline) -> dict:
    """
    Tek bir metin için duygu tahmini yapar.

    Args:
        text: Ham yorum metni.
        pipeline: Eğitilmiş model pipeline'ı.

    Returns:
        dict: {"text": ..., "sentiment": ..., "clean_text": ...}
    """
    from src.preprocessing import preprocess_text

    clean = preprocess_text(text)
    prediction = pipeline.predict([clean])[0]

    emoji_map = {"pozitif": "😊", "nötr": "😐", "negatif": "😠"}

    return {
        "text": text,
        "clean_text": clean,
        "sentiment": prediction,
        "emoji": emoji_map.get(prediction, ""),
    }


# ============================================================
#  ANA ÇALIŞTIRMA
# ============================================================
def main():
    """Tam baseline eğitim ve karşılaştırma pipeline'ını çalıştırır."""
    print("=" * 60)
    print("  🚀 Baseline Duygu Analizi — SVM vs Naive Bayes")
    print("=" * 60)

    # 1. Veri hazırla
    X_train, X_test, y_train, y_test, label_names = load_and_prepare_data()

    # 2. Modelleri oluştur
    models = {
        "SVM (LinearSVC)": create_svm_pipeline(),
        "Naive Bayes": create_nb_pipeline(),
    }

    # 3. Eğit ve değerlendir
    all_results = {}
    for name, pipeline in models.items():
        trained = train_model(pipeline, X_train, y_train, name)
        results = evaluate_model(trained, X_test, y_test, label_names, name)
        save_model(trained, name)
        all_results[name] = results

    # 4. Karşılaştırma tablosu
    print("\n" + "=" * 60)
    print("  🏆 MODEL KARŞILAŞTIRMA TABLOSU")
    print("=" * 60)
    comparison = pd.DataFrame({
        name: {
            "Accuracy": f"{r['accuracy']:.4f}",
            "F1 (Macro)": f"{r['f1_macro']:.4f}",
            "F1 (Weighted)": f"{r['f1_weighted']:.4f}",
        }
        for name, r in all_results.items()
    }).T
    print(comparison.to_string())

    # 5. En iyi modeli belirle
    best_model_name = max(all_results, key=lambda k: all_results[k]["f1_macro"])
    print(f"\n   🥇 En iyi model: {best_model_name}")
    print(f"      F1-Macro: {all_results[best_model_name]['f1_macro']:.4f}")

    # 6. Demo tahminler
    print("\n" + "=" * 60)
    print("  🔮 Demo Tahminler")
    print("=" * 60)
    best_pipeline = models[best_model_name]
    demo_texts = [
        "Harika bir ürün, çok memnun kaldım tavsiye ederim",
        "Kargo çok geç geldi, ürün kötü kalitede",
        "Fiyatına göre idare eder, beklentimi karşıladı",
    ]
    for text in demo_texts:
        result = predict_sentiment(text, best_pipeline)
        print(f"   {result['emoji']} '{text}'")
        print(f"      → {result['sentiment']}")
        print()

    return all_results


if __name__ == "__main__":
    main()
