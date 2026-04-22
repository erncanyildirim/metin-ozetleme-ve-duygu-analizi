"""
📏 Ortak Değerlendirme Metrikleri Modülü
Sorumlu: Eren Can

Bu modül hem sınıflandırma hem özetleme görevleri için
ortak değerlendirme fonksiyonları sağlar.

Metrikler:
    - Sınıflandırma: Accuracy, F1-Score (macro/weighted), Precision, Recall
    - Özetleme: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore

Kullanım:
    from src.evaluation import evaluate_classification, evaluate_summarization
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
#  1) SINIFLANDIRMA DEĞERLENDİRMESİ
# ============================================================
def evaluate_classification(
    y_true,
    y_pred,
    label_names: list = None,
    model_name: str = "Model",
    print_report: bool = True,
) -> dict:
    """
    Sınıflandırma modelinin performans metriklerini hesaplar.

    Args:
        y_true: Gerçek etiketler.
        y_pred: Tahmin edilen etiketler.
        label_names: Etiket isimleri listesi.
        model_name: Modelin adı (raporlama için).
        print_report: Raporu ekrana yazdır.

    Returns:
        dict: Tüm metrikler.
    """
    results = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if print_report:
        print(f"\n{'='*55}")
        print(f"  📊 {model_name} — Sınıflandırma Raporu")
        print(f"{'='*55}")
        print(f"  Accuracy       : {results['accuracy']:.4f}")
        print(f"  F1 (Macro)     : {results['f1_macro']:.4f}")
        print(f"  F1 (Weighted)  : {results['f1_weighted']:.4f}")
        print(f"  Precision      : {results['precision_macro']:.4f}")
        print(f"  Recall         : {results['recall_macro']:.4f}")
        print()

        report_str = classification_report(
            y_true, y_pred,
            target_names=label_names,
            zero_division=0,
        )
        print(report_str)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=label_names)
        if label_names:
            cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
            print("  Confusion Matrix:")
            print(f"  {cm_df.to_string()}")

        results["confusion_matrix"] = cm

    return results


# ============================================================
#  2) ÖZETLEME DEĞERLENDİRMESİ — ROUGE
# ============================================================
def evaluate_summarization_rouge(
    references: list,
    predictions: list,
    model_name: str = "Model",
    print_report: bool = True,
) -> dict:
    """
    Özetleme modelinin ROUGE skorlarını hesaplar.

    Gerekli kütüphane: pip install rouge-score

    Args:
        references: Referans özetler listesi.
        predictions: Üretilen özetler listesi.
        model_name: Modelin adı.
        print_report: Raporu yazdır.

    Returns:
        dict: ROUGE metrikleri.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("⚠️  rouge-score kütüphanesi bulunamadı.")
        print("   Kurulum: pip install rouge-score")
        return {}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,  # Türkçe için stemmer kapalı
    )

    all_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        for key in all_scores:
            all_scores[key].append(scores[key].fmeasure)

    results = {
        "model_name": model_name,
        "rouge1": np.mean(all_scores["rouge1"]),
        "rouge2": np.mean(all_scores["rouge2"]),
        "rougeL": np.mean(all_scores["rougeL"]),
    }

    if print_report:
        print(f"\n{'='*55}")
        print(f"  📏 {model_name} — ROUGE Skorları")
        print(f"{'='*55}")
        print(f"  ROUGE-1 : {results['rouge1']:.4f}")
        print(f"  ROUGE-2 : {results['rouge2']:.4f}")
        print(f"  ROUGE-L : {results['rougeL']:.4f}")

    return results


# ============================================================
#  3) ÖZETLEME DEĞERLENDİRMESİ — BERTScore
# ============================================================
def evaluate_summarization_bertscore(
    references: list,
    predictions: list,
    model_name: str = "Model",
    lang: str = "tr",
    print_report: bool = True,
) -> dict:
    """
    Özetleme modelinin BERTScore değerlerini hesaplar.

    Gerekli kütüphane: pip install bert-score

    Args:
        references: Referans özetler.
        predictions: Üretilen özetler.
        model_name: Model adı.
        lang: Dil kodu (Türkçe için "tr").
        print_report: Raporu yazdır.

    Returns:
        dict: BERTScore metrikleri (Precision, Recall, F1).
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("⚠️  bert-score kütüphanesi bulunamadı.")
        print("   Kurulum: pip install bert-score")
        return {}

    print(f"   ⏳ BERTScore hesaplanıyor ({model_name})...")

    P, R, F1 = bert_score_fn(
        predictions,
        references,
        lang=lang,
        verbose=False,
    )

    results = {
        "model_name": model_name,
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }

    if print_report:
        print(f"\n{'='*55}")
        print(f"  🧠 {model_name} — BERTScore")
        print(f"{'='*55}")
        print(f"  Precision : {results['bertscore_precision']:.4f}")
        print(f"  Recall    : {results['bertscore_recall']:.4f}")
        print(f"  F1        : {results['bertscore_f1']:.4f}")

    return results


# ============================================================
#  4) TÜM ÖZETLEMEMETRİKLERİNİ BİRLEŞTİR
# ============================================================
def evaluate_summarization(
    references: list,
    predictions: list,
    model_name: str = "Model",
    use_bertscore: bool = True,
    print_report: bool = True,
) -> dict:
    """
    ROUGE ve BERTScore'u tek seferde hesaplar.

    Args:
        references: Referans özetler.
        predictions: Üretilen özetler.
        model_name: Model adı.
        use_bertscore: BERTScore da hesaplansın mı.
        print_report: Raporu yazdır.

    Returns:
        dict: Birleşik metrikler.
    """
    results = evaluate_summarization_rouge(
        references, predictions, model_name, print_report
    )

    if use_bertscore:
        bert_results = evaluate_summarization_bertscore(
            references, predictions, model_name, print_report=print_report
        )
        results.update(bert_results)

    return results


# ============================================================
#  5) MODEL KARŞILAŞTIRMA TABLOSU
# ============================================================
def compare_models(results_list: list, task: str = "classification") -> pd.DataFrame:
    """
    Birden fazla modelin sonuçlarını karşılaştırma tablosu olarak döndürür.

    Args:
        results_list: evaluate_classification veya evaluate_summarization
                      çıktılarının listesi.
        task: "classification" veya "summarization".

    Returns:
        pd.DataFrame: Karşılaştırma tablosu.
    """
    if task == "classification":
        rows = []
        for r in results_list:
            rows.append({
                "Model": r.get("model_name", "?"),
                "Accuracy": f"{r.get('accuracy', 0):.4f}",
                "F1 (Macro)": f"{r.get('f1_macro', 0):.4f}",
                "F1 (Weighted)": f"{r.get('f1_weighted', 0):.4f}",
                "Precision": f"{r.get('precision_macro', 0):.4f}",
                "Recall": f"{r.get('recall_macro', 0):.4f}",
            })
    else:
        rows = []
        for r in results_list:
            row = {
                "Model": r.get("model_name", "?"),
                "ROUGE-1": f"{r.get('rouge1', 0):.4f}",
                "ROUGE-2": f"{r.get('rouge2', 0):.4f}",
                "ROUGE-L": f"{r.get('rougeL', 0):.4f}",
            }
            if "bertscore_f1" in r:
                row["BERTScore F1"] = f"{r['bertscore_f1']:.4f}"
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("Model")

    print(f"\n{'='*55}")
    print(f"  🏆 Model Karşılaştırma Tablosu ({task})")
    print(f"{'='*55}")
    print(df.to_string())

    return df


# ============================================================
#  DEMO
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  📏 Değerlendirme Modülü — Demo")
    print("=" * 60)

    # Demo: Sınıflandırma
    y_true = ["pozitif", "negatif", "nötr", "pozitif", "negatif", "pozitif"]
    y_pred = ["pozitif", "negatif", "pozitif", "pozitif", "nötr",   "pozitif"]

    clf_results = evaluate_classification(
        y_true, y_pred,
        label_names=["negatif", "nötr", "pozitif"],
        model_name="Demo Model",
    )

    # Demo: ROUGE (basit test)
    refs = ["Bu ürün çok kaliteli ve güzel", "Kargo hızlı geldi memnunum"]
    preds = ["Ürün kaliteli güzel", "Kargo hızlı geldi"]

    rouge_results = evaluate_summarization_rouge(
        refs, preds, model_name="Demo Özetleme"
    )

    print("\n✅ Demo tamamlandı!")
