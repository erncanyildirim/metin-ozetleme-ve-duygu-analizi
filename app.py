"""
🖥️ Streamlit Web Arayüzü
Türkçe E-Ticaret Ürün Yorumları — Duygu Analizi & Metin Özetleme

Çalıştırma:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import sys
import time

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
#  SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="Türkçe Duygu Analizi & Metin Özetleme",
    page_icon="🇹🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  ÖZEL CSS
# ============================================================
st.markdown("""
<style>
    /* Ana başlık */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Sonuç kartları */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .result-card-dark {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a2e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        color: white;
    }

    /* Duygu etiketleri */
    .sentiment-pozitif {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    .sentiment-negatif {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    .sentiment-nötr {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #aaa;
        padding: 2rem 0 1rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  MODEL YÜKLEME (cache ile — sadece 1 kez yüklenir)
# ============================================================
@st.cache_resource
def load_baseline_model():
    """Kaydedilmiş baseline (SVM) modelini yükler."""
    import joblib
    models_dir = os.path.join(os.path.dirname(__file__), "models")

    # Önce SVM, yoksa NB dene
    for name in ["svm_(linearsvc)", "naive_bayes", "svm__linearsvc_"]:
        path = os.path.join(models_dir, f"{name}.joblib")
        if os.path.exists(path):
            return joblib.load(path), name
    return None, None


@st.cache_resource
def load_transformer_model():
    """Hamza'nın Transformer modelini yükler (varsa)."""
    try:
        from src.transformer_model import load_sentiment_model, load_summarization_model
        sentiment_model = load_sentiment_model()
        summarization_model = load_summarization_model()
        return sentiment_model, summarization_model
    except Exception:
        return None, None


def preprocess_text_cached(text: str) -> str:
    """Ön işleme fonksiyonunu çağırır."""
    from src.preprocessing import preprocess_text
    return preprocess_text(text)


# ============================================================
#  DUYGU ANALİZİ FONKSİYONU
# ============================================================
def analyze_sentiment(texts: list, model_type: str = "baseline") -> list:
    """
    Metin listesi için duygu analizi yapar.

    Args:
        texts: Yorum metni listesi.
        model_type: "baseline" veya "transformer".

    Returns:
        list[dict]: Her yorum için sonuç.
    """
    results = []
    emoji_map = {"pozitif": "😊", "nötr": "😐", "negatif": "😠"}

    if model_type == "baseline":
        pipeline, _ = load_baseline_model()
        if pipeline is None:
            # Model yoksa basit kural tabanlı fallback
            for text in texts:
                clean = preprocess_text_cached(text)
                sentiment = _rule_based_sentiment(clean)
                results.append({
                    "text": text,
                    "clean_text": clean,
                    "sentiment": sentiment,
                    "emoji": emoji_map.get(sentiment, ""),
                    "method": "Kural Tabanlı (model bulunamadı)",
                })
            return results

        for text in texts:
            clean = preprocess_text_cached(text)
            pred = pipeline.predict([clean])[0]
            results.append({
                "text": text,
                "clean_text": clean,
                "sentiment": pred,
                "emoji": emoji_map.get(pred, ""),
                "method": "SVM (Baseline)",
            })

    elif model_type == "transformer":
        sentiment_model, _ = load_transformer_model()
        if sentiment_model is None:
            st.warning("⚠️ Transformer modeli henüz yüklenmedi. Baseline kullanılıyor.")
            return analyze_sentiment(texts, model_type="baseline")

        try:
            for text in texts:
                result = sentiment_model(text)
                label = result[0]["label"]
                # Label'ı Türkçe'ye çevir
                label_map = {
                    "POSITIVE": "pozitif", "NEGATIVE": "negatif", "NEUTRAL": "nötr",
                    "positive": "pozitif", "negative": "negatif", "neutral": "nötr",
                    "POS": "pozitif", "NEG": "negatif", "NEU": "nötr",
                    "1 star": "negatif", "2 stars": "negatif",
                    "3 stars": "nötr", "4 stars": "pozitif", "5 stars": "pozitif",
                }
                sentiment = label_map.get(label, label.lower())
                results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "emoji": emoji_map.get(sentiment, ""),
                    "confidence": result[0].get("score", 0),
                    "method": "Transformer (BERTurk)",
                })
        except Exception as e:
            st.error(f"Transformer hatası: {e}")
            return analyze_sentiment(texts, model_type="baseline")

    return results


def _rule_based_sentiment(text: str) -> str:
    """Model yüklenemezse basit kural tabanlı duygu analizi."""
    positive_words = {
        "güzel", "harika", "mükemmel", "süper", "kaliteli", "memnun",
        "tavsiye", "beğendim", "sevdim", "muhteşem", "başarılı",
        "iyi", "hızlı", "sağlam", "uygun", "teşekkür", "teşekkürler",
        "memnunum", "ederim", "bayıldım", "kusursuz", "enfes",
    }
    negative_words = {
        "kötü", "berbat", "rezalet", "bozuk", "hasarlı", "geç",
        "pişman", "iade", "almayın", "kırık", "sahte", "düşük",
        "yetersiz", "vasat", "korkunç", "felaket", "eksik",
        "sorun", "problem", "çöp", "leş", "boktan", "dandik",
    }
    words = set(text.lower().split())
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)

    if pos_count > neg_count:
        return "pozitif"
    elif neg_count > pos_count:
        return "negatif"
    else:
        return "nötr"


# ============================================================
#  ÖZETLEMEFONKSİYONU
# ============================================================
def summarize_texts(texts: list) -> str:
    """
    Birden fazla yorumu tek paragraflık özete dönüştürür.

    Önce Transformer (T5) dener, yoksa extractive fallback kullanır.
    """
    _, summarization_model = load_transformer_model()

    combined_text = " ".join(texts)

    if summarization_model is not None:
        try:
            summary = summarization_model(
                combined_text[:1024],  # Token limiti
                max_length=150,
                min_length=30,
                do_sample=False,
            )
            return summary[0]["summary_text"]
        except Exception:
            pass

    # Fallback: Extractive özetleme (en önemli cümleleri seç)
    return _extractive_summary(texts)


def _extractive_summary(texts: list, top_n: int = 3) -> str:
    """
    Basit extractive özetleme: en uzun ve bilgi yoğun cümleleri seçer.
    Transformer modeli yokken kullanılır.
    """
    all_sentences = []
    for text in texts:
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".")]
        all_sentences.extend([s for s in sentences if len(s.split()) >= 3])

    if not all_sentences:
        return "Yeterli metin bulunamadı."

    # Kelime frekansına göre cümle skoru hesapla
    word_freq = {}
    for sentence in all_sentences:
        for word in sentence.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1

    scored = []
    for sentence in all_sentences:
        words = sentence.lower().split()
        if len(words) == 0:
            continue
        score = sum(word_freq.get(w, 0) for w in words) / len(words)
        scored.append((score, sentence))

    scored.sort(reverse=True)
    selected = [s for _, s in scored[:top_n]]
    summary = ". ".join(selected)

    if not summary.endswith("."):
        summary += "."

    return summary


# ============================================================
#  ANA ARAYÜZ
# ============================================================
def main():
    # --- Başlık ---
    st.markdown('<h1 class="main-title">🇹🇷 Türkçe Duygu Analizi & Metin Özetleme</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Hepsiburada ürün yorumlarını yapay zeka ile analiz edin</p>',
        unsafe_allow_html=True,
    )

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Hepsiburada_logo.svg/512px-Hepsiburada_logo.svg.png", width=180)
        st.markdown("---")
        st.markdown("### ⚙️ Ayarlar")

        model_choice = st.radio(
            "Model Seçimi:",
            ["Baseline (SVM)", "Transformer (BERTurk)"],
            index=0,
            help="Baseline: Hızlı, hafif | Transformer: Daha doğru, yavaş",
        )

        st.markdown("---")
        st.markdown("### 👥 Proje Ekibi")
        st.markdown("- **Eren Can** — Veri & Baseline")
        st.markdown("- **Hamza** — NLP & Transformer")
        st.markdown("---")
        st.markdown("### 📊 Metrikler")
        st.markdown("- Duygu: Accuracy, F1")
        st.markdown("- Özet: ROUGE, BERTScore")

    # --- Ana İçerik ---
    tab1, tab2 = st.tabs(["💬 Duygu Analizi", "📝 Metin Özetleme"])

    # ==================== TAB 1: Duygu Analizi ====================
    with tab1:
        st.markdown("### 💬 Ürün Yorumlarının Duygu Analizi")
        st.markdown(
            "Aşağıya bir veya birden fazla ürün yorumu girin. "
            "**Her yorum ayrı bir satırda** olmalıdır."
        )

        default_text = (
            "Ürün gerçekten harika, çok memnun kaldım. Herkese tavsiye ederim.\n"
            "Kargo çok geç geldi ve ürün hasarlıydı. Kesinlikle almayın.\n"
            "Fiyatına göre idare eder, beklentiyi karşılıyor."
        )

        user_input = st.text_area(
            "Ürün yorumlarını girin:",
            value=default_text,
            height=200,
            placeholder="Her satıra bir yorum yazın...",
        )

        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            analyze_btn = st.button("🔍 Analiz Et", type="primary", use_container_width=True)

        if analyze_btn and user_input.strip():
            texts = [t.strip() for t in user_input.strip().split("\n") if t.strip()]
            model_type = "baseline" if "Baseline" in model_choice else "transformer"

            with st.spinner("Analiz ediliyor..."):
                results = analyze_sentiment(texts, model_type=model_type)

            # --- Genel Duygu Özeti ---
            sentiments = [r["sentiment"] for r in results]
            pos_count = sentiments.count("pozitif")
            neg_count = sentiments.count("negatif")
            neu_count = sentiments.count("nötr")
            total = len(sentiments)

            st.markdown("---")
            st.markdown("### 📊 Genel Duygu Durumu")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam Yorum", total)
            c2.metric("😊 Pozitif", f"{pos_count} ({pos_count/total*100:.0f}%)")
            c3.metric("😐 Nötr", f"{neu_count} ({neu_count/total*100:.0f}%)")
            c4.metric("😠 Negatif", f"{neg_count} ({neg_count/total*100:.0f}%)")

            # Genel duygu
            if pos_count > neg_count and pos_count > neu_count:
                overall = "😊 Genel olarak **POZİTİF** bir duygu hâkim."
            elif neg_count > pos_count and neg_count > neu_count:
                overall = "😠 Genel olarak **NEGATİF** bir duygu hâkim."
            else:
                overall = "😐 Genel olarak **NÖTR / KARIŞIK** bir duygu hâkim."

            st.info(overall)

            # Progress bar
            if total > 0:
                st.markdown("**Duygu Dağılımı:**")
                col_p, col_n, col_neu = st.columns(3)
                col_p.progress(pos_count / total, text=f"Pozitif {pos_count/total*100:.0f}%")
                col_n.progress(neg_count / total if neg_count > 0 else 0, text=f"Negatif {neg_count/total*100:.0f}%")
                col_neu.progress(neu_count / total if neu_count > 0 else 0, text=f"Nötr {neu_count/total*100:.0f}%")

            # --- Detaylı Sonuçlar ---
            st.markdown("---")
            st.markdown("### 📋 Detaylı Sonuçlar")

            for i, r in enumerate(results, 1):
                sentiment = r["sentiment"]
                emoji = r["emoji"]
                color_map = {
                    "pozitif": "#d4edda",
                    "negatif": "#f8d7da",
                    "nötr": "#fff3cd",
                }
                bg = color_map.get(sentiment, "#f0f0f0")

                st.markdown(
                    f"""
                    <div style="background-color: {bg}; padding: 1rem 1.5rem;
                    border-radius: 12px; margin: 0.5rem 0;">
                        <strong>Yorum {i}:</strong> {r['text']}<br>
                        <span style="font-size: 1.3rem;">{emoji}</span>
                        <span class="sentiment-{sentiment}"> {sentiment.upper()} </span>
                        <small style="color: #888;"> — {r.get('method', '')}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        elif analyze_btn:
            st.warning("⚠️ Lütfen en az bir yorum girin.")

    # ==================== TAB 2: Metin Özetleme ====================
    with tab2:
        st.markdown("### 📝 Çoklu Metin Özetleme")
        st.markdown(
            "Birden fazla ürün yorumunu girin, yapay zeka tek paragraflık bir özet üretsin."
        )

        summary_input = st.text_area(
            "Özetlenecek yorumları girin:",
            height=250,
            placeholder="Birden fazla yorum girin (her satıra bir yorum)...",
            key="summary_input",
        )

        col_s1, col_s2 = st.columns([1, 5])
        with col_s1:
            summarize_btn = st.button("📝 Özetle", type="primary", use_container_width=True)

        if summarize_btn and summary_input.strip():
            texts = [t.strip() for t in summary_input.strip().split("\n") if t.strip()]

            with st.spinner("Özet oluşturuluyor..."):
                summary = summarize_texts(texts)

            st.markdown("---")
            st.markdown("### 📄 Özet")
            st.success(summary)

            # Ayrıca bu yorumların duygu analizini de göster
            st.markdown("---")
            st.markdown("### 💬 Yorumların Duygu Durumu")
            model_type = "baseline" if "Baseline" in model_choice else "transformer"
            results = analyze_sentiment(texts, model_type=model_type)

            sentiments = [r["sentiment"] for r in results]
            pos = sentiments.count("pozitif")
            neg = sentiments.count("negatif")
            total = len(sentiments)

            if total > 0:
                st.markdown(
                    f"**{total}** yorumdan **{pos}** tanesi pozitif, "
                    f"**{neg}** tanesi negatif."
                )

        elif summarize_btn:
            st.warning("⚠️ Lütfen özetlenecek metin girin.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        '<p class="footer">🎓 Türkçe E-Ticaret Duygu Analizi & Metin Özetleme Projesi<br>'
        'Eren Can & Hamza — 2024/2025</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
