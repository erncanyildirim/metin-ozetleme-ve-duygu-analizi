# 🇹🇷 Türkçe E-Ticaret Ürün Yorumları: Duygu Analizi ve Çoklu Metin Özetleme

> Hepsiburada ürün yorumları üzerinde **Duygu Analizi (Sentiment Analysis)** ve **Çoklu Metin Özetleme (Multi-Document Summarization)** gerçekleştiren bir NLP projesidir.

## 👥 Proje Ekibi

| Kişi | Görev Alanı |
|------|------------|
| **Eren Can** | Veri İndirme, Ön İşleme, SVM/Naive Bayes Baseline Modeli |
| **Hamza** | NLP Ön İşleme, BERTurk/T5 Transformer Modeli |

## 📁 Proje Yapısı

```
metin-ozetleme-ve-duygu-analizi/
├── data/                  # Veri setleri (gitignore ile hariç tutulur)
├── notebooks/             # Jupyter Notebook deneyleri
├── src/                   # Python kaynak kodları
│   ├── __init__.py
│   ├── data_loader.py     # Veri indirme ve yükleme (Eren Can)
│   ├── preprocessing.py   # NLP ön işleme fonksiyonları (Hamza)
│   ├── baseline_model.py  # SVM / Naive Bayes modeli (Eren Can)
│   ├── transformer_model.py # BERTurk / T5 modeli (Hamza)
│   └── evaluation.py      # Ortak değerlendirme metrikleri
├── models/                # Eğitilmiş model dosyaları (gitignore ile hariç tutulur)
├── app.py                 # Streamlit web arayüzü
├── requirements.txt       # Bağımlılıklar
├── .gitignore
└── README.md
```

## 🚀 Kurulum

```bash
# Repoyu klonla
git clone https://github.com/erncanyildirim/metin-ozetleme-ve-duygu-analizi.git
cd metin-ozetleme-ve-duygu-analizi

# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları kur
pip install -r requirements.txt

# nbstripout'u aktif et (Jupyter çakışmalarını önler)
nbstripout --install
```

## 🧪 Kullanım

```bash
# Streamlit arayüzünü başlat
streamlit run app.py
```

## 📊 Değerlendirme Metrikleri

| Görev | Metrikler |
|-------|----------|
| Duygu Analizi | Accuracy, F1-Score |
| Metin Özetleme | ROUGE-1, ROUGE-2, ROUGE-L, BERTScore |

## 📝 Lisans

Bu proje eğitim amaçlıdır.
