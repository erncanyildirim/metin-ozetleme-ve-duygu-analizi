# TÜRKÇE E-TİCARET ÜRÜN YORUMLARI ÜZERİNDE DUYGU ANALİZİ VE ÇOKLU METİN ÖZETLEME

**Bitirme Projesi Raporu**

---

**Hazırlayanlar:**
Eren Can YILDIRIM
Hamza [SOYADI]

**Dersler:**
Doğal Dil İşleme & Derin Öğrenme

**Danışman:**
[Hocanın Adı]

**Tarih:**
2026 — Bahar Dönemi

---

## İÇİNDEKİLER

1. [GİRİŞ](#1-giriş)
   1.1. Problemin Tanımı
   1.2. Projenin Amacı ve Kapsamı
   1.3. Motivasyon
   1.4. Katkılar
2. [ÖNCEKİ ÇALIŞMALAR (LİTERATÜR TARAMASI)](#2-önceki-çalışmalar-literatür-taraması)
   2.1. Duygu Analizi Alanındaki Çalışmalar
   2.2. Metin Özetleme Alanındaki Çalışmalar
   2.3. Türkçe NLP Çalışmaları
   2.4. Karşılaştırmalı Tablo
3. [MALZEME VE YÖNTEM](#3-malzeme-ve-yöntem)
   3.1. Kullanılan Veri Seti
   3.2. Veri Ön İşleme
   3.3. Özellik Çıkarımı (TF-IDF)
   3.4. Kullanılan Modeller
   3.5. Değerlendirme Metrikleri
   3.6. Donanım ve Yazılım Ortamı
4. [DENEY VE BULGULAR](#4-deney-ve-bulgular)
   4.1. Hipotezler
   4.2. Deney Tasarımı
   4.3. Veri Seti İstatistikleri
   4.4. Duygu Analizi Sonuçları
   4.5. Metin Özetleme Sonuçları
   4.6. Karşılaştırmalı Analiz
   4.7. Tartışma
5. [SONUÇ VE ÖNERİLER](#5-sonuç-ve-öneriler)
   5.1. Projenin Başarı Değerlendirmesi
   5.2. Karşılaşılan Zorluklar
   5.3. Gelecek Çalışmalar
6. [KAYNAKÇA](#6-kaynakça)
7. [EKLER](#7-ekler)

---

## 1. GİRİŞ

### 1.1. Problemin Tanımı

E-ticaret sektörünün hızlı büyümesiyle birlikte, kullanıcı yorumları hem tüketiciler hem de satıcılar için kritik bir bilgi kaynağı hâline gelmiştir. Türkiye'nin en büyük e-ticaret platformlarından biri olan Hepsiburada üzerinde her gün binlerce ürün yorumu yayınlanmaktadır. Ancak bu yorumların manuel olarak okunup analiz edilmesi pratik değildir. Bu durumda iki temel problem ortaya çıkmaktadır:

1. **Duygu Analizi (Sentiment Analysis) Problemi:** Bir yorumun pozitif, nötr veya negatif bir duygu içerip içermediğini otomatik olarak tespit etme ihtiyacı.
2. **Çoklu Metin Özetleme (Multi-Document Summarization) Problemi:** Bir ürün hakkındaki onlarca/yüzlerce yorumun ana fikrini birkaç cümlelik bir özette toparlama ihtiyacı.

Türkçe, sondan eklemeli (agglutinative) yapısı, zengin morfolojisi ve sınırlı NLP kaynakları sebebiyle bu problemleri İngilizce'ye göre daha zorlu hâle getirmektedir.

### 1.2. Projenin Amacı ve Kapsamı

Bu projenin amacı, Hepsiburada'dan toplanmış Türkçe ürün yorumları üzerinde:

- **(i)** Klasik makine öğrenmesi yöntemleri (SVM, Naive Bayes) ile derin öğrenme yöntemlerini (BERTurk Transformer modeli) **duygu sınıflandırma** görevinde karşılaştırmak,
- **(ii)** Aynı ürüne dair birden çok yorumun, mT5 (multilingual T5) tabanlı bir Transformer modeli ile **çoklu özetlenmesini** sağlamak,
- **(iii)** Elde edilen sistemleri etkileşimli bir **Streamlit web arayüzünde** son kullanıcıya sunmaktır.

### 1.3. Motivasyon

- Türkçe için açık kaynaklı son kullanıcı odaklı duygu analizi araçları sınırlıdır.
- E-ticaret platformlarındaki ürün önerme sistemleri için yorum analizi kritik bir ön adımdır.
- Klasik (TF-IDF + SVM) yaklaşımların hâlâ rekabetçi olup olmadığını derin öğrenme alternatifleriyle karşılaştırmak akademik olarak değerlidir.

### 1.4. Katkılar

Bu çalışmanın temel katkıları:

1. Hepsiburada veri seti üzerinde **uçtan uca** bir Türkçe duygu analizi pipeline'ı önerilmiş ve açık kaynak olarak yayımlanmıştır.
2. **TF-IDF + Linear SVM** baseline modeli ile **BERTurk** transformer modelinin doğruluk/zaman maliyeti karşılaştırması yapılmıştır.
3. Yorumlar üzerinde **çoklu özetleme** için mT5-small modelinin Türkçe performansı raporlanmıştır.
4. Sonuçlar, kullanıcıların kendi yorumlarını test edebileceği bir **Streamlit web arayüzü** ile somutlaştırılmıştır.

---

## 2. ÖNCEKİ ÇALIŞMALAR (LİTERATÜR TARAMASI)

### 2.1. Duygu Analizi Alanındaki Çalışmalar

**Klasik makine öğrenmesi yaklaşımları:**

- **Pang ve Lee (2008)** — *"Opinion Mining and Sentiment Analysis"* başlıklı temel referans çalışmasında SVM, Naive Bayes ve Maximum Entropy modellerini film yorumları üzerinde karşılaştırmış, TF-IDF + SVM'in en iyi sonucu verdiğini göstermiştir.
- **Joachims (1998)** — "*Text Categorization with Support Vector Machines*" makalesiyle SVM'i metin sınıflandırma için altın standart olarak konumlandırmıştır.

**Derin öğrenme yaklaşımları:**

- **Devlin et al. (2018)** — *"BERT: Pre-training of Deep Bidirectional Transformers"* (NAACL 2019) çalışmasıyla bağlamsal kelime gömme yöntemini tanıtmış, duygu analizinde state-of-the-art sonuçlar elde etmiştir.
- **Liu et al. (2019)** — *"RoBERTa"* ve sonrasında çıkan **DistilBERT**, **ALBERT** gibi varyasyonlar BERT'in performansını artırmıştır.

### 2.2. Metin Özetleme Alanındaki Çalışmalar

- **Lin (2004)** — *"ROUGE: A Package for Automatic Evaluation of Summaries"* — Özetlemenin standart değerlendirme metriği ROUGE'u tanıtmıştır.
- **See et al. (2017)** — *"Get To The Point: Summarization with Pointer-Generator Networks"* — Soyut (abstractive) özetlemede pointer-generator mimarisini tanıtmıştır.
- **Raffel et al. (2020)** — *"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)"* — Tüm NLP görevlerini "text-to-text" olarak modelleyen T5 mimarisini önermiştir. mT5 (multilingual T5) Türkçe dâhil 101 dili desteklemektedir.
- **Lewis et al. (2020)** — *"BART"* — Hem soyut özetleme hem de denoising autoencoding için kullanılan bir model.

### 2.3. Türkçe NLP Çalışmaları

- **Schweter (2020)** — *"BERTurk"* — Türkçe Wikipedia ve OSCAR korpuslarında ön-eğitilen BERT modeli (`dbmdz/bert-base-turkish-cased`). Türkçe duygu analizinde standart başlangıç noktası hâline gelmiştir.
- **Savcı ve Üsküdarlı (2022)** — Türkçe Twitter verisi üzerinde BERTurk + bilstm hibrit modeli, klasik yöntemlere göre yaklaşık %5 F1 artışı raporlamıştır.
- **Akın (2007)** — *"Zemberek-NLP"* — Türkçe morfolojik analiz ve kök bulma için referans araç.

### 2.4. GitHub ve Medium Kaynakları

- **HuggingFace Transformers** kütüphanesi (`transformers` paketi) — BERTurk, mT5 gibi modelleri kolayca yüklemeyi sağlar.
- **savasy/bert-base-turkish-sentiment-cased** — Hugging Face hub'da yayınlanmış, ürün/film yorumlarında ince ayarlı Türkçe duygu modeli.
- **Medium: "Turkish Sentiment Analysis with BERT"** — Türkçe duygu analizini adım adım anlatan blog post serisi.

### 2.5. Karşılaştırmalı Tablo

| Çalışma | Yıl | Veri Seti | Yöntem | Sonuç (F1/Acc) |
|---------|-----|-----------|--------|-----------------|
| Pang & Lee | 2008 | IMDB | TF-IDF + SVM | ~%88 Acc |
| Devlin et al. | 2018 | SST-2 | BERT | %94.9 Acc |
| Schweter | 2020 | Twitter TR | BERTurk | %85 F1 |
| Savcı & Üsküdarlı | 2022 | Twitter TR | BERTurk + BiLSTM | %88 F1 |
| **Bu Çalışma** | 2026 | Hepsiburada | SVM + BERTurk | **[Sonuç buraya]** |

---

## 3. MALZEME VE YÖNTEM

### 3.1. Kullanılan Veri Seti

**Veri seti:** Hepsiburada Product Reviews and Ratings Dataset
**Kaynak:** Kaggle — `yusufkesmenn/hepsiburada-product-reviews-and-ratings-dataset`
**Boyut:** ~250.000 yorum (tahmini)
**Alanlar:** Ürün adı, kullanıcı yorumu, 1-5 yıldız puanı

**Neden bu veri seti?**

1. **Türkçe ve Gerçek Hayata Yakın:** Türkçe e-ticaret yorumları doğal dilde, argo ve yazım hataları içerir.
2. **Etiketli:** Yıldız puanı sayesinde duygu etiketi otomatik üretilebilir (1-2: negatif, 3: nötr, 4-5: pozitif).
3. **Büyük Hacim:** Hem klasik hem derin öğrenme modelleri için yeterli örnek sayısı.
4. **Çok Boyutlu Kullanılabilir:** Hem duygu sınıflandırma hem de çoklu özetleme için aynı veri kullanılabilir.

### 3.2. Veri Ön İşleme

Türkçe için özel olarak uygulanan ön işleme adımları (`src/preprocessing.py`):

1. **Küçük harfe çevirme** (Türkçe karakter farkındalığıyla: İ→i, I→ı)
2. **URL ve e-posta temizliği**
3. **HTML etiket temizliği**
4. **Emoji temizliği** (Unicode aralıkları ile)
5. **Noktalama işareti temizliği** (Türkçe karakterleri koruyarak)
6. **Türkçe stop-word filtresi** (özellikle duygu taşıyan kelimeler — "değil", "yok", "ama" — bilinçli olarak listede tutulmamıştır)
7. **Tekrarlayan karakter düzeltimi** ("çoook" → "çook")
8. **Fazla boşluk temizliği**

### 3.3. Özellik Çıkarımı: TF-IDF

Baseline modelleri için **TF-IDF (Term Frequency–Inverse Document Frequency)** vektörizasyonu kullanılmıştır:

- **Max özellik:** 50.000 kelime
- **N-gram aralığı:** Unigram + Bigram (1, 2)
- **Sublinear TF:** Aktif (logaritmik ölçekleme)
- **Min df:** 2 (en az 2 dokümanda geçen kelimeler)
- **Max df:** 0.95 (çok yaygın kelimeler atılır)

### 3.4. Kullanılan Modeller

#### 3.4.1. Baseline Model 1: Linear SVM

- **Algoritma:** sklearn `LinearSVC`
- **Düzenleme parametresi (C):** 1.0
- **Sınıf ağırlığı:** Dengesiz sınıflar için "balanced"
- **Max iterasyon:** 10.000

#### 3.4.2. Baseline Model 2: Multinomial Naive Bayes

- **Algoritma:** sklearn `MultinomialNB`
- **Alpha (Laplace smoothing):** 0.1

#### 3.4.3. Transformer Model 1: BERTurk (Duygu Analizi)

- **Mimari:** `dbmdz/bert-base-turkish-cased` (Stefan Schweter, 2020)
- **Parametreler:** 110M parametre, 12 katman, 768 gizli boyut
- **Fine-tuning:** Hepsiburada veri setinin %80'i üzerinde 3 epoch
- **Optimizer:** AdamW (learning rate: 2e-5)
- **Batch size:** 16 (Colab T4 GPU sınırı)

#### 3.4.4. Transformer Model 2: mT5-small (Özetleme)

- **Mimari:** `google/mt5-small` (multilingual T5, Raffel et al. 2020)
- **Parametreler:** 300M parametre, 101 dil
- **Görev:** Aynı ürüne ait yorumların ortak özetini üretme
- **Generation parametreleri:** beam search (num_beams=4), max_length=150, min_length=30

### 3.5. Değerlendirme Metrikleri

**Duygu Sınıflandırma için:**

- **Accuracy:** Doğru tahmin edilen örneklerin oranı.
- **Precision (Macro):** Tüm sınıflar için ortalama doğruluk.
- **Recall (Macro):** Tüm sınıflar için ortalama duyarlılık.
- **F1-Score (Macro / Weighted):** Dengesiz veri seti için Macro F1 birincil metrik.
- **Confusion Matrix:** Sınıflar arası karışıklıkların görselleştirilmesi.

**Özetleme için:**

- **ROUGE-1, ROUGE-2, ROUGE-L:** Referans özet ile üretilen özet arasındaki n-gram örtüşmesi.
- **BERTScore (F1):** Anlamsal benzerlik için kontekstüel kelime gömme tabanlı metrik.

### 3.6. Donanım ve Yazılım Ortamı

| Bileşen | Detay |
|---------|-------|
| Donanım | Google Colab — NVIDIA Tesla T4 GPU (16GB) |
| İşletim Sistemi | Ubuntu 22.04 (Colab varsayılan) |
| Python sürümü | 3.10 |
| Temel kütüphaneler | `scikit-learn 1.3+`, `pandas 2.0+`, `transformers 4.30+`, `torch 2.0+`, `rouge-score`, `bert-score` |
| Sunum | Streamlit 1.28+ |

---

## 4. DENEY VE BULGULAR

> **NOT:** Bu bölüm modeller eğitildikten sonra **gerçek sayılar ve grafiklerle** doldurulacaktır.

### 4.1. Hipotezler

- **H1:** TF-IDF + Linear SVM, dengeli bir veri setinde >%85 F1-Macro skoru elde edebilir.
- **H2:** BERTurk fine-tune edildiğinde baseline'a göre en az **+5 puan** F1 iyileştirmesi sağlar.
- **H3:** Türkçe stop-word'leri çıkarma, **F1-Macro** üzerinde **+1-2 puan** etkilidir; ancak duygu taşıyan kelimeleri ("değil", "ama") tutmak kritiktir.
- **H4:** mT5-small, küçük boyutuna rağmen ROUGE-1 ≥ 0.30 üzerinde özetleme yapabilir.
- **H5:** Yorum sayısı arttıkça (çoklu özetleme), özet kalitesi artar.

### 4.2. Deney Tasarımı

- **Train/Test bölme:** 80/20 stratified split (sınıf dengesini korur)
- **Cross-Validation:** Baseline modeller için 5-fold CV
- **Seed:** `random_state=42` (tekrar üretilebilirlik için)

### 4.3. Veri Seti İstatistikleri

> Aşağıdaki tablolar `notebooks/01_veri_kesfi.ipynb` çıktısından elde edilecek.

- Toplam yorum sayısı: **[X]**
- Ortalama yorum uzunluğu: **[Y]** kelime
- Sınıf dağılımı:
  - Pozitif (4-5 yıldız): **[A]** (%)
  - Nötr (3 yıldız): **[B]** (%)
  - Negatif (1-2 yıldız): **[C]** (%)

*Şekil 4.1:* Sınıf dağılımı pasta grafiği — **[grafik eklenecek]**
*Şekil 4.2:* Kelime sayısı histogramı — **[grafik eklenecek]**
*Şekil 4.3:* En sık 20 kelime barplot — **[grafik eklenecek]**

### 4.4. Duygu Analizi Sonuçları

#### 4.4.1. Baseline Sonuçları

| Model | Accuracy | Precision | Recall | F1 (Macro) | F1 (Weighted) |
|-------|----------|-----------|--------|------------|---------------|
| Naive Bayes | [.] | [.] | [.] | [.] | [.] |
| Linear SVM | [.] | [.] | [.] | [.] | [.] |

*Şekil 4.4:* Baseline modellerin confusion matrix karşılaştırması — **[grafik eklenecek]**

#### 4.4.2. Transformer Sonuçları

| Model | Accuracy | F1 (Macro) | Eğitim Süresi |
|-------|----------|------------|---------------|
| BERTurk (3 epoch) | [.] | [.] | [.] dakika |

*Şekil 4.5:* BERTurk eğitim/doğrulama kayıp grafiği — **[grafik eklenecek]**

#### 4.4.3. Karşılaştırma

| Model | F1 (Macro) | Tahmin Süresi (1000 yorum) | Bellek |
|-------|-----------|-----------------------------|--------|
| SVM | [.] | [.] sn | ~50MB |
| BERTurk | [.] | [.] sn | ~440MB |

### 4.5. Metin Özetleme Sonuçları

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 |
|-------|---------|---------|---------|--------------|
| Extractive Baseline | [.] | [.] | [.] | [.] |
| mT5-small | [.] | [.] | [.] | [.] |

*Şekil 4.6:* Örnek özetleme çıktıları — referans vs üretilen — **[tablo eklenecek]**

### 4.6. Karşılaştırmalı Analiz

> Bu bölümde:
> - SVM vs BERTurk için **t-test** veya **McNemar** istatistiksel anlamlılık testi
> - Yanlış sınıflandırılan örneklerden **hata analizi** (ironi, alaycılık, karışık yorumlar)
> - Sınıf bazlı analiz: hangi sınıfta hangi model daha başarılı?

### 4.7. Tartışma

- Hipotezlerin doğrulanma durumu
- Beklenmeyen bulgular
- Veri setindeki ön yargılar (örneğin 5 yıldızlı yorumların çokluğu)

---

## 5. SONUÇ VE ÖNERİLER

### 5.1. Projenin Başarı Değerlendirmesi

> Bu bölüm Bölüm 1'de tanımlanan amaçların gerçekleşme oranına göre yazılacak.

- ✅ Türkçe yorumlar üzerinde uçtan uca duygu analizi sistemi kuruldu.
- ✅ Klasik ve derin öğrenme modelleri karşılaştırıldı.
- ✅ Çoklu metin özetleme prototipi geliştirildi.
- ✅ Streamlit ile etkileşimli demo arayüzü sunuldu.

### 5.2. Karşılaşılan Zorluklar

- Türkçe NLP için sınırlı önişleme aracı (Zemberek alternatifi az)
- Veri setindeki etiket dengesizliği (5 yıldız yorumların aşırı çoğunluğu)
- Colab'ın oturum süre sınırları (uzun eğitimler için yetersiz)
- Türkçe için ROUGE'un Türkçe stemmer eksikliği nedeniyle kısmi geçerliliği

### 5.3. Gelecek Çalışmalar

1. **Daha büyük transformer modelleri:** BERTurk-large veya XLM-RoBERTa-large denenebilir.
2. **Daha kaliteli özetleme:** mT5-base veya Türkçe-spesifik BART modelleriyle iyileştirme.
3. **Aspect-Based Sentiment Analysis:** Sadece "ürün iyi/kötü" değil, "kargo geç ama ürün iyi" gibi yönsel analizler.
4. **İronisi ve alaycılık tespiti:** Türkçe ironi veri seti oluşturup model ince ayarı.
5. **Real-time deployment:** Streamlit yerine FastAPI + Docker ile production sistem.
6. **Çapraz alan transfer:** Hepsiburada modelini Trendyol, n11 vb. platformlara transfer ederek genelleştirme.
7. **Açıklanabilirlik (XAI):** SHAP/LIME ile hangi kelimelerin tahmini etkilediğini görselleştirme.

---

## 6. KAYNAKÇA

1. Pang, B., & Lee, L. (2008). *Opinion Mining and Sentiment Analysis*. Foundations and Trends in Information Retrieval, 2(1–2), 1–135.
2. Joachims, T. (1998). *Text Categorization with Support Vector Machines: Learning with Many Relevant Features*. ECML.
3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.
4. Schweter, S. (2020). *BERTurk - BERT models for Turkish*. Zenodo. https://doi.org/10.5281/zenodo.3770924
5. Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR, 21(140), 1-67.
6. Xue, L., et al. (2021). *mT5: A massively multilingual pre-trained text-to-text transformer*. NAACL-HLT.
7. Lin, C. Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. Text Summarization Branches Out.
8. Zhang, T., et al. (2019). *BERTScore: Evaluating Text Generation with BERT*. ICLR.
9. See, A., Liu, P. J., & Manning, C. D. (2017). *Get To The Point: Summarization with Pointer-Generator Networks*. ACL.
10. Hugging Face Transformers Library. https://huggingface.co/docs/transformers
11. Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP Demos.
12. Akın, A. A., & Akın, M. D. (2007). *Zemberek, an open source NLP framework for Turkic languages*. Structure.
13. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.

---

## 7. EKLER

### Ek A: Proje GitHub Deposu

https://github.com/erncanyildirim/metin-ozetleme-ve-duygu-analizi

### Ek B: Streamlit Demo

`streamlit run app.py` ile yerelde çalıştırılabilir.

### Ek C: Eğitim Süreleri

| Model | Donanım | Eğitim Süresi |
|-------|---------|----------------|
| Linear SVM | CPU | [.] dakika |
| Naive Bayes | CPU | [.] saniye |
| BERTurk | T4 GPU | [.] dakika |
| mT5-small fine-tune | T4 GPU | [.] dakika |

### Ek D: Hiperparametre Detayları

> Tüm modellerin tam parametre listeleri burada verilecek.

---

**[Rapor sonu]**
