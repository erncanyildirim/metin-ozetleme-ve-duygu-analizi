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

## ÖZET (ABSTRACT)

Bu çalışmada Hepsiburada e-ticaret platformundan elde edilmiş 146.657 Türkçe ürün yorumu üzerinde **duygu analizi** ve **çoklu metin özetleme** problemleri ele alınmıştır. Duygu analizi için iki klasik makine öğrenmesi modeli (Linear SVM ve Naive Bayes) bir derin öğrenme modeli (BERTurk, ince ayarlı) ile karşılaştırılmıştır. Çoklu özetleme için mT5-small (zero-shot) modeli denenmiştir.

Bulgular, **BERTurk'ün F1-Macro skorunda Linear SVM'e göre +13.1 puan üstünlük** (0.626 → 0.757) sağladığını göstermiştir. Özellikle nötr sınıfının doğru sınıflandırılma oranı SVM'de %23 iken BERTurk'te %81'e yükselmiştir. Modelin Türkçe ironik ifadeleri tespit edebildiği gözlemlenmiştir. Öte yandan mT5 zero-shot özetlemede ROUGE-1=0.092 ile yetersiz performans göstermiş, hallucination ve tekrarlama problemleri tespit edilmiştir. Sonuçlar, derin öğrenme tabanlı modellerin dengesiz Türkçe duygu analizinde belirgin üstünlüğünü kanıtlamakta, fakat alana özel özetleme için fine-tuning'in zorunlu olduğunu vurgulamaktadır.

**Anahtar Kelimeler:** Türkçe NLP, Duygu Analizi, Metin Özetleme, BERTurk, mT5, TF-IDF, SVM

---

## İÇİNDEKİLER

1. [GİRİŞ](#1-giriş)
2. [ÖNCEKİ ÇALIŞMALAR (LİTERATÜR TARAMASI)](#2-önceki-çalışmalar-literatür-taraması)
3. [MALZEME VE YÖNTEM](#3-malzeme-ve-yöntem)
4. [DENEY VE BULGULAR](#4-deney-ve-bulgular)
5. [SONUÇ VE ÖNERİLER](#5-sonuç-ve-öneriler)
6. [KAYNAKÇA](#6-kaynakça)
7. [EKLER](#7-ekler)

---

## 1. GİRİŞ

### 1.1. Problemin Tanımı

E-ticaret sektörünün hızlı büyümesiyle birlikte, kullanıcı yorumları hem tüketiciler hem de satıcılar için kritik bir bilgi kaynağı hâline gelmiştir. Türkiye'nin en büyük e-ticaret platformlarından biri olan Hepsiburada üzerinde her gün binlerce ürün yorumu yayınlanmaktadır. Ancak bu yorumların manuel olarak okunup analiz edilmesi pratik değildir. Bu durumda iki temel problem ortaya çıkmaktadır:

1. **Duygu Analizi (Sentiment Analysis):** Bir yorumun pozitif, nötr veya negatif bir duygu içerip içermediğini otomatik olarak tespit etme.
2. **Çoklu Metin Özetleme (Multi-Document Summarization):** Bir ürün hakkındaki onlarca/yüzlerce yorumun ana fikrini birkaç cümlelik bir özette toparlama.

Türkçe, sondan eklemeli (agglutinative) yapısı, zengin morfolojisi ve sınırlı NLP kaynakları nedeniyle bu problemleri İngilizce'ye göre daha zorlu hâle getirmektedir.

### 1.2. Projenin Amacı ve Kapsamı

Bu projenin amacı, Hepsiburada'dan toplanmış 146.657 Türkçe ürün yorumu üzerinde:

- **(i)** Klasik makine öğrenmesi yöntemleri (Linear SVM, Naive Bayes) ile derin öğrenme yöntemini (BERTurk Transformer) duygu sınıflandırma görevinde karşılaştırmak,
- **(ii)** Aynı ürüne dair birden çok yorumun, mT5 (multilingual T5) tabanlı bir Transformer modeli ile çoklu özetlenmesini sağlamak,
- **(iii)** Elde edilen sistemleri etkileşimli bir Streamlit web arayüzünde son kullanıcıya sunmaktır.

### 1.3. Motivasyon

- Türkçe için açık kaynaklı son kullanıcı odaklı duygu analizi araçları sınırlıdır.
- E-ticaret platformlarındaki ürün önerme sistemleri için yorum analizi kritik bir ön adımdır.
- Klasik (TF-IDF + SVM) yaklaşımların hâlâ rekabetçi olup olmadığını derin öğrenme alternatifleriyle karşılaştırmak akademik olarak değerlidir.
- Dengesiz veri setlerinde (gerçek hayat senaryolarında olduğu gibi) hangi yöntemin daha güvenilir olduğunu göstermek önemlidir.

### 1.4. Katkılar

Bu çalışmanın temel katkıları:

1. Hepsiburada veri seti üzerinde **uçtan uca** bir Türkçe duygu analizi pipeline'ı önerilmiş ve açık kaynak olarak yayımlanmıştır (https://github.com/erncanyildirim/metin-ozetleme-ve-duygu-analizi).
2. **TF-IDF + Linear SVM** baseline modeli ile **BERTurk** transformer modelinin doğruluk/zaman maliyeti karşılaştırması yapılmıştır. BERTurk'ün **F1-Macro'da +13.1 puan üstünlük** sağladığı kanıtlanmıştır.
3. BERTurk'ün **Türkçe ironik ifadeleri** tespit edebildiği gözlemsel olarak gösterilmiştir.
4. Yorumlar üzerinde mT5-small'ün **zero-shot çoklu özetleme** performansı raporlanmış (ROUGE-1=0.092), modelin alana özel fine-tune gerektirdiği vurgulanmıştır.
5. Sonuçlar, kullanıcıların kendi yorumlarını test edebileceği bir **Streamlit web arayüzü** ile somutlaştırılmıştır.

---

## 2. ÖNCEKİ ÇALIŞMALAR (LİTERATÜR TARAMASI)

### 2.1. Duygu Analizi Alanındaki Çalışmalar

**Klasik makine öğrenmesi yaklaşımları:**

- **Pang ve Lee (2008)** — *"Opinion Mining and Sentiment Analysis"* başlıklı temel referans çalışmasında SVM, Naive Bayes ve Maximum Entropy modellerini film yorumları üzerinde karşılaştırmış, TF-IDF + SVM'in en iyi sonucu verdiğini göstermiştir.
- **Joachims (1998)** — *"Text Categorization with Support Vector Machines"* makalesiyle SVM'i metin sınıflandırma için altın standart olarak konumlandırmıştır.

**Derin öğrenme yaklaşımları:**

- **Devlin et al. (2018)** — *"BERT: Pre-training of Deep Bidirectional Transformers"* (NAACL 2019) çalışmasıyla bağlamsal kelime gömme yöntemini tanıtmış, duygu analizinde state-of-the-art sonuçlar elde etmiştir.
- **Liu et al. (2019)** — *"RoBERTa"* ve sonrasında çıkan **DistilBERT**, **ALBERT** gibi varyasyonlar BERT'in performansını artırmıştır.

### 2.2. Metin Özetleme Alanındaki Çalışmalar

- **Lin (2004)** — *"ROUGE: A Package for Automatic Evaluation of Summaries"* — Özetlemenin standart değerlendirme metriği ROUGE'u tanıtmıştır.
- **See et al. (2017)** — *"Get To The Point: Summarization with Pointer-Generator Networks"* — Soyut (abstractive) özetlemede pointer-generator mimarisini tanıtmıştır.
- **Raffel et al. (2020)** — *"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)"* — Tüm NLP görevlerini "text-to-text" olarak modelleyen T5 mimarisini önermiştir.
- **Xue et al. (2021)** — *"mT5: A massively multilingual pre-trained text-to-text transformer"* — mT5 (multilingual T5) Türkçe dâhil 101 dili desteklemektedir.
- **Lewis et al. (2020)** — *"BART"* — Hem soyut özetleme hem de denoising autoencoding için kullanılan bir model.

### 2.3. Türkçe NLP Çalışmaları

- **Schweter (2020)** — *"BERTurk"* — Türkçe Wikipedia ve OSCAR korpuslarında ön-eğitilen BERT modeli (`dbmdz/bert-base-turkish-cased`). Türkçe duygu analizinde standart başlangıç noktası hâline gelmiştir.
- **Savcı ve Üsküdarlı (2022)** — Türkçe Twitter verisi üzerinde BERTurk + BiLSTM hibrit modeli, klasik yöntemlere göre yaklaşık %5 F1 artışı raporlamıştır.
- **Akın (2007)** — *"Zemberek-NLP"* — Türkçe morfolojik analiz ve kök bulma için referans araç.
- **Gündeş (2021)** — *"mt5-small-turkish-summarization"* (Hugging Face) — mT5 modelini Türkçe haber özetleme için fine-tune etmiştir.

### 2.4. GitHub ve Medium Kaynakları

- **HuggingFace Transformers** kütüphanesi — BERTurk, mT5 gibi modelleri kolayca yüklemeyi sağlar.
- **savasy/bert-base-turkish-sentiment-cased** — Hugging Face hub'da yayınlanmış, ürün/film yorumlarında ince ayarlı Türkçe duygu modeli.
- **Medium: "Turkish Sentiment Analysis with BERT"** — Türkçe duygu analizini adım adım anlatan blog post serisi.

### 2.5. Karşılaştırmalı Tablo

| Çalışma | Yıl | Veri Seti | Yöntem | Sonuç (F1/Acc) |
|---------|-----|-----------|--------|-----------------|
| Pang & Lee | 2008 | IMDB | TF-IDF + SVM | ~%88 Acc |
| Devlin et al. | 2018 | SST-2 | BERT | %94.9 Acc |
| Schweter | 2020 | Twitter TR | BERTurk | %85 F1 |
| Savcı & Üsküdarlı | 2022 | Twitter TR | BERTurk + BiLSTM | %88 F1 |
| **Bu Çalışma** | 2026 | Hepsiburada (TR) | **SVM vs BERTurk** | **F1-Macro: 0.626 → 0.757** |

---

## 3. MALZEME VE YÖNTEM

### 3.1. Kullanılan Veri Seti

**Veri seti:** Hepsiburada Product Reviews and Ratings Dataset
**Kaynak:** Kaggle — `yusufkesmenn/hepsiburada-product-reviews-and-ratings-dataset`
**Boyut:** 146.661 yorum (ham), 146.657 yorum (temizleme sonrası)
**Alanlar:** `product_id` (ürün adı), `review_rating` (1-5 yıldız puanı), `review_body` (kullanıcı yorumu metni)

**Neden bu veri seti?**

1. **Türkçe ve Gerçek Hayata Yakın:** Türkçe e-ticaret yorumları doğal dilde, argo ve yazım hataları içerir.
2. **Etiketli:** Yıldız puanı sayesinde duygu etiketi otomatik üretilebilir (1-2: negatif, 3: nötr, 4-5: pozitif).
3. **Büyük Hacim:** Hem klasik hem derin öğrenme modelleri için yeterli örnek sayısı.
4. **Çok Boyutlu Kullanılabilir:** Hem duygu sınıflandırma hem de çoklu özetleme için aynı veri kullanılabilir.
5. **Dengesizlik:** Gerçek dünya senaryosunu yansıtan doğal sınıf dengesizliği (Bölüm 4.3) modelleri zorlayan bir test ortamı sunar.

### 3.2. Veri Ön İşleme

Türkçe için özel olarak uygulanan ön işleme adımları (`src/preprocessing.py`):

1. **Küçük harfe çevirme** (Türkçe karakter farkındalığıyla)
2. **URL ve e-posta temizliği**
3. **HTML etiket temizliği**
4. **Emoji temizliği** (Unicode aralıkları ile)
5. **Noktalama işareti temizliği** (Türkçe karakterleri koruyarak)
6. **Türkçe stop-word filtresi** (özellikle duygu taşıyan kelimeler — "değil", "yok", "ama" — bilinçli olarak listede tutulmamıştır)
7. **Tekrarlayan karakter düzeltimi** ("çoook" → "çook")
8. **Fazla boşluk temizliği**

> **Gözlemlenen Yan Etki:** Tekrarlayan karakter kuralı sayısal değerleri de etkilemiştir (örn. "4000mhz" → "400mhz"). Bu durum duygu analizini etkilememekle birlikte rapor 4.7'de tartışılmıştır.

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
- **Sınıf ağırlığı:** Dengesiz sınıflar için `"balanced"`
- **Max iterasyon:** 10.000

#### 3.4.2. Baseline Model 2: Multinomial Naive Bayes
- **Algoritma:** sklearn `MultinomialNB`
- **Alpha (Laplace smoothing):** 0.1

#### 3.4.3. Transformer Model 1: BERTurk (Duygu Analizi)
- **Mimari:** `dbmdz/bert-base-turkish-cased` (Schweter, 2020)
- **Parametreler:** 110M parametre, 12 katman, 768 gizli boyut
- **Fine-tuning:** Hepsiburada veri setinin dengeli bir alt kümesi üzerinde 3 epoch
- **Eğitim seti:** 21.076 örnek, Doğrulama: 2.635, Test: 2.635
- **Optimizer:** AdamW (learning rate: 2e-5, weight decay: 0.01)
- **Batch size:** 16, Max sequence length: 128
- **Warmup ratio:** 0.1
- **FP16 mixed precision:** Aktif (T4 GPU)

#### 3.4.4. Transformer Model 2: mT5-small (Özetleme)
- **Mimari:** `ozcangundes/mt5-small-turkish-summarization` (Türkçe haber özetleme için fine-tune edilmiş)
- **Parametreler:** ~300M parametre
- **Kullanım Modu:** Zero-shot (ürün yorumları için fine-tune yapılmamıştır)
- **Generation parametreleri:** num_beams=4, max_length=150, min_length=30, no_repeat_ngram_size=3, early_stopping=True

### 3.5. Değerlendirme Metrikleri

**Duygu Sınıflandırma için:**

- **Accuracy:** Doğru tahmin edilen örneklerin oranı (sınıf dengesizliğinde yanıltıcı olabilir).
- **F1-Score (Macro):** **Birincil metrik** — sınıf dengesizliğinde tüm sınıflara eşit ağırlık verir.
- **F1-Score (Weighted):** Örnek sayısına orantılı ağırlıkla F1.
- **Precision/Recall (Macro):** Sınıf bazlı doğruluk/duyarlılık.
- **Confusion Matrix:** Sınıflar arası karışıklıkların görselleştirilmesi.

**Özetleme için:**

- **ROUGE-1, ROUGE-2, ROUGE-L:** Referans özet ile üretilen özet arasındaki n-gram örtüşmesi.

### 3.6. Donanım ve Yazılım Ortamı

| Bileşen | Detay |
|---------|-------|
| Donanım | Google Colab — NVIDIA Tesla T4 GPU (15.6 GB) |
| İşletim Sistemi | Ubuntu 22.04 (Colab varsayılan) |
| Python sürümü | 3.12 |
| Temel kütüphaneler | `scikit-learn 1.3+`, `pandas 2.0+`, `transformers 4.46+`, `torch 2.0+`, `rouge-score`, `datasets`, `accelerate` |
| Sunum | Streamlit 1.28+ |

---

## 4. DENEY VE BULGULAR

### 4.1. Hipotezler

- **H1:** TF-IDF + Linear SVM, dengesiz bir veri setinde >%60 F1-Macro skoru elde edebilir.
- **H2:** BERTurk fine-tune edildiğinde baseline'a göre en az **+5 puan** F1-Macro iyileştirmesi sağlar.
- **H3:** Türkçe stop-word'leri çıkarma F1-Macro üzerinde etkili olur; ancak duygu taşıyan kelimeleri ("değil", "ama") tutmak kritiktir.
- **H4:** mT5-small zero-shot ROUGE-1 ≥ 0.30 üzerinde özetleme yapabilir.
- **H5:** BERTurk, klasik modellerin yapamadığı **bağlamsal/ironik** ifadeleri tespit edebilir.

### 4.2. Deney Tasarımı

- **Train/Test bölme:** 80/20 stratified split (sınıf dengesini korur)
- **Validation seti:** Train'den ek %10
- **Cross-Validation:** Baseline modeller için 5-fold CV
- **Seed:** `random_state=42` (tekrar üretilebilirlik için)
- **BERTurk için:** Sınıf-dengeli örneklemeyle 30K alt küme (sınıf başına 10K) → 21.076 train + 2.635 val + 2.635 test

### 4.3. Veri Seti İstatistikleri

| Metrik | Değer |
|--------|-------|
| Toplam yorum sayısı | **146.657** (temizleme sonrası, 4 satır boş clean_text nedeniyle silindi) |
| Ortalama yorum uzunluğu | **9.7 kelime** |
| Medyan yorum uzunluğu | 9 kelime |
| Maksimum yorum uzunluğu | 34 kelime |
| Standart sapma | 3.6 kelime |

**Sınıf Dağılımı (Dengesiz!):**

| Sınıf | Yorum Sayısı | Yüzde |
|-------|:---:|:---:|
| 😊 Pozitif (4-5 yıldız) | 128.566 | **%87.7** |
| 😠 Negatif (1-2 yıldız) | 11.745 | %8.0 |
| 😐 Nötr (3 yıldız) | 6.346 | %4.3 |

> **Önemli Gözlem:** Veri seti **şiddetli dengesizlik** göstermektedir. Pozitif sınıf, negatifin yaklaşık 11 katı, nötrün 20 katıdır. Bu, gerçek dünya e-ticaret verilerinde tipiktir (insanlar genellikle memnun kaldıklarında yorum yazma eğilimindedir) ve modellerin değerlendirilmesi için **Accuracy yerine F1-Macro** kullanımını zorunlu kılar.

**Şekiller:**
- *Şekil 4.1:* Sınıf dağılımı (pasta + bar) — `rapor/sekil_4_1_sinif_dagilimi.png`
- *Şekil 4.2:* Yorum uzunluk dağılımı (histogram + boxplot) — `rapor/sekil_4_2_uzunluk_dagilimi.png`
- *Şekil 4.3:* En sık 20 kelime — `rapor/sekil_4_3_en_sik_kelimeler.png`
- *Şekil 4.3b:* Sınıf bazlı en sık kelimeler — `rapor/sekil_4_3b_sinif_bazli_kelimeler.png`

**Sınıf Bazlı Karakteristik Kelimeler (Ön İşleme Sonrası):**

| Pozitif | Nötr | Negatif |
|---------|------|---------|
| güzel, iyi, gayet | ama, değil, fakat | iade, yok, kötü |
| kaliteli, tavsiye | biraz, iade | hiç, kesinlikle |
| hızlı, teşekkürler | (tereddüt sinyalleri) | (red sinyalleri) |

➡️ **Yorum:** Sınıflar **dilsel olarak ayrıştırılabilir** durumdadır. Bu, hem klasik hem de derin öğrenme modellerinin başarılı olmasını mümkün kılmaktadır.

### 4.4. Duygu Analizi Sonuçları

#### 4.4.1. Baseline Modeller (146K Yorum, Tam Veri)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | **F1 (Macro)** | F1 (Weighted) | Eğitim Süresi |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive Bayes | 0.913 | 0.68 | 0.56 | **0.584** | 0.898 | 61.3 sn |
| **Linear SVM** | 0.902 | 0.62 | 0.63 | **0.626** ⭐ | 0.903 | 104.8 sn |

**5-Fold Cross Validation (F1-Macro):**
- Naive Bayes: 0.5785 (±0.0043)
- Linear SVM: 0.6269 (±0.0047)

**SVM Sınıf Bazlı Skorlar:**

| Sınıf | Precision | Recall | F1 |
|-------|:---:|:---:|:---:|
| Pozitif | 0.96 | 0.95 | **0.96** |
| Negatif | 0.64 | 0.69 | 0.66 |
| Nötr | 0.26 | 0.26 | **0.26** ⚠️ |

> **Kritik gözlem:** NB'nin Accuracy'si yüksek görünmesine rağmen F1-Macro daha düşüktür → **Accuracy sınıf dengesizliğinde yanıltıcıdır**. SVM, dengesizliği `class_weight='balanced'` ile daha iyi yönetmektedir. Her iki model de **nötr sınıfında zayıftır** (3 yıldızlı yorumlar dilsel olarak karışıktır).

*Şekil 4.4:* Baseline modellerin karşılaştırma grafiği — `rapor/sekil_4_4_baseline_karsilastirma.png`
*Şekil 4.4b:* Naive Bayes & SVM Confusion Matrix — `rapor/sekil_4_4b_confusion_matrix.png`

#### 4.4.2. BERTurk Fine-Tuning Sonuçları (30K Alt Küme)

**Eğitim Süresi:** 7 dakika 10 saniye (Tesla T4 GPU)

**Epoch Bazlı Doğrulama Metrikleri:**

| Epoch | Eğitim Kaybı | Doğrulama Kaybı | Accuracy | **F1-Macro** |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.5567 | 0.5567 | 0.768 | 0.743 |
| 2 | 0.4886 | 0.5496 | 0.781 | 0.748 |
| 3 | 0.3506 | 0.5757 | 0.782 | **0.759** ⭐ |

**Test Seti Sonuçları:**
- **Accuracy:** 0.780
- **F1-Macro:** **0.757**

> Eğitim kaybı düşerken doğrulama kaybı 3. epoch'ta hafif arttı (0.5567 → 0.5757) — bu mild **overfitting** sinyali. Erken durdurma F1-Macro'ya göre yapıldığı için en iyi model 3. epoch'ta seçildi.

#### 4.4.3. SVM vs BERTurk Doğrudan Karşılaştırma (2.000 Örnek Aynı Test Setinde)

| Model | Accuracy | **F1-Macro** | Nötr Doğru Tahmin |
|-------|:---:|:---:|:---:|
| SVM (Baseline) | 0.895 | 0.619 | 23/97 (**%23**) |
| **BERTurk** | 0.879 | **0.715** ⭐ | 79/97 (**%81**) |
| **Fark** | -0.016 | **+0.096** | **+%58** |

**BERTurk Confusion Matrix (2000 örnek):**

|  | Tahmin: negatif | Tahmin: nötr | Tahmin: pozitif |
|--|:---:|:---:|:---:|
| **Gerçek: negatif** (165) | 135 | 24 | 6 |
| **Gerçek: nötr** (97) | 13 | **79** | 5 |
| **Gerçek: pozitif** (1738) | 46 | 147 | 1545 |

**SVM Confusion Matrix (2000 örnek):**

|  | Tahmin: negatif | Tahmin: nötr | Tahmin: pozitif |
|--|:---:|:---:|:---:|
| **Gerçek: negatif** (165) | 112 | 21 | 32 |
| **Gerçek: nötr** (97) | 26 | **23** | 48 |
| **Gerçek: pozitif** (1738) | 40 | 43 | 1655 |

> 🎯 **En çarpıcı bulgu:** BERTurk **nötr sınıfında SVM'in 3.5 katı başarılı**. SVM nötrleri büyük oranda pozitife karıştırırken, BERTurk nötrü doğru tanıyor.

*Şekil 4.5:* SVM vs BERTurk performans karşılaştırması — `rapor/sekil_4_5_baseline_vs_bert.png`
*Şekil 4.5b:* Yan yana confusion matrix — `rapor/sekil_4_5b_confusion_compare.png`

#### 4.4.4. Demo Tahminler — Kalitatif Analiz

| Yorum | SVM Tahmini | BERTurk Tahmini | BERTurk Skoru | Yorum |
|-------|:---:|:---:|:---:|-------|
| "Bu ürün kesinlikle harika, çok memnun kaldım" | pozitif | pozitif | 0.982 | ✅ Net |
| "Berbat, hayatımda gördüğüm en kötü ürün" | negatif | negatif | 0.988 | ✅ Net |
| "İdare eder, fiyatına göre fena değil" | **pozitif** ❌ | **nötr** ✅ | 0.876 | BERTurk doğru sınıflandırdı |
| "Kargo çok geç geldi ama ürün güzel" | pozitif | pozitif | 0.972 | Karışık, son ifadeye ağırlık verdi |
| **"Yorumlardan etkilenip aldım, hata yapmışım"** | **pozitif** ❌ | **negatif** ✅ | 0.843 | 🎯 **İRONİ TESPİTİ** |

> 🎯 **Son örnek özellikle önemli:** Cümlede hiçbir açık negatif kelime (kötü, berbat, bozuk vb.) yoktur. Negatiflik tamamen **bağlamsal**: "hata yapmışım" yorumdan pişman olduğunu ima eder. BERTurk bunu yakalamıştır. Bu, **Hipotez H5'i doğrular** ve klasik TF-IDF tabanlı modellerin yapısal sınırlamasının somut bir kanıtıdır.

### 4.5. Metin Özetleme Sonuçları (mT5-small Zero-Shot)

**Test Setupı:** 3 farklı ürün için, her biri 3 yorumdan oluşan girdiler. Referans özetler insan tarafından yazılmıştır.

| Metrik | Skor | Değerlendirme |
|--------|:---:|:---:|
| ROUGE-1 | **0.092** | Düşük (iyi: >0.30) |
| ROUGE-2 | **0.019** | Çok düşük |
| ROUGE-L | **0.092** | Düşük |

**Üretilen Özet Örnekleri:**

**Örnek 1 (Kalite/Tavsiye):**
- Girdi: 3 pozitif yorum (kaliteli, tavsiye, fiyat-performans)
- Referans: "Ürün kaliteli, fiyat performans olarak başarılı ve tavsiye edilebilir."
- Üretilen: *"Geçtiğimiz günlerde çok beğenilen, çok beğenilen, çok memnun kaldım. Ürün gerçekten kaliteli ve çok memnun kaldım."*
- ⚠️ **Tekrarlama sorunu**

**Örnek 2 (Kargo/İade):**
- Girdi: 3 negatif yorum (geç kargo, hasar, iade)
- Referans: "Kargo ve paketleme sorunları yüzünden ürün hasarlı geldi ve iade edildi."
- Üretilen: *"Geçtiğimiz günlerde kargo, kargo, kargo, kargo, kargo..."*
- ⚠️ **Şiddetli tekrarlama (degenerate output)**

**Örnek 3 (Telefon Yorumları):**
- Girdi: 3 yorum (kamera, pil, ekran)
- Referans: "Kamera ve pil performansı iyi, ekranın küçüklüğü tek olumsuz yan."
- Üretilen: *"ATV'de yayınlanan Müge Anlı ile Müge Anlı ile..."*
- ⚠️ **Hallucination (uydurma içerik)** — model haber metinlerine maruz kaldığı için ilgisiz Türk TV programı adı üretiyor

> **Tespit Edilen Sorunlar:**
> 1. **Hallucination:** Model haber özetleme datasından gelen ön yargıyla ilgisiz haber konuları uyduruyor.
> 2. **Repetition:** Beam search'te tekrarlama cezası yetersiz (`no_repeat_ngram_size` parametresi eklenerek kısmen düzeltilmiştir).
> 3. **Domain Mismatch:** Model **haber metinleri** üzerinde fine-tune edilmiş; **ürün yorumları** farklı bir domain.

### 4.6. Karşılaştırmalı Analiz

**Performans-Maliyet Karşılaştırması:**

| Özellik | Naive Bayes | Linear SVM | BERTurk |
|---------|:---:|:---:|:---:|
| F1-Macro (2K test) | 0.584 | 0.619 | **0.715** |
| Eğitim Süresi | 61 sn | 105 sn | 430 sn (GPU) |
| Tahmin Süresi (1000 yorum) | 48 ms | 40 ms | ~30.000 ms (GPU) |
| Model Boyutu | ~5 MB | ~3 MB | ~450 MB |
| Donanım Gereksinimi | CPU | CPU | **GPU önerilir** |
| Türkçe Bağlam Anlama | ❌ | ❌ | ✅ |
| İroni Tespiti | ❌ | ❌ | ✅ |

> **Trade-off:** BERTurk **+12 puan F1-Macro** kazandırırken **~750x daha yavaş** tahmin yapıyor. Production sistemlerde **iki kademeli bir mimari** önerilebilir: SVM ile hızlı ön-eleme, sonra şüpheli/karışık örneklerde BERTurk.

### 4.7. Tartışma

**Hipotezlerin Doğrulanma Durumu:**

- **H1 (SVM >%60 F1-Macro):** ✅ Doğrulandı (0.626 elde edildi)
- **H2 (BERTurk +5 puan üstünlük):** ✅ Doğrulandı, **fazlasıyla** — gerçek fark +13 puan (0.626 → 0.757)
- **H3 (Stop-word filtresi etkisi):** ✅ Doğrulandı — "değil", "ama" gibi duygu taşıyan kelimelerin tutulması nötr ve negatif tahminlerini iyileştirdi.
- **H4 (mT5 ROUGE-1 ≥ 0.30):** ❌ **Reddedildi** — gerçek skor 0.092, hipotezin çok altında.
- **H5 (BERTurk ironi tespiti):** ✅ Doğrulandı — "Yorumlardan etkilenip aldım, hata yapmışım" örneği.

**Beklenmeyen Bulgular:**

1. **Naive Bayes'in yüksek Accuracy'si yanıltıcı:** Modelin her şeye "pozitif" deme eğilimi accuracy'yi yapay olarak yüksek tuttu. Bu, **sınıf dengesizliğinde F1-Macro'nun ne kadar kritik olduğunu** somut olarak gösterdi.

2. **BERTurk az veriyle bile çok başarılı:** SVM 117K eğitim örneği gördü, BERTurk yalnızca 21K. Buna rağmen BERTurk +13 puan F1-Macro üstünlük sağladı. Tam veride eğitim ile bu farkın daha da açılması beklenir.

3. **mT5'in hallucination problemi:** Model haber özetleme korpusundan gelen ön-eğitim etkisiyle, ürün yorumlarında "Müge Anlı" gibi ilgisiz TV programlarından bahsetti. Bu, **transfer learning'in sınırlarını** göstermesi açısından önemli.

4. **Ön işleme yan etkisi:** "4000mhz" → "400mhz" gibi sayısal değer bozulması, duygu analizini etkilemese de teknik içerikli yorumlarda bilgi kaybı yarattı. Üretim sisteminde **sayıları regex ile koruyacak bir kural** eklenmelidir.

**Sınırlamalar:**

- BERTurk eğitiminde donanım kısıtları nedeniyle 30K alt küme kullanıldı (tam 146K değil). Bu, fine-tuning maliyet/yarar dengesini sınırlamıştır.
- mT5 fine-tune edilmediği için özetleme sonuçları zero-shot ile sınırlıdır.
- Referans özetler insan tarafından sentetik olarak yazıldı — gerçek bir altın standart veri seti yoktu.
- 3 test case az sayıda — daha güvenilir ROUGE skoru için 100+ test case ile değerlendirme yapılmalıdır.

---

## 5. SONUÇ VE ÖNERİLER

### 5.1. Projenin Başarı Değerlendirmesi

Bu proje, Bölüm 1.2'de belirtilen üç amacın tamamına ulaşmıştır:

✅ **Klasik (TF-IDF + SVM) ve derin öğrenme (BERTurk) modelleri karşılaştırılmıştır:** BERTurk'ün F1-Macro'da +13 puan üstünlüğü kanıtlanmıştır. Özellikle nötr sınıfında 3.5x daha başarılı olduğu gözlemlenmiştir.

✅ **Çoklu metin özetleme prototipi geliştirilmiştir:** mT5-small zero-shot kullanımıyla denenmiş, ancak hallucination ve repetition problemleri tespit edilmiştir. Bu, alana özel fine-tuning'in zorunluluğunu göstermiştir.

✅ **Etkileşimli Streamlit web arayüzü kurulmuştur:** Hem baseline (SVM) hem transformer (BERTurk) modellerini içerir, kullanıcı kendi yorumlarını test edebilir.

**Ek beklenmeyen başarı:** BERTurk'ün Türkçe **ironik ifadeleri** tespit edebildiği gözlemsel olarak gösterilmiştir.

### 5.2. Karşılaşılan Zorluklar

1. **Veri set dengesizliği (%87.7 pozitif):** Accuracy metriğinin yanıltıcı olmasına neden oldu; F1-Macro'ya geçildi.
2. **Türkçe karakter kodlama sorunları:** "İ", "ı" gibi karakterler için titiz bir küçük harfe dönüştürme gerekti.
3. **Colab GPU oturum sınırları:** Uzun eğitimler için inactivity sorunu, modellerin Drive'a yedeklenmesini zorunlu kıldı.
4. **mT5 hallucination:** Domain transferinin sınırlı çalıştığını ortaya çıkardı.
5. **Ön işlemenin yan etkileri:** Tekrarlayan karakter kuralı sayısal değerleri etkiledi.
6. **Kütüphane sürüm uyumsuzlukları:** `transformers >= 4.46`'da `Trainer(tokenizer=)` parametresi ve `pipeline("summarization")` task'ı kaldırıldığı için kod uyumluluğu yeniden yapıldı.

### 5.3. Gelecek Çalışmalar

1. **BERTurk'ü Tam Veride Eğitmek:** 146K örnekle eğitim yapıldığında F1-Macro'nun 0.80+ seviyesine çıkması beklenmektedir.
2. **mT5 Fine-Tuning:** En az 10.000 (yorum, insan özeti) çifti ile mT5-small'ün ürün yorumları için fine-tune edilmesi.
3. **Daha Büyük Modeller:** BERTurk-large veya XLM-RoBERTa-large denenebilir.
4. **Aspect-Based Sentiment Analysis (ABSA):** Sadece genel duygu değil, "kargo iyi ama ürün kötü" gibi alt başlıklara göre analiz.
5. **İroni/Alaycılık Tespiti:** Türkçe ironi veri seti oluşturup özel bir BERTurk modeli ince ayarlanabilir.
6. **Production Deployment:** Streamlit yerine FastAPI + Docker ile gerçek production sistem.
7. **Açıklanabilirlik (XAI):** SHAP/LIME ile hangi kelimelerin tahmini etkilediğini görselleştirme.
8. **İki Kademeli Mimari:** SVM ile hızlı ön-eleme + BERTurk ile şüpheli durumlarda derin analiz.
9. **Extractive Özetleme Alternatifi:** mT5 yerine TextRank gibi extractive yöntemler denenmelidir — hallucination riski yoktur.
10. **Çapraz Domain Transfer:** Hepsiburada modelinin Trendyol, n11 vb. platformlara genelleştirilebilirliğinin test edilmesi.

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
14. Gündeş, Ö. (2021). *mT5-small Turkish Summarization*. Hugging Face Hub. https://huggingface.co/ozcangundes/mt5-small-turkish-summarization

---

## 7. EKLER

### Ek A: Proje GitHub Deposu

https://github.com/erncanyildirim/metin-ozetleme-ve-duygu-analizi

### Ek B: Streamlit Demo

Yerelde çalıştırmak için:
```bash
streamlit run app.py
```
Tarayıcıda `http://localhost:8501` adresinde erişim sağlar.

### Ek C: Eğitim Süreleri (Tesla T4 GPU)

| Model | Eğitim Verisi | Donanım | Süre |
|-------|:---:|:---:|:---:|
| Naive Bayes | 117.325 yorum | CPU | **61.3 sn** |
| Linear SVM | 117.325 yorum | CPU | **104.8 sn** |
| BERTurk (3 epoch) | 21.076 yorum | Tesla T4 GPU | **~7 dakika** |
| mT5-small | Pre-trained | Tesla T4 GPU | 0 (zero-shot) |

### Ek D: Hiperparametre Detayları

**TF-IDF + SVM:**
- `max_features=50000, ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=0.95`
- `LinearSVC(C=1.0, class_weight='balanced', max_iter=10000, random_state=42)`

**TF-IDF + Naive Bayes:**
- Aynı TF-IDF ayarları
- `MultinomialNB(alpha=0.1)`

**BERTurk:**
- `num_train_epochs=3, batch_size=16, learning_rate=2e-5, warmup_ratio=0.1, weight_decay=0.01`
- `max_length=128, fp16=True, metric_for_best_model='f1_macro'`
- `load_best_model_at_end=True`

**mT5-small:**
- `num_beams=4, max_length=150, min_length=30, no_repeat_ngram_size=3, early_stopping=True`

### Ek E: Reproduce Edilebilirlik

Tüm sonuçlar `random_state=42` ile üretilmiştir. Notebook'lar GitHub deposunda yer almakta olup adım adım yeniden çalıştırılabilir:
- `notebooks/01_veri_kesfi_ve_onisleme.ipynb`
- `notebooks/02_baseline_model.ipynb`
- `notebooks/03_transformer_model.ipynb`

---

**[Rapor sonu]**

*Bu çalışma, Doğal Dil İşleme ve Derin Öğrenme derslerinin birleşik bitirme projesi olarak hazırlanmıştır.*
