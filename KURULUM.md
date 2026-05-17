# 🚀 Adım Adım Kurulum & Çalıştırma Rehberi

Bu rehber, projeyi sıfırdan kurmanı ve **Google Colab** üzerinde çalıştırmanı sağlar. Hiçbir Python/ML bilgin olmasa bile bu adımları takip ederek projeyi tamamlayabilirsin.

> 🎯 **Toplam süre tahmini:** 1.5 - 2 saat (modelleri eğitmek dâhil)

---

## 📋 İçindekiler

1. [Kaggle Hesabı Oluşturma](#1-kaggle-hesab%C4%B1-olu%C5%9Fturma)
2. [GitHub'a Projeyi Yükleme](#2-github-projeyi-y%C3%BCkleme)
3. [Google Colab'da Notebook Açma](#3-google-colabda-notebook-a%C3%A7ma)
4. [GPU'yu Aktifleştirme](#4-gpuyu-aktifle%C5%9Ftirme)
5. [Aşama 1: Veri Keşfi](#5-a%C5%9Fama-1-veri-ke%C5%9Ffi)
6. [Aşama 2: Baseline Modeller](#6-a%C5%9Fama-2-baseline-modeller)
7. [Aşama 3: Transformer Modelleri](#7-a%C5%9Fama-3-transformer-modelleri)
8. [Aşama 4: Streamlit Arayüzü](#8-a%C5%9Fama-4-streamlit-aray%C3%BCz%C3%BC)
9. [Aşama 5: Rapor Yazımı](#9-a%C5%9Fama-5-rapor-yaz%C4%B1m%C4%B1)
10. [Sık Karşılaşılan Sorunlar](#10-s%C4%B1k-kar%C5%9F%C4%B1la%C5%9F%C4%B1lan-sorunlar)

---

## 1. Kaggle Hesabı Oluşturma

Veri setini indirebilmek için Kaggle API anahtarına ihtiyacın var. **Ücretsiz** ve 3 dakika sürer.

### 1.1. Hesap Aç
1. https://www.kaggle.com adresine git
2. Sağ üstte **Register** → "Register with Email" (veya Google ile gir)
3. Adın, soyadın ve şifreni girip kayıt ol
4. E-postaya gelen doğrulama linkine tıkla

### 1.2. API Anahtarı Al
1. Sağ üstte avatarına tıkla → **Settings**
2. Aşağıya kaydır → **API** bölümü
3. **Create New Token** butonuna bas
4. `kaggle.json` adında bir dosya bilgisayarına inecek

> 💡 Bu dosyada `{"username": "...", "key": "..."}` formatında veriler var. Bu dosyayı Colab'a yükleyeceksin (Notebook 1'in 1. hücresinde otomatik buton açılıyor).

### 1.3. Veri Setine Erişim
1. https://www.kaggle.com/datasets/yusufkesmenn/hepsiburada-product-reviews-and-ratings-dataset adresine git
2. Sağ üstte **"I understand and accept"** olabilir, kabul et (varsa)
3. Artık API ile indirilebilir

---

## 2. GitHub'a Projeyi Yükleme

Eğer projeyi henüz GitHub'a yüklemediysen:

### 2.1. GitHub Repository Aç
1. https://github.com adresine git, giriş yap
2. Sağ üstte **+** → **New repository**
3. Repository adı: `metin-ozetleme-ve-duygu-analizi`
4. **Public** seç (Colab'dan erişebilmek için)
5. **Create repository**

### 2.2. Lokal Projeyi Push Et
Terminal'de proje klasöründe:

```bash
git remote add origin https://github.com/KULLANICI_ADIN/metin-ozetleme-ve-duygu-analizi.git
git push -u origin main
```

> ⚠️ Eğer remote zaten varsa hata alırsın, `git remote -v` ile kontrol et.

---

## 3. Google Colab'da Notebook Açma

### 3.1. Colab'a Git
1. https://colab.research.google.com adresine git
2. Google hesabınla giriş yap (Gmail varsa hazır)

### 3.2. Notebook'u Aç (2 yöntem)

**Yöntem A — GitHub'dan doğrudan:**
1. Colab açıldığında çıkan menüde **GitHub** sekmesi
2. Arama kutusuna repo URL'sini yapıştır
3. `notebooks/01_veri_kesfi_ve_onisleme.ipynb` aç

**Yöntem B — Manuel yükleme:**
1. **Dosya → Notebook yükle**
2. `notebooks/01_veri_kesfi_ve_onisleme.ipynb` dosyasını seç

---

## 4. GPU'yu Aktifleştirme

> ⚠️ **Transformer notebook (03) için MUTLAKA GPU gerekli.** Baseline notebook (02) için isteğe bağlı.

1. Üst menüde **Çalışma Zamanı** (Runtime)
2. **Çalışma Zamanı Türünü Değiştir** (Change runtime type)
3. **Donanım hızlandırıcı (Hardware accelerator):** **T4 GPU** seç
4. **Kaydet** (Save)

> 💡 Ücretsiz Colab'da haftada ~15-30 saat GPU veriliyor. Bittiyse 24 saat bekle veya başka hesap kullan.

---

## 5. Aşama 1: Veri Keşfi

**Notebook:** `01_veri_kesfi_ve_onisleme.ipynb`
**Süre:** ~5 dakika
**GPU gerekli mi:** Hayır

### Yapacakların:

1. **0. Hücre — Kurulum:** Sadece Colab'daysan ÇALIŞTIR. GitHub'dan repoyu klonlar.

2. **1. Hücre — Kaggle API:** Çalıştır → açılan butonla **kaggle.json**'ı yükle.

3. **2. Hücre — İndirme:** Veri setini indirir (~50MB, 1-2 dakika).

4. **3. Hücre — EDA:** Veri istatistikleri.

5. **4-7. Hücreler:** Grafikler oluşturulur ve `rapor/sekil_*.png` olarak kaydedilir.

6. **Son Hücre:** `data/hepsiburada_clean.csv` üretilir.

> 🎯 **Rapor için gerekli çıktılar:**
> - Veri seti istatistikleri (toplam yorum, sınıf dağılımı, ortalama uzunluk)
> - `rapor/sekil_4_1_sinif_dagilimi.png`
> - `rapor/sekil_4_2_uzunluk_dagilimi.png`
> - `rapor/sekil_4_3_en_sik_kelimeler.png`

### Önemli: Colab'da Drive'a Yedekleme

Veri seti büyük olduğu için Colab oturumu kapatılınca silinir. **Mutlaka Drive'a yedekle:**

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/nlp_projesi
!cp data/hepsiburada_clean.csv /content/drive/MyDrive/nlp_projesi/
```

Sonraki notebooklarda bu dosyayı tekrar kullanabilirsin:
```python
!cp /content/drive/MyDrive/nlp_projesi/hepsiburada_clean.csv data/
```

---

## 6. Aşama 2: Baseline Modeller

**Notebook:** `02_baseline_model.ipynb`
**Süre:** ~10-15 dakika
**GPU gerekli mi:** Hayır (sklearn CPU yeterli)

### Yapacakların:

1. **0. Hücre:** Colab kurulumu + Drive'dan veri çek (eğer önceki adımda yüklediysen)

2. **1. Hücre:** Veriyi yükle, ön işle, train/test böl

3. **2. Hücre — Naive Bayes:**
   - 5-fold cross validation
   - Test seti değerlendirmesi
   - Model kaydı: `models/naive_bayes.joblib`

4. **3. Hücre — Linear SVM:**
   - Aynı işlemler
   - Model kaydı: `models/svm_(linearsvc).joblib`

5. **4-7. Hücreler:** Karşılaştırma tabloları, confusion matrix grafikleri.

> 🎯 **Rapor için:**
> - Tablo 4.4.1 (Baseline sonuçları) — `comparison` DataFrame çıktısı
> - `rapor/sekil_4_4_baseline_karsilastirma.png`
> - `rapor/sekil_4_4b_confusion_matrix.png`

### Modelleri Drive'a Yedekle:
```python
!cp -r models /content/drive/MyDrive/nlp_projesi/
```

---

## 7. Aşama 3: Transformer Modelleri

**Notebook:** `03_transformer_model.ipynb`
**Süre:** ~20-40 dakika (GPU ile, 30K örnek için)
**GPU gerekli mi:** **EVET, MUTLAKA**

### ÖNEMLİ — Önce GPU'yu Aç!

Çalışma Zamanı → T4 GPU seçtiğinden emin ol. İlk hücrede CUDA olduğunu doğrula:
```
🖥️  CUDA mevcut mu : True
   GPU adı       : Tesla T4
   GPU belleği   : 15.0 GB
```

### Yapacakların:

1. **0. Hücre:** Kurulum + GPU kontrolü

2. **1. Hücre — BERTurk Fine-Tuning:**
   - ⚠️ Bu hücre **15-30 dakika** sürer (Colab T4 + 30K örnek)
   - İlk eğitimde `sample_size=30000` öneririm
   - Tatmin edici sonuç alırsan `sample_size=None` ile tam veri ile eğit

3. **2. Hücre:** Demo tahminler — modelin nasıl çalıştığını gör

4. **3. Hücre — Karşılaştırma:** SVM ile aynı test setinde performans

5. **4. Hücre — mT5 Özetleme:** Birden fazla yorumu özetle

6. **5. Hücre — ROUGE:** Özetleme metrikleri

7. **6. Hücre:** Modelleri Drive'a yedekle

> 🎯 **Rapor için:**
> - Tablo 4.4.2 (BERTurk sonuçları)
> - Tablo 4.4.3 (SVM vs BERTurk)
> - Tablo 4.5 (ROUGE skorları)
> - `rapor/sekil_4_5_baseline_vs_bert.png`
> - `rapor/sekil_4_5b_confusion_compare.png`

### BERTurk Modelini Yedeklemek (önemli!)

Model ~440MB. Drive'a yedeklemezsen Colab oturumu kapanınca kaybolur:
```python
!cp -r models/berturk-sentiment /content/drive/MyDrive/nlp_projesi/
```

---

## 8. Aşama 4: Streamlit Arayüzü

**Dosya:** `app.py`
**Çalıştırma yeri:** **LOKAL** (Mac'inde) — Colab'da Streamlit çalıştırmak zor.

### 8.1. Lokal Kurulum (Mac)

Terminal'de proje klasörüne git:
```bash
cd /Users/eren/Desktop/metin-ozetleme-ve-duygu-analizi

# Sanal ortam oluştur (önerilen)
python3 -m venv venv
source venv/bin/activate

# Bağımlılıkları kur
pip install -r requirements.txt
```

### 8.2. Modelleri Drive'dan İndir

Colab'da eğittiğin modelleri Drive'dan indir:

1. https://drive.google.com → `nlp_projesi/models` klasörünü indir
2. İndirdiğin `models/` klasörünü proje kök dizinine kopyala
3. Klasör yapısı şöyle olmalı:
```
metin-ozetleme-ve-duygu-analizi/
├── models/
│   ├── svm_(linearsvc).joblib
│   ├── naive_bayes.joblib
│   └── berturk-sentiment/
│       ├── config.json
│       ├── model.safetensors
│       └── ...
```

### 8.3. Çalıştır

```bash
streamlit run app.py
```

Tarayıcıda otomatik olarak http://localhost:8501 açılır.

### Test Senaryoları:

**Duygu Analizi sekmesinde:**
- "Bu ürün gerçekten harika, çok memnun kaldım" → Pozitif beklenir
- "Kargo geç geldi, ürün hasarlıydı" → Negatif beklenir
- "Fiyatına göre idare eder" → Nötr beklenir

**Metin Özetleme sekmesinde:**
Bir ürün için 5-6 yorum girip özet üretmesini iste.

> 🎯 **Rapor için:** Streamlit ekran görüntüleri al, `rapor/` klasörüne koy. Bunlar **Ek B** (Streamlit Demo) bölümüne girecek.

---

## 9. Aşama 5: Rapor Yazımı

Rapor iskeleti `rapor/RAPOR.md` dosyasında hazır. Şimdi **Bölüm 4 (Deney ve Bulgular)** kısmını gerçek sayılarla doldurman gerek.

### 9.1. Doldurulacak Yerler

`RAPOR.md` içinde `[.]` veya `[X]` gibi placeholder'ları kendi sonuçlarınla değiştir:

- **Bölüm 4.3:** Veri seti istatistikleri (Notebook 1'in son hücresi)
- **Bölüm 4.4:** Duygu analizi sonuçları (Notebook 2 ve 3)
- **Bölüm 4.5:** Özetleme sonuçları (Notebook 3)
- **Bölüm 5.1:** Başarı değerlendirmesi

### 9.2. Word'e Dönüştürme

Markdown raporu Word'e dönüştürmek için (Mac'te):

**Yöntem A — Pandoc (komut satırı):**
```bash
brew install pandoc
pandoc rapor/RAPOR.md -o rapor/RAPOR.docx
```

**Yöntem B — Online araç:**
- https://word2md.com/ veya https://stackedit.io kullan
- Markdown'ı yapıştır → Word olarak indir

**Yöntem C — VSCode eklentisi:**
- "Markdown All in One" eklentisini kur → Export to PDF/Word

### 9.3. Görselleri Eklerken

Tüm `rapor/sekil_*.png` dosyalarını Word'e drag-drop ile ekle.

---

## 10. Sık Karşılaşılan Sorunlar

### ❌ "Kaggle 401 Unauthorized"
- `~/.kaggle/kaggle.json` yanlış izinlerle. `chmod 600 ~/.kaggle/kaggle.json` çalıştır.

### ❌ "CUDA out of memory"
- Notebook 3'te `batch_size=16` → **`batch_size=8`** yap.
- Veya `sample_size=30000` → daha küçük yap (örn. 10000).

### ❌ "ModuleNotFoundError: No module named 'src'"
- Colab'da yanlış klasördesin. `%cd metin-ozetleme-ve-duygu-analizi` ile gir.

### ❌ Colab oturumu kapandı, modeller kayboldu
- Drive'a yedekleme yapmadın. Bir dahaki sefere mutlaka yedekle.
- Modelleri tekrar eğitmek zorundasın.

### ❌ Streamlit "Model bulunamadı" diyor
- `models/` klasöründe .joblib dosyaları yok. Colab'dan indirip yerine koy.

### ❌ Hocaya teslim ederken "BERT'i ben mi eğittim?" sorusu
- **EVET**, sen eğittin. Notebook 3 zaten eğitiyor. Sadece "model dosyasını Colab'da ürettim" demek yeterli.

### ❌ Veri seti çok büyük, bilgisayarım donuyor
- Tam veri yerine `sample_size` ile çalış (30K-50K yeterli).
- Veya tam veriyi Colab'da kullan, lokalde sadece Streamlit testi yap.

---

## 🎯 Tüm Süreç — Zaman Çizelgesi

| Gün | Görev | Süre |
|-----|-------|------|
| Gün 1 | Kaggle hesabı + Colab kurulumu + Notebook 1 | 1 saat |
| Gün 1 | Notebook 2 (Baseline) | 30 dk |
| Gün 2 | Notebook 3 (Transformer — GPU eğitimi) | 1-2 saat |
| Gün 2 | Streamlit lokal test | 30 dk |
| Gün 3 | Rapor doldurma + grafik ekleme | 2-3 saat |
| Gün 3 | Word formatı + ekran görüntüleri + sunum | 1 saat |

**Toplam:** ~8 saat (3 güne yayılırsa rahat).

---

## 📞 Yardım Lazımsa

- **Kod hatası:** Hatanın tam metnini bana ver, çözüm bulalım.
- **Rapor içeriği:** Hangi bölümü daha ayrıntılı yazmak istiyorsan söyle.
- **Sunum:** Final sunumu için ayrı bir PowerPoint hazırlanabilir.

🚀 Başarılar!
