## Açıklama

Makine Öğrenmesi (BLM5110) dersi kapsamında Scikit-learn kullanılarak geliştirilen
özellik seçimi projesi. Online News Popularity veri kümesi kullanılarak haberlerin
popülerliğini tahmin eden bir ikili sınıflandırma modeli geliştirilmiştir.

Projede üç farklı özellik seçimi yöntemi uygulanmıştır:
- Filtreleme Yöntemi (Filter Method): Pearson Korelasyonu
- Sarmalayıcı Yöntem (Wrapper Method): Recursive Feature Elimination (RFE)
- Gömülü Yöntem (Embedded Method): Random Forest Özellik Önemi

## Gereksinimler

Gerekli kütüphaneleri yüklemek için:

```
pip install -r requirements.txt
```

Veya manuel olarak:
```
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

## Veri Seti

`dataset/OnlineNewsPopularity.csv` dosyası Online News Popularity veri kümesini içerir.
Veri kümesi 39.644 haber makalesi ve 59 özellik içermektedir.

Veri kümesine aşağıdaki linkten erişebilirsiniz:
https://www.kaggle.com/datasets/thehapyone/uci-online-news-popularity-data-set

İndirilen CSV dosyasını `dataset/` klasörüne koyun.

## Çalıştırma

### Eğitim:

Tüm özellik seçimi yöntemlerini uygulamak ve modelleri eğitmek için:

```
python train.py
```

Bu komut:
- Veriyi yükler ve ön işler (%80 eğitim, %20 test)
- Hedef değişkeni ikili sınıfa dönüştürür (shares >= 1400 → 1, değilse 0)
- Üç farklı özellik seçimi yöntemi uygular (her biri için 15 özellik)
- 5-fold cross validation kullanarak Lojistik Regresyon eğitir
- Regularizasyon parametresi (C) optimizasyonu yapar
- Optimum özellik sayısını arar
- Sonuçları `results/` klasörüne kaydeder

### Değerlendirme:

Eğitilmiş modeli test seti üzerinde değerlendirmek için:

```
python eval.py
```

Bu komut:
- Kaydedilmiş modeli yükler
- Test seti üzerinde tahmin yapar
- Accuracy, precision, recall ve f-score metriklerini hesaplar
- Confusion matrix'i gösterir
- Örnek tahminleri gösterir

## Dosya Düzeni

```
MLHW2/
  ├── train.py              # Ana eğitim scripti
  ├── eval.py               # Değerlendirme scripti
  ├── model.py              # Lojistik Regresyon model sınıfı
  ├── dataset.py            # Veri yükleme ve ön işleme
  ├── feature_selection.py  # Özellik seçimi yöntemleri
  ├── metrics.py            # Değerlendirme metrikleri
  ├── utils.py              # Görselleştirme fonksiyonları
  ├── requirements.txt      # Gerekli Python kütüphaneleri
  ├── README.txt            # Bu dosya
  ├── dataset/
  │   └── OnlineNewsPopularity.csv  # Veri seti
  └── results/
      ├── metrics.txt              # Detaylı metrik sonuçları
      ├── results_table.csv        # Sonuç tablosu
      ├── confusion_matrix.png     # En iyi model confusion matrix
      ├── results_comparison.png   # Yöntem karşılaştırma grafiği
      ├── feature_comparison.csv   # Özellik karşılaştırma tablosu
      └── best_model.joblib        # Kaydedilmiş model
```

## İşlem Adımları

### 1. Veri Ön İşleme
- Ayırt edici bilgi içermeyen `url` sütunu çıkarılır
- Hedef değişken `shares` için medyan değeri (1400) eşik olarak alınır
- shares >= 1400 ise 1 (Popüler), değilse 0 (Popüler Değil) olarak etiketlenir
- Veri %80 eğitim, %20 test olarak bölünür
- Eğitim verisi üzerinde 5-fold cross validation uygulanır

### 2. Özellik Seçimi Yöntemleri

**Filtreleme Yöntemi (Filter Method):**
- Pearson korelasyonu kullanarak hedef değişken ile korelasyonu hesaplar
- Yüksek korelasyonlu özellik çiftlerini eler (multicollinearity azaltma)
- En yüksek korelasyonlu 15 özelliği seçer

**Sarmalayıcı Yöntem (Wrapper Method):**
- Recursive Feature Elimination (RFE) kullanır
- Lojistik Regresyon ile birlikte çalışır
- İteratif olarak en az önemli özellikleri eler
- En önemli 15 özelliği seçer

**Gömülü Yöntem (Embedded Method):**
- Random Forest sınıflandırıcı (100 ağaç) kullanır
- Özellik önem skorlarını hesaplar
- En yüksek önem skoruna sahip 15 özelliği seçer

### 3. Model Eğitimi ve Değerlendirme
- Tüm yöntemler için Lojistik Regresyon kullanılır
- 5-fold Stratified Cross Validation ile eğitim yapılır
- Aşırı öğrenme kontrolü için regularizasyon parametresi (C) optimize edilir
- Test seti üzerinde final değerlendirme yapılır

## Model Detayları

- **Model:** Lojistik Regresyon (Scikit-learn)
- **Optimizasyon:** LBFGS solver
- **Regularizasyon:** L2 (Ridge) - C parametresi {0.001, 0.01, 0.1, 1, 10, 100}
- **Cross-Validation:** 5-fold Stratified
- **Veri Bölme:** %80 eğitim, %20 test (stratified)
- **Karar Eşiği:** 0.5

## Değerlendirme Metrikleri

Model performansı şu metriklerle ölçülür:

- **Accuracy:** (TP + TN) / (TP + TN + FP + FN)
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)
- **Confusion Matrix:** Gerçek vs Tahmin sınıf karşılaştırması

## Sonuç Tablosu

```
Yöntem           Özellik Sayısı   Doğruluk(Accuracy)   F1-Skoru   Eğitim Süresi
Tüm Özellikler   59               0.6562               0.6849     3.49s
Filtreleme       15               0.6436               0.6776     0.54s
Sarmalayıcı      15               0.6464               0.6779     1.19s
Gömülü           15               0.6396               0.6823     0.79s
Optimal (RFE)    30               0.6529               0.6827     2.06s
```

## Notlar

- Tüm işlemler için Scikit-learn kütüphanesi kullanılmıştır
- Görselleştirmeler Matplotlib ve Seaborn ile oluşturulmuştur
- Model ağırlıkları ve sonuçlar `results/` klasörüne otomatik kaydedilir
- Kod modüler yapıda organize edilmiştir

## Yazar
Mohammed Izedin Mohammed STUDENT_ID_REMOVED
Makine Öğrenmesi Dersi - 2. Ödev
2025-2026 Güz Yarıyılı
