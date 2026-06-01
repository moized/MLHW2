"""
Değerlendirme scripti.
Eğitilmiş modeli yükler ve test seti üzerinde değerlendirir.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

# Uyarıları kapat
warnings.filterwarnings('ignore')

from dataset import prepare_dataset
from model import LogisticRegressionModel
from utils import plot_confusion_matrix


def main():
    """Ana değerlendirme fonksiyonu."""
    
    print("="*70)
    print("MODEL DEĞERLENDİRME")
    print("="*70)
    
    # Dosya yolları
    data_path = "dataset/OnlineNewsPopularity.csv"
    model_path = "results/best_model.joblib"
    
    # Veri dosyasını kontrol et
    if not os.path.exists(data_path):
        print(f"\nHATA: Veri dosyası bulunamadı: {data_path}")
        sys.exit(1)
    
    # Model dosyasını kontrol et
    if not os.path.exists(model_path):
        print(f"\nHATA: Model dosyası bulunamadı: {model_path}")
        print("Lütfen önce 'python train.py' komutunu çalıştırın.")
        sys.exit(1)
    
    # Veriyi hazırla
    print("\nVeri yükleniyor...")
    data = prepare_dataset(data_path, threshold=1400, test_size=0.2)
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Modeli yükle
    print("\nModel yükleniyor...")
    model = LogisticRegressionModel.load_model(model_path)
    
    # Eğitim bilgilerini göster
    if model.cv_scores:
        print("\n" + "="*60)
        print("EĞİTİM BİLGİLERİ")
        print("="*60)
        print(f"Cross-Validation Accuracy: {model.cv_scores['accuracy']['mean']:.4f}")
        print(f"Cross-Validation F1-Score: {model.cv_scores['f1']['mean']:.4f}")
        print(f"Eğitim süresi: {model.training_time:.2f} saniye")
    
    # Test seti üzerinde değerlendir
    print("\n" + "="*60)
    print("TEST SETİ DEĞERLENDİRMESİ")
    print("="*60)
    
    # Modelin kullandığı özellikleri al
    model_features = model.model.feature_names_in_
    
    # Test setini modelin özelliklerine göre filtrele
    X_test_subset = X_test[model_features]
    
    results = model.evaluate(X_test_subset, y_test)
    
    print(f"\nTest Seti Sonuçları:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"              Tahmin: 0    Tahmin: 1")
    print(f"  Gerçek: 0      {cm[0,0]:5d}        {cm[0,1]:5d}")
    print(f"  Gerçek: 1      {cm[1,0]:5d}        {cm[1,1]:5d}")
    
    # True Negative, False Positive, False Negative, True Positive
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negative (TN):  {tn}")
    print(f"  False Positive (FP): {fp}")
    print(f"  False Negative (FN): {fn}")
    print(f"  True Positive (TP):  {tp}")
    
    # Örnek tahminler
    print("\n" + "="*60)
    print("ÖRNEK TAHMİNLER (İlk 10 test örneği)")
    print("="*60)
    
    y_pred = results['predictions'][:10]
    y_prob = results['probabilities'][:10]
    y_true = y_test.values[:10]
    
    print(f"{'#':>3} {'Gerçek':>8} {'Tahmin':>8} {'Olasılık':>10} {'Durum':>10}")
    print("-"*45)
    
    for i in range(10):
        status = "✓ Doğru" if y_true[i] == y_pred[i] else "✗ Yanlış"
        label_true = "Popüler" if y_true[i] == 1 else "Değil"
        label_pred = "Popüler" if y_pred[i] == 1 else "Değil"
        print(f"{i+1:3d} {label_true:>8} {label_pred:>8} {y_prob[i]:>10.4f} {status:>10}")
    
    # Confusion matrix görselleştir
    plot_confusion_matrix(
        cm,
        title="Test Seti Confusion Matrix",
        save_path="results/eval_confusion_matrix.png"
    )
    
    print("\nDeğerlendirme tamamlandı!")
    
    return results


if __name__ == "__main__":
    main()
