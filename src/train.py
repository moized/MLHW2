"""
Ana eğitim scripti.
Tüm özellik seçimi yöntemlerini uygular ve sonuçları karşılaştırır.
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Modülleri import et
from dataset import prepare_dataset
from feature_selection import (
    filter_method_pearson,
    wrapper_method_rfe,
    embedded_method_random_forest,
    find_optimal_features_rfe,
    compare_selected_features
)
from model import train_and_evaluate, find_best_regularization
from utils import (
    plot_confusion_matrix,
    plot_feature_comparison,
    plot_results_comparison,
    save_results_to_file,
    create_results_table
)


def main():
    """Ana eğitim fonksiyonu."""
    
    print("="*70)
    print("MAKINE ÖĞRENMESI ÖDEV 2 - ÖZELLİK SEÇİMİ")
    print("Online News Popularity Veri Kümesi")
    print("="*70)
    
    # Veri dosyası yolu
    data_path = "dataset/OnlineNewsPopularity.csv"
    
    # Veri dosyasını kontrol et
    if not os.path.exists(data_path):
        print(f"\nHATA: Veri dosyası bulunamadı: {data_path}")
        print("Lütfen veri setini Kaggle'dan indirip 'dataset/' klasörüne koyun:")
        print("https://www.kaggle.com/datasets/thehapyone/uci-online-news-popularity-data-set")
        sys.exit(1)
    
    # Sonuçları saklamak için sözlük
    all_results = {}
    selected_features_dict = {}
    
    # =========================================================================
    # 1. VERİ HAZIRLAMA
    # =========================================================================
    print("\n" + "#"*70)
    print("# ADIM 1: VERİ HAZIRLAMA")
    print("#"*70)
    
    data = prepare_dataset(data_path, threshold=1400, test_size=0.2)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    print(f"\nToplam özellik sayısı: {len(feature_names)}")
    
    # =========================================================================
    # 2. ÖZELLİK SEÇİMİ
    # =========================================================================
    print("\n" + "#"*70)
    print("# ADIM 2: ÖZELLİK SEÇİMİ")
    print("#"*70)
    
    n_features_to_select = 15
    
    # 2a. Filtreleme Yöntemi (Pearson Korelasyonu)
    filter_features, filter_corr_df = filter_method_pearson(
        X_train, y_train, n_features=n_features_to_select
    )
    selected_features_dict['Filtreleme'] = filter_features
    
    # 2b. Sarmalayıcı Yöntem (RFE)
    wrapper_features, wrapper_ranking = wrapper_method_rfe(
        X_train, y_train, n_features=n_features_to_select
    )
    selected_features_dict['Sarmalayıcı'] = wrapper_features
    
    # 2c. Gömülü Yöntem (Random Forest)
    embedded_features, importance_df = embedded_method_random_forest(
        X_train, y_train, n_features=n_features_to_select
    )
    selected_features_dict['Gömülü'] = embedded_features
    
    # Özellik karşılaştırması
    comparison_df = compare_selected_features(
        filter_features, wrapper_features, embedded_features
    )
    
    # =========================================================================
    # 3. REGULARIZASYON PARAMETRESI ARAMA
    # =========================================================================
    print("\n" + "#"*70)
    print("# ADIM 3: REGULARİZASYON PARAMETRESİ ARAMA")
    print("#"*70)
    
    best_C = find_best_regularization(X_train, y_train)
    
    # =========================================================================
    # 4. MODEL EĞİTİMİ VE DEĞERLENDİRME
    # =========================================================================
    print("\n" + "#"*70)
    print("# ADIM 4: MODEL EĞİTİMİ VE DEĞERLENDİRME")
    print("#"*70)
    
    # 4a. Tüm özelliklerle eğitim
    all_results['Tüm Özellikler'] = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        selected_features=None,
        method_name="Tüm Özellikler",
        C=best_C
    )
    
    # 4b. Filtreleme yöntemi özellikleriyle eğitim
    all_results['Filtreleme'] = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        selected_features=filter_features,
        method_name="Filtreleme (Pearson)",
        C=best_C
    )
    
    # 4c. Sarmalayıcı yöntem özellikleriyle eğitim
    all_results['Sarmalayıcı'] = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        selected_features=wrapper_features,
        method_name="Sarmalayıcı (RFE)",
        C=best_C
    )
    
    # 4d. Gömülü yöntem özellikleriyle eğitim
    all_results['Gömülü'] = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        selected_features=embedded_features,
        method_name="Gömülü (Random Forest)",
        C=best_C
    )
    
    # =========================================================================
    # 5. EN İYİ YÖNTEM İLE OPTİMUM ÖZELLİK SAYISI
    # =========================================================================
    print("\n" + "#"*70)
    print("# ADIM 5: OPTİMUM ÖZELLİK SAYISI ARAMA")
    print("#"*70)
    
    # En iyi yöntemi bul
    best_method = max(
        ['Filtreleme', 'Sarmalayıcı', 'Gömülü'],
        key=lambda m: all_results[m]['test_results']['accuracy']
    )
    print(f"\nEn başarılı yöntem: {best_method}")
    
    # En iyi yöntemle optimum özellik sayısını bul
    if best_method == 'Sarmalayıcı':
        optimal_n, optimal_features, search_results = find_optimal_features_rfe(
            X_train, y_train, min_features=5, max_features=30, step=5
        )
    else:
        # Diğer yöntemler için de RFE kullan (en yaygın)
        optimal_n, optimal_features, search_results = find_optimal_features_rfe(
            X_train, y_train, min_features=5, max_features=30, step=5
        )
    
    selected_features_dict['Optimal'] = optimal_features
    
    # Optimum özelliklerle eğitim
    all_results['Optimal'] = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        selected_features=optimal_features,
        method_name=f"Optimal ({optimal_n} özellik)",
        C=best_C
    )
    
    # =========================================================================
    # 6. SONUÇLARI KAYDET
    # =========================================================================
    print("\n" + "#"*70)
    print("# ADIM 6: SONUÇLARI KAYDET")
    print("#"*70)
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Sonuç tablosu oluştur
    results_table = create_results_table(all_results)
    print("\n" + "="*70)
    print("SONUÇ TABLOSU")
    print("="*70)
    print(results_table.to_string(index=False))
    
    # Sonuçları dosyaya kaydet
    save_results_to_file(
        all_results, 
        selected_features_dict, 
        comparison_df,
        os.path.join(results_dir, "metrics.txt")
    )
    
    # En iyi model için confusion matrix
    # Not: Ödev gereği 3 özellik seçimi yöntemi arasından en başarılısı seçilir
    # "Tüm Özellikler" baseline olarak değerlendirilir
    feature_selection_methods = ['Filtreleme', 'Sarmalayıcı', 'Gömülü', 'Optimal']
    best_fs_method = max(
        feature_selection_methods,
        key=lambda m: all_results[m]['test_results']['accuracy']
    )
    print(f"\nÖzellik seçimi yöntemleri arasında en başarılı: {best_fs_method}")
    print(f"Test Accuracy: {all_results[best_fs_method]['test_results']['accuracy']:.4f}")
    
    # Confusion matrix görselleştir - En iyi özellik seçimi yöntemi için
    plot_confusion_matrix(
        all_results[best_fs_method]['test_results']['confusion_matrix'],
        title=f"Confusion Matrix - {best_fs_method}",
        save_path=os.path.join(results_dir, "confusion_matrix.png")
    )
    
    # Sonuç karşılaştırma grafiği
    plot_results_comparison(
        all_results,
        save_path=os.path.join(results_dir, "results_comparison.png")
    )
    
    # Özellik karşılaştırma tablosu
    comparison_df.to_csv(
        os.path.join(results_dir, "feature_comparison.csv"), 
        index=False,
        encoding='utf-8-sig'
    )
    
    # Sonuç tablosunu CSV olarak kaydet
    results_table.to_csv(
        os.path.join(results_dir, "results_table.csv"),
        index=False,
        encoding='utf-8-sig'
    )
    
    # En iyi özellik seçimi modelini kaydet
    all_results[best_fs_method]['model'].save_model(
        os.path.join(results_dir, "best_model.joblib")
    )
    
    print(f"\nTüm sonuçlar '{results_dir}/' klasörüne kaydedildi.")
    print("\nEğitim tamamlandı!")
    
    return all_results, selected_features_dict


if __name__ == "__main__":
    main()
