"""
Görselleştirme ve yardımcı fonksiyonlar modülü.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


# Türkçe karakter desteği için font ayarı
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100


def plot_confusion_matrix(cm: np.ndarray, 
                          title: str = "Confusion Matrix",
                          save_path: Optional[str] = None):
    """
    Confusion matrix görselleştirir.
    
    Argümanlar:
        cm -- Confusion matrix (2x2 numpy array)
        title -- Grafik başlığı
        save_path -- Kayıt yolu (None ise sadece gösterir)
    """
    plt.figure(figsize=(8, 6))
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negatif (0)', 'Pozitif (1)'],
                yticklabels=['Negatif (0)', 'Pozitif (1)'],
                annot_kws={'size': 14})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Gercek Deger', fontsize=12)
    plt.xlabel('Tahmin Edilen Deger', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix kaydedildi: {save_path}")
    
    plt.close()


def plot_feature_comparison(comparison_df: pd.DataFrame,
                            save_path: Optional[str] = None):
    """
    Özellik seçimi karşılaştırmasını görselleştirir.
    
    Argümanlar:
        comparison_df -- Karşılaştırma DataFrame'i
        save_path -- Kayıt yolu
    """
    # Her üç yöntemde de seçilen özellikleri say
    filter_count = (comparison_df['Filtreleme'] == '✓').sum()
    wrapper_count = (comparison_df['Sarmalayıcı'] == '✓').sum()
    embedded_count = (comparison_df['Gömülü'] == '✓').sum()
    
    # Ortak özellikleri say
    common_2 = (comparison_df['Ortak_Sayısı'] >= 2).sum()
    common_3 = (comparison_df['Ortak_Sayısı'] == 3).sum()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sol: Yöntem başına özellik sayısı
    methods = ['Filtreleme', 'Sarmalayici', 'Gomulu']
    counts = [filter_count, wrapper_count, embedded_count]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    axes[0].bar(methods, counts, color=colors, edgecolor='black')
    axes[0].set_title('Yontem Basina Secilen Ozellik Sayisi', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Ozellik Sayisi')
    axes[0].set_ylim(0, max(counts) + 2)
    
    for i, v in enumerate(counts):
        axes[0].text(i, v + 0.5, str(v), ha='center', fontsize=12, fontweight='bold')
    
    # Sağ: Ortak özellikler
    labels = ['Yalnizca 1 Yontem', '2 Yontemde Ortak', '3 Yontemde Ortak']
    only_1 = len(comparison_df) - common_2
    only_2 = common_2 - common_3
    values = [only_1, only_2, common_3]
    colors_pie = ['#bdc3c7', '#f39c12', '#27ae60']
    
    axes[1].pie(values, labels=labels, autopct='%1.1f%%', colors=colors_pie,
                explode=(0, 0.05, 0.1), shadow=True)
    axes[1].set_title('Ozellik Ortaklik Orani', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Ozellik karsilastirmasi kaydedildi: {save_path}")
    
    plt.close()


def plot_results_comparison(all_results: Dict,
                            save_path: Optional[str] = None):
    """
    Tüm yöntemlerin sonuçlarını karşılaştırır.
    
    Argümanlar:
        all_results -- Tüm sonuçlar sözlüğü
        save_path -- Kayıt yolu
    """
    methods = list(all_results.keys())
    
    # Metrikleri çıkar
    accuracies = [all_results[m]['test_results']['accuracy'] for m in methods]
    f1_scores = [all_results[m]['test_results']['f1'] for m in methods]
    n_features = [all_results[m]['n_features'] for m in methods]
    train_times = [all_results[m]['cv_results']['training_time'] for m in methods]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sol üst: Accuracy
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(methods)))
    bars1 = axes[0, 0].bar(range(len(methods)), accuracies, color=colors, edgecolor='black')
    axes[0, 0].set_title('Test Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(range(len(methods)))
    axes[0, 0].set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
    axes[0, 0].set_ylim(0.5, max(accuracies) + 0.05)
    
    for bar, acc in zip(bars1, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.4f}', ha='center', fontsize=10)
    
    # Sağ üst: F1-Score
    bars2 = axes[0, 1].bar(range(len(methods)), f1_scores, color=colors, edgecolor='black')
    axes[0, 1].set_title('Test F1-Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_xticks(range(len(methods)))
    axes[0, 1].set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
    axes[0, 1].set_ylim(0.5, max(f1_scores) + 0.05)
    
    for bar, f1 in zip(bars2, f1_scores):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{f1:.4f}', ha='center', fontsize=10)
    
    # Sol alt: Özellik sayısı
    bars3 = axes[1, 0].bar(range(len(methods)), n_features, color=colors, edgecolor='black')
    axes[1, 0].set_title('Ozellik Sayisi', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Ozellik Sayisi')
    axes[1, 0].set_xticks(range(len(methods)))
    axes[1, 0].set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
    
    for bar, n in zip(bars3, n_features):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(n), ha='center', fontsize=10)
    
    # Sağ alt: Eğitim süresi
    bars4 = axes[1, 1].bar(range(len(methods)), train_times, color=colors, edgecolor='black')
    axes[1, 1].set_title('Egitim Suresi', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Sure (saniye)')
    axes[1, 1].set_xticks(range(len(methods)))
    axes[1, 1].set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
    
    for bar, t in zip(bars4, train_times):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{t:.2f}s', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sonuc karsilastirmasi kaydedildi: {save_path}")
    
    plt.close()


def create_results_table(all_results: Dict) -> pd.DataFrame:
    """
    Sonuç tablosu oluşturur.
    
    Argümanlar:
        all_results -- Tüm sonuçlar sözlüğü
        
    Dönüş:
        Sonuç tablosu DataFrame
    """
    rows = []
    for method, result in all_results.items():
        row = {
            'Yontem': method,
            'Ozellik_Sayisi': result['n_features'],
            'Accuracy': f"{result['test_results']['accuracy']:.4f}",
            'F1_Skoru': f"{result['test_results']['f1']:.4f}",
            'Precision': f"{result['test_results']['precision']:.4f}",
            'Recall': f"{result['test_results']['recall']:.4f}",
            'Egitim_Suresi': f"{result['cv_results']['training_time']:.2f}s"
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_results_to_file(all_results: Dict,
                         selected_features_dict: Dict,
                         comparison_df: pd.DataFrame,
                         filepath: str):
    """
    Tüm sonuçları metin dosyasına kaydeder.
    
    Argümanlar:
        all_results -- Tüm sonuçlar sözlüğü
        selected_features_dict -- Seçilen özellikler sözlüğü
        comparison_df -- Özellik karşılaştırma DataFrame'i
        filepath -- Kayıt dosya yolu
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MAKINE OGRENMESI ODEV 2 - OZELLIK SECIMI SONUCLARI\n")
        f.write("Online News Popularity Veri Kumesi\n")
        f.write("="*70 + "\n\n")
        
        # Sonuç tablosu
        f.write("1. SONUC TABLOSU\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Yontem':<20} {'Ozellik':<10} {'Accuracy':<12} {'F1-Skoru':<12} {'Egitim Suresi':<15}\n")
        f.write("-"*70 + "\n")
        
        for method, result in all_results.items():
            f.write(f"{method:<20} {result['n_features']:<10} "
                   f"{result['test_results']['accuracy']:<12.4f} "
                   f"{result['test_results']['f1']:<12.4f} "
                   f"{result['cv_results']['training_time']:<15.2f}s\n")
        
        f.write("-"*70 + "\n\n")
        
        # Her yöntem için detaylı sonuçlar
        f.write("2. DETAYLI SONUCLAR\n")
        f.write("-"*70 + "\n\n")
        
        for method, result in all_results.items():
            f.write(f"[{method}]\n")
            f.write(f"  Ozellik Sayisi: {result['n_features']}\n")
            f.write(f"  Cross-Validation Sonuclari:\n")
            f.write(f"    Accuracy:  {result['cv_results']['accuracy']['mean']:.4f} "
                   f"(+/- {result['cv_results']['accuracy']['std']:.4f})\n")
            f.write(f"    F1-Score:  {result['cv_results']['f1']['mean']:.4f} "
                   f"(+/- {result['cv_results']['f1']['std']:.4f})\n")
            f.write(f"  Test Sonuclari:\n")
            f.write(f"    Accuracy:  {result['test_results']['accuracy']:.4f}\n")
            f.write(f"    F1-Score:  {result['test_results']['f1']:.4f}\n")
            f.write(f"    Precision: {result['test_results']['precision']:.4f}\n")
            f.write(f"    Recall:    {result['test_results']['recall']:.4f}\n")
            f.write(f"  Egitim Suresi: {result['cv_results']['training_time']:.2f} saniye\n\n")
        
        # En iyi yöntem için confusion matrix
        best_method = max(all_results.keys(), 
                         key=lambda m: all_results[m]['test_results']['accuracy'])
        cm = all_results[best_method]['test_results']['confusion_matrix']
        
        f.write("3. EN BASARILI YONTEM ICIN CONFUSION MATRIX\n")
        f.write("-"*70 + "\n")
        f.write(f"Yontem: {best_method}\n\n")
        f.write("              Tahmin: 0    Tahmin: 1\n")
        f.write(f"  Gercek: 0      {cm[0,0]:5d}        {cm[0,1]:5d}\n")
        f.write(f"  Gercek: 1      {cm[1,0]:5d}        {cm[1,1]:5d}\n\n")
        
        tn, fp, fn, tp = cm.ravel()
        f.write(f"  True Negative (TN):  {tn}\n")
        f.write(f"  False Positive (FP): {fp}\n")
        f.write(f"  False Negative (FN): {fn}\n")
        f.write(f"  True Positive (TP):  {tp}\n\n")
        
        # Seçilen özellikler
        f.write("4. SECILEN OZELLIKLER\n")
        f.write("-"*70 + "\n\n")
        
        for method, features in selected_features_dict.items():
            f.write(f"[{method}] ({len(features)} ozellik):\n")
            for i, feat in enumerate(features, 1):
                f.write(f"  {i:2d}. {feat}\n")
            f.write("\n")
        
        # Ortak özellikler
        f.write("5. ORTAK OZELLIKLER\n")
        f.write("-"*70 + "\n")
        
        if 'Filtreleme' in selected_features_dict and \
           'Sarmalayıcı' in selected_features_dict and \
           'Gömülü' in selected_features_dict:
            common = set(selected_features_dict['Filtreleme']) & \
                     set(selected_features_dict['Sarmalayıcı']) & \
                     set(selected_features_dict['Gömülü'])
            
            f.write(f"\nHer uc yontemde ortak ozellikler ({len(common)} adet):\n")
            for feat in sorted(common):
                f.write(f"  - {feat}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Sonuclar basariyla kaydedildi.\n")
    
    print(f"Sonuclar kaydedildi: {filepath}")


def plot_learning_curve(train_scores: List[float],
                        val_scores: List[float],
                        title: str = "Learning Curve",
                        save_path: Optional[str] = None):
    """
    Öğrenme eğrisi çizer.
    
    Argümanlar:
        train_scores -- Eğitim skorları
        val_scores -- Doğrulama skorları
        title -- Grafik başlığı
        save_path -- Kayıt yolu
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_scores) + 1)
    
    plt.plot(epochs, train_scores, 'b-', label='Egitim', linewidth=2)
    plt.plot(epochs, val_scores, 'r-', label='Dogrulama', linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Skor', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Ogrenme egrisi kaydedildi: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test
    import numpy as np
    
    # Örnek confusion matrix
    cm = np.array([[3000, 500], [400, 3100]])
    plot_confusion_matrix(cm, title="Test Confusion Matrix")
    
    print("Utils modulu test edildi.")
