"""
Veri yükleme ve ön işleme modülü.
Online News Popularity veri kümesini yükler ve işler.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    CSV dosyasından veri kümesini yükler.
    
    Argümanlar:
        filepath -- Veri dosyasının yolu
        
    Dönüş:
        DataFrame olarak veri kümesi
    """
    print(f"Veri yükleniyor: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Veri boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df


def preprocess_data(df: pd.DataFrame, threshold: int = 1400) -> tuple:
    """
    Veri ön işleme adımlarını uygular:
    - URL ve timedelta gibi ayırt edici olmayan özellikleri kaldırır
    - shares sütununu ikili sınıfa dönüştürür (>= threshold ise 1, değilse 0)
    
    Argümanlar:
        df -- Ham veri DataFrame'i
        threshold -- Popülerlik eşik değeri (default 1400)
        
    Dönüş:
        X (özellikler), y (hedef değişken), feature_names (özellik isimleri)
    """
    print("\nVeri ön işleme başlıyor...")
    
    # Boşlukları sütun isimlerinden önce temizle
    df.columns = df.columns.str.strip()
    
    # Kaldırılacak sütunlar (ayırt edici bilgi içermeyenler)
    # Not: Ödev tablosunda 59 özellik belirtilmiş, bu yüzden sadece url kaldırılıyor
    # timedelta da non-predictive olmasına rağmen, ödev gereksinimlerine uyum için tutuldu
    columns_to_drop = ['url']
    
    # Mevcut sütunları kontrol et ve kaldır
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        print(f"Kaldırılan sütunlar: {existing_cols_to_drop}")
    
    # Hedef değişkeni ayır
    if 'shares' not in df.columns:
        raise ValueError("'shares' sütunu bulunamadı!")
    
    # İkili sınıflandırma için hedef değişkeni oluştur
    y = (df['shares'] >= threshold).astype(int)
    print(f"\nHedef değişken dağılımı (eşik={threshold}):")
    print(f"  Popüler (1): {y.sum()} ({100*y.mean():.2f}%)")
    print(f"  Popüler Değil (0): {len(y) - y.sum()} ({100*(1-y.mean()):.2f}%)")
    
    # Özellikleri ayır (shares hariç)
    X = df.drop(columns=['shares'])
    feature_names = X.columns.tolist()
    
    print(f"\nÖzellik sayısı: {len(feature_names)}")
    
    return X, y, feature_names


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
               random_state: int = 42) -> tuple:
    """
    Veriyi eğitim ve test setlerine böler.
    
    Argümanlar:
        X -- Özellik matrisi
        y -- Hedef değişken
        test_size -- Test seti oranı (default 0.2)
        random_state -- Rastgelelik tohumu
        
    Dönüş:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nVeri bölme tamamlandı:")
    print(f"  Eğitim seti: {len(X_train)} örnek")
    print(f"  Test seti: {len(X_test)} örnek")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Özellikleri standartlaştırır (Z-score normalizasyonu).
    
    Argümanlar:
        X_train -- Eğitim özellikleri
        X_test -- Test özellikleri
        
    Dönüş:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("\nÖzellikler standartlaştırıldı (Z-score normalizasyonu)")
    
    return X_train_scaled, X_test_scaled, scaler


def prepare_dataset(filepath: str, threshold: int = 1400, 
                    test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Tüm veri hazırlama adımlarını uygular.
    
    Argümanlar:
        filepath -- Veri dosyasının yolu
        threshold -- Popülerlik eşik değeri
        test_size -- Test seti oranı
        random_state -- Rastgelelik tohumu
        
    Dönüş:
        Hazırlanmış veri sözlüğü
    """
    # Veri yükleme
    df = load_data(filepath)
    
    # Ön işleme
    X, y, feature_names = preprocess_data(df, threshold)
    
    # Bölme
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Ölçekleme
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'X_train_original': X_train,
        'X_test_original': X_test
    }


if __name__ == "__main__":
    # Test
    data_path = "dataset/OnlineNewsPopularity.csv"
    if os.path.exists(data_path):
        data = prepare_dataset(data_path)
        print(f"\nVeri hazırlama tamamlandı!")
        print(f"Eğitim seti boyutu: {data['X_train'].shape}")
        print(f"Test seti boyutu: {data['X_test'].shape}")
    else:
        print(f"Veri dosyası bulunamadı: {data_path}")
        print("Lütfen veri setini 'dataset/' klasörüne indirin.")
