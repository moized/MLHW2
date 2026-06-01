"""
Özellik seçimi modülü.
Filtreleme, Sarmalayıcı ve Gömülü yöntemlerini uygular.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple


def filter_method_pearson(X: pd.DataFrame, y: pd.Series, 
                          n_features: int = 15,
                          correlation_threshold: float = 0.8) -> Tuple[List[str], pd.DataFrame]:
    """
    Filtreleme Yöntemi: Pearson Korelasyonu kullanarak özellik seçimi.
    
    1. Hedef ile korelasyonu hesapla
    2. Yüksek korelasyonlu özellik çiftlerini ele (multicollinearity)
    3. En iyi n_features özelliği seç
    
    Argümanlar:
        X -- Özellik matrisi
        y -- Hedef değişken
        n_features -- Seçilecek özellik sayısı
        correlation_threshold -- Özellikler arası korelasyon eşiği
        
    Dönüş:
        selected_features -- Seçilen özellik isimleri listesi
        correlation_df -- Korelasyon değerleri DataFrame'i
    """
    print("\n" + "="*60)
    print("FİLTRELEME YÖNTEMİ (Pearson Korelasyonu)")
    print("="*60)
    
    # Hedef ile korelasyonları hesapla
    correlations = X.corrwith(y).abs()
    
    # Özellikler arası korelasyon matrisini hesapla
    feature_corr = X.corr().abs()
    
    # Yüksek korelasyonlu özellik çiftlerini bul ve birini ele
    features_to_keep = list(X.columns)
    
    for i in range(len(feature_corr.columns)):
        for j in range(i + 1, len(feature_corr.columns)):
            if feature_corr.iloc[i, j] > correlation_threshold:
                col_i = feature_corr.columns[i]
                col_j = feature_corr.columns[j]
                
                # Her iki özellik de hala listede mi kontrol et
                if col_i in features_to_keep and col_j in features_to_keep:
                    # Hedef ile düşük korelasyonlu olanı çıkar
                    if correlations[col_i] < correlations[col_j]:
                        features_to_keep.remove(col_i)
                    else:
                        features_to_keep.remove(col_j)
    
    print(f"Yüksek korelasyonlu özellikler elendikten sonra: {len(features_to_keep)} özellik")
    
    # Kalan özelliklerden hedef ile en yüksek korelasyonlu olanları seç
    remaining_corr = correlations[features_to_keep].sort_values(ascending=False)
    selected_features = remaining_corr.head(n_features).index.tolist()
    
    # Sonuçları DataFrame olarak hazırla
    correlation_df = pd.DataFrame({
        'Özellik': selected_features,
        'Hedef_Korelasyon': [correlations[f] for f in selected_features]
    })
    
    print(f"\nSeçilen en iyi {n_features} özellik:")
    for i, (feat, corr) in enumerate(zip(selected_features, correlation_df['Hedef_Korelasyon']), 1):
        print(f"  {i:2d}. {feat}: {corr:.4f}")
    
    return selected_features, correlation_df


def wrapper_method_rfe(X: pd.DataFrame, y: pd.Series, 
                       n_features: int = 15,
                       random_state: int = 42) -> Tuple[List[str], np.ndarray]:
    """
    Sarmalayıcı Yöntem: Recursive Feature Elimination (RFE) kullanarak özellik seçimi.
    
    Lojistik Regresyon ile birlikte RFE uygular.
    
    Argümanlar:
        X -- Özellik matrisi
        y -- Hedef değişken
        n_features -- Seçilecek özellik sayısı
        random_state -- Rastgelelik tohumu
        
    Dönüş:
        selected_features -- Seçilen özellik isimleri listesi
        feature_ranking -- Özellik sıralaması
    """
    print("\n" + "="*60)
    print("SARMALAYICI YÖNTEM (RFE - Recursive Feature Elimination)")
    print("="*60)
    
    # Lojistik Regresyon modeli oluştur
    estimator = LogisticRegression(
        max_iter=1000, 
        random_state=random_state,
        solver='lbfgs',
        n_jobs=-1
    )
    
    # RFE uygula
    print(f"RFE uygulanıyor (hedef: {n_features} özellik)...")
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=1,
        verbose=0
    )
    
    rfe.fit(X, y)
    
    # Seçilen özellikleri al
    selected_mask = rfe.support_
    selected_features = X.columns[selected_mask].tolist()
    feature_ranking = rfe.ranking_
    
    print(f"\nSeçilen en iyi {n_features} özellik:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")
    
    return selected_features, feature_ranking


def embedded_method_random_forest(X: pd.DataFrame, y: pd.Series,
                                   n_features: int = 15,
                                   n_estimators: int = 100,
                                   random_state: int = 42) -> Tuple[List[str], pd.DataFrame]:
    """
    Gömülü Yöntem: Random Forest özellik önem skoru kullanarak özellik seçimi.
    
    Argümanlar:
        X -- Özellik matrisi
        y -- Hedef değişken
        n_features -- Seçilecek özellik sayısı
        n_estimators -- Ağaç sayısı
        random_state -- Rastgelelik tohumu
        
    Dönüş:
        selected_features -- Seçilen özellik isimleri listesi
        importance_df -- Özellik önem skorları DataFrame'i
    """
    print("\n" + "="*60)
    print("GÖMÜLÜ YÖNTEM (Random Forest Özellik Önemi)")
    print("="*60)
    
    # Random Forest modeli oluştur ve eğit
    print(f"Random Forest eğitiliyor ({n_estimators} ağaç)...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=10  # Hız için derinlik sınırı
    )
    
    rf.fit(X, y)
    
    # Özellik önem skorlarını al
    importances = rf.feature_importances_
    
    # DataFrame oluştur ve sırala
    importance_df = pd.DataFrame({
        'Özellik': X.columns,
        'Önem_Skoru': importances
    }).sort_values('Önem_Skoru', ascending=False)
    
    # En iyi n_features özelliği seç
    selected_features = importance_df.head(n_features)['Özellik'].tolist()
    
    print(f"\nSeçilen en iyi {n_features} özellik:")
    for i, row in importance_df.head(n_features).iterrows():
        idx = importance_df.head(n_features).index.tolist().index(i) + 1
        print(f"  {idx:2d}. {row['Özellik']}: {row['Önem_Skoru']:.4f}")
    
    return selected_features, importance_df


def find_optimal_features_rfe(X: pd.DataFrame, y: pd.Series,
                               min_features: int = 5,
                               max_features: int = 30,
                               step: int = 5,
                               random_state: int = 42) -> Tuple[int, List[str], dict]:
    """
    RFE kullanarak optimum özellik sayısını bulur.
    
    Argümanlar:
        X -- Özellik matrisi
        y -- Hedef değişken
        min_features -- Minimum özellik sayısı
        max_features -- Maksimum özellik sayısı
        step -- Artış miktarı
        random_state -- Rastgelelik tohumu
        
    Dönüş:
        optimal_n -- Optimum özellik sayısı
        optimal_features -- Optimum özellik listesi
        results -- Tüm sonuçlar
    """
    from sklearn.model_selection import cross_val_score
    
    print("\n" + "="*60)
    print("OPTİMUM ÖZELLİK SAYISI ARAMA (RFE)")
    print("="*60)
    
    results = {}
    best_score = 0
    optimal_n = min_features
    
    for n in range(min_features, max_features + 1, step):
        print(f"\n{n} özellik test ediliyor...")
        
        # RFE ile özellik seç
        estimator = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1)
        rfe = RFE(estimator=estimator, n_features_to_select=n, step=1)
        rfe.fit(X, y)
        
        # Seçilen özelliklerle cross-validation
        X_selected = X.iloc[:, rfe.support_]
        model = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1)
        scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        
        results[n] = {
            'mean_accuracy': mean_score,
            'std': scores.std(),
            'features': X.columns[rfe.support_].tolist()
        }
        
        print(f"  Accuracy: {mean_score:.4f} (+/- {scores.std():.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            optimal_n = n
    
    print(f"\n*** Optimum özellik sayısı: {optimal_n} (Accuracy: {best_score:.4f}) ***")
    
    return optimal_n, results[optimal_n]['features'], results


def compare_selected_features(filter_features: List[str],
                               wrapper_features: List[str],
                               embedded_features: List[str]) -> pd.DataFrame:
    """
    Farklı yöntemlerle seçilen özellikleri karşılaştırır.
    
    Argümanlar:
        filter_features -- Filtreleme yöntemi ile seçilen özellikler
        wrapper_features -- Sarmalayıcı yöntem ile seçilen özellikler
        embedded_features -- Gömülü yöntem ile seçilen özellikler
        
    Dönüş:
        Karşılaştırma tablosu DataFrame
    """
    # Tüm benzersiz özellikleri topla
    all_features = set(filter_features) | set(wrapper_features) | set(embedded_features)
    
    # Karşılaştırma tablosu oluştur
    comparison = []
    for feat in sorted(all_features):
        row = {
            'Özellik': feat,
            'Filtreleme': '✓' if feat in filter_features else '',
            'Sarmalayıcı': '✓' if feat in wrapper_features else '',
            'Gömülü': '✓' if feat in embedded_features else '',
            'Ortak_Sayısı': sum([feat in filter_features, 
                                feat in wrapper_features, 
                                feat in embedded_features])
        }
        comparison.append(row)
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('Ortak_Sayısı', ascending=False)
    
    # Ortak özellikleri bul
    common_features = set(filter_features) & set(wrapper_features) & set(embedded_features)
    
    print("\n" + "="*60)
    print("ÖZELLİK SEÇİMİ KARŞILAŞTIRMASI")
    print("="*60)
    print(f"\nHer üç yöntemde ortak özellikler ({len(common_features)} adet):")
    for feat in common_features:
        print(f"  - {feat}")
    
    return comparison_df


if __name__ == "__main__":
    # Test için örnek veri
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=30, 
                               n_informative=15, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(30)])
    y = pd.Series(y)
    
    # Yöntemleri test et
    filter_feat, _ = filter_method_pearson(X, y, n_features=10)
    wrapper_feat, _ = wrapper_method_rfe(X, y, n_features=10)
    embedded_feat, _ = embedded_method_random_forest(X, y, n_features=10)
    
    # Karşılaştır
    compare_selected_features(filter_feat, wrapper_feat, embedded_feat)
