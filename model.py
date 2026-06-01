"""
Model eğitimi ve değerlendirme modülü.
Lojistik Regresyon modeli ile 5-fold cross-validation uygular.
"""

import numpy as np
import pandas as pd
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from typing import List, Dict, Tuple, Optional


class LogisticRegressionModel:
    """
    Lojistik Regresyon modeli sınıfı.
    5-fold cross-validation ile eğitim ve değerlendirme yapar.
    """
    
    def __init__(self, 
                 max_iter: int = 1000,
                 C: float = 1.0,
                 random_state: int = 42,
                 solver: str = 'lbfgs'):
        """
        Model başlatıcı.
        
        Argümanlar:
            max_iter -- Maksimum iterasyon sayısı
            C -- Regularizasyon parametresi (küçük = güçlü regularizasyon)
            random_state -- Rastgelelik tohumu
            solver -- Optimizasyon algoritması
        """
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state
        self.solver = solver
        self.model = None
        self.cv_scores = None
        self.training_time = None
    
    def train_with_cv(self, X: pd.DataFrame, y: pd.Series, 
                      n_folds: int = 5) -> Dict:
        """
        5-fold cross-validation ile model eğitir.
        
        Argümanlar:
            X -- Eğitim özellikleri
            y -- Eğitim hedef değişkeni
            n_folds -- Fold sayısı (default 5)
            
        Dönüş:
            Cross-validation sonuçları sözlüğü
        """
        print(f"\n{'='*60}")
        print(f"{n_folds}-Fold Cross Validation Eğitimi")
        print(f"{'='*60}")
        print(f"Veri boyutu: {X.shape[0]} örnek, {X.shape[1]} özellik")
        print(f"Regularizasyon (C): {self.C}")
        
        start_time = time.time()
        
        # Stratified K-Fold oluştur
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                            random_state=self.random_state)
        
        # Model oluştur
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            C=self.C,
            random_state=self.random_state,
            solver=self.solver,
            n_jobs=-1
        )
        
        # Cross-validation skorları
        accuracy_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1')
        precision_scores = cross_val_score(self.model, X, y, cv=cv, scoring='precision')
        recall_scores = cross_val_score(self.model, X, y, cv=cv, scoring='recall')
        
        # Tüm veri üzerinde modeli eğit
        self.model.fit(X, y)
        
        self.training_time = time.time() - start_time
        
        self.cv_scores = {
            'accuracy': {
                'mean': accuracy_scores.mean(),
                'std': accuracy_scores.std(),
                'all': accuracy_scores
            },
            'f1': {
                'mean': f1_scores.mean(),
                'std': f1_scores.std(),
                'all': f1_scores
            },
            'precision': {
                'mean': precision_scores.mean(),
                'std': precision_scores.std(),
                'all': precision_scores
            },
            'recall': {
                'mean': recall_scores.mean(),
                'std': recall_scores.std(),
                'all': recall_scores
            },
            'training_time': self.training_time
        }
        
        # Sonuçları yazdır
        print(f"\nCross-Validation Sonuçları:")
        print(f"  Accuracy:  {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std():.4f})")
        print(f"  F1-Score:  {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
        print(f"  Precision: {precision_scores.mean():.4f} (+/- {precision_scores.std():.4f})")
        print(f"  Recall:    {recall_scores.mean():.4f} (+/- {recall_scores.std():.4f})")
        print(f"  Eğitim süresi: {self.training_time:.2f} saniye")
        
        return self.cv_scores
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Model performansını değerlendirir.
        
        Argümanlar:
            X -- Test özellikleri
            y -- Test hedef değişkeni
            
        Dönüş:
            Değerlendirme sonuçları sözlüğü
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi! Önce train_with_cv() çağırın.")
        
        # Tahmin yap
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        # Metrikleri hesapla
        results = {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Tahmin yapar.
        
        Argümanlar:
            X -- Özellik matrisi
            
        Dönüş:
            Tahminler
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi!")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Olasılık tahmini yapar.
        
        Argümanlar:
            X -- Özellik matrisi
            
        Dönüş:
            Olasılık tahminleri
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi!")
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str):
        """
        Modeli dosyaya kaydeder.
        
        Argümanlar:
            filepath -- Kayıt dosya yolu
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        save_dict = {
            'model': self.model,
            'cv_scores': self.cv_scores,
            'training_time': self.training_time,
            'params': {
                'max_iter': self.max_iter,
                'C': self.C,
                'random_state': self.random_state,
                'solver': self.solver
            }
        }
        joblib.dump(save_dict, filepath)
        print(f"Model kaydedildi: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LogisticRegressionModel':
        """
        Kaydedilmiş modeli yükler.
        
        Argümanlar:
            filepath -- Model dosya yolu
            
        Dönüş:
            Yüklenen model
        """
        save_dict = joblib.load(filepath)
        
        instance = cls(
            max_iter=save_dict['params']['max_iter'],
            C=save_dict['params']['C'],
            random_state=save_dict['params']['random_state'],
            solver=save_dict['params']['solver']
        )
        instance.model = save_dict['model']
        instance.cv_scores = save_dict['cv_scores']
        instance.training_time = save_dict['training_time']
        
        print(f"Model yüklendi: {filepath}")
        return instance


def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       selected_features: Optional[List[str]] = None,
                       method_name: str = "Tüm Özellikler",
                       C: float = 1.0) -> Dict:
    """
    Model eğitir ve değerlendirir.
    
    Argümanlar:
        X_train -- Eğitim özellikleri
        y_train -- Eğitim hedef değişkeni
        X_test -- Test özellikleri
        y_test -- Test hedef değişkeni
        selected_features -- Kullanılacak özellik listesi (None ise tümü)
        method_name -- Yöntem adı (raporlama için)
        C -- Regularizasyon parametresi
        
    Dönüş:
        Eğitim ve test sonuçları sözlüğü
    """
    print(f"\n{'#'*60}")
    print(f"# {method_name}")
    print(f"{'#'*60}")
    
    # Özellik seçimi uygula
    if selected_features is not None:
        X_train_subset = X_train[selected_features]
        X_test_subset = X_test[selected_features]
        n_features = len(selected_features)
    else:
        X_train_subset = X_train
        X_test_subset = X_test
        n_features = X_train.shape[1]
    
    print(f"Özellik sayısı: {n_features}")
    
    # Model oluştur ve eğit
    model = LogisticRegressionModel(C=C)
    cv_results = model.train_with_cv(X_train_subset, y_train)
    
    # Test seti üzerinde değerlendir
    print(f"\nTest Seti Değerlendirmesi:")
    test_results = model.evaluate(X_test_subset, y_test)
    
    print(f"  Accuracy:  {test_results['accuracy']:.4f}")
    print(f"  F1-Score:  {test_results['f1']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall:    {test_results['recall']:.4f}")
    
    return {
        'method_name': method_name,
        'n_features': n_features,
        'features': selected_features if selected_features else list(X_train.columns),
        'cv_results': cv_results,
        'test_results': test_results,
        'model': model
    }


def find_best_regularization(X_train: pd.DataFrame, y_train: pd.Series,
                             C_values: List[float] = [0.001, 0.01, 0.1, 1, 10, 100],
                             n_folds: int = 5) -> float:
    """
    En iyi regularizasyon parametresini bulur.
    
    Argümanlar:
        X_train -- Eğitim özellikleri
        y_train -- Eğitim hedef değişkeni
        C_values -- Denenecek C değerleri
        n_folds -- Fold sayısı
        
    Dönüş:
        En iyi C değeri
    """
    print("\n" + "="*60)
    print("REGULARİZASYON PARAMETRESİ ARAMA")
    print("="*60)
    
    best_score = 0
    best_C = 1.0
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for C in C_values:
        model = LogisticRegression(max_iter=1000, C=C, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        
        print(f"C={C:6.3f}: Accuracy = {mean_score:.4f} (+/- {scores.std():.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_C = C
    
    print(f"\n*** En iyi C değeri: {best_C} (Accuracy: {best_score:.4f}) ***")
    return best_C


if __name__ == "__main__":
    # Test için örnek veri
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model eğit ve değerlendir
    results = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    print("\nConfusion Matrix:")
    print(results['test_results']['confusion_matrix'])
