"""
Değerlendirme metrikleri modülü.
Sınıflandırma performansını ölçmek için metrik fonksiyonları içerir.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Karmaşıklık matrisi (confusion matrix) bileşenlerini hesaplar.
    
    Argümanlar:
        y_true -- Gerçek etiketler
        y_pred -- Tahmin edilen etiketler
    
    Dönüş:
        tp -- Doğru pozitifler (True Positives)
        tn -- Doğru negatifler (True Negatives)
        fp -- Yanlış pozitifler (False Positives)
        fn -- Yanlış negatifler (False Negatives)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tp, tn, fp, fn


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Doğruluk (accuracy) skorunu hesaplar.
    
    Argümanlar:
        y_true -- Gerçek etiketler
        y_pred -- Tahmin edilen etiketler
    
    Dönüş:
        Doğruluk skoru: (TP + TN) / (TP + TN + FP + FN)
    """
    return accuracy_score(y_true, y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Kesinlik (precision) skorunu hesaplar.
    
    Argümanlar:
        y_true -- Gerçek etiketler
        y_pred -- Tahmin edilen etiketler
    
    Dönüş:
        Kesinlik skoru: TP / (TP + FP)
    """
    return precision_score(y_true, y_pred, zero_division=0)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Duyarlılık (recall/sensitivity) skorunu hesaplar.
    
    Argümanlar:
        y_true -- Gerçek etiketler
        y_pred -- Tahmin edilen etiketler
    
    Dönüş:
        Duyarlılık skoru: TP / (TP + FN)
    """
    return recall_score(y_true, y_pred, zero_division=0)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 skorunu hesaplar.
    
    Argümanlar:
        y_true -- Gerçek etiketler
        y_pred -- Tahmin edilen etiketler
    
    Dönüş:
        F1 skoru: 2 × (Precision × Recall) / (Precision + Recall)
    """
    return f1_score(y_true, y_pred, zero_division=0)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Modeli tüm metriklerle değerlendirir.
    
    Argümanlar:
        y_true -- Gerçek etiketler
        y_pred -- Tahmin edilen etiketler
    
    Dönüş:
        Tüm metrikleri içeren sözlük
    """
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy(y_true, y_pred),
        'precision': precision(y_true, y_pred),
        'recall': recall(y_true, y_pred),
        'f1': f1(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def print_metrics(metrics: Dict, set_name: str = "Test") -> None:
    """
    Metrikleri formatlanmış şekilde yazdırır.
    
    Argümanlar:
        metrics -- evaluate_model fonksiyonundan dönen metrik sözlüğü
        set_name -- Veri seti adı (örn: "Eğitim", "Test")
    """
    print(f"\n{set_name} Seti Metrikleri:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    print(f"\n{set_name} Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"              Tahmin: 0    Tahmin: 1")
    print(f"  Gerçek: 0      {cm[0,0]:5d}        {cm[0,1]:5d}")
    print(f"  Gerçek: 1      {cm[1,0]:5d}        {cm[1,1]:5d}")


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                               target_names: list = None) -> str:
    """
    Detaylı sınıflandırma raporu döndürür.
    
    Argümanlar:
        y_true -- Gerçek etiketler
        y_pred -- Tahmin edilen etiketler
        target_names -- Sınıf isimleri
    
    Dönüş:
        Formatlanmış sınıflandırma raporu
    """
    if target_names is None:
        target_names = ['Popüler Değil (0)', 'Popüler (1)']
    
    return classification_report(y_true, y_pred, target_names=target_names)
