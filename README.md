# 📰 Feature Selection Framework for Online News Popularity Classification

## 📌 Overview
This project implements a rigorous, three-tiered feature selection pipeline engineered to optimize binary classification boundaries on the multi-dimensional *Online News Popularity* dataset. Rather than feeding raw, noisy data streams into models, this repository delivers systematic programmatic methods to reduce dimensionality, mitigate overfitting, and map feature importances using statistical and algorithmic criteria.

---

## 📁 Project Structure
The repository strictly adheres to clean production-grade machine learning folder layouts matching the actual execution architecture:

```text
MLHW2/
├── dataset/           # Ignored raw source data (OnlineNewsPopularity.csv)
├── src/               # Ingested analytical Python modules
│   ├── train.py       # Controls cross-validation loops and feature grid iterations
│   ├── eval.py        # Compiles performance charts and confusion matrices
│   ├── feature_selection.py # Houses Filter, Wrapper, and Embedded statistical methods
│   ├── dataset.py     # Preprocesses token sheets and calculates index slices
│   ├── model.py       # Custom Logistic Regression wrappers and cross-entropy matrices
│   ├── metrics.py     # Evaluation scorers (Accuracy, Precision, Recall, F1)
│   └── utils.py       # Visualization helpers and boundary plotting engines
├── results/           # Cached operational metrics and evaluation sheets (Local Only)
└── requirements.txt   # Core pipeline dependencies