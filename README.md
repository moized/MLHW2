# MLHW2 - Feature Selection

Machine learning homework project for comparing feature selection methods on the Online News Popularity dataset. The project trains Logistic Regression models using all features, filter-based selection, wrapper-based selection, embedded selection, and an optimized RFE feature subset.

## Setup

```bash
pip install -r requirements.txt
```

Download `OnlineNewsPopularity.csv` and place it in:

```text
dataset/OnlineNewsPopularity.csv
```

Dataset source: https://www.kaggle.com/datasets/thehapyone/uci-online-news-popularity-data-set

## Usage

Train models and generate results:

```bash
python train.py
```

Evaluate the saved best model:

```bash
python eval.py
```

Generated outputs are written to `results/` and are intentionally excluded from version control.

## Project Structure

```text
dataset.py             Data loading and preprocessing
feature_selection.py   Feature selection methods
model.py               Model training and persistence
metrics.py             Evaluation metrics
utils.py               Plotting and result helpers
train.py               Training pipeline
eval.py                Evaluation script
```
