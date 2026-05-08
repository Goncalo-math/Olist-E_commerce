# Olist Customer Satisfaction · Prediction Pipeline

> Brazilian e-commerce · NLP + tabular ML · negative review early warning

![Python](https://img.shields.io/badge/Python-3.10+-3C3489?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-0F6E56?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-854F0B?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-444441?style=flat-square)

End-to-end analysis and prediction system built on the **Olist Brazilian E-Commerce dataset** (100k orders, 2016–2018). The project explores the drivers of customer dissatisfaction and trains a classifier to **flag orders likely to receive a negative review before it is posted**, enabling proactive intervention by customer success teams.

## What this repo does

| | |
|---|---|
| **Exploratory analysis** | Distribution of scores, delivery delays, geographic patterns, and category-level satisfaction |
| **Feature engineering** | Delivery delay, freight ratio, payment type, seller metrics, and time-based signals |
| **Binary classifier** | XGBoost model trained to predict review scores ≤ 2 (negative) before submission |
| **SHAP explainability** | Per-order explanations of what drove each negative prediction |

## Dataset

9 relational tables covering orders, reviews, payments, customers, sellers, products, and geolocation for ~100k orders placed on the Olist marketplace between 2016 and 2018.

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `imbalanced-learn` · `shap` · `matplotlib` · `seaborn` · `jupyter`

## Quick start

```bash
git clone https://github.com/Goncalo-math/
Olist-E_commerce
cd olist-satisfaction
pip install -r requirements.txt
# place the 9 CSVs in data/raw/
jupyter notebook notebooks/01_eda.ipynb
```
