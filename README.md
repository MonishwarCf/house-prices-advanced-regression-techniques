# House Price Prediction using Linear Regression

Linear Regression model trained on the Kaggle Ames Housing dataset with manual feature engineering and custom 5-fold cross-validation.

---

## Overview

This project focuses on improving Linear Regression performance through manual data cleaning, feature engineering, and validation instead of automated pipelines.

Main goals:

* Reduce prediction instability
* Handle outliers manually
* Improve generalization across different data splits

---

## Key Improvements

* Improved R² score from ~0.35 to a stable range of 0.72–0.82
* Identified and removed high-leverage outliers (indices 271 and 120)
* Prevented negative house price predictions caused by extreme rows
* Built a manual 5-fold cross-validation loop without automated CV utilities

---

## Data Processing

### Neighborhood Binning

Grouped 25+ neighborhoods into 3 tiers based on mean sale price.

### Ordinal Encoding

Mapped categorical quality features into numeric scales:

| Rating | Value |
| ------ | ----- |
| Ex     | 4     |
| Gd     | 3     |
| TA     | 2     |
| Fa     | 1     |
| Po     | 0     |

### Missing Value Handling

| Feature       | Method        |
| ------------- | ------------- |
| `MasVnrArea`  | Filled with 0 |
| `LotFrontage` | Median        |
| `GarageYrBlt` | Median        |

### Feature Reduction

Dropped highly correlated features such as:

* `GarageArea`
* `1stFlrSF`

to reduce multicollinearity.

### Outlier Filtering

Removed houses with:

* `SalePrice > 500000`

to reduce skew in linear weights.

---

## Cross-Validation

Implemented manual 5-fold cross-validation to evaluate:

* Fold-wise R²
* RMSE variance
* Model stability across shuffled splits

This helped identify unstable folds and diagnose leverage effects from extreme rows.

---

## Final Results

| Metric     | Value                        |
| ---------- | ---------------------------- |
| R² Score   | 0.7239                       |
| RMSE       | $40,706.38                   |
| Test Split | 80/20 Manual Shuffle & Slice |

---

## Observations

* A small number of outliers can significantly distort Linear Regression weights
* Random train/test splits can produce misleading performance
* Data quality and feature engineering can improve simple models substantially

---

## Project Structure

```text
├── notebooks/
│   └── training and validation notebooks
├── data/
│   └── Kaggle Ames Housing dataset
└── README.md
```

---

## Dataset

Dataset:
[Kaggle Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques?utm_source=chatgpt.com)

---

## Requirements

* pandas
* numpy
* scikit-learn

---

## Run

```bash
git clone <repo-link>
cd <project-folder>
```

Run the notebook to train the model and view manual cross-validation logs.
