# ML-Loan-Predictor-Analysis

A machine learning project applying classification and regression techniques to real-world loan approval data to predict client creditworthiness and maximum loan amounts.

## Project Overview

This project applies machine learning techniques to a real-world loan approval dataset containing 58,645 client records. It covers two prediction tasks:

- **Case Study A — Classification:** Predicting whether a loan application will be approved or rejected using Logistic Regression, K-Nearest Neighbours and Naive Bayes classifiers.
- **Case Study B — Regression:** Predicting the maximum loan amount for approved clients using Decision Tree Regressors.

## Repository Structure

| File | Description |
|------|-------------|
| Data_Understanding_Preprocessing.ipynb | Data understanding and preprocessing — missing value imputation, encoding and dataset preparation |
| Classification_Modelling_Tuning.ipynb | Classification modelling — LR, KNN, Naive Bayes with hyperparameter tuning via GridSearchCV |
| Ensemble_Regression_Decision_Trees.ipynb | Ensemble classifier (soft voting) and regression decision trees (DT-1 fully grown, DT-2 pruned) |

## Key Results

### Classification (Case Study A)
| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| Logistic Regression | 0.8902 | 0.4965 | 0.8714 |
| KNN (k=7) Best | 0.9142 | 0.6400 | 0.8549 |
| Naive Bayes | 0.8493 | 0.5322 | 0.8548 |
| Voting Ensemble (LR + NB) | 0.8795 | 0.5651 | — |

### Regression (Case Study B)
| Model | MAE (GBP) | R-Squared |
|-------|-----------|-----------|
| DT-1 Fully Grown Best | 1213.79 | 0.9822 |
| DT-2 Pruned (depth=4) | 11801.04 | 0.8650 |

**Client 60256 Predicted Maximum Loan Amount: GBP 83097.00**

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Google Colab

## Author

**Shane Rowell Fernando**
