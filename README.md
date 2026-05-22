# Leakage-Free Stacked Ensemble Learning for Multi-Task Cascading Failure Prediction in Power Grids

A machine learning framework for predicting cascading failures in power systems using a leakage-free stacked ensemble architecture.  
This project performs simultaneous prediction of:

- Binary blackout occurrence
- Cascade severity classification
- Load-shed magnitude regression

The framework combines:
- Random Forest (RF)
- XGBoost (XGB)
- TabTransformer

with 5-Fold Out-of-Fold (OOF) stacking for robust and leakage-free learning.

---

## 📌 Project Overview

Cascading failures in power grids can propagate within milliseconds, causing large-scale blackouts and infrastructure instability. Traditional contingency analysis methods are computationally expensive for real-time applications.

This project introduces a fast and scalable ML-based prediction framework trained on synthetic IEEE 118-bus cascading-failure simulations.

Key highlights:
- Leakage-free training protocol
- Multi-task learning pipeline
- Stacked ensemble architecture
- Large-scale synthetic dataset (42,199 scenarios)
- Real-time inference (~1.2 ms per scenario)

---

## 🧠 Tasks Performed

### 1. Binary Blackout Detection
Predict whether a blackout will occur.

### 2. Cascade Severity Classification
Classify blackout severity into:
- Small
- Medium
- Large

### 3. Load Shed Regression
Predict total load shed magnitude (MW).

---

## ⚙️ Model Architecture

### Base Models
- Random Forest
- XGBoost
- TabTransformer

### Meta Models
- Logistic Regression (Classification)
- Ridge Regression (Regression)

### Stacking Strategy
- 5-Fold Out-of-Fold (OOF) Stacking
- Leakage-free meta-feature generation

---

## 📊 Dataset Information

### Power System
- IEEE 118-Bus System

### Total Scenarios
- 42,199 synthetic cascading-failure scenarios

### Features
- 593 pre-cascade observable features

Feature groups include:
- Bus voltages
- Frequency deviations
- Line flows
- Line status
- Scenario metadata

### Leakage Prevention
The following post-cascade variables are strictly excluded:
- `cascade_depth`
- `blackout_size_buses`

---

## 📈 Results

### Binary Blackout Detection
| Metric | Score |
|---|---|
| Accuracy | 99.53% |
| F1-Macro | 0.9939 |
| ROC-AUC | 0.9999 |

### Severity Classification
| Metric | Score |
|---|---|
| Accuracy | 98.67% |
| F1-Macro | 0.9813 |

### Load Shed Regression
| Metric | Score |
|---|---|
| RMSE | 25.1 MW |
| R² Score | 0.9982 |

---

## 🛠️ Tech Stack

### Programming Language
- Python 3.10

### Libraries & Frameworks
- Scikit-learn
- XGBoost
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 📂 Project Structure

```bash
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│
├── models/
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── stacking/
│
├── results/
│   ├── figures/
│   ├── metrics/
│
├── requirements.txt
├── README.md
└── main.py
