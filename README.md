# 💳 FinShield — Intelligent Fraud & Anomaly Detection System

FinShield is a machine learning–powered fraud detection system for financial transactions. It uses feature engineering and a supervised LightGBM model to detect suspicious transactions with very high accuracy. The project also includes a Streamlit web app to interactively test transactions and view fraud risk in real time.

## 🚀 Features

- 📊 Exploratory Data Analysis (EDA) on real-world transaction data  
- 🛠️ Feature engineering based on balance inconsistencies and transaction patterns  
- 🤖 Supervised fraud detection using LightGBM  
- ⚖️ Handles extreme class imbalance using class weighting  
- 📈 Evaluation using ROC-AUC, Precision, Recall  
- 🖥️ Streamlit web app for real-time fraud risk prediction  
- 💾 Trained model saved and loaded using joblib  
- 🔍 Model explainability using feature importance  

## 🧠 Problem Statement

Financial fraud is rare, evolving, and costly. Rule-based systems struggle to capture complex patterns in transaction behavior. The goal of FinShield is to predict the probability that a transaction is fraudulent based on transaction amount and balance behavior, and provide a risk score that can be used for automated blocking or manual review.

## 📂 Dataset

This project uses the PaySim: A Financial Mobile Money Simulator dataset. It simulates real mobile money transactions and is highly imbalanced (~0.1% fraud). Features include transaction type, amount, sender and receiver balances before and after the transaction. The target variable is isFraud (1 = Fraud, 0 = Normal).


---

## 🛠️ Approach

### 1. Exploratory Data Analysis (EDA)
- Analyzed class imbalance  
- Studied fraud distribution across transaction types  
- Compared amount and balance patterns for fraud vs normal transactions  
- Identified strong fraud signals in balance inconsistencies  

### 2. Feature Engineering
Created new features such as:
- `orig_balance_diff = oldbalanceOrg - newbalanceOrig`  
- `dest_balance_diff = newbalanceDest - oldbalanceDest`  
- Flags for zero balances  
- One-hot encoding for transaction type  
- Scaling of numerical features  

### 3. Unsupervised Models (Baseline)
- Isolation Forest  
- Local Outlier Factor (LOF)  

These models performed close to random due to extreme class imbalance and overlapping distributions. This shows that unsupervised anomaly detection alone is not sufficient for this problem.

### 4. Supervised Model (Production Model)

- Model: **LightGBM Classifier**  
- Handles:
  - Non-linear feature interactions  
  - Large datasets  
  - Class imbalance using `scale_pos_weight`  
- Evaluation metrics:
  - **ROC-AUC** (primary)  
  - Precision, Recall, F1-score  
  - Confusion Matrix  

---

## 📈 Results

- ✅ ROC-AUC ≈ **0.995**  
- ✅ Recall for fraud ≈ **0.99** (catches almost all frauds)  
- ⚠️ Precision is lower (expected in highly imbalanced fraud detection)  
- 🎯 The model is excellent at ranking transactions by risk  

In real-world fraud systems, high recall is prioritized to avoid missing fraudulent transactions, even at the cost of some false positives.

---

## ⚖️ Threshold Tuning

Instead of using the default 0.5 threshold, different thresholds can be tested to:
- Increase precision (fewer false alarms), or  
- Increase recall (catch more fraud)  

This allows the system to be adapted to different business requirements.

---

## 🖥️ Streamlit App

A simple web UI is provided to:
- Enter transaction details  
- Get:
  - Fraud probability (%)
  - Risk level: **Low / Medium / High**  

Run the app:

```bash
streamlit run app.py

