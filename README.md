# Transaction Analysis & Operational Risk Insights

**Author:** Samuel Adegboyega  
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn  
**Dataset:** [PaySim Synthetic Financial Dataset — Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

## Problem Statement

This project analyses transaction data to identify inefficiencies, peak loads, and potential operational risks in a payment system. Using exploratory data analysis, anomaly detection, and machine learning, we surface patterns that inform operational decisions — including when system strain is most likely, where failure rates concentrate, and which transactions carry fraud risk.

**Business Questions:**
- When do peak transaction loads occur, and what operational risks do they create?
- What is the fraud rate and when does it worsen?
- Which transactions are anomalous and potentially fraudulent?
- Can we build a model to accurately predict fraudulent transactions?

---

## Approach

### 1. Data Cleaning
- Removed duplicate records
- Converted step variable to hour-of-day and day features
- Engineered balance-unchanged flag for failed transaction detection

### 2. Exploratory Data Analysis
- Transaction volume trends by hour and day
- Transaction type distribution
- Fraud vs legitimate breakdown
- Amount distribution and outlier analysis
- Peak failure and fraud periods

### 3. Anomaly Detection
- **Z-Score method:** flagged transactions with |Z| > 3 as outliers
- **IQR method:** flagged transactions outside Q1 - 1.5×IQR / Q3 + 1.5×IQR bounds
- Cross-referenced anomalies with fraud labels to validate detection quality

### 4. Machine Learning — Fraud Detection
- **Logistic Regression** (baseline)
- **Random Forest Classifier** (primary model)
- Evaluation: Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

---

## Key Insights

| Finding | Operational Implication |
|---|---|
| Peak transaction volume at specific hours | Load balancing and autoscaling required at peak periods |
| Fraud concentrates in TRANSFER and CASH_OUT types | Enhanced authentication needed for high-risk transaction types |
| Anomalous (high Z-score) transactions have higher fraud rates | Real-time anomaly flags are viable screening signals |
| Random Forest achieves strong precision/recall on fraud | Production-grade model is feasible with regular retraining |
| Balance-unchanged transactions cluster with failures | Processing validation checks needed in pipeline |

---

## Recommendations

1. **Implement load balancing** during peak transaction hours to prevent system strain
2. **Deploy real-time anomaly monitoring** — flag high-value outliers automatically
3. **Prioritise fraud screening** on TRANSFER and CASH_OUT transactions above thresholds
4. **Build a production fraud scoring model** with regular retraining on recent data
5. **Monitor balance-unchanged transactions** as indicators of processing failures

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Hydrosamueldc/transaction-risk-analysis
cd transaction-risk-analysis

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# 3. Download dataset
# Go to: https://www.kaggle.com/datasets/ealaxi/paysim1
# Download and place PS_20174392719_1491204439457_log.csv in this folder

# 4. Run notebook
jupyter notebook transaction_analysis.ipynb
```

---

## Business Relevance

This analysis mirrors the kind of work done in financial institution Operations and Technology teams — identifying where payment systems are under pressure, flagging transactions that deviate from normal patterns, and building predictive models to automate risk detection. The insights translate directly into actionable operational decisions: when to scale infrastructure, which transactions need enhanced review, and where to focus compliance monitoring.

---

*Samuel Adegboyega | [LinkedIn](https://linkedin.com/in/adegboyega-samuel-1a302b203)*



