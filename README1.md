# Transaction Analysis & Operational Risk Insights

**Author:** Samuel Adegboyega  
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn  
**Dataset:** [PaySim Synthetic Financial Dataset - Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

## Project Overview

This project analyzes mobile money transaction data to identify operational risk, suspicious transaction behavior, and fraud patterns. The analysis combines exploratory data analysis, anomaly detection, and machine learning to answer practical business questions around transaction volume, failure signals, fraud concentration, and model-based fraud prediction.

The work is based on the PaySim synthetic financial dataset, which simulates mobile money transactions using patterns inspired by real financial activity. Because the dataset is synthetic, it is useful for fraud detection experimentation without exposing real customer data.

## Business Questions

- When do peak transaction loads occur, and what operational risks do they create?
- What is the overall fraud rate in the transaction system?
- Which transaction types are most associated with fraud?
- Do unusually large transactions show higher fraud risk?
- Can a machine learning model predict fraudulent transactions accurately?
- Which balance or transaction features are useful for risk monitoring?

---

## Dataset Summary

The dataset contains **6,362,620 transactions** across **743 hourly time steps**. Each `step` represents one hour, so the dataset covers approximately 31 days of simulated activity.

| Metric | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Original columns | 11 |
| Added analysis features | 3 |
| Duplicate rows | 0 |
| Missing values | 0 |
| Fraudulent transactions | 8,213 |
| Fraud rate | 0.129% |
| Average transaction amount | 179,861.90 |
| Average fraudulent amount | 1,467,967.30 |

## Data Dictionary

| Field | Meaning |
|---|---|
| `step` | Time unit of the transaction. Each step represents one hour from the start of the simulation. |
| `type` | Transaction category. Values include `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, and `TRANSFER`. |
| `amount` | Value of the transaction. |
| `nameOrig` | Identifier for the customer or account that initiated the transaction. |
| `oldbalanceOrig` | Sender's balance before the transaction. |
| `newbalanceOrig` | Sender's balance after the transaction. |
| `nameDest` | Identifier for the recipient customer or merchant. |
| `oldbalanceDest` | Recipient's balance before the transaction. |
| `newbalanceDest` | Recipient's balance after the transaction. |
| `isFraud` | Target variable. `1` means the transaction is fraudulent; `0` means it is legitimate. |
| `isFlaggedFraud` | Rule-based fraud flag from the simulation. `1` means the transaction was flagged by the system. |

## Engineered Features

| Feature | Meaning |
|---|---|
| `hour` | Hour of day derived from `step % 24`. Used to study daily transaction patterns. |
| `day` | Day number derived from `step // 24`. Used to study transaction volume over time. |
| `balance_unchanged` | Flag showing whether the sender's balance stayed the same after a positive transaction amount. This can indicate unusual processing behavior or failed balance updates. |
| `amount_zscore` | Standardized transaction amount used to detect unusually large transactions. |
| `type_encoded` | Numeric encoding of transaction type for machine learning models. |

---

## Approach

### 1. Data Cleaning

- Checked for missing values and duplicate records.
- Standardized the sender balance column name from `oldbalanceOrg` to `oldbalanceOrig`.
- Created time-based features for hour and day.
- Created a balance-unchanged indicator to help identify suspicious or failed balance behavior.

### 2. Exploratory Data Analysis

- Analyzed transaction volume by hour and day.
- Compared transaction type distribution.
- Measured legitimate and fraudulent transaction proportions.
- Compared transaction amounts for legitimate and fraudulent transactions.
- Studied fraud concentration by hour and transaction type.

### 3. Anomaly Detection

- Used Z-score detection to flag transactions with unusually high amounts.
- Used IQR detection to identify transactions outside normal amount ranges.
- Compared anomaly groups against the fraud label to test whether outliers are useful risk signals.

### 4. Machine Learning

- Built a Logistic Regression model as a baseline.
- Built a Random Forest Classifier as the main fraud prediction model.
- Evaluated models with precision, recall, F1-score, ROC-AUC, and confusion matrix.

---

## Key Observations

| Observation | What It Means |
|---|---|
| Fraud is rare: only 0.129% of transactions are fraudulent. | The dataset is highly imbalanced, so accuracy alone is not a reliable model metric. Precision, recall, F1-score, and ROC-AUC are more useful. |
| Peak transaction volume occurs around 19:00. | Operations teams should monitor capacity and service reliability around this period. |
| Fraud peaks around 05:00. | Early-morning activity may require stronger monitoring or alert thresholds. |
| Fraud is concentrated in `TRANSFER` and `CASH_OUT` transactions. | These transaction types should receive extra attention in fraud rules and model features. |
| Fraudulent transactions have a much higher average amount than normal transactions. | Transaction amount is a strong risk signal, especially when combined with balance movement. |
| Z-score anomalies represent 0.706% of transactions but have a 3.75% fraud rate. | Very large transactions are much riskier than the dataset average. |
| IQR anomalies represent 5.314% of transactions and have a 1.14% fraud rate. | Broader outlier detection captures more suspicious transactions but with more false positives. |
| The Random Forest model outperformed Logistic Regression. | Non-linear models can better capture complex fraud patterns in amount, balance, type, and time features. |

## Model Results

| Model | Fraud Precision | Fraud Recall | Fraud F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.96 | 0.72 | 0.82 | 0.9656 |
| Random Forest | 0.98 | 0.97 | 0.98 | 0.9990 |

The Random Forest model achieved the strongest performance on the sampled test set. This suggests that transaction amount, balance changes, transaction type, and time features provide useful signals for fraud detection.

---

## Recommendations

1. **Monitor peak hours closely:** Transaction volume is highest around 19:00, so infrastructure scaling and operational support should be strongest around that period.
2. **Prioritize high-risk transaction types:** Apply enhanced checks to `TRANSFER` and `CASH_OUT` transactions, especially when amounts are unusually high.
3. **Use anomaly detection as an early warning layer:** Z-score and IQR flags can help identify transactions that deserve extra review before model scoring.
4. **Track balance inconsistencies:** Transactions where balances do not change as expected should be reviewed as possible processing issues or suspicious records.
5. **Use recall-focused model monitoring:** Since fraud is rare, the model should be evaluated for its ability to catch fraud while keeping false positives manageable.
6. **Retrain models regularly:** Fraud behavior changes over time, so the model should be retrained and validated with newer transaction data.

---

## Suggested README Additions

To make this project even stronger, the README could also include:

- **Project structure:** A short list explaining the notebook, generated charts, and any future scripts.
- **Visual examples:** Add selected plots such as transaction volume, fraud by type, anomaly detection, and model confusion matrix.
- **Limitations:** Explain that PaySim is synthetic, so results should be treated as analytical practice rather than production banking evidence.
- **Future improvements:** Include ideas such as SMOTE/class weighting, threshold tuning, cost-based evaluation, SHAP explainability, and model deployment.
- **Dashboard or app link:** If you later build a Streamlit dashboard, link it here so users can explore the analysis interactively.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Hydrosamueldc/Transaction-Risk-Analysis.git
cd Transaction-Risk-Analysis

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn kagglehub jupyter

# 3. Run the notebook
jupyter notebook transaction_analysis.ipynb
```

The notebook downloads the PaySim dataset directly from Kaggle using `kagglehub`. On first use, Kaggle may require credentials depending on your local setup.

---

## Business Relevance

This analysis reflects the type of work done in financial operations, fraud analytics, and risk monitoring teams. It shows how transaction data can be used to identify system pressure, detect suspicious behavior, and build predictive models that support faster decision-making.

The findings can help teams decide when to scale infrastructure, which transaction types need stricter review, and how anomaly detection can support fraud investigation.

---

*Samuel Adegboyega | [LinkedIn](https://linkedin.com/in/adegboyega-samuel-1a302b203)*
