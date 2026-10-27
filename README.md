# Logistic Regression — Customer Churn Early-Warning System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)

## Overview

A churn prediction model for a telecom company, built with logistic regression. It uses customer demographics, service usage, contract, and billing data to estimate churn risk, so the business can intervene before customers leave.

## Dataset

The project uses the Telco Customer Churn dataset (`https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data`): **7,043 customers** and **21 fields**.

### Feature Groups

| Group | Fields |
|------|------|
| **Customer profile** | customerID, gender, SeniorCitizen, Partner, Dependents |
| **Service usage** | tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| **Contract and billing** | Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| **Target** | Churn (Yes/No) |

### Data Characteristics

- 7,043 records, 21 fields
- Churn rate: about 26.5% (1,869 customers)
- A small number of missing values in `TotalCharges`; these were handled

## Tech Stack

- **Python 3.11+**
- **pandas** — data processing
- **numpy** — numerical computing
- **matplotlib / seaborn** — visualization
- **scikit-learn** — modeling

## Project Structure

```
.
├── README.md                              # this file
├── customer_churn_prediction.ipynb        # main notebook
├── churn_prediction_model.pkl             # saved model
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # dataset
└── scaler.pkl                             # saved scaler
```

## What It Does

### 1. Data Preprocessing

- **Column names**: strip leading/trailing spaces
- **Missing values**: convert `TotalCharges` to numeric and drop rows with missing values
- **Label encoding**: map Churn Yes/No to 1/0
- **One-hot encoding**: encode 15 categorical variables, producing 30 features

### 2. Model

- **Algorithm**: logistic regression
- **Class balance**: `class_weight='balanced'`
- **Split**: 80% train (5,625 rows), 20% test (1,407 rows)
- **Scaling**: StandardScaler

### 3. Evaluation

| Metric | Value |
|------|------|
| Accuracy | 72.64% |
| Precision | 49.09% |
| Recall | 79.68% |
| F1 score | 60.75% |
| AUC-ROC | **83.50%** |

### 4. Risk Tiers

Customers are grouped by predicted churn probability:

| Tier | Probability | Suggested action |
|----------|----------|----------|
| **High** | ≥ 70% | Act immediately; offer targeted retention deals |
| **Medium** | 40%–70% | Follow up regularly; improve satisfaction |
| **Low** | < 40% | Normal service; occasional check-ins |

### 5. Visualizations

- Churn distribution
- Confusion matrix heatmap
- ROC curve
- Top 15 feature importances
- Risk tier distribution

## Usage

```bash
jupyter notebook customer_churn_prediction.ipynb
```

Run the cells in order to walk through the analysis and results interactively.

## Key Findings

1. **High recall** — the model catches 79.68% of potential churners, which suits an early-warning use case.
2. **AUC-ROC of 83.50%** — good separation between churners and non-churners.
3. **Main drivers**:
   - Contract type (month-to-month customers churn more)
   - Tenure (new customers are at higher risk)
   - Internet service type (fiber users churn more)
   - Payment method (electronic check users are at higher risk)

## Business Recommendations

1. **Focus on high-risk customers** — reach out proactively with retention offers.
2. **Rework contract incentives** — encourage longer contracts and reduce month-to-month share.
3. **Improve onboarding** — strengthen support during the first three months.
4. **Promote add-ons** — push online security and tech support to increase stickiness.

## Saving and Loading the Model

```python
import joblib

# Save
joblib.dump(model, 'churn_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load
loaded_model = joblib.load('churn_prediction_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
```

## Contributing

Issues and pull requests are welcome.

## License

MIT

---

## Contact

**Dr. Ailan Anjuxi (谙弆悕)**

- Email: anjuxi.ME@outlook.com
- SIP: sip:anjuxi@sip.linphone.org

---

*A teaching demo for data science and machine learning. The dataset comes from a public source.*
