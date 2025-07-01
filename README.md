# Credit Risk Modeling Project

## Business Context & Regulatory Compliance

### **1. Basel II’s Emphasis on Risk Measurement**
**Key Requirements:**
- Banks must hold capital reserves proportional to asset risk
- **Internal Ratings-Based (IRB) Approach** requires estimation of:
  - Probability of Default (PD)
  - Loss Given Default (LGD) 
  - Exposure at Default (EAD)
  - Effective Maturity (M)

**Model Implications:**
✅ **Interpretability Mandatory** - Must explain feature impacts  
✅ **Full Documentation** - From data to deployment  
✅ **Governance Framework** - Validation, monitoring, stress testing  

> 🔍 *Basel II prioritizes explainable, transparent models over black-box approaches*

---

### **2. Proxy Variables for Default**
**Common Proxies When True Default Data is Unavailable:**
- 90+ days past due
- Account charge-offs
- Write-off status

**Business Risks:**
| Risk | Mitigation Strategy |
|------|---------------------|
| Label Misalignment | Align proxy with regulatory default definition |
| Model Bias | Validate against actual default outcomes |
| Regulatory Rejection | Document proxy justification thoroughly |
| Stakeholder Misinterpretation | Clear communication of limitations |

> 🧠 *Proxy variables are practical but require careful implementation*

---

### **3. Model Selection Trade-offs**
| Criteria | Logistic Regression | Gradient Boosting |
|----------|---------------------|-------------------|
| Interpretability | ✅ High | ❌ Low |
| Regulatory Acceptance | ✅ Easy | ❌ Challenging |
| Predictive Power | ❌ Moderate | ✅ High |
| Maintenance | ✅ Simple | ❌ Complex |

**Best Practice:**  
⚖️ *Use interpretable models for compliance, complex models for benchmarking*

---

## Data EDA

Number of column and row: (95662, 16) 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 95662 entries, 0 to 95661
Data columns (total 16 columns):

### 🔢 Summary Statistics:

| Column Name            | Count   | Unique | Top Value              | Frequency |
|------------------------|---------|--------|-------------------------|-----------|
| TransactionId          | 95,662  | 95,662 | TransactionId_76871    | 1         |
| BatchId                | 95,662  | 94,809 | BatchId_67019          | 28        |
| AccountId              | 95,662  | 3,633  | AccountId_4841         | 30,893    |
| SubscriptionId         | 95,662  | 3,627  | SubscriptionId_3829    | 32,630    |
| CustomerId             | 95,662  | 3,742  | CustomerId_7343        | 4,091     |
| CurrencyCode           | 95,662  | 1      | UGX                    | 95,662    |
| CountryCode            | 95,662  | NaN    | NaN                    | NaN       |
| ProviderId             | 95,662  | 6      | ProviderId_4           | 38,189    |
| ProductId              | 95,662  | 23     | ProductId_6            | 32,635    |
| ProductCategory        | 95,662  | 9      | financial_services     | 45,405    |
| ChannelId              | 95,662  | 4      | ChannelId_3            | 56,935    |
| Amount                 | 95,662  | NaN    | NaN                    | NaN       |
| Value                  | 95,662  | NaN    | NaN                    | NaN       |
| TransactionStartTime   | 95,662  | 94,556 | 2018-12-24T16:30:13Z   | 17        |
| PricingStrategy        | 95,662  | NaN    | NaN                    | NaN       |
| FraudResult            | 95,662  | NaN    | NaN                    | NaN       |

| Column Name           | Mean        | Std Dev       | Min         | 25%     | 50%     |
|-----------------------|-------------|---------------|-------------|---------|---------|
| TransactionId         | NaN         | NaN           | NaN         | NaN     | NaN     |
| BatchId               | NaN         | NaN           | NaN         | NaN     | NaN     |
| AccountId             | NaN         | NaN           | NaN         | NaN     | NaN     |
| SubscriptionId        | NaN         | NaN           | NaN         | NaN     | NaN     |
| CustomerId            | NaN         | NaN           | NaN         | NaN     | NaN     |
| CurrencyCode          | NaN         | NaN           | NaN         | NaN     | NaN     |
| CountryCode           | 256.0       | 0.0           | 256.0       | 256.0   | 256.0   |
| ProviderId            | NaN         | NaN           | NaN         | NaN     | NaN     |
| ProductId             | NaN         | NaN           | NaN         | NaN     | NaN     |
| ProductCategory       | NaN         | NaN           | NaN         | NaN     | NaN     |
| ChannelId             | NaN         | NaN           | NaN         | NaN     | NaN     |
| Amount                | 6,717.85    | 123,306.80    | -1,000,000  | -50.0   | 1,000.0 |
| Value                 | 9,900.58    | 123,122.09    | 2.0         | 275.0   | 1,000.0 |
| TransactionStartTime  | NaN         | NaN           | NaN         | NaN     | NaN     |
| PricingStrategy       | 2.26        | 0.73          | 0.0         | 2.0     | 2.0     |
| FraudResult           | 0.002       | 0.0449        | 0.0         | 0.0     | 0.0     |

🔍 A few quick observations:
- Most categorical ID-type fields have NaNs here because these summary stats only apply to numerical data.
- `Amount` and `Value` have massive standard deviations compared to their means—might want to check for extreme outliers.
- `CountryCode` seems to be a single constant value (256.0), which suggests low variance—maybe all records are from Uganda?
- `FraudResult` is very close to 0 throughout, implying most transactions were likely not flagged.
 


## Technical Implementation

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/yourusername/credit-risk-model/ci.yml) 
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

### Project Structure
```
credit-risk-model/
├── .github/workflows/ci.yml    # CI/CD pipelines
├── data/                       # .gitignored raw data
├── notebooks/                  # Jupyter notebooks
├── src/                        # Core application code
│   ├── api/                    # FastAPI implementation
│   ├── data_processing.py      # Feature engineering
│   ├── train.py                # Model training
│   └── predict.py              # Inference logic
├── tests/                      # Unit tests
├── Dockerfile                  # Containerization
└── requirements.txt            # Dependencies
```

### Key Features
**Data Processing Pipeline:**
```python
def preprocess_data(df):
    """Applies WoE encoding and feature engineering"""
    df = handle_missing_values(df)
    df = apply_woe_binning(df)
    return df
```

**API Endpoint:**
```python
@app.post("/predict")
async def predict_risk(input: RiskInput):
    """Returns PD estimate with explanation"""
    prediction = model.predict_proba([input.features])
    return {
        "PD": prediction[0][1],
        "key_factors": get_top_features(input)
    }
```

## Getting Started

### Installation
```bash
git clone https://github.com/yourusername/credit-risk-model.git
cd credit-risk-model
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Running Services
**Training Pipeline:**
```bash
python src/train.py --data_path=data/processed/loan_data.csv
```

**API Server:**
```bash
uvicorn src.api.main:app --reload
```
*Access interactive docs:* http://localhost:8000/docs

## Documentation
| Notebook | Description |
|----------|-------------|
| [1.0-eda.ipynb](notebooks/) | Exploratory analysis |
| [2.0-feature-engineering.ipynb](notebooks/) | WoE transformation |

## Contributing
1. Create an issue describing proposed changes
2. Fork the repository
3. Submit a PR with:
   - Test coverage
   - Updated documentation
   - Type hints for new code

---