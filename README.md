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