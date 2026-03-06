# Universal Bank - Personal Loan Analysis Dashboard

Interactive Streamlit dashboard performing 4 types of data analysis on Universal Bank's personal loan campaign.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Click Deploy

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `UniversalBank.csv` | Source dataset (5,000 customers) |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Dark theme configuration |

## Analysis Sections

1. **Descriptive** - Customer demographics and loan distribution
2. **Diagnostic** - Key drivers of loan acceptance
3. **Predictive** - ML model comparison (Logistic Regression, Decision Tree, Random Forest)
4. **Confusion Matrix Deep Dive** - Detailed error analysis with business impact
5. **Prescriptive** - Actionable targeting recommendations
