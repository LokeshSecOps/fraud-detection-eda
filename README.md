# Credit Card Transaction Fraud Detection

Exploratory data analysis and anomaly flagging on a synthetic credit card transaction dataset. Built to demonstrate practical data analytics, feature engineering, and interpretable fraud detection — the kind of work done by Credit & Fraud Risk (CFR) teams.
   
## What it does
<img width="1991" height="1434" alt="image" src="https://github.com/user-attachments/assets/9ea215b6-8c3c-452d-9b39-0be5662a69b1" />
    
     
1. **Generates a realistic synthetic dataset** — 10,000 transactions across 200 users, 8 merchant categories, spanning one year. Fraud patterns are embedded: high-amount outliers, odd-hour transactions (1–4am), and rapid velocity bursts.

2. **Engineers analytical features** from raw data:
   - Per-user baseline (mean & std of their historical spend)
   - Amount Z-score — how far a transaction deviates from that user's own average
   - Odd-hour flag — transactions between 1am and 4am
   - Transaction velocity — number of transactions by the same user in the same 1-hour window

3. **Flags anomalies using interpretable rules** — every rule is explicit and auditable, matching how real risk teams build defensible systems:
   - `flag_high_amount`: Z-score > 3.0
   - `flag_odd_hour`: transaction between 1am–4am
   - `flag_velocity`: >4 transactions in the same hour by the same user

4. **Visualizes patterns** across 4 charts (see `fraud_analysis.png`):
   - Amount distribution: fraud vs normal
   - Flag outcome breakdown (true positive / false positive / missed / clean)
   - Hourly transaction volume with odd-hour window highlighted
   - Z-score scatter: flagged vs clean transactions

5. **Exports flagged transactions** to CSV (`flagged_transactions.csv`) with all feature columns for downstream review.

## Results (on synthetic data)

| Metric | Value |
|---|---|
| Total transactions | 10,000 |
| Known fraud (ground truth) | 500 (5%) |
| Flagged as suspicious | ~2,088 (20.9%) |
| True positives | 278 |
| Precision | 13.3% |
| Recall | 55.6% |

Precision/recall trade-off is intentional: this is a **first-pass screening layer**, not the final decision. In a real pipeline, flagged transactions go to a second-stage model or human review queue — high recall is more important than low false positives at this stage.

## Tech stack

- Python 3.12
- Pandas — data manipulation, feature engineering, groupby aggregations
- NumPy — statistical computation (Z-scores, distributions)
- Matplotlib — all visualizations

## How to run

```bash
git clone https://github.com/LokeshSecOps/fraud-detection-eda
cd fraud-detection-eda
pip install pandas numpy matplotlib
python fraud_detection.py
```

Outputs:
- `fraud_analysis.png` — 4-panel visualization
- `flagged_transactions.csv` — all flagged records with feature columns

## Design decisions

**Why rule-based instead of ML?**
Interpretability matters in financial risk. A logistic regression or gradient boosting model might get slightly better numbers, but a risk analyst needs to explain to a regulator *why* a transaction was flagged. Rules like "Z-score > 3.0" and "transaction at 3am" are self-evident. This also reflects how real CFR teams often start: transparent heuristics first, ML models later as a complement.

**Why Z-score per user rather than global?**
A $500 transaction is normal for a frequent traveller but alarming for someone who usually spends $40. User-level baselines dramatically reduce false positives compared to a global threshold.

**Why synthetic data?**
Real card transaction data is PII-sensitive and not publicly distributable. The synthetic generator produces statistically realistic patterns (realistic merchant category mix, natural spending distributions, embedded fraud signatures) without any privacy concerns.
