"""
Credit Card Transaction Fraud Detection — Exploratory Analysis & Anomaly Flagging
Author: Lokesh Sharma

Approach:
  - Generate a realistic synthetic transaction dataset with embedded fraud patterns
  - Engineer features: velocity, amount deviation, odd-hour flag, merchant risk
  - Flag anomalies using Z-score + rule-based heuristics (interpretable, auditable)
  - Visualize distributions and flagged transactions
  - Export flagged transactions to CSV for downstream review 
 
No black-box models: every rule is explicit and auditable — matching how real
risk teams build interpretable, regulatorily-defensible fraud systems.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import random
import os

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. GENERATE SYNTHETIC DATASET ────────────────────────────────────────────

N_USERS       = 200
N_NORMAL      = 9_500
N_FRAUD       = 500          # 5% fraud rate (realistic for demonstration)
START_DATE    = datetime(2024, 1, 1)

MERCHANT_CATEGORIES = {
    "Grocery":       {"weight": 0.25, "avg_amount": 60,  "std": 30},
    "Fuel":          {"weight": 0.15, "avg_amount": 50,  "std": 20},
    "Restaurant":    {"weight": 0.20, "avg_amount": 40,  "std": 25},
    "Electronics":   {"weight": 0.10, "avg_amount": 300, "std": 200},
    "Pharmacy":      {"weight": 0.10, "avg_amount": 35,  "std": 20},
    "Travel":        {"weight": 0.05, "avg_amount": 800, "std": 400},
    "Online Retail": {"weight": 0.10, "avg_amount": 120, "std": 80},
    "ATM":           {"weight": 0.05, "avg_amount": 150, "std": 100},
}

categories    = list(MERCHANT_CATEGORIES.keys())
cat_weights   = [MERCHANT_CATEGORIES[c]["weight"] for c in categories]

user_ids      = [f"U{str(i).zfill(4)}" for i in range(N_USERS)]

def sample_amount(category):
    info = MERCHANT_CATEGORIES[category]
    amt  = np.random.normal(info["avg_amount"], info["std"])
    return max(1.0, round(amt, 2))

def random_timestamp(start, days=365):
    delta   = timedelta(days=np.random.randint(0, days),
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60))
    return start + delta

# Normal transactions
normal_rows = []
for _ in range(N_NORMAL):
    cat = np.random.choice(categories, p=cat_weights)
    normal_rows.append({
        "transaction_id": f"T{_:06d}",
        "user_id":        np.random.choice(user_ids),
        "category":       cat,
        "amount":         sample_amount(cat),
        "timestamp":      random_timestamp(START_DATE),
        "is_fraud":       0,
    })

# Fraud transactions — three realistic patterns
fraud_rows = []
for i in range(N_FRAUD):
    pattern = np.random.choice(["high_amount", "odd_hour", "velocity"], p=[0.4, 0.35, 0.25])
    cat = np.random.choice(categories, p=cat_weights)

    if pattern == "high_amount":
        # Unusually large amount (4–10x normal)
        base   = MERCHANT_CATEGORIES[cat]["avg_amount"]
        amount = round(base * np.random.uniform(4, 10), 2)
        ts     = random_timestamp(START_DATE)

    elif pattern == "odd_hour":
        # Transaction between 1am–4am
        day    = timedelta(days=np.random.randint(0, 365))
        hour   = np.random.randint(1, 5)
        ts     = START_DATE + day + timedelta(hours=hour,
                                               minutes=np.random.randint(0, 60))
        amount = sample_amount(cat)

    else:  # velocity — multiple rapid transactions (same user, same hour)
        day    = timedelta(days=np.random.randint(0, 365))
        ts     = START_DATE + day + timedelta(hours=np.random.randint(8, 22),
                                               minutes=np.random.randint(0, 10))
        amount = sample_amount(cat)

    fraud_rows.append({
        "transaction_id": f"F{i:06d}",
        "user_id":        np.random.choice(user_ids),
        "category":       cat,
        "amount":         amount,
        "timestamp":      ts,
        "is_fraud":       1,
    })

df = pd.DataFrame(normal_rows + fraud_rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"]      = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Mon … 6=Sun


# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────

# Per-user statistics (mean & std of their normal spending)
user_stats = (
    df[df["is_fraud"] == 0]
    .groupby("user_id")["amount"]
    .agg(user_mean="mean", user_std="std")
    .reset_index()
)
df = df.merge(user_stats, on="user_id", how="left")
df["user_std"] = df["user_std"].fillna(df["amount"].std())

# Z-score: how many std-devs is this txn from the user's own average?
df["amount_zscore"] = (df["amount"] - df["user_mean"]) / df["user_std"].clip(lower=1)

# Odd-hour flag: 1am–4am
df["odd_hour_flag"] = df["hour"].between(1, 4).astype(int)

# Transaction velocity: # of txns by same user in same 1-hour window
df_sorted = df.sort_values("timestamp")
df["velocity_1h"] = (
    df_sorted.groupby("user_id")["timestamp"]
    .transform(lambda s: s.expanding().count() -
               s.shift(1, fill_value=s.iloc[0]).expanding().count())
    .clip(lower=0)
)
# Simpler velocity proxy: transactions per user per hour-bucket
hour_bucket  = df["timestamp"].dt.floor("h").astype(str)
df["vel_key"] = df["user_id"] + "_" + hour_bucket
vel_counts    = df.groupby("vel_key")["transaction_id"].transform("count")
df["velocity_1h"] = vel_counts


# ── 3. ANOMALY FLAGGING (rule-based + Z-score) ────────────────────────────────

# Rules — each is independently auditable:
#   R1: amount Z-score > 3.0  (far above user's own baseline)
#   R2: odd hour flag          (1am–4am)
#   R3: velocity in 1h > 4    (more than 4 txns in same hour)

df["flag_high_amount"]  = (df["amount_zscore"] > 3.0).astype(int)
df["flag_odd_hour"]     = df["odd_hour_flag"]
df["flag_velocity"]     = (df["velocity_1h"] > 4).astype(int)

df["flag_count"]        = df["flag_high_amount"] + df["flag_odd_hour"] + df["flag_velocity"]
df["flagged_suspicious"]= (df["flag_count"] >= 1).astype(int)

# Summary
total          = len(df)
total_flagged  = df["flagged_suspicious"].sum()
true_positives = df[(df["flagged_suspicious"] == 1) & (df["is_fraud"] == 1)].shape[0]
false_positives= df[(df["flagged_suspicious"] == 1) & (df["is_fraud"] == 0)].shape[0]
total_fraud    = df["is_fraud"].sum()
precision      = true_positives / total_flagged if total_flagged else 0
recall         = true_positives / total_fraud   if total_fraud   else 0

print("=" * 55)
print("FRAUD DETECTION ANALYSIS — SUMMARY")
print("=" * 55)
print(f"Total transactions:       {total:,}")
print(f"Known fraud (ground truth):{total_fraud:,} ({total_fraud/total*100:.1f}%)")
print(f"Flagged as suspicious:    {total_flagged:,} ({total_flagged/total*100:.1f}%)")
print(f"True positives:           {true_positives:,}")
print(f"False positives:          {false_positives:,}")
print(f"Precision:                {precision:.2%}")
print(f"Recall:                   {recall:.2%}")
print("=" * 55)


# ── 4. VISUALISATIONS ────────────────────────────────────────────────────────

plt.style.use("seaborn-v0_8-whitegrid")
TEAL   = "#0F6E5C"
RED    = "#C0392B"
GREY   = "#95A5A6"
ORANGE = "#E67E22"

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Credit Card Transaction Fraud Detection — Exploratory Analysis",
             fontsize=16, fontweight="bold", y=1.01)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Amount distribution (log scale) — fraud vs normal ────────────────
ax1 = fig.add_subplot(gs[0, :2])
bins = np.linspace(0, df["amount"].quantile(0.995), 60)
ax1.hist(df[df["is_fraud"]==0]["amount"], bins=bins,
         alpha=0.7, color=TEAL, label="Normal", density=True)
ax1.hist(df[df["is_fraud"]==1]["amount"], bins=bins,
         alpha=0.7, color=RED,  label="Fraud",  density=True)
ax1.set_title("Transaction Amount Distribution (Fraud vs Normal)", fontweight="bold")
ax1.set_xlabel("Amount (INR)")
ax1.set_ylabel("Density")
ax1.legend()
ax1.set_xlim(0, df["amount"].quantile(0.995))

# ── Plot 2: Donut — flagged breakdown ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
labels = ["True Positive", "False Positive", "Missed Fraud", "Clean"]
vals   = [
    true_positives,
    false_positives,
    total_fraud - true_positives,
    total - total_flagged - (total_fraud - true_positives),
]
colors = [RED, ORANGE, "#8E44AD", TEAL]
wedges, texts, autotexts = ax2.pie(
    vals, labels=labels, colors=colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(width=0.5), pctdistance=0.75,
    textprops={"fontsize": 8},
)
ax2.set_title("Flag Outcome Breakdown", fontweight="bold")

# ── Plot 3: Hourly transaction count — fraud vs normal ───────────────────────
ax3 = fig.add_subplot(gs[1, :2])
hourly_normal = df[df["is_fraud"]==0].groupby("hour").size()
hourly_fraud  = df[df["is_fraud"]==1].groupby("hour").size()
hours = range(24)
ax3.bar(hours, [hourly_normal.get(h, 0) for h in hours],
        color=TEAL, alpha=0.8, label="Normal")
ax3.bar(hours, [hourly_fraud.get(h,  0) for h in hours],
        bottom=[hourly_normal.get(h, 0) for h in hours],
        color=RED,  alpha=0.8, label="Fraud")
ax3.axvspan(1, 4, alpha=0.1, color=RED, label="Odd-hour window (1–4am)")
ax3.set_title("Transactions by Hour of Day", fontweight="bold")
ax3.set_xlabel("Hour of Day")
ax3.set_ylabel("Transaction Count")
ax3.set_xticks(hours)
ax3.legend(fontsize=8)

# ── Plot 4: Z-score — flagged vs not flagged ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.scatter(
    df[df["flagged_suspicious"]==0]["amount_zscore"].clip(-2, 10),
    df[df["flagged_suspicious"]==0]["amount"],
    alpha=0.15, s=8, color=GREY, label="Normal"
)
ax4.scatter(
    df[df["flagged_suspicious"]==1]["amount_zscore"].clip(-2, 10),
    df[df["flagged_suspicious"]==1]["amount"],
    alpha=0.5,  s=12, color=RED, label="Flagged"
)
ax4.axvline(3.0, color=ORANGE, linestyle="--", linewidth=1.2, label="Z-score threshold (3.0)")
ax4.set_title("Amount vs Z-Score\n(Flagged Transactions)", fontweight="bold")
ax4.set_xlabel("Amount Z-Score")
ax4.set_ylabel("Amount (INR)")
ax4.legend(fontsize=7)
ax4.set_ylim(0, df["amount"].quantile(0.995))
ax4.set_xlim(-2, 10)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "fraud_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nVisualization saved → {out_path}")


# ── 5. EXPORT FLAGGED TRANSACTIONS ───────────────────────────────────────────

flagged_export = df[df["flagged_suspicious"] == 1][[
    "transaction_id", "user_id", "timestamp", "category", "amount",
    "amount_zscore", "flag_high_amount", "flag_odd_hour", "flag_velocity",
    "flag_count", "is_fraud"
]].sort_values("flag_count", ascending=False)

csv_path = os.path.join(OUTPUT_DIR, "flagged_transactions.csv")
flagged_export.to_csv(csv_path, index=False)
print(f"Flagged transactions exported → {csv_path}")
print(f"  Total flagged: {len(flagged_export):,}")
print(f"  Top-flagged (flag_count = 3): {(flagged_export['flag_count']==3).sum()}")

print("\nTop 10 highest-risk transactions:")
print(flagged_export.head(10)[["transaction_id","user_id","category","amount",
                                 "amount_zscore","flag_count","is_fraud"]].to_string(index=False))
