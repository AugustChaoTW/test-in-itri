"""Phase 1: AI4I 2020 Predictive Maintenance EDA (exploratory run) - for stats collection"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (12, 6)
sns.set_style("whitegrid")

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

df = pd.read_csv("/root/data/ai4i2020.csv")

print("=" * 60)
print("STEP 1: DATA LOADING")
print("=" * 60)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Dtypes:\n", df.dtypes)
print("\nMemory usage:", df.memory_usage(deep=True).sum() / 1024, "KB")

print("\n" + "=" * 60)
print("STEP 2: HEAD / DESCRIBE")
print("=" * 60)
print(df.head())
print("\nDescribe (numeric):\n", df.describe())

print("\n" + "=" * 60)
print("STEP 3: MISSING / DUPLICATES")
print("=" * 60)
print("Missing values:\n", df.isna().sum())
print("Total missing:", df.isna().sum().sum())
print("Duplicates:", df.duplicated().sum())

print("\n" + "=" * 60)
print("STEP 4: TARGET DISTRIBUTION")
print("=" * 60)
print("Machine failure counts:\n", df["Machine failure"].value_counts())
print("Failure rate:", df["Machine failure"].mean())

failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
print("\nFailure mode counts:")
for c in failure_modes:
    print(f"  {c}: {df[c].sum()}")

print("\n" + "=" * 60)
print("STEP 5: CATEGORICAL FEATURES")
print("=" * 60)
print("Product ID value counts:\n", df["Product ID"].value_counts().head(20))
print("Type value counts:\n", df["Type"].value_counts())
print("\nFailure rate by Type:\n", df.groupby("Type")["Machine failure"].mean())

print("\n" + "=" * 60)
print("STEP 6: NUMERICAL SUMMARY BY FAILURE")
print("=" * 60)
num_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]"]
for c in num_cols:
    print(f"\n--- {c} ---")
    print(df.groupby("Machine failure")[c].describe().round(3))

print("\n" + "=" * 60)
print("STEP 7: CORRELATION MATRIX")
print("=" * 60)
corr = df.corr(numeric_only=True)
print(corr.round(3))

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("/tmp/corr_heatmap.png")
plt.close()

print("\n" + "=" * 60)
print("STEP 8: IQR OUTLIERS (per feature, by failure flag groups)")
print("=" * 60)
for c in num_cols + ["Machine failure"]:
    q1 = df[c].quantile(0.25)
    q3 = df[c].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[c] < lo) | (df[c] > hi)).sum()
    print(f"  {c}: Q1={q1:.3f} Q3={q3:.3f} IQR={iqr:.3f} bounds=[{lo:.3f}, {hi:.3f}] outliers={n_out}")

print("\n" + "=" * 60)
print("STEP 9: DERIVED FEATURES")
print("=" * 60)
df["Power (Nm·rad/s)"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * np.pi / (60 * 2)) / (2 * np.pi) * 2  # placeholder
# power = torque * angular velocity = torque * rpm * pi / 30
df["Power (Nm*rad/s)"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * np.pi / 30) / (2 * np.pi)  # Nm*rad/s = Nm * rad/s ; torque*omega
df["Temp diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Power (W)"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * np.pi / 30)
for c in ["Machine failure", "Power (W)", "Temp diff"]:
    print(f"  {c}: corr with all numeric = {df[[c] + num_cols].corr().iloc[0, -2:].round(3).to_dict() if False else ''}")
print("Correlations with failure:")
for c in num_cols + ["Power (W)", "Temp diff"]:
    print(f"  {c}: {df[c].corr(df['Machine failure']).round(4)}")

print("\nALL_STATS_COLLECTED")
