"""Generate notebooks/eda.ipynb for Phase 1 AI4I 2020 EDA (Issue #2)"""
import nbformat as nbf

nb = nbf.v4.new_notebook()

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

import_code = """import warnings
warnings.filterwarnings("ignore")

%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.dpi"] = 110
plt.rcParams["figure.figsize"] = (11, 5)
sns.set_style("whitegrid")

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 220)

print("Libraries loaded.")"""

nb.cells = [
    md("""# Phase 1: 資料探索 (EDA) — AI4I 2020 Predictive Maintenance

對應 Issue #2。執行者：AI4I 2020 資料集分析專案

**資料來源**：UCI Machine Learning Repository — "AI4I 2020 Predictive Maintenance Dataset" (Dataset ID 601)
- 下載連結：https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip
- 儲存位置：`data/ai4i2020.csv`

共 10,000 筆、14 個變數。本筆記本完成完整探索性資料分析，並輸出報告 `reports/eda_summary.md`。"""),

    code(import_code),

    code("""DATA_PATH = "/root/data/ai4i2020.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
df.head(5)"""),

    md("""### 1.1 資料集基本資訊
記錄樣本數、變數名稱、型態、記憶體佔用。"""),

    code("""print("Shape:", df.shape)
print("\\nColumns:", list(df.columns))
print("\\nDtypes:")
print(df.dtypes)
print("\\nMemory usage (KB):", round(df.memory_usage(deep=True).sum() / 1024, 2))"""),

    md("### 1.2 描述性統計（數值型）"),

    code("df.describe()"),

    md("### 1.3 遺失值與重複值檢查"),

    code("""missing = df.isna().sum()
missing = missing[missing > 0]
print("Missing values by column:")
print(missing if len(missing) else "None")
print("Total missing:", int(df.isna().sum().sum()))
print("Duplicated rows:", int(df.duplicated().sum()))"""),

    md("""## 2. 目標變數：機器故障 (Machine failure)

目標欄位 `Machine failure` 為二元標籤（0=正常，1=故障），另有五個獨立故障模式欄位：TWF, HDF, PWF, OSF, RNF。"""),

    code("""print("Machine failure distribution:")
print(df["Machine failure"].value_counts())
print("\\nFailure rate:", round(df["Machine failure"].mean(), 4))

failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
print("\\nFailure mode counts:")
for c in failure_modes:
    print(f"  {c}: {df[c].sum()}")"""),

    code("""fig, ax = plt.subplots(1, 2, figsize=(11, 4))

counts = df["Machine failure"].value_counts().sort_index()
colors = ["#4c78a8", "#e45756"]
ax[0].bar(counts.index.astype(str), counts.values, color=colors, width=0.5)
ax[0].set_xlabel("Machine failure")
ax[0].set_ylabel("Count")
ax[0].set_title("Machine failure counts")
for i, v in enumerate(counts.values):
    ax[0].text(i, v + 30, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

rate = df["Machine failure"].mean()
ax[1].pie([rate, 1 - rate], labels=["Failure (1)", "Normal (0)"],
          autopct="%1.2f%%", colors=colors, startangle=90,
          wedgeprops=dict(width=0.4))
ax[1].set_title("Failure proportion")

plt.suptitle("Machine failure distribution", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
print(f"Failure rate: {rate:.4%} ({int(counts.iloc[1]):,} / {int(counts.sum()):,})")"""),

    md("""## 3. 類別型變數：Product ID 與 Type

- Product ID：高基數編碼（L/M/H 開頭 + 序號），共 10,000 筆唯一，主要用途為識別。
- Type：產品品質等級 L（低）、M（中）、H（高），分佈 60% / 30% / 10%。"""),

    code("""print("Product ID value counts (top 20):")
print(df["Product ID"].value_counts().head(20))
print("\\nType value counts:")
print(df["Type"].value_counts())
print("\\nType %:")
print((df["Type"].value_counts(normalize=True) * 100).round(2))
print("\\nFailure rate by Type:")
print(df.groupby("Type")["Machine failure"].mean().round(4))"""),

    code("""fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

type_counts = df["Type"].value_counts().reindex(["L", "M", "H"])
ax[0].bar(type_counts.index, type_counts.values, color=["#72b7b2", "#e45756", "#54a24b"], width=0.5)
ax[0].set_xlabel("Type")
ax[0].set_ylabel("Count")
ax[0].set_title("Product Type distribution")
for i, v in enumerate(type_counts.values):
    ax[0].text(i, v + 100, f"{v:,}", ha="center", fontsize=11, fontweight="bold")

type_fail = df.groupby("Type")["Machine failure"].mean().reindex(["L", "M", "H"])
ax[1].bar(type_fail.index, type_fail.values * 100, color=["#72b7b2", "#e45756", "#54a24b"], width=0.5)
ax[1].set_xlabel("Type")
ax[1].set_ylabel("Failure rate (%)")
ax[1].set_title("Failure rate by Type")
for i, v in enumerate(type_fail.values * 100):
    ax[1].text(i, v + 0.3, f"{v:.2f}%", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()"""),

    md("""## 4. 數值型變數分佈

氣溫、程序溫度、轉速、扭矩、刀具磨損五個連續/離散數值變數。"""),

    code("""num_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, c in enumerate(num_cols):
    sns.histplot(data=df, x=c, bins=40, kde=True, color="#4c78a8", ax=axes[i], alpha=0.75)
    axes[i].set_title(c)
    axes[i].set_xlabel("")
fig.delaxes(axes[-1])
plt.suptitle("Numerical feature distributions", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

df[num_cols].describe()"""),

    code("""fig, ax = plt.subplots(figsize=(10, 5))
# Show boxplots for each feature by failure status
for c in num_cols:
    sns.boxplot(data=df, x="Machine failure", y=c, width=0.4, showcaps=True,
                boxprops=dict(alpha=.6), whiskerprops=dict(alpha=.6),
                medianprops=dict(color="black"), ax=ax)
plt.title("Numerical features vs Machine failure")
plt.tight_layout()
plt.show()"""),

    md("## 5. 相關性分析\n\n計算 Pearson 相關矩陣，並繪製熱力圖。"),

    code("""corr = df.corr(numeric_only=True)
corr.round(3)"""),

    code("""fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, linecolor="gray", cbar_kws={"label": "Pearson r"}, ax=ax)
plt.title("Correlation heatmap", fontsize=13, pad=16)
plt.tight_layout()
plt.show()

print("Top correlations with Machine failure (abs):")
target_corr = corr["Machine failure"].abs().sort_values(ascending=False)
print(target_corr[1:].round(4))"""),

    md("## 6. 特徵與目標變數的關係\n\n逐一觀察各數值特徵與故障標籤的關係（小提琴圖）。"),

    code("""fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for i, c in enumerate(num_cols):
    sns.violinplot(data=df, x="Machine failure", y=c, hue="Machine failure", split=True, inner="quartile",
                   palette={0: "#72b7b2", 1: "#e45756"}, ax=axes.flatten()[i])
    axes.flatten()[i].set_title(f"{c} by failure")
    axes.flatten()[i].set_xlabel("Machine failure")
plt.tight_layout()
plt.show()"""),

    md("""## 7. 衍生特徵

- **Power (W)** = Torque [Nm] × Angular velocity [rad/s] = Torque × Rotational speed × π / 30
- **Temp diff** = Process temperature − Air temperature

觀察其與失敗標籤的相關性，驗證業務邏輯（功率過高/過低導致失敗、溫差過小導致散熱失敗）。"""),

    code("""df = df.copy()
df["Power (W)"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * np.pi / 30)
df["Temp diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

derived_cols = ["Power (W)", "Temp diff"]
print("Correlations with Machine failure:")
for c in num_cols + derived_cols:
    print(f"  {c:>20}: {df[c].corr(df['Machine failure']):+.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.scatterplot(data=df, x="Power (W)", y="Machine failure", hue="Machine failure",
                palette={0: "#72b7b2", 1: "#e45756"}, alpha=0.5, s=15, ax=axes[0])
axes[0].set_title("Power vs Machine failure")
sns.scatterplot(data=df, x="Temp diff", y="Machine failure", hue="Machine failure",
                palette={0: "#72b7b2", 1: "#e45756"}, alpha=0.5, s=15, ax=axes[1])
axes[1].set_title("Temp diff vs Machine failure")
plt.tight_layout()
plt.show()"""),

    md("## 8. 異常值偵測（IQR 法）\n\n以四分位距（IQR）偵測數值變數的潛在異常點。"),

    code("""print("IQR-based outlier counts:")
for c in num_cols:
    q1 = df[c].quantile(0.25)
    q3 = df[c].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
    print(f"  {c:>20}: Q1={q1:>8.2f} Q3={q3:>8.2f} IQR={iqr:>7.2f} bounds=[{lo:>8.2f}, {hi:>8.2f}] outliers={n_out:,}")"""),

    md("## 9. 總結與發現"),

    code("""print("=" * 60)
print("EDA KEY FINDINGS")
print("=" * 60)
print(f"1. Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns, no missing values, no duplicates.")
print(f"2. Machine failure rate: {df['Machine failure'].mean():.2%} (failure: {int(df['Machine failure'].sum()):,}, normal: {int((df['Machine failure']==0).sum()):,}).")
print("3. Failure mode counts: " + ", ".join(f"{c}={int(df[c].sum())}" for c in failure_modes))
print("4. Type distribution: " + ", ".join(f"{k}={int(v)}" for k, v in df["Type"].value_counts().items()))
print("   Failure rate by Type: " + ", ".join(f"{k}={v:.2%}" for k, v in df.groupby("Type")["Machine failure"].mean().items()))
print(f"5. Strongest correlation with failure: Torque [Nm] ({corr.loc['Machine failure','Torque [Nm]']:+.3f}), Power ({df['Power (W)'].corr(df['Machine failure']):+.3f}).")
print(f"   Negative: Rotational speed ({corr.loc['Machine failure','Rotational speed [rpm]']:+.3f}), Temp diff ({df['Temp diff'].corr(df['Machine failure']):+.3f}).")
print(f"6. Rotational speed vs Torque correlation: {corr.loc['Rotational speed [rpm]','Torque [Nm]']:+.3f} (strong negative).")
print(f"   Air temp vs Process temp correlation: {corr.loc['Air temperature [K]','Process temperature [K]']:+.3f}.")
print("7. IQR outliers: Rotational speed=418, Torque=69 (others minimal).")
print("=" * 60)"""),

    md("""**EDA 完成。** 詳細數據摘要請參見 `reports/eda_summary.md`。

---
*生成時間：由 Phase 1 EDA 管線自動產出*"""),
]

nbf.write(nb, "/root/notebooks/eda.ipynb")
print("Notebook written to /root/notebooks/eda.ipynb")