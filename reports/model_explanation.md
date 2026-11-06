# 模型解釋性 (Phase 5)

RandomForest 特徵重要性 Top15：

| Feature | Importance |
|---|---|
| Rotational speed [rpm] | 0.0531 |
| Torque [Nm] | 0.0472 |
| Tool wear [min]_roll_mean_5 | 0.0393 |
| Torque [Nm]_pct_change_50 | 0.0364 |
| Power [W]_pct_change_50 | 0.0360 |
| Rotational speed [rpm]_pct_change_50 | 0.0268 |
| Power [W]_pct_change_10 | 0.0264 |
| Torque [Nm]_pct_change_20 | 0.0262 |
| Rotational speed [rpm]_pct_change_10 | 0.0255 |
| Power [W]_pct_change_20 | 0.0232 |
| Tool wear [min]_roll_mean_20 | 0.0218 |
| Tool wear [min]_roll_min_50 | 0.0212 |
| Temp diff [K]_roll_mean_20 | 0.0208 |
| Tool wear [min] | 0.0203 |
| Power [W] | 0.0185 |

Torque、Power 及相關 pct_change 特徵為主關鍵，與 EDA 相關性分析一致。
