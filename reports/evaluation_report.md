# AI4I 2020 模型評估報告

TDD Phase3/4 產出，測試見 `tests/test_models.py` (5 passed)。

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| logistic | 0.8900 | 0.2164 | 0.8529 | 0.3452 | 0.9404 |
| random_forest | 0.9805 | 0.9677 | 0.4412 | 0.6061 | 0.9655 |
| xgboost | 0.9830 | 0.7024 | 0.8676 | 0.7763 | 0.9650 |

- XGBoost F1 最高 (0.776)，ROC-AUC 0.965，建議為主模型
- Logistic 高 Recall 低 Precision；RF 高 Precision 低 Recall
- 資料不平衡以 class_weight/scale_pos_weight 處理
