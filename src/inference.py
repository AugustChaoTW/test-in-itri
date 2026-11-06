"""Phase 5: 推論流程 - 載入主模型 XGBoost 預測故障機率。"""
import pickle
def predict_proba(model_path, X):
    with open(model_path, "rb") as f: m = pickle.load(f)
    return m.predict_proba(X)[:, 1]
if __name__ == "__main__":
    from src.models.train import load_processed
    X, y, _ = load_processed()
    p = predict_proba("models/saved/xgboost.pkl", X.head(5))
    print(p)
