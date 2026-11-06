"""Phase 3: 模型建構 - TDD 實作版。"""
import pickle, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

TARGET = "Machine failure"

def load_processed(path="data/processed/ai4i2020_processed.csv"):
    df = pd.read_csv(path)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    cols = X.columns.tolist()
    return X, y, cols

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_baseline_models(X_train, y_train):
    from xgboost import XGBClassifier
    if hasattr(X_train, "values"): X_train = X_train.values
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logistic = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    logistic.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=50, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    xgb = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, eval_metric="logloss", scale_pos_weight=spw, n_jobs=-1)
    xgb.fit(X_train, y_train)
    return {"logistic": logistic, "random_forest": rf, "xgboost": xgb}

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc = 0.0
    return {"accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc}

def save_model(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f: pickle.dump(model, f)

def load_model(path):
    with open(path, "rb") as f: return pickle.load(f)

if __name__ == "__main__":
    X, y, cols = load_processed()
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    models = train_baseline_models(X_tr, y_tr)
    for n, m in models.items():
        print(n, evaluate_model(m, X_te, y_te))
        save_model(m, f"models/saved/{n}.pkl")
