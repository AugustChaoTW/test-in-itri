import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_load_processed_has_target_and_features():
    from src.models.train import load_processed
    X, y, cols = load_processed("data/processed/ai4i2020_processed.csv")
    assert X.shape[0] == y.shape[0] == 10000
    assert len(cols) == X.shape[1]
    assert set(y.unique()) <= {0, 1}
    assert "Machine failure" not in cols  # target not as feature

def test_split_data_shapes_and_no_overlap():
    from src.models.train import load_processed, split_data
    X, y, _ = load_processed("data/processed/ai4i2020_processed.csv")
    X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2, random_state=42)
    assert X_tr.shape[0] + X_te.shape[0] == X.shape[0]
    assert len(X_te) == 2000
    assert set(y_tr.unique()) <= {0, 1}

def test_train_baseline_models_predicts():
    from src.models.train import load_processed, split_data, train_baseline_models
    X, y, _ = load_processed("data/processed/ai4i2020_processed.csv")
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    models = train_baseline_models(X_tr, y_tr)
    assert set(models.keys()) >= {"logistic", "random_forest", "xgboost"}
    for name, m in models.items():
        pred = m.predict(X_te.head(5))
        assert len(pred) == 5

def test_evaluate_returns_required_metrics():
    from src.models.train import load_processed, split_data, train_baseline_models, evaluate_model
    X, y, _ = load_processed("data/processed/ai4i2020_processed.csv")
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    models = train_baseline_models(X_tr, y_tr)
    metrics = evaluate_model(models["random_forest"], X_te, y_te)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert k in metrics and 0 <= metrics[k] <= 1

def test_save_and_load_model_roundtrip(tmp_path):
    import pickle
    from src.models.train import load_processed, split_data, train_baseline_models, save_model, load_model
    X, y, _ = load_processed("data/processed/ai4i2020_processed.csv")
    X_tr, _, y_tr, _ = split_data(X, y)
    models = train_baseline_models(X_tr, y_tr)
    path = str(tmp_path / "model.pkl")
    save_model(models["xgboost"], path)
    loaded = load_model(path)
    assert len(loaded.predict(X.head(3))) == 3
