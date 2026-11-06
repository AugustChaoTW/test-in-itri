import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent

def test_ble001_no_blind_exception():
    text = (ROOT / "src/models/train.py").read_text()
    assert "except Exception:" not in text

def test_f401_no_unused_pickle_in_test_models():
    text = (ROOT / "tests/test_models.py").read_text()
    assert "import pickle" not in text

def test_ruf059_no_unused_y_te():
    text = (ROOT / "tests/test_models.py").read_text()
    assert "y_tr, y_te =" not in text
    assert "X_tr, X_te, y_tr, y_te" not in text

def test_perf102_use_values():
    text = (ROOT / "tests/test_models.py").read_text()
    assert "models.items()" not in text
    assert "models.values()" in text

def test_c408_no_dict_call_in_notebooks():
    for nb in ["notebooks/eda.ipynb", "notebooks/eda_executed.ipynb"]:
        text = (ROOT / nb).read_text()
        assert "dict(" not in text, f"{nb} contains dict()"
