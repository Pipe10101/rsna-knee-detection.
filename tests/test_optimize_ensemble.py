import json
import os
import sys
import pandas as pd
import pytest

import importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("optimize_ensemble_weights", os.path.join(ROOT, "scripts", "optimize_ensemble_weights.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["scripts.optimize_ensemble_weights"] = mod
spec.loader.exec_module(mod)
optimize_weights = mod.optimize_weights

def test_optimize_weights_atomic_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    train_csv = data_dir / "train.csv"
    train_df = pd.DataFrame({"StudyInstanceUID": ["001", "002"], "ACL": [1, 0]})
    train_df.to_csv(train_csv, index=False)
    
    oof_csv = tmp_path / "oof.csv"
    oof_df = pd.DataFrame({"StudyInstanceUID": ["001", "002"], "fold": [0, 1], "ACL_pred": [0.9, 0.1]})
    oof_df.to_csv(oof_csv, index=False)
    
    optimize_weights(str(oof_csv), data_dir=str(data_dir))
    
    weights_path = tmp_path / "results" / "ensemble_weights.json"
    assert weights_path.exists()
    assert not (tmp_path / "results" / "ensemble_weights.json.tmp").exists()
    
    with open(weights_path) as f:
        data = json.load(f)
    assert "fold_0" in data and "fold_1" in data
