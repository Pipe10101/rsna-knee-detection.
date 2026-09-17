import json
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.select_ensemble import main

def test_select_ensemble_atomic_write(tmp_path, monkeypatch):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run1.mkdir()
    run2.mkdir()
    
    uids = np.array(["001", "002", "003", "004", "005"])
    y = np.array([[1, 0] * 6, [0, 1] * 6, [1, 0] * 6, [0, 1] * 6, [1, 0] * 6], dtype=float)
    w = np.ones((5, 12))
    
    np.savez(run1 / "oof_fold_0.npz", uids=uids[:3], logits=np.full((3, 12), 0.8), y=y[:3], w=w[:3])
    np.savez(run1 / "oof_fold_1.npz", uids=uids[3:], logits=np.full((2, 12), 0.7), y=y[3:], w=w[3:])
    
    np.savez(run2 / "oof_fold_0.npz", uids=uids[:3], logits=np.full((3, 12), 0.6), y=y[:3], w=w[:3])
    np.savez(run2 / "oof_fold_1.npz", uids=uids[3:], logits=np.full((2, 12), 0.5), y=y[3:], w=w[3:])
    
    out_file = str(tmp_path / "ensemble.json")
    monkeypatch.setattr(sys, "argv", ["select_ensemble.py", str(run1), str(run2), "--out", out_file])
    main()
    
    assert os.path.exists(out_file)
    assert not os.path.exists(out_file + ".tmp")
    with open(out_file) as f:
        data = json.load(f)
    assert "members" in data
