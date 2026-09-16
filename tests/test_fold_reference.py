"""The grouped-fold split must be INVARIANT to --labels.

assign_grouped_folds stratifies on the y_ columns, so deriving folds from the
training labels CSV moved ~66% of studies across folds the moment a different
labels file was passed (relabel arm, 2026-08-24: uid-ovl 0.34).
fold_reference_frame pins the split to the canonical v4 labels.
"""
import importlib.util
import os
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.folds import assign_grouped_folds  # noqa: E402
from src.llm_labels import ID_COL, LABELS, build_targets, load_llm_labels  # noqa: E402

CANON = os.path.join(ROOT, "data_subset", "labels_external", "stevenleehans",
                     "llm_labels_v4_blend.csv")


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_slotknee", os.path.join(ROOT, "scripts", "train_slotknee.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def train_mod():
    return _load_train_module()


@pytest.fixture(scope="module")
def gold_df():
    return pd.read_csv(os.path.join(ROOT, "data_subset", "train_gold.csv"))


def _targets(labels_csv, gold_df, n=200):
    df = build_targets(load_llm_labels([labels_csv]), gold_df=gold_df, gold_weight=8.0)
    df = df.head(n).reset_index(drop=True)
    df["group"] = ["g%03d" % (i // 2) for i in range(len(df))]
    return df


@pytest.mark.skipif(not os.path.exists(CANON), reason="canonical labels not on disk")
def test_fold_split_invariant_to_labels(train_mod, gold_df, tmp_path):
    # A "relabel" CSV: canonical labels with a chunk of cells perturbed.
    canon_raw = pd.read_csv(CANON)
    perturbed = canon_raw.copy()
    lab_cols = [c for c in perturbed.columns if c != ID_COL][:6]
    perturbed.loc[perturbed.index[:150], lab_cols] = 0.05
    alt_csv = str(tmp_path / "llm_labels_alt.csv")
    perturbed.to_csv(alt_csv, index=False)

    base_df = _targets(CANON, gold_df)
    alt_df = _targets(alt_csv, gold_df)
    assert not np.allclose(base_df[[f"y_{LABELS[0]}"]].to_numpy(),
                           alt_df[[f"y_{LABELS[0]}"]].to_numpy()), "perturbation was a no-op"

    args = Namespace(data_dir=os.path.join(ROOT, "data_subset"),
                     labels=[alt_csv], gold_weight=8.0)
    fold_df = train_mod.fold_reference_frame(alt_df, alt_df["group"], args, gold_df)

    # y_ columns for splitting come from the canonical file, aligned to alt_df's rows...
    y_cols = [f"y_{l}" for l in LABELS]
    assert list(fold_df[ID_COL].astype(str)) == list(alt_df[ID_COL].astype(str))
    assert np.allclose(fold_df[y_cols].to_numpy(dtype=float),
                       base_df[y_cols].to_numpy(dtype=float), equal_nan=True)
    # ...so the resulting split is identical to the canonical one.
    f_alt = assign_grouped_folds(fold_df, n_folds=5, seed=42)
    f_base = assign_grouped_folds(base_df, n_folds=5, seed=42)
    assert (f_alt == f_base).all()


@pytest.mark.skipif(not os.path.exists(CANON), reason="canonical labels not on disk")
def test_canonical_labels_skip_reference(train_mod, gold_df):
    # Default arms (--labels == canonical) must keep the historical split bit-identically:
    # the frame is returned untouched.
    df = _targets(CANON, gold_df)
    args = Namespace(data_dir=os.path.join(ROOT, "data_subset"),
                     labels=[CANON], gold_weight=8.0)
    assert train_mod.fold_reference_frame(df, df["group"], args, gold_df) is df


def test_corrupt_sample_submission_still_seeds(tmp_path):
    """infer_slotknee must write a scoreable template even when sample_submission.csv
    is present but unreadable (it seeds BEFORE loading checkpoints, so the run below
    fails at checkpoint load — after the seed)."""
    import subprocess
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample_submission.csv").write_bytes(b"\x00\x89garbage\xff\xfe\n\x00")
    pd.DataFrame({"StudyInstanceUID": ["1.2.3", "4.5.6"]}).to_csv(
        data_dir / "test.csv", index=False)
    out = tmp_path / "submission.csv"
    r = subprocess.run(["python3", "scripts/infer_slotknee.py",
                        "--data-dir", str(data_dir),
                        "--ckpt", str(tmp_path / "missing.pt"),
                        "--out", str(out), "--decode-workers", "0"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0  # missing checkpoint: the run itself fails...
    assert out.exists()       # ...but a scoreable 0.5 template was already written
    df = pd.read_csv(out)
    assert list(df["StudyInstanceUID"].astype(str)) == ["1.2.3", "4.5.6"]
    assert (df.iloc[:, 1:].to_numpy() == 0.5).all()
