"""The checkpoint's slot_layout must carry the TRAINING cache's anchor band (trim_frac) and crop size,
and infer_slotknee._decode_study must rebuild the study with them: a model trained on the central-block
(t35) cache read with the default 0.15 band would be a silent train/test mismatch (found 2026-09-09)."""
import importlib.util, json, os, subprocess, sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import slots                                   # noqa: E402

DATA_DIR = os.path.join(ROOT, "data_subset")
BUILDER = os.path.join(ROOT, "scripts", "build_slot_cache.py")
TRAINER = os.path.join(ROOT, "scripts", "train_slotknee.py")
_spec = importlib.util.spec_from_file_location("infer_slotknee", os.path.join(ROOT, "scripts", "infer_slotknee.py"))
infer = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(infer)


@pytest.mark.skipif(not os.path.isdir(DATA_DIR), reason="data_subset not available locally")
def test_slot_layout_round_trips_trim_frac_and_crop_mm(tmp_path):
    cache = str(tmp_path / "cache_t35")
    subprocess.run([sys.executable, BUILDER, "--data-dir", DATA_DIR, "--split", "train", "--out", cache,
                    "--limit", "4", "--P", "224", "--G", "4", "--T", "1", "--trim-frac", "0.35", "--workers", "1"],
                   check=True, cwd=ROOT)
    c = slots.SlotCache(cache, "train")
    assert c.trim_frac == 0.35 and c.crop_mm == 140.0 and c.G == 4 and c.T == 1
    out = str(tmp_path / "out")
    subprocess.run([sys.executable, TRAINER, "--cache", cache, "--data-dir", DATA_DIR, "--folds", "0", "--epochs", "1",
                    "--bs", "2", "--lr-head", "3e-4", "--lr-backbone", "5e-5", "--trainable-blocks", "1", "--out", out,
                    "--seed", "42", "--amp", "cpu", "--max-studies", "4", "--aux-weight", "0.2"], check=True, cwd=ROOT)
    layout = torch.load(os.path.join(out, "fold_0_best.pt"), map_location="cpu")["slot_layout"]
    assert layout["trim_frac"] == 0.35 and layout["crop_mm"] == 140.0 and layout["G"] == 4 and layout["T"] == 1
    # _decode_study must rebuild with THAT band: equal to a 0.35 build, different from a 0.15 build
    uid = c.uids[0]
    study_dir = os.path.join(DATA_DIR, "train_images", uid)
    lookup = slots.series_lookup(None, uid) if hasattr(slots, "series_lookup") else None
    xs, mask, ms, err = infer._decode_study((study_dir, lookup, 224, layout, [0]))
    assert err is None, err
    x35, m35, _ = slots.build_study_tensor(study_dir, series_df=lookup, P=224, G=4, T=1, trim_frac=0.35)
    x15, m15, _ = slots.build_study_tensor(study_dir, series_df=lookup, P=224, G=4, T=1, trim_frac=0.15)
    assert np.array_equal(xs[0], x35) and np.array_equal(mask, m35)
    assert not np.array_equal(x35, x15), "the two bands must select different slices on a real study"
    # the cached row is the 0.35 build too (same code path at cache time)
    assert np.array_equal(np.asarray(c[0][0]), x35)


@pytest.mark.skipif(not os.path.isdir(DATA_DIR), reason="data_subset not available locally")
def test_infer_refuses_mixed_layout_checkpoints(tmp_path):
    """Members trained on different anchor bands (or any other layout key) must not be scored on
    one decode: _check_layouts exits naming the offender; identical layouts pass."""
    src = os.path.join(ROOT, "data_subset")
    cache = str(tmp_path / "cache"); out = str(tmp_path / "out")
    subprocess.run([sys.executable, BUILDER, "--data-dir", src, "--split", "train", "--out", cache,
                    "--limit", "3", "--P", "224", "--G", "3", "--T", "1", "--workers", "1"], check=True, cwd=ROOT)
    subprocess.run([sys.executable, TRAINER, "--cache", cache, "--data-dir", src, "--folds", "0", "--epochs", "1",
                    "--bs", "2", "--trainable-blocks", "1", "--out", out, "--seed", "42", "--amp", "cpu",
                    "--max-studies", "3", "--aux-weight", "0.2"], check=True, cwd=ROOT)
    a = os.path.join(out, "fold_0_best.pt")
    ck = torch.load(a, map_location="cpu")
    same = os.path.join(out, "same.pt"); torch.save(ck, same)
    ck["slot_layout"] = dict(ck["slot_layout"], trim_frac=0.35)
    other = os.path.join(out, "t35.pt"); torch.save(ck, other)
    lay = infer._check_layouts([a, same])
    assert lay["trim_frac"] == 0.15 and lay["crop_mm"] == 140.0 and lay["G"] == 3
    with pytest.raises(SystemExit) as e:
        infer._check_layouts([a, same, other])
    assert "t35.pt" in str(e.value) and "trim_frac" in str(e.value)
    # a pre-2026-09-09 checkpoint (no trim_frac key) is the default band and groups with the 0.15 one
    old = dict(torch.load(a, map_location="cpu")); old["slot_layout"] = {k: v for k, v in old["slot_layout"].items() if k not in ("trim_frac", "crop_mm")}
    legacy = os.path.join(out, "legacy.pt"); torch.save(old, legacy)
    assert infer._check_layouts([a, legacy])["trim_frac"] == 0.15
