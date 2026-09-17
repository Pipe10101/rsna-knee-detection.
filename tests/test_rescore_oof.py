"""scripts/rescore_oof.py: multi-bag re-scoring of a fold's held-out studies from the cache.

Mirrors test_slotknee_pipeline's micro fixture (6 studies, 1 epoch, CPU) and checks the tool
runs end to end, scores both read-outs, and writes an OOF-style npz.
"""
import json, os, subprocess, sys
import numpy as np, pytest

_ROOT_FOR_IMAGES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Tests below build a cache from real DICOMs: skip when the image subset is absent (the CSVs
# and label files can be present without it).
_HAVE_IMAGES = os.path.isdir(os.path.join(_ROOT_FOR_IMAGES, "data_subset", "train_images")) or \
               os.path.isdir(os.path.join(_ROOT_FOR_IMAGES, "data_subset", "train_series"))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(not _HAVE_IMAGES, reason="data_subset/train_images not present")
def test_rescore_oof_micro(tmp_path):
    data_dir = "data_subset"
    if not os.path.isdir(os.path.join(ROOT, data_dir)):
        pytest.skip("data_subset not available locally")
    cache_dir, out_dir = os.path.join(tmp_path, "cache"), os.path.join(tmp_path, "out")
    subprocess.run(["python3", "scripts/build_slot_cache.py", "--data-dir", data_dir, "--split", "train",
                    "--out", cache_dir, "--limit", "6", "--P", "224", "--workers", "1"], check=True, cwd=ROOT)
    subprocess.run(["python3", "scripts/train_slotknee.py", "--cache", cache_dir, "--data-dir", data_dir,
                    "--folds", "0", "--epochs", "1", "--bs", "2", "--trainable-blocks", "1", "--out", out_dir,
                    "--seed", "42", "--amp", "cpu", "--max-studies", "6"], check=True, cwd=ROOT)
    ckpt, oof, out = (os.path.join(out_dir, "fold_0_best.pt"), os.path.join(out_dir, "oof_fold_0.npz"),
                      os.path.join(tmp_path, "oof_fold_0_bags.npz"))
    r = subprocess.run([sys.executable, "scripts/rescore_oof.py", "--cache", cache_dir, "--ckpt", ckpt, "--oof", oof,
                        "--bag", "2", "--bags", "2", "--bs", "2", "--device", "cpu", "--out", out],
                       check=True, cwd=ROOT, capture_output=True, text=True)
    rep = json.loads(r.stdout[r.stdout.index("{"): r.stdout.rindex("}") + 1])
    z = np.load(out, allow_pickle=True)
    n = len(np.load(oof, allow_pickle=True)["uids"])
    assert rep["n"] == n and z["logits"].shape == (n, 12) and z["logits_full"].shape == (n, 12)
    assert np.isfinite(z["logits"]).all() and np.isfinite(z["logits_full"]).all()
    # a 2-of-3 bag is a genuinely different read-out from all 3 anchors
    assert not np.allclose(z["logits"], z["logits_full"])
    # --bag must be < G: the tool refuses rather than silently running full-anchor twice
    bad = subprocess.run([sys.executable, "scripts/rescore_oof.py", "--cache", cache_dir, "--ckpt", ckpt, "--oof", oof,
                          "--bag", "3", "--bags", "2", "--device", "cpu"], cwd=ROOT, capture_output=True, text=True)
    assert bad.returncode != 0 and "must be in 1.." in (bad.stdout + bad.stderr)
