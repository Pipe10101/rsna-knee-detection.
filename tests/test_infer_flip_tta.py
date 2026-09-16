"""infer_slotknee.py --flip-tta: the submission path scores the laterality-mirrored study too."""
import os, sys, subprocess
import numpy as np, pandas as pd, pytest, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import importlib.util
spec = importlib.util.spec_from_file_location("infer_slotknee", os.path.join(ROOT, "scripts", "infer_slotknee.py"))
infer = importlib.util.module_from_spec(spec); spec.loader.exec_module(infer)
from src import slots as S                                    # noqa: E402
from src.llm_labels import LABELS                             # noqa: E402


def test_infer_mirror_matches_cache_normalisation_and_perm_is_involution():
    x = torch.randint(0, 255, (1, 6, 3, 1, 8, 8), dtype=torch.uint8)
    names = ["SAG_FS", "COR_FS", "AX_FS", "SAG_T1", "COR_T1", "AX_T1"]
    ref = S.apply_laterality(x[0].numpy(), "R", slot_names=names)
    assert np.array_equal(infer._mirror_study(x, names)[0].numpy(), ref)
    perm = infer._lat_perm(LABELS)
    assert [perm[perm[i]] for i in range(len(perm))] == list(range(len(perm)))
    assert LABELS[perm[LABELS.index("Lateral OA")]] == "Medial OA" and perm[LABELS.index("ACL")] == LABELS.index("ACL")


def test_infer_flip_tta_micro(tmp_path):
    data_dir = "data_subset"
    if not os.path.isdir(os.path.join(ROOT, data_dir)):
        pytest.skip("data_subset not available locally")
    cache_dir, out_dir = os.path.join(tmp_path, "cache"), os.path.join(tmp_path, "out")
    subprocess.run(["python3", "scripts/build_slot_cache.py", "--data-dir", data_dir, "--split", "train",
                    "--out", cache_dir, "--limit", "6", "--P", "224", "--workers", "1"], check=True, cwd=ROOT)
    subprocess.run(["python3", "scripts/train_slotknee.py", "--cache", cache_dir, "--data-dir", data_dir,
                    "--folds", "0", "--epochs", "1", "--bs", "2", "--trainable-blocks", "1", "--out", out_dir,
                    "--seed", "42", "--amp", "cpu", "--max-studies", "6"], check=True, cwd=ROOT)
    ck = os.path.join(out_dir, "fold_0_best.pt")
    assert torch.load(ck, map_location="cpu")["slot_layout"]["slot_names"]   # what --flip-tta relies on
    plain, tta = os.path.join(tmp_path, "plain.csv"), os.path.join(tmp_path, "tta.csv")
    common = ["python3", "scripts/infer_slotknee.py", "--data-dir", data_dir, "--ckpt", ck, "--P", "224",
              "--bs", "2", "--amp", "cpu", "--decode-workers", "0"]
    subprocess.run(common + ["--out", plain], check=True, cwd=ROOT)
    subprocess.run(common + ["--out", tta, "--flip-tta"], check=True, cwd=ROOT)
    a, b = pd.read_csv(plain), pd.read_csv(tta)
    assert list(a.columns) == list(b.columns) and list(a.iloc[:, 0]) == list(b.iloc[:, 0])
    va, vb = a.iloc[:, 1:].values, b.iloc[:, 1:].values
    assert (vb > 0).all() and (vb < 1).all()
    assert not np.allclose(va, vb)                                # the mirrored view changes the read-out
