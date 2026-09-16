"""Laterality-mirror augmentation (--flip-swap) and mirror TTA (rescore_oof --flip-tta).

The cache normalises every study to a left knee (COR/AX columns flipped, SAG anchor order
reversed for right knees).  Re-applying that transform is an exact mirror whose medial and
lateral structures swap sides; the four medial/lateral labels must swap with it, and the
transform must be an involution.  These tests pin that contract without any real data.
"""
import os, sys, json, subprocess
import numpy as np, pytest, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.llm_labels import LABELS                       # noqa: E402
from src import slots as S                              # noqa: E402
import importlib.util
spec = importlib.util.spec_from_file_location("rescore_oof", os.path.join(ROOT, "scripts", "rescore_oof.py"))
rescore = importlib.util.module_from_spec(spec); spec.loader.exec_module(rescore)

SLOTS = ["SAG_FS", "COR_FS", "AX_FS", "SAG_T1", "COR_T1", "AX_T1"]


def test_lat_perm_swaps_exactly_the_four_paired_labels_and_is_an_involution():
    perm = rescore.lat_perm(LABELS)
    swapped = {LABELS[i] for i in range(len(LABELS)) if perm[i] != i}
    assert swapped == {"Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"}
    assert [perm[perm[i]] for i in range(len(perm))] == list(range(len(perm)))
    assert LABELS[perm[LABELS.index("Medial Meniscus")]] == "Lateral Meniscus"
    assert LABELS[perm[LABELS.index("Medial OA")]] == "Lateral OA"
    assert perm[LABELS.index("MCL")] == LABELS.index("MCL")           # MCL does not change side


def test_mirror_matches_cache_laterality_normalisation_and_is_an_involution():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, size=(6, 4, 1, 8, 8), dtype=np.uint8)           # [S, G, T, P, P]
    # reference: what the cache does to a RIGHT knee
    ref = S.apply_laterality(x, "R", slot_names=SLOTS)
    got = rescore.mirror_study(torch.from_numpy(x)[None], SLOTS)[0].numpy()
    assert np.array_equal(got, ref), "mirror_study must reproduce apply_laterality exactly"
    twice = rescore.mirror_study(rescore.mirror_study(torch.from_numpy(x)[None], SLOTS), SLOTS)[0].numpy()
    assert np.array_equal(twice, x)
    # plane semantics: COR flips columns, SAG reverses the anchor axis, pixels themselves untouched
    assert np.array_equal(got[1], x[1][..., ::-1])
    assert np.array_equal(got[0], x[0][::-1])


def test_trainer_mirror_uses_kept_slot_names(tmp_path):
    """The dataset's _mirror must follow the KEPT slot names (per-plane specialists subset slots)."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib
    train = importlib.import_module("train_slotknee")
    ds = train.SlotKneeDataset.__new__(train.SlotKneeDataset)
    ds.slot_names_kept = ["COR_FS", "COR_T1"]                              # a coronal specialist
    x = torch.arange(2 * 3 * 1 * 4 * 4, dtype=torch.uint8).reshape(2, 3, 1, 4, 4)
    m = ds._mirror(x)
    assert torch.equal(m, x.flip(-1))                                       # both kept slots are coronal
    ds.slot_names_kept = ["SAG_FS", "SAG_T1"]
    assert torch.equal(ds._mirror(x), x.flip(1))                            # sagittal: anchor axis


def test_flip_swap_micro_run(tmp_path):
    """--flip-swap 0.5 trains end to end on the 6-study fixture; --flip-tta rescoring runs on its output."""
    data_dir = "data_subset"
    if not os.path.isdir(os.path.join(ROOT, data_dir)):
        pytest.skip("data_subset not available locally")
    cache_dir, out_dir = os.path.join(tmp_path, "cache"), os.path.join(tmp_path, "out")
    subprocess.run(["python3", "scripts/build_slot_cache.py", "--data-dir", data_dir, "--split", "train",
                    "--out", cache_dir, "--limit", "6", "--P", "224", "--workers", "1"], check=True, cwd=ROOT)
    subprocess.run(["python3", "scripts/train_slotknee.py", "--cache", cache_dir, "--data-dir", data_dir,
                    "--folds", "0", "--epochs", "1", "--bs", "2", "--trainable-blocks", "1", "--out", out_dir,
                    "--seed", "42", "--amp", "cpu", "--max-studies", "6", "--flip-swap", "0.5"], check=True, cwd=ROOT)
    r = subprocess.run([sys.executable, "scripts/rescore_oof.py", "--cache", cache_dir,
                        "--ckpt", os.path.join(out_dir, "fold_0_best.pt"), "--oof", os.path.join(out_dir, "oof_fold_0.npz"),
                        "--bag", "2", "--bags", "1", "--bs", "2", "--device", "cpu", "--flip-tta"],
                       check=True, cwd=ROOT, capture_output=True, text=True)
    rep = json.loads(r.stdout[r.stdout.index("{"): r.stdout.rindex("}") + 1])
    assert rep["flip_tta"] is True and rep["n"] > 0


def test_medial_zoom_slot_is_dropped_on_every_mirror_pass():
    """A medial-centred zoom slot (COR_FS_Z80M) is anchored on the medial compartment, so its mirror is
    NOT lateral anatomy: every mirror helper must zero it and clear its mask instead of flipping it."""
    from src.slots import is_medial_slot
    assert [is_medial_slot(n) for n in ("COR_FS_Z80M", "SAG_FS_Z87.5M", "COR_FS_Z80J", "COR_FS_Z", "COR_FS", "COR_T1_Z100")] == \
        [True, True, False, False, False, False]
    ispec = importlib.util.spec_from_file_location("infer_slotknee", os.path.join(ROOT, "scripts", "infer_slotknee.py"))
    infer = importlib.util.module_from_spec(ispec); ispec.loader.exec_module(infer)
    names = SLOTS + ["COR_FS_Z80M"]
    torch.manual_seed(0)
    x = torch.randint(0, 256, (1, 7, 3, 1, 8, 8), dtype=torch.uint8)
    m = torch.ones(1, 7, dtype=torch.uint8)
    # rescore_oof: base slots mirror exactly as before, the medial slot is zeroed and masked out
    xr = rescore.mirror_study(x, names)
    assert torch.equal(xr[:, :6], rescore.mirror_study(x[:, :6], SLOTS)) and int(xr[:, 6].max()) == 0
    mr = rescore.mirror_mask(m, names)
    assert mr[:, :6].tolist() == [[1] * 6] and int(mr[0, 6]) == 0 and int(m[0, 6]) == 1     # input untouched
    # infer_slotknee: identical semantics
    assert torch.equal(infer._mirror_study(x, names), xr) and torch.equal(infer._mirror_mask(m, names), mr)
    # trainer dataset ([S, G, T, P, P] + [S] mask)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib as _il
    train = _il.import_module("train_slotknee")
    ds = train.SlotKneeDataset.__new__(train.SlotKneeDataset)
    ds.slot_names_kept = names
    xt, mt = ds._mirror(x[0]), ds._mirror_mask(m[0])
    ds.slot_names_kept = SLOTS
    assert torch.equal(xt[:6], ds._mirror(x[0][:6])) and int(xt[6].max()) == 0
    assert mt.tolist() == [1] * 6 + [0] and m[0].tolist() == [1] * 7
