"""Auxiliary slot-identity head (SlotKneeS(aux_slotid=True), trainer --aux-slotid)."""
import os, sys, json, subprocess
import pytest, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.slotknee import SlotKneeS   # noqa: E402


def _tiny(**kw):
    return SlotKneeS(P=28, T=1, n_slots=6, max_groups=8, trainable_blocks=1, pretrained=False,
                     backbone="vit_small_patch14_dinov2.lvd142m", mixer_layers=1, **kw)


def test_slotid_logits_shape_and_hparams_roundtrip():
    m = _tiny(aux_slotid=True).eval()
    x = torch.randint(0, 255, (2, 6, 3, 1, 28, 28), dtype=torch.uint8)
    mask = torch.ones(2, 6)
    mask[0, 5] = 0
    with torch.no_grad():
        out = m(x, mask)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (2, 12)
    assert m.last_slotid_logits is not None and tuple(m.last_slotid_logits.shape) == (2, 6, 3, 6)
    assert m.hparams["aux_slotid"] is True
    # the checkpoint contract used by infer: rebuild from hparams and load the state dict strictly
    hp = {k: v for k, v in m.hparams.items() if k not in ("pretrained", "pretrained_path")}
    m2 = SlotKneeS(pretrained=False, **hp)
    m2.load_state_dict(m.state_dict())
    assert m2.slotid_head is not None


def test_default_has_no_slotid_head():
    m = _tiny()
    assert m.slotid_head is None and m.hparams["aux_slotid"] is False
    with torch.no_grad():
        m.eval()(torch.zeros(1, 6, 2, 1, 28, 28, dtype=torch.uint8), torch.ones(1, 6))
    assert m.last_slotid_logits is None


def test_aux_slotid_micro_run(tmp_path):
    data_dir = "data_subset"
    if not os.path.isdir(os.path.join(ROOT, data_dir)):
        pytest.skip("data_subset not available locally")
    cache_dir, out_dir = os.path.join(tmp_path, "cache"), os.path.join(tmp_path, "out")
    subprocess.run(["python3", "scripts/build_slot_cache.py", "--data-dir", data_dir, "--split", "train",
                    "--out", cache_dir, "--limit", "6", "--P", "224", "--workers", "1"], check=True, cwd=ROOT)
    subprocess.run(["python3", "scripts/train_slotknee.py", "--cache", cache_dir, "--data-dir", data_dir,
                    "--folds", "0", "--epochs", "1", "--bs", "2", "--trainable-blocks", "1", "--out", out_dir,
                    "--seed", "42", "--amp", "cpu", "--max-studies", "6", "--aux-slotid", "0.1"], check=True, cwd=ROOT)
    ck = torch.load(os.path.join(out_dir, "fold_0_best.pt"), map_location="cpu")
    assert ck["hparams"]["aux_slotid"] is True
    with open(os.path.join(out_dir, "fold_0_log.json")) as f:
        assert json.load(f)[0]["train_loss"] == json.load(open(os.path.join(out_dir, "fold_0_log.json")))[0]["train_loss"]
