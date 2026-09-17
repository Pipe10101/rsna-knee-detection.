"""Tests for SlotKnee-S pipeline."""
import importlib.util
import json
import math
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src.llm_labels import LABELS, y_col, w_col
from src.slotknee import SlotKneeS

_ROOT_FOR_IMAGES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Tests below build a cache from real DICOMs: skip when the image subset is absent (the CSVs
# and label files can be present without it).
_HAVE_IMAGES = os.path.isdir(os.path.join(_ROOT_FOR_IMAGES, "data_subset", "train_images")) or \
               os.path.isdir(os.path.join(_ROOT_FOR_IMAGES, "data_subset", "train_series"))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the training script as a module (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "train_slotknee", os.path.join(ROOT, "scripts", "train_slotknee.py"))
train_slotknee = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_slotknee)


class _FakeCache:
    """Minimal SlotCache stand-in: background 50, bright stripe at cols 20-23."""

    def __init__(self, N=4, G=3, T=3, P=28):
        self.N, self.G, self.T, self.P = N, G, T, P
        self.uids = ["u%d" % i for i in range(N)]
        self.done = np.ones(N, dtype=bool)
        self.x = np.full((N, 6, G, T, P, P), 50, dtype=np.uint8)
        self.x[..., 20:24] = 250
        self.mask = np.ones((N, 6), dtype=np.uint8)
        self.mask[:, 5] = 0
        self.x[:, 5] = 0

    def __getitem__(self, i):
        return self.x[i], self.mask[i]


def _targets_df(uids):
    df = pd.DataFrame({"StudyInstanceUID": uids})
    for lab in LABELS:
        df[y_col(lab)] = 0.7
    for lab in LABELS:
        df[w_col(lab)] = 0.4
    df["is_gold"] = False
    return df


def test_augment_no_hflip_and_dropout_semantics():
    cache = _FakeCache()
    ds = train_slotknee.SlotKneeDataset(cache, _targets_df(cache.uids),
                                        is_train=True, slot_dropout=0.0)
    random.seed(0)
    saw_group_drop = False
    for _ in range(100):
        x, mask, y, w, is_gold = ds[0]
        assert x.dtype == torch.uint8 and tuple(x.shape) == (6, 3, 3, 28, 28)
        assert mask.dtype == torch.uint8
        # slot_dropout=0: the slot mask is never altered by augmentation
        assert torch.equal(mask, torch.from_numpy(cache.mask[0]))
        for s in range(5):  # slot 5 is absent
            for g in range(3):
                col_means = x[s, g].float().mean(dim=(0, 1))
                rng = float(col_means.max() - col_means.min())
                if rng < 30:
                    # group dropout zeroed this group (brightness may later lift
                    # it to a constant); the slot mask stays untouched either way
                    saw_group_drop = True
                    continue
                # NO hflip: the bright stripe must stay in the right half
                # (translation moves it by at most 4 px; a flip would move it
                # to columns 4-7).
                assert int(col_means.argmax()) > 14
    assert saw_group_drop  # p=0.2 per item; P(miss in 100 draws) ~ 2e-10

    # Slot-dropout zeroes pixels AND mask, but never the last present slot.
    ds_sd = train_slotknee.SlotKneeDataset(cache, _targets_df(cache.uids),
                                           is_train=True, slot_dropout=0.9)
    random.seed(1)
    saw_slot_drop = False
    for _ in range(20):
        x, mask, _, _, _ = ds_sd[0]
        assert int(mask.sum()) >= 1
        for s in range(6):
            if cache.mask[0, s] and not mask[s]:
                saw_slot_drop = True
                assert x[s].sum() == 0
    assert saw_slot_drop


def test_collate_shapes():
    from torch.utils.data import DataLoader
    cache = _FakeCache()
    ds = train_slotknee.SlotKneeDataset(cache, _targets_df(cache.uids), is_train=False)
    x, mask, y, w, is_gold = next(iter(DataLoader(ds, batch_size=2)))
    assert x.dtype == torch.uint8 and tuple(x.shape) == (2, 6, 3, 3, 28, 28)
    assert tuple(mask.shape) == (2, 6)
    assert y.dtype == torch.float32 and tuple(y.shape) == (2, 12)
    assert w.dtype == torch.float32 and tuple(w.shape) == (2, 12)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = SlotKneeS(P=28, T=3, d=32, trainable_blocks=1, pretrained=False,
                      drop_path=0.0)
    model.eval()
    x = torch.randint(0, 256, (2, 6, 2, 3, 28, 28), dtype=torch.uint8)
    mask = torch.ones(2, 6)
    with torch.no_grad():
        ref = model(x, mask)
    path = os.path.join(tmp_path, "ckpt.pt")
    torch.save({"state_dict": model.state_dict(), "hparams": model.hparams}, path)

    ckpt = torch.load(path, map_location="cpu")
    hparams = {k: v for k, v in ckpt["hparams"].items()
               if k not in ("pretrained", "pretrained_path")}
    m2 = SlotKneeS(pretrained=False, **hparams)
    m2.load_state_dict(ckpt["state_dict"])
    m2.eval()
    with torch.no_grad():
        out = m2(x, mask)
    assert torch.equal(ref, out)


# pytest-timeout is not installed; the < 90 s budget is enforced by CI wall-clock.
@pytest.mark.skipif(not _HAVE_IMAGES, reason="data_subset/train_images not present")
def test_slotknee_pipeline(tmp_path):
    """End-to-end micro-pipeline test in < 90s on CPU."""
    data_dir = "data_subset"
    if not os.path.isdir(data_dir):
        pytest.skip("data_subset not available locally")

    cache_dir = os.path.join(tmp_path, "cache")
    out_dir = os.path.join(tmp_path, "out")
    sub_csv = os.path.join(tmp_path, "submission.csv")

    # 1. Build cache
    subprocess.run([
        "python3", "scripts/build_slot_cache.py",
        "--data-dir", data_dir,
        "--split", "train",
        "--out", cache_dir,
        "--limit", "6",
        "--P", "224",
        "--workers", "1"
    ], check=True, cwd=ROOT)

    # 2. Train
    subprocess.run([
        "python3", "scripts/train_slotknee.py",
        "--cache", cache_dir,
        "--data-dir", data_dir,
        "--folds", "0",
        "--epochs", "1",
        "--bs", "2",
        "--lr-head", "3e-4",
        "--lr-backbone", "5e-5",
        "--trainable-blocks", "1",
        "--out", out_dir,
        "--seed", "42",
        "--amp", "cpu",  # Force CPU
        "--max-studies", "6",
        "--aux-weight", "0.2"
    ], check=True, cwd=ROOT)

    # Aux slot loss ran and stayed finite; the checkpoint records the aux head
    # so infer rebuilds the same architecture (tuple forward) below.
    with open(os.path.join(out_dir, "fold_0_log.json")) as f:
        log = json.load(f)
    assert math.isfinite(log[0]["train_loss"])
    ckpt = torch.load(os.path.join(out_dir, "fold_0_best.pt"), map_location="cpu")
    assert ckpt["hparams"]["aux_slot_logits"] is True

    # 3. Infer
    subprocess.run([
        "python3", "scripts/infer_slotknee.py",
        "--data-dir", data_dir,
        "--ckpt", os.path.join(out_dir, "fold_0_best.pt"),
        "--out", sub_csv,
        "--P", "224",
        "--bs", "2",
        "--amp", "cpu",
        "--decode-workers", "0"
    ], check=True, cwd=ROOT)

    # 4. Validate output: aligned to sample_submission, values in (0, 1)
    assert os.path.exists(sub_csv)
    df = pd.read_csv(sub_csv)
    sample = pd.read_csv(os.path.join(ROOT, data_dir, "sample_submission.csv"))
    assert list(df.columns) == list(sample.columns)
    assert list(df.iloc[:, 0]) == list(sample.iloc[:, 0])

    vals = df.iloc[:, 1:].values
    assert (vals > 0.0).all()
    assert (vals < 1.0).all()


# --------------------------------------------------------------------------- #
# 2026-08-24 audit: mixup safety and the cached-activation path
# --------------------------------------------------------------------------- #

def test_mixup_batch_scales_uint8_before_blending():
    """_normalise only divides by 255 for INTEGER inputs; a float blend of raw
    uint8 pixels would reach the encoder 255x too large."""
    torch.manual_seed(0)
    x = torch.randint(0, 256, (4, 6, 2, 3, 8, 8), dtype=torch.uint8)
    mask = torch.ones(4, 6, dtype=torch.uint8)
    y = torch.rand(4, 12)
    w = torch.rand(4, 12)
    lam, idx = 0.7, torch.tensor([1, 0, 3, 2])
    mx, mmask, my, mw, md = train_slotknee.mixup_batch(x, mask, y, w, None, lam, idx)
    assert mx.is_floating_point()
    assert float(mx.max()) <= 1.0 + 1e-6
    expected = lam * (x.float() / 255.0) + (1 - lam) * (x[idx].float() / 255.0)
    assert torch.allclose(mx, expected, atol=1e-6)
    assert torch.allclose(my, lam * y + (1 - lam) * y[idx])
    assert torch.allclose(mw, lam * w + (1 - lam) * w[idx])
    assert md is None
    # float input (cached activations) must NOT be rescaled
    xf = torch.rand(4, 6, 2, 5, 16)
    mxf, _, _, _, _ = train_slotknee.mixup_batch(xf, mask, y, w, None, lam, idx)
    assert torch.allclose(mxf, lam * xf + (1 - lam) * xf[idx], atol=1e-6)


def test_mixup_batch_ors_masks():
    """Blended pixels contain every slot present in EITHER study, so the mask is
    OR (the old AND hid slots whose pixels were still in the blend)."""
    x = torch.randint(0, 256, (2, 6, 2, 3, 8, 8), dtype=torch.uint8)
    mask = torch.tensor([[1, 1, 0, 0, 1, 1],
                         [1, 0, 1, 0, 0, 1]], dtype=torch.uint8)
    y = torch.rand(2, 12)
    w = torch.rand(2, 12)
    idx = torch.tensor([1, 0])
    _, mmask, _, _, _ = train_slotknee.mixup_batch(x, mask, y, w, None, 0.6, idx)
    assert torch.equal(mmask, mask | mask[idx])


def test_mixup_batch_nan_safe_distill():
    """A study without teacher probabilities (NaN row) must not poison its mix
    partner; NaN survives only where both sides are NaN."""
    x = torch.randint(0, 256, (4, 6, 2, 3, 8, 8), dtype=torch.uint8)
    mask = torch.ones(4, 6, dtype=torch.uint8)
    y = torch.rand(4, 12)
    w = torch.rand(4, 12)
    dist = torch.rand(4, 12)
    dist[1] = float("nan")
    dist[3] = float("nan")
    lam, idx = 0.7, torch.tensor([1, 0, 3, 2])   # pairs (0,1) and (2,3)
    _, _, _, _, md = train_slotknee.mixup_batch(x, mask, y, w, dist.clone(), lam, idx)
    assert torch.allclose(md[0], dist[0])   # valid row keeps its own targets
    assert torch.allclose(md[1], dist[0])   # NaN row inherits its partner's
    assert torch.allclose(md[2], dist[2])
    assert torch.allclose(md[3], dist[2])
    assert not torch.isnan(md).any()
    # both sides NaN -> stays NaN (the distill loss masks those cells)
    all_nan = torch.full((4, 12), float("nan"))
    _, _, _, _, md2 = train_slotknee.mixup_batch(x, mask, y, w, all_nan, lam, idx)
    assert torch.isnan(md2).all()


def test_cached_activation_forward_matches_uncached():
    """forward(is_cached=True) on encode_frozen tokens == the raw-pixel forward
    (the frozen prefix is deterministic in eval mode); also pins the token count
    formula: (P/14)^2 patches + prefix tokens, NOT a hardcoded 257."""
    torch.manual_seed(0)
    model = SlotKneeS(P=28, T=3, d=32, trainable_blocks=1, pretrained=False,
                      drop_path=0.0)
    model.eval()
    B, S, G = 2, 6, 2
    x = torch.randint(0, 256, (B, S, G, 3, 28, 28), dtype=torch.uint8)
    mask = torch.ones(B, S, dtype=torch.uint8)
    mask[0, 3] = 0
    with torch.no_grad():
        ref = model(x, mask)
        ref = ref[0] if isinstance(ref, tuple) else ref
        toks = model.encode_frozen(x.reshape(B * S * G, 3, 28, 28))
        n_tok = (28 // 14) ** 2 + model.num_prefix_tokens
        assert tuple(toks.shape) == (B * S * G, n_tok, model.embed_dim)
        cached = toks.reshape(B, S, G, n_tok, model.embed_dim)
        out = model(cached, mask, is_cached=True)
        out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(ref, out, atol=1e-5)


def test_open_cached_activations_meta_and_mismatch(tmp_path):
    cache = _FakeCache()  # N=4, G=3, S defaults to 6 via getattr
    path = os.path.join(tmp_path, "act.memmap")
    shape = (4, 6, 3, 5, 16)
    arr = np.memmap(path, dtype=np.float16, mode="w+", shape=shape)
    arr[:] = 1.0
    arr.flush()
    meta = {"shape": list(shape), "dtype": "float16", "P": 28, "T": 3, "S": 6,
            "G": 3, "n_tokens": 5, "embed_dim": 16,
            "backbone": "vit_small_patch14_dinov2.lvd142m", "trainable_blocks": 1}
    with open(path + ".json", "w") as f:
        json.dump(meta, f)

    mm = train_slotknee.open_cached_activations(path, cache)
    assert mm.shape == shape and mm.dtype == np.float16

    # matching settings pass
    train_slotknee.check_cached_activations(
        path, cache, trainable_blocks=1, backbone="vit_small_patch14_dinov2.lvd142m")
    # trainable-blocks mismatch would silently skip encoder blocks -> refused
    with pytest.raises(SystemExit):
        train_slotknee.check_cached_activations(
            path, cache, trainable_blocks=4, backbone="vit_small_patch14_dinov2.lvd142m")
    with pytest.raises(SystemExit):
        train_slotknee.check_cached_activations(
            path, cache, trainable_blocks=1, backbone="vit_base_patch14_dinov2.lvd142m")
    # missing file is a hard error, not a silent fall-back to uncached training
    with pytest.raises(SystemExit):
        train_slotknee.check_cached_activations(
            os.path.join(tmp_path, "nope.memmap"), cache, 1, "x")
    # shape/cache mismatch is refused at open time
    meta["shape"] = [5, 6, 3, 5, 16]
    with open(path + ".json", "w") as f:
        json.dump(meta, f)
    with pytest.raises(SystemExit):
        train_slotknee.open_cached_activations(path, cache)


def test_yw_schema_csv_detected(tmp_path):
    """pseudo_fill.py output (y_/w_ columns) must bypass load_llm_labels, which
    would match y_ columns as probabilities and drop the baked-in weights."""
    p = os.path.join(tmp_path, "pseudo.csv")
    df = pd.DataFrame({"StudyInstanceUID": ["u0", "u1"]})
    for lab in LABELS:
        df[y_col(lab)] = 0.7
    for lab in LABELS:
        df[w_col(lab)] = 0.3
    df.to_csv(p, index=False)
    assert train_slotknee._is_yw_csv(p)

    q = os.path.join(tmp_path, "probs.csv")
    df2 = pd.DataFrame({"StudyInstanceUID": ["u0"]})
    for lab in LABELS:
        df2[lab] = 0.5
    df2.to_csv(q, index=False)
    assert not train_slotknee._is_yw_csv(q)
    assert not train_slotknee._is_yw_csv(os.path.join(tmp_path, "missing.csv"))
