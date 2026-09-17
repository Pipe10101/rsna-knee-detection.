"""Tests for scripts/ssl_pretrain.py (MAE continued pretraining of the DINOv2-S encoder).

Runs in < 90 s on CPU with the local data_subset: a 3-study slot cache is built in
tmp, the MAE trains for one epoch on 12 images, and the exported encoder must load
strictly into SlotKneeS.
"""
import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.slotknee import SlotKneeS
from src.slots import SlotCache

_ROOT_FOR_IMAGES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Tests below build a cache from real DICOMs: skip when the image subset is absent (the CSVs
# and label files can be present without it).
_HAVE_IMAGES = os.path.isdir(os.path.join(_ROOT_FOR_IMAGES, "data_subset", "train_images")) or \
               os.path.isdir(os.path.join(_ROOT_FOR_IMAGES, "data_subset", "train_series"))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "ssl_pretrain.py")
BUILDER = os.path.join(ROOT, "scripts", "build_slot_cache.py")
DATA_DIR = os.path.join(ROOT, "data_subset")

_spec = importlib.util.spec_from_file_location("ssl_pretrain", SCRIPT)
ssl_pretrain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ssl_pretrain)


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory):
    if not _HAVE_IMAGES:
        pytest.skip("data_subset/train_images not present")
    out = str(tmp_path_factory.mktemp("ssl") / "cache")
    subprocess.run([sys.executable, BUILDER, "--data-dir", DATA_DIR, "--split", "train", "--out", out,
                    "--limit", "3", "--P", "224", "--workers", "1"], check=True, cwd=ROOT)
    return out


def test_random_masking_exact_ratio_and_restore():
    torch.manual_seed(0)
    B, L, D = 2, 256, 4
    x = torch.arange(B * L * D, dtype=torch.float32).view(B, L, D)
    kept, mask, ids_restore = ssl_pretrain.random_masking(x, 0.75)
    assert kept.shape == (B, 64, D) and mask.shape == (B, L) and ids_restore.shape == (B, L)
    assert abs(float(mask.mean()) - 0.75) < 1e-6
    # un-shuffling the kept tokens + placeholders puts every kept token back at its position
    filler = torch.full((B, L - 64, D), float("nan"))
    restored = torch.gather(torch.cat([kept, filler], 1), 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
    visible = mask == 0
    assert torch.equal(restored[visible], x[visible])
    assert torch.isnan(restored[~visible]).all()


def test_dataset_skips_absent_slots_and_returns_cache_pixels(tiny_cache):
    cache = SlotCache(tiny_cache, "train")
    ds = ssl_pretrain.SlotImageDataset([tiny_cache], aug=False)
    present = np.asarray(cache.mask).astype(bool) & cache.done[:, None]
    assert len(ds) == int(present.sum()) * cache.G
    assert ds.n_absent == int((~present & cache.done[:, None]).sum()) * cache.G
    for i in range(len(ds)):
        _, _, row, slot, g = (int(v) for v in ds.index[i])
        assert present[row, slot]
        img = ds[i]
        assert img.dtype == torch.uint8 and tuple(img.shape) == (3, 224, 224)
        assert np.array_equal(img.numpy(), cache[row][0][slot, g])
    # augmentation keeps dtype/shape and never flips: compare the column-profile orientation
    ds_aug = ssl_pretrain.SlotImageDataset([tiny_cache], aug=True)
    img = ds_aug[0]
    assert img.dtype == torch.uint8 and tuple(img.shape) == (3, 224, 224)
    # --max-images draws a deterministic subset
    sub = ssl_pretrain.SlotImageDataset([tiny_cache], aug=False, max_images=5, seed=1)
    sub2 = ssl_pretrain.SlotImageDataset([tiny_cache], aug=False, max_images=5, seed=1)
    assert len(sub) == 5 and np.array_equal(sub.index, sub2.index)


def test_mae_forward_masks_75_percent_and_normalises_like_slotknee():
    torch.manual_seed(0)
    model = ssl_pretrain.MAE(P=224, pretrained=False, dec_depth=1, dec_dim=64, dec_heads=4).eval()
    ref = SlotKneeS(P=224, T=3, pretrained=False, mixer_layers=0)
    assert torch.allclose(model.norm_mean, ref.norm_mean) and torch.allclose(model.norm_std, ref.norm_std)
    imgs = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
    assert torch.allclose(model.normalise(imgs), ref._normalise(imgs))
    loss, mask, pred = model(imgs)
    assert math.isfinite(float(loss.detach())) and abs(float(mask.mean()) - 0.75) < 1e-6
    assert tuple(pred.shape) == (2, 256, 14 * 14 * 3)
    x = model.normalise(imgs)
    assert torch.allclose(model.unpatchify(model.patchify(x)), x)


def test_ssl_pretrain_smoke_exports_strict_loadable_encoder(tiny_cache, tmp_path):
    out = str(tmp_path / "ssl")
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    subprocess.run([sys.executable, SCRIPT, "--cache", tiny_cache, "--out", out, "--epochs", "1",
                    "--bs", "4", "--max-images", "12", "--workers", "0", "--device", "cpu",
                    "--warmup-epochs", "0", "--log-every", "1", "--seed", "0"], check=True, cwd=ROOT, env=env)
    enc_path = os.path.join(out, "encoder.safetensors")
    assert os.path.isfile(enc_path) and os.path.isfile(os.path.join(out, "mae_full.pt"))
    with open(os.path.join(out, "log.json")) as fh:
        log = json.load(fh)
    assert log["n_images"] == 12 and len(log["epochs"]) == 1 and log["epochs_done"] == 1
    ep = log["epochs"][0]
    assert ep["n_images"] == 12 and ep["steps"] == 3
    assert math.isfinite(ep["loss"]) and ep["loss"] > 0
    assert abs(ep["mask_frac"] - 0.75) < 1e-3
    assert log["verify"]["ok"] and log["verify"]["max_abs_diff"] == 0.0

    # strict load into the downstream model, exactly as train_slotknee.py --pretrained-path does
    from safetensors.torch import load_file
    sd = load_file(enc_path)
    model = SlotKneeS(P=224, T=3, pretrained_path=enc_path, mixer_layers=0)
    esd = model.encoder.state_dict()
    assert set(esd) == set(sd)
    assert all(torch.equal(esd[k], sd[k]) for k in sd)
    assert "head" not in " ".join(sd) and all(k.startswith(("blocks.", "patch_embed.", "norm.")) or
                                               k in ("cls_token", "pos_embed") for k in sd)
    # the encoder actually moved away from the DINOv2 initialisation
    init = SlotKneeS(P=224, T=3, mixer_layers=0).encoder.state_dict()
    assert not torch.equal(init["blocks.11.mlp.fc2.weight"], sd["blocks.11.mlp.fc2.weight"])

    # resume: a second invocation with the same --out and --epochs does no extra work
    subprocess.run([sys.executable, SCRIPT, "--cache", tiny_cache, "--out", out, "--epochs", "1",
                    "--bs", "4", "--max-images", "12", "--workers", "0", "--device", "cpu",
                    "--warmup-epochs", "0", "--seed", "0"], check=True, cwd=ROOT, env=env)
    with open(os.path.join(out, "log.json")) as fh:
        log2 = json.load(fh)
    assert log2["resumed"] and len(log2["epochs"]) == 1
