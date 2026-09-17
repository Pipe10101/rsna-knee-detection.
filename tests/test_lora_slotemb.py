"""LoRA across all encoder blocks (+ patch embedding) and the in-encoder slot embedding."""
import os, sys, subprocess
import pytest, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.slotknee import SlotKneeS, LoRALinear, LoRAConv2d   # noqa: E402

_ROOT_FOR_IMAGES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Tests below build a cache from real DICOMs: skip when the image subset is absent (the CSVs
# and label files can be present without it).
_HAVE_IMAGES = os.path.isdir(os.path.join(_ROOT_FOR_IMAGES, "data_subset", "train_images")) or \
               os.path.isdir(os.path.join(_ROOT_FOR_IMAGES, "data_subset", "train_series"))



def _tiny(**kw):
    return SlotKneeS(P=28, T=1, n_slots=6, max_groups=8, trainable_blocks=1, pretrained=False,
                     backbone="vit_small_patch14_dinov2.lvd142m", mixer_layers=1, **kw)


def test_lora_wraps_every_block_freezes_base_and_trains_only_lora_and_norms():
    m = _tiny(lora_rank=4, lora_alpha=8.0)
    enc = m.encoder
    assert m.n_frozen_blocks == 0 and m.trainable_blocks == m.depth          # no frozen prefix under LoRA
    assert all(isinstance(b.attn.qkv, LoRALinear) and isinstance(b.attn.proj, LoRALinear) for b in enc.blocks)
    assert isinstance(enc.patch_embed.proj, LoRAConv2d)
    for b in enc.blocks:                                                      # base weights frozen, LoRA trainable
        assert not b.attn.qkv.base.weight.requires_grad and b.attn.qkv.lora_A.requires_grad
        assert b.norm1.weight.requires_grad and b.norm2.weight.requires_grad  # LayerNorm affines train
    names = [n for n, p in m.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    assert all(("lora_" in n) or n.endswith(("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"))
               or ".norm.weight" in n or ".norm.bias" in n for n in names), names[:5]
    in_ch = enc.patch_embed.proj.base.in_channels                          # = T (1 here, 3 for triplets)
    expected = sum(4 * (384 + 1152) + 4 * (384 + 384) for _ in range(12)) + 4 * in_ch * 14 * 14 + 384 * 4
    assert m.lora_params == expected
    assert m.hparams["lora_rank"] == 4 and m.hparams["slot_tok_embed"] is False


def test_lora_step0_equals_base_model_and_checkpoint_roundtrips():
    torch.manual_seed(0)
    base = _tiny().eval()
    lora = _tiny(lora_rank=4).eval()
    # copy base weights into the LoRA model's frozen bases (state dict keys differ by ".base")
    sd = {k.replace("attn.qkv.", "attn.qkv.base.").replace("attn.proj.", "attn.proj.base.").replace("patch_embed.proj.", "patch_embed.proj.base."): v
          for k, v in base.state_dict().items()}
    missing, unexpected = lora.load_state_dict(sd, strict=False)
    assert not unexpected and all("lora_" in k for k in missing)
    x = torch.randint(0, 255, (1, 6, 2, 1, 28, 28), dtype=torch.uint8); mask = torch.ones(1, 6)
    with torch.no_grad():
        a = base(x, mask); b = lora(x, mask)
    a = a[0] if isinstance(a, tuple) else a; b = b[0] if isinstance(b, tuple) else b
    assert torch.allclose(a, b, atol=1e-5)                                   # lora_B = 0 -> identical at step 0
    hp = {k: v for k, v in lora.hparams.items() if k not in ("pretrained", "pretrained_path")}
    again = SlotKneeS(pretrained=False, **hp); again.load_state_dict(lora.state_dict())   # strict
    groups = lora.param_groups(1e-4, 3e-4)
    assert sum(len(g["params"]) for g in groups) == sum(1 for p in lora.parameters() if p.requires_grad)
    assert all(abs(g["lr"] - 1e-4) < 1e-12 for g in groups if g["name"].startswith("blocks."))   # uniform LR under LoRA


def test_slot_tok_embed_is_zero_init_and_receives_gradient():
    torch.manual_seed(0)
    base = _tiny().eval(); se = _tiny(slot_tok_embed=True).eval()
    se.load_state_dict(base.state_dict(), strict=False)
    x = torch.randint(0, 255, (1, 6, 2, 1, 28, 28), dtype=torch.uint8); mask = torch.ones(1, 6)
    with torch.no_grad():
        a = base(x, mask); b = se(x, mask)
    a = a[0] if isinstance(a, tuple) else a; b = b[0] if isinstance(b, tuple) else b
    assert torch.allclose(a, b, atol=1e-5)                                   # zero-init: no change at step 0
    se.train(); out = se(x, mask); (out[0] if isinstance(out, tuple) else out).sum().backward()
    assert se.slot_tok_embed.weight.grad is not None and se.slot_tok_embed.weight.grad.abs().sum() > 0
    assert se.hparams["slot_tok_embed"] is True


@pytest.mark.skipif(not _HAVE_IMAGES, reason="data_subset/train_images not present")
def test_lora_slotemb_micro_run(tmp_path):
    data_dir = "data_subset"
    if not os.path.isdir(os.path.join(ROOT, data_dir)):
        pytest.skip("data_subset not available locally")
    cache_dir, out_dir = os.path.join(tmp_path, "cache"), os.path.join(tmp_path, "out")
    subprocess.run(["python3", "scripts/build_slot_cache.py", "--data-dir", data_dir, "--split", "train",
                    "--out", cache_dir, "--limit", "6", "--P", "224", "--workers", "1"], check=True, cwd=ROOT)
    subprocess.run(["python3", "scripts/train_slotknee.py", "--cache", cache_dir, "--data-dir", data_dir,
                    "--folds", "0", "--epochs", "1", "--bs", "2", "--out", out_dir, "--seed", "42", "--amp", "cpu",
                    "--max-studies", "6", "--lora-rank", "4", "--slot-tok-embed", "--lr-backbone", "1e-4"], check=True, cwd=ROOT)
    ck = torch.load(os.path.join(out_dir, "fold_0_best.pt"), map_location="cpu")
    assert ck["hparams"]["lora_rank"] == 4 and ck["hparams"]["slot_tok_embed"] is True
    hp = {k: v for k, v in ck["hparams"].items() if k not in ("pretrained", "pretrained_path")}
    SlotKneeS(pretrained=False, **hp).load_state_dict(ck["state_dict"])       # the infer-side contract
