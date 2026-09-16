"""Tests for ``src/slotknee.py`` (SlotKnee-S, spec §4).

Runs offline on CPU in well under 60 s: the DINOv2-S weights come from the local HF
cache (``HF_HUB_OFFLINE=1`` is set by ``conftest.py``).  One default model is shared
by the cheap tests; the P=252 / T=5 / checkpointing / weight-routing tests build
their own small variants.
"""

import glob
import math
import os
import subprocess
import sys
import time

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import PatchSizeError  # noqa: E402
from src.slotknee import DEFAULT_BACKBONE, SlotKneeS  # noqa: E402

N_LABELS, N_SLOTS = 12, 6
#: non-encoder state_dict keys of the head before the attention temperature was added
LEGACY_HEAD_KEYS = {"norm_mean", "norm_std", "proj.0.weight", "proj.0.bias", "proj.1.weight",
                    "proj.1.bias", "slot_embed", "base_queries", "group_map", "label_weight",
                    "label_bias"}


def _cached_weights():
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m/"
        "snapshots/*/model.safetensors"))
    return hits[0] if hits else None


def _inputs(B=2, G=3, T=3, P=224, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 256, (B, N_SLOTS, G, T, P, P), generator=g, dtype=torch.uint8)
    mask = torch.ones(B, N_SLOTS)
    return x, mask


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return SlotKneeS()


# --------------------------------------------------------------------------- #
# shape / speed / normalisation
# --------------------------------------------------------------------------- #

def test_forward_shape_and_speed(model):
    x, mask = _inputs(B=2, G=3)
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x, mask)
    elapsed = time.perf_counter() - t0
    assert out.shape == (2, N_LABELS)
    assert torch.isfinite(out).all()
    assert elapsed < 10.0, f"forward took {elapsed:.1f}s"
    assert model.last_n_encoded == 2 * N_SLOTS * 3


def test_uint8_equals_float_over_255(model):
    x, mask = _inputs(B=1, G=2)
    model.eval()
    with torch.no_grad():
        a = model(x, mask)
        b = model(x.float() / 255.0, mask)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_normalisation_buffers_come_from_pretrained_cfg(model):
    cfg = model.encoder.pretrained_cfg
    assert model.norm_mean.shape == (1, 3, 1, 1)
    assert torch.allclose(model.norm_mean.flatten(), torch.tensor(cfg["mean"]))
    assert torch.allclose(model.norm_std.flatten(), torch.tensor(cfg["std"]))
    assert "norm_mean" in model.state_dict() and "norm_std" in model.state_dict()


# --------------------------------------------------------------------------- #
# masked slot attention
# --------------------------------------------------------------------------- #

def test_masked_slot_receives_exactly_zero_attention(model):
    x, mask = _inputs(B=3, G=2)
    mask[0, 2] = 0
    mask[1, [0, 1, 5]] = 0
    mask[2, :] = 0                      # no slot at all -> uniform fallback
    model.eval()
    with torch.no_grad():
        out = model(x, mask)
    assert torch.isfinite(out).all()
    attn = model.slot_attention()
    assert attn.shape == (3, N_LABELS, N_SLOTS)
    assert torch.allclose(attn.sum(-1), torch.ones(3, N_LABELS), atol=1e-6)
    assert (attn[0, :, 2] == 0).all()
    assert (attn[0, :, [0, 1, 3, 4, 5]] > 0).all()
    assert (attn[1, :, [0, 1, 5]] == 0).all()
    assert torch.allclose(attn[2], torch.full((N_LABELS, N_SLOTS), 1.0 / N_SLOTS), atol=1e-6)
    # absent slots are not encoded at all
    assert model.last_n_encoded == (5 + 3 + 0) * 2


def test_absent_slot_pixels_cannot_change_the_output(model):
    x, mask = _inputs(B=1, G=2)
    mask[0, 4] = 0
    x2 = x.clone()
    x2[0, 4] = 255 - x2[0, 4]
    model.eval()
    with torch.no_grad():
        a, b = model(x, mask), model(x2, mask)
    assert torch.equal(a, b)


def test_slot_attention_before_forward_raises():
    m = SlotKneeS(pretrained=False, trainable_blocks=0, drop_path=0.0)
    with pytest.raises(RuntimeError):
        m.slot_attention()


# --------------------------------------------------------------------------- #
# freezing / gradients / LLRD
# --------------------------------------------------------------------------- #

def test_gradients_flow_only_to_trainable_tail(model):
    x, mask = _inputs(B=1, G=2)
    model.train()
    model.zero_grad(set_to_none=True)
    model(x, mask).sum().backward()

    enc = model.encoder
    depth, k = len(enc.blocks), model.trainable_blocks
    assert k == 4
    for blk in list(enc.blocks)[: depth - k]:
        for p in blk.parameters():
            assert p.requires_grad is False
            assert p.grad is None
    for name in ("patch_embed", "norm_pre"):
        for p in getattr(enc, name).parameters():
            assert p.requires_grad is False and p.grad is None
    assert enc.pos_embed.grad is None and enc.cls_token.grad is None
    for blk in list(enc.blocks)[depth - k:]:
        for p in blk.parameters():
            assert p.requires_grad and p.grad is not None
    for p in enc.norm.parameters():
        assert p.grad is not None
    for name in ("proj", "mixer", "slot_embed", "group_embed", "base_queries", "attn_tau",
                 "label_weight", "label_bias"):
        obj = getattr(model, name)
        for p in (obj.parameters() if isinstance(obj, torch.nn.Module) else [obj]):
            assert p.grad is not None, name
    model.zero_grad(set_to_none=True)


def test_frozen_prefix_stays_in_eval_mode(model):
    model.train()
    depth, k = model.depth, model.trainable_blocks
    assert all(not blk.training for blk in list(model.encoder.blocks)[: depth - k])
    assert all(blk.training for blk in list(model.encoder.blocks)[depth - k:])
    assert not model.encoder.patch_embed.training
    assert model.proj.training and model.head_dropout.training
    model.eval()
    assert not any(m.training for m in model.modules())


def test_param_groups_llrd(model):
    lr_b, lr_h, llrd = 1e-4, 1e-3, 0.8
    groups = model.param_groups(lr_b, lr_h, llrd=llrd)
    by_name = {g["name"]: g for g in groups}
    depth = model.depth
    for i in range(depth - model.trainable_blocks, depth):
        assert by_name[f"blocks.{i}"]["lr"] == pytest.approx(lr_b * llrd ** (depth - 1 - i))
    assert f"blocks.{depth - model.trainable_blocks - 1}" not in by_name
    assert by_name["norm"]["lr"] == pytest.approx(lr_b)
    assert by_name["head"]["lr"] == pytest.approx(lr_h)
    assert by_name["blocks.11"]["lr"] > by_name["blocks.10"]["lr"] > by_name["blocks.8"]["lr"]

    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter appears in two groups"
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert set(ids) == trainable
    head_ids = {id(p) for p in by_name["head"]["params"]}
    assert id(model.base_queries) in head_ids and id(model.label_weight) in head_ids
    assert id(model.group_embed) in head_ids and id(model.slot_embed) in head_ids
    assert id(model.attn_tau) in head_ids
    assert all(id(p) in head_ids for p in model.proj.parameters())
    assert all(id(p) in head_ids for p in model.mixer.parameters())
    torch.optim.AdamW(groups, lr=lr_h, weight_decay=0.01)  # torch accepts the extra "name" key


def test_trainable_parameter_count(model):
    trainable, total = model.trainable_parameters()
    assert 22_000_000 < total < 23_000_000
    # 4 ViT-S blocks (~1.77M each) + norm + projection + queries
    # + group/mixer head (~0.53M, measured 529,152)
    assert 7_500_000 < trainable < 8_200_000
    assert trainable < total


# --------------------------------------------------------------------------- #
# chunking / checkpointing
# --------------------------------------------------------------------------- #

def test_chunked_equals_unchunked(model):
    x, mask = _inputs(B=2, G=2)
    model.eval()
    old = model.encoder_chunk
    try:
        with torch.no_grad():
            model.encoder_chunk = 5
            a = model(x, mask)
            model.encoder_chunk = 10_000
            b = model(x, mask)
            model.encoder_chunk = 0          # <= 0: everything in one call
            c = model(x, mask)
    finally:
        model.encoder_chunk = old
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
    assert torch.allclose(b, c, atol=1e-6)


def test_grad_checkpointing_matches_plain_backward():
    torch.manual_seed(1)
    m = SlotKneeS(trainable_blocks=2, drop_path=0.0, head_dropout=0.0)
    x, mask = _inputs(B=1, G=1, seed=3)
    m.train()

    def run(enable):
        torch.manual_seed(7)   # identical mixer-dropout draws in both runs
        m.set_grad_checkpointing(enable)
        assert m.encoder.grad_checkpointing is enable
        m.zero_grad(set_to_none=True)
        out = m(x, mask)
        out.sum().backward()
        grads = [p.grad.clone() for p in m.encoder.blocks[-1].parameters()]
        return out.detach().clone(), grads

    out_a, g_a = run(False)
    out_b, g_b = run(True)
    assert torch.allclose(out_a, out_b, atol=1e-5)
    assert all(torch.allclose(ga, gb, atol=1e-5) for ga, gb in zip(g_a, g_b))
    assert all(g.abs().sum() > 0 for g in g_b), "checkpointed tail received no gradient"


# --------------------------------------------------------------------------- #
# resolution / channels / weights routing
# --------------------------------------------------------------------------- #

def test_p252_builds_and_runs():
    m = SlotKneeS(P=252, trainable_blocks=1, drop_path=0.0).eval()
    assert tuple(m.encoder.patch_embed.grid_size) == (18, 18)
    assert m.encoder.pos_embed.shape[1] == 18 * 18 + 1
    x, mask = _inputs(B=1, G=1, P=252)
    with torch.no_grad():
        out = m(x, mask)
    assert out.shape == (1, N_LABELS) and torch.isfinite(out).all()


def test_p_not_multiple_of_patch_is_rejected():
    with pytest.raises(PatchSizeError):
        SlotKneeS(P=230, pretrained=False)
    with pytest.raises(ValueError):
        SlotKneeS(P=230, pretrained=False)


def test_t5_adapts_patch_embedding_and_normalisation():
    m = SlotKneeS(T=5, trainable_blocks=0, drop_path=0.0).eval()
    assert tuple(m.encoder.patch_embed.proj.weight.shape) == (384, 5, 14, 14)
    assert m.norm_mean.shape == (1, 5, 1, 1)
    x, mask = _inputs(B=1, G=1, T=5)
    with torch.no_grad():
        assert m(x, mask).shape == (1, N_LABELS)


def test_wrong_input_layout_is_rejected(model):
    x, mask = _inputs(B=1, G=1)
    with pytest.raises(ValueError):
        model(x[:, :, 0], mask)                         # 5-D
    with pytest.raises(ValueError):
        model(x[:, :4], mask)                           # 4 slots
    with pytest.raises(ValueError):
        model(x, mask[:, :3])                           # mask shape


@pytest.mark.skipif(_cached_weights() is None, reason="no local DINOv2-S checkpoint")
def test_pretrained_path_file_and_directory_are_routed_offline():
    from safetensors.torch import load_file
    ckpt = _cached_weights()
    ref = load_file(ckpt)["blocks.0.attn.qkv.weight"]
    for path in (ckpt, os.path.dirname(ckpt)):
        m = SlotKneeS(pretrained_path=path, pretrained=False, trainable_blocks=0, drop_path=0.0)
        assert m.pretrained is True
        got = m.encoder.blocks[0].attn.qkv.weight.detach()
        assert torch.allclose(got, ref), f"weights from {path} were not loaded"


def test_pretrained_false_is_random_but_builds():
    m = SlotKneeS(pretrained=False, trainable_blocks=0, drop_path=0.0)
    assert m.pretrained is False
    assert m.hparams["backbone"] == DEFAULT_BACKBONE
    x, mask = _inputs(B=1, G=1)
    with torch.no_grad():
        assert m.eval()(x, mask).shape == (1, N_LABELS)


def test_state_dict_round_trip_without_pretrained_weights(model):
    m = SlotKneeS(**model.hparams, pretrained=False)
    missing, unexpected = m.load_state_dict(model.state_dict(), strict=True)
    assert not missing and not unexpected
    x, mask = _inputs(B=1, G=1, seed=5)
    mask[0, 1] = 0
    model.eval(); m.eval()
    with torch.no_grad():
        assert torch.allclose(model(x, mask), m(x, mask), atol=1e-6)


# --------------------------------------------------------------------------- #
# head extensions: group tokens, mixer, aux logits, fp16 safety
# --------------------------------------------------------------------------- #

def test_legacy_config_matches_old_checkpoint_layout():
    """token_level="slot", mixer_layers=0 must have exactly the pre-extension
    parameter set, so checkpoints saved before the head extensions load strict."""
    build = lambda: SlotKneeS(token_level="slot", mixer_layers=0, mil_pool="none",
                              pretrained=False, drop_path=0.0)
    m = build()
    non_encoder = {k for k in m.state_dict() if not k.startswith("encoder.")}
    assert non_encoder == LEGACY_HEAD_KEYS | {"attn_tau"}
    missing, unexpected = build().load_state_dict(m.state_dict(), strict=True)
    assert not missing and not unexpected
    # a checkpoint from before attn_tau existed: loads strict, tau filled with 1.0
    old = {k: v for k, v in m.state_dict().items()
           if k in LEGACY_HEAD_KEYS or k.startswith("encoder.")}
    m_old = build()
    with pytest.warns(RuntimeWarning, match="attn_tau"):
        missing, unexpected = m_old.load_state_dict(old, strict=True)
    assert not missing and not unexpected
    assert torch.equal(m_old.attn_tau.detach(), torch.ones(N_LABELS))
    assert m.group_embed is None and m.mixer is None and m.aux_head is None

    x, mask = _inputs(B=1, G=2)
    mask[0, 5] = 0
    m.eval()
    with torch.no_grad():
        out = m(x, mask)
    assert isinstance(out, torch.Tensor) and out.shape == (1, N_LABELS)
    attn = m.slot_attention()
    assert attn.shape == (1, N_LABELS, N_SLOTS) and (attn[0, :, 5] == 0).all()
    with pytest.raises(RuntimeError):
        m.group_attention()


def test_group_attention_zero_at_group_granularity(model):
    G = 3
    x, mask = _inputs(B=2, G=G)
    mask[0, [1, 4]] = 0
    model.eval()
    with torch.no_grad():
        model(x, mask)
    ga = model.group_attention()
    assert ga.shape == (2, N_LABELS, N_SLOTS, G)
    assert (ga[0, :, [1, 4], :] == 0).all()
    assert (ga[0, :, [0, 2, 3, 5], :] > 0).all()
    assert (ga[1] > 0).all()
    assert torch.allclose(ga.sum(dim=(-1, -2)), torch.ones(2, N_LABELS), atol=1e-5)
    assert torch.allclose(model.slot_attention(), ga.sum(dim=-1), atol=1e-7)


def test_mixer_changes_logits_but_not_shapes(model):
    x, mask = _inputs(B=1, G=2)
    model.eval()
    saved = model.mixer
    try:
        with torch.no_grad():
            with_mixer = model(x, mask)
            model.mixer = None
            without = model(x, mask)
    finally:
        model.mixer = saved
    assert with_mixer.shape == without.shape == (1, N_LABELS)
    assert not torch.allclose(with_mixer, without, atol=1e-3), "mixer path is dead"


def test_aux_slot_logits_shape_and_return_type_bc(model):
    x, mask = _inputs(B=2, G=2)
    # default: plain tensor, so the training script's forward(x, mask) keeps working
    model.eval()
    with torch.no_grad():
        assert isinstance(model(x, mask), torch.Tensor)

    m = SlotKneeS(aux_slot_logits=True, pretrained=False, drop_path=0.0)
    assert m.hparams["aux_slot_logits"] is True
    m.eval()
    with torch.no_grad():
        out = m(x, mask)
    assert isinstance(out, tuple) and len(out) == 2
    logits, aux = out
    assert logits.shape == (2, N_LABELS) and aux.shape == (2, N_SLOTS, N_LABELS)
    assert torch.isfinite(aux).all()
    m.train()
    m.zero_grad(set_to_none=True)
    logits, aux = m(x, mask)
    (logits.sum() + 0.2 * aux.sum()).backward()   # the documented aux weight
    assert m.aux_head.weight.grad is not None
    assert m.base_queries.grad is not None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast_low_precision_five_absent_slots(model, dtype):
    x, _ = _inputs(B=1, G=3)
    mask = torch.zeros(1, N_SLOTS)
    mask[0, 3] = 1
    model.eval()
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=dtype):
            out = model(x, mask)
    except RuntimeError as exc:
        pytest.skip(f"cpu autocast {dtype} unsupported here: {exc}")
    assert torch.isfinite(out).all(), f"non-finite logits under {dtype}"
    ga = model.group_attention()
    assert (ga[0, :, [0, 1, 2, 4, 5], :] == 0).all(), "absent slots got attention"
    assert torch.allclose(model.slot_attention()[0, :, 3].float(),
                          torch.ones(N_LABELS), atol=1e-2)


def test_new_head_adds_fewer_than_1m_params(model):
    legacy = SlotKneeS(token_level="slot", mixer_layers=0, pretrained=False, drop_path=0.0)
    _, total_new = model.trainable_parameters()
    _, total_old = legacy.trainable_parameters()
    delta = total_new - total_old
    assert 0 < delta < 1_000_000, f"new head adds {delta:,} params"
    # measured: 529,152 = group_embed (8*256) + one mixer layer (~527k)


def test_g_above_max_groups_is_rejected_before_encoding(model):
    x = torch.zeros(1, N_SLOTS, model.max_groups + 1, 3, 224, 224, dtype=torch.uint8)
    with pytest.raises(ValueError):
        model(x, torch.ones(1, N_SLOTS))


# --------------------------------------------------------------------------- #
# attention temperature / entropy / diagnostics
# --------------------------------------------------------------------------- #

def test_default_build_keys_match_previous_layout_plus_tau(model):
    """Default build: `attn_tau` is the only added key; a default-layout checkpoint
    saved before it existed loads strict (tau filled with 1.0) and gives identical logits."""
    sd = model.state_dict()
    assert "attn_tau" in sd and sd["attn_tau"].shape == (N_LABELS,)
    assert torch.equal(sd["attn_tau"], torch.ones(N_LABELS))
    assert model.hparams["attn_tau_init"] == 1.0 and model.hparams["attn_entropy_weight"] == 0.0
    pre_tau = {k: v for k, v in sd.items() if k != "attn_tau"}
    m = SlotKneeS(**model.hparams, pretrained=False)
    with pytest.warns(RuntimeWarning, match="attn_tau"):
        missing, unexpected = m.load_state_dict(pre_tau, strict=True)
    assert not missing and not unexpected
    x, mask = _inputs(B=1, G=2, seed=11)
    mask[0, 0] = 0
    model.eval()
    m.eval()
    with torch.no_grad():
        assert torch.equal(model(x, mask), m(x, mask))
    assert model.last_attention_entropy is None   # weight 0: nothing retained


def _clone_with(model, **overrides):
    """Same weights as `model`, different constructor flags (tau filled by the loader)."""
    hp = dict(model.hparams)
    hp.update(overrides)
    m = SlotKneeS(**hp, pretrained=False)
    sd = {k: v for k, v in model.state_dict().items() if k != "attn_tau"}
    with pytest.warns(RuntimeWarning, match="attn_tau"):
        m.load_state_dict(sd, strict=True)
    return m.eval()


def test_attn_tau_init_changes_logits_not_shapes(model):
    m3 = _clone_with(model, attn_tau_init=3.0)
    assert torch.equal(m3.attn_tau.detach(), torch.full((N_LABELS,), 3.0))
    assert m3.hparams["attn_tau_init"] == 3.0
    x, mask = _inputs(B=2, G=3)
    mask[1, 2] = 0
    model.eval()
    with torch.no_grad():
        a, b = model(x, mask), m3(x, mask)
    assert a.shape == b.shape == (2, N_LABELS)
    assert not torch.allclose(a, b, atol=1e-4), "temperature has no effect on the logits"
    assert (m3.slot_attention()[1, :, 2] == 0).all()


def test_attention_entropy_in_unit_range_and_sharper_with_higher_tau(model):
    m1 = _clone_with(model, attn_entropy_weight=0.1)                     # tau 1.0
    m3 = _clone_with(model, attn_entropy_weight=0.1, attn_tau_init=3.0)
    x, mask = _inputs(B=2, G=3, seed=4)
    mask[0, [1, 5]] = 0
    with torch.no_grad():
        m1(x, mask)
        e1 = m1.last_attention_entropy
        m3(x, mask)
        e3 = m3.last_attention_entropy
    assert e1.dim() == 0 and 0.0 <= float(e1) <= 1.0
    assert 0.0 <= float(e3) <= 1.0
    assert float(e3) < float(e1), (float(e1), float(e3))
    # differentiable in train mode so the training script can add it to the loss
    m1.train()
    m1.zero_grad(set_to_none=True)
    m1(x, mask)
    (0.1 * m1.last_attention_entropy).backward()
    assert m1.attn_tau.grad is not None and m1.base_queries.grad is not None


def test_attention_report_shapes(model):
    G = 3
    x, mask = _inputs(B=3, G=G, seed=9)
    mask[0, [0, 4]] = 0
    model.train()
    rep = model.attention_report(x, mask)
    assert model.training, "attention_report must restore the module mode"
    model.eval()
    assert rep["anchor"].shape == (N_LABELS, G)
    assert rep["slot"].shape == (N_LABELS, N_SLOTS)
    assert rep["slot_present"].shape == (N_LABELS, N_SLOTS)
    assert rep["tau"].shape == (N_LABELS,)
    assert rep["n_studies"] == 3
    assert 0.0 <= rep["entropy"] <= 1.0
    assert torch.allclose(rep["anchor"].sum(-1), torch.ones(N_LABELS), atol=1e-5)
    assert torch.allclose(rep["slot"].sum(-1), torch.ones(N_LABELS), atol=1e-5)
    assert (rep["slot_present"] >= rep["slot"] - 1e-6).all()
    legacy = SlotKneeS(token_level="slot", mixer_layers=0, pretrained=False, drop_path=0.0)
    rep2 = legacy.attention_report(x, mask)
    assert rep2["anchor"] is None and rep2["slot"].shape == (N_LABELS, N_SLOTS)


# --------------------------------------------------------------------------- #
# MIL pooling of per-token logits
# --------------------------------------------------------------------------- #

def _mil_clone(model, **overrides):
    """Fixture weights plus a fresh (random-init) instance head and tau_mil."""
    hp = dict(model.hparams)
    hp.update(overrides)
    m = SlotKneeS(**hp, pretrained=False)
    m.load_state_dict(model.state_dict(), strict=False)   # mil_head / tau_mil keep their init
    return m.eval()


def test_mil_pool_none_is_default_and_leaves_logits_and_keys_unchanged(model):
    assert model.hparams["mil_pool"] == "none" and model.hparams["mil_alpha"] == 0.5
    assert model.mil_head is None and model.tau_mil is None
    assert not any("mil" in k for k in model.state_dict())
    m0 = SlotKneeS(**dict(model.hparams, mil_pool="none"), pretrained=False)
    missing, unexpected = m0.load_state_dict(model.state_dict(), strict=True)
    assert not missing and not unexpected
    x, mask = _inputs(B=2, G=2, seed=21)
    mask[0, 3] = 0
    model.eval()
    m0.eval()
    with torch.no_grad():
        assert torch.equal(model(x, mask), m0(x, mask))
    assert model.last_mil_logits is None and model.last_token_logits is None


@pytest.mark.parametrize("pool", ["lse", "max"])
def test_mil_pool_changes_logits_not_shapes(model, pool):
    m = _mil_clone(model, mil_pool=pool)
    assert m.hparams["mil_pool"] == pool
    assert tuple(m.mil_head.weight.shape) == (N_LABELS, model.d) and m.tau_mil.shape == ()
    _, total_new = m.trainable_parameters()
    _, total_old = model.trainable_parameters()
    assert total_new - total_old == N_LABELS * model.d + N_LABELS + 1      # 3,085 for d=256
    x, mask = _inputs(B=3, G=3, seed=22)
    mask[1, [0, 5]] = 0
    mask[2, :] = 0                                     # no slot at all: must stay finite
    model.eval()
    with torch.no_grad():
        a, b = model(x, mask), m(x, mask)
    assert a.shape == b.shape == (3, N_LABELS)
    assert torch.isfinite(b).all()
    assert not torch.allclose(a, b, atol=1e-4), "MIL path is dead"
    assert m.last_mil_logits.shape == (3, N_LABELS)
    assert m.last_token_logits.shape == (3, N_SLOTS * 3, N_LABELS)
    head_ids = {id(p) for p in m.param_groups(1e-4, 1e-3)[-1]["params"]}
    assert id(m.mil_head.weight) in head_ids and id(m.mil_head.bias) in head_ids
    assert id(m.tau_mil) in head_ids
    # aux contract unchanged: main logits are the mixed ones
    ma = _mil_clone(model, mil_pool=pool, aux_slot_logits=True)
    with torch.no_grad():
        out = ma(x, mask)
    assert isinstance(out, tuple) and out[0].shape == (3, N_LABELS) and out[1].shape == (3, N_SLOTS, N_LABELS)


def test_lse_pool_rises_monotonically_as_tau_mil_decreases(model):
    G = 3
    m = _mil_clone(model, mil_pool="lse", mil_alpha=1.0)    # study logits == MIL logits
    x, mask = _inputs(B=1, G=G, seed=23)
    mask[0, [2, 4]] = 0
    present = (mask[0] > 0).repeat_interleave(G)             # slot-major token order
    n_present = int(present.sum())
    taus = [100.0, 4.0, 1.0, 0.25, 0.05]
    outs = {}
    with torch.no_grad():
        for tau in taus:
            m.tau_mil.data.fill_(tau)
            outs[tau] = m(x, mask)[0].clone()
    tok = m.last_token_logits[0]                             # [n_tokens, L], independent of tau
    zmax, zmean = tok[present].max(dim=0).values, tok[present].mean(dim=0)
    for hi, lo in zip(taus[:-1], taus[1:]):                  # tau decreasing -> logit non-decreasing
        assert (outs[lo] >= outs[hi] - 1e-6).all(), (hi, lo)
    j = int((zmax - zmean).argmax())                         # label with the most dominant token
    assert outs[0.05][j] > outs[100.0][j] + 1e-3
    assert torch.allclose(outs[100.0], zmean, atol=0.05)                         # high tau -> mean
    assert (outs[0.05] <= zmax + 1e-5).all()                                     # low tau -> max
    assert (outs[0.05] >= zmax - 0.05 * math.log(n_present) - 1e-5).all()
    # max pooling is the tau -> 0 limit
    mm = _mil_clone(model, mil_pool="max", mil_alpha=1.0)
    mm.mil_head.load_state_dict(m.mil_head.state_dict())
    with torch.no_grad():
        assert torch.allclose(mm(x, mask)[0], zmax, atol=1e-5)


def test_absent_tokens_have_zero_influence_even_when_encoded(model):
    """encode_absent=True makes absent slots carry real, changing token features, so
    this exercises the masking in the mixer, the query attention and the MIL pooling."""
    m = _mil_clone(model, mil_pool="lse", seq_mix="gru", encode_absent=True)
    x, mask = _inputs(B=1, G=2, seed=24)
    mask[0, [1, 4]] = 0
    x2 = x.clone()
    x2[0, [1, 4]] = 255 - x2[0, [1, 4]]
    absent = (mask[0] == 0).repeat_interleave(2)
    with torch.no_grad():
        a = m(x, mask)
        tok_a = m.last_token_logits.clone()
        b = m(x2, mask)
        tok_b = m.last_token_logits.clone()
    assert not torch.equal(tok_a[0, absent], tok_b[0, absent]), "perturbation did not reach the tokens"
    assert torch.allclose(tok_a[0, ~absent], tok_b[0, ~absent], atol=1e-6)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(m.last_mil_logits, m.last_mil_logits.clone())
    # gradient: nothing flows from absent tokens into the instance head via the pooling
    m.train()
    m.zero_grad(set_to_none=True)
    out = m(x, mask)
    out.sum().backward()
    assert torch.isfinite(m.tau_mil.grad).all() and torch.isfinite(m.mil_head.weight.grad).all()


def test_mil_lse_finite_under_low_precision_autocast_with_five_absent_slots(model):
    m = _mil_clone(model, mil_pool="lse", seq_mix="gru")
    m.tau_mil.data.fill_(0.25)      # masked-after-division path with |finfo.min / tau| overflow risk
    x, _ = _inputs(B=1, G=3, seed=25)
    mask = torch.zeros(1, N_SLOTS)
    mask[0, 2] = 1
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = m(x, mask)
    except RuntimeError as exc:
        pytest.skip(f"cpu autocast bf16 unsupported here: {exc}")
    assert torch.isfinite(out).all() and torch.isfinite(m.last_mil_logits).all()


# --------------------------------------------------------------------------- #
# sequence mixing along the anchor axis
# --------------------------------------------------------------------------- #

def test_seq_mix_none_is_default_and_unchanged(model):
    assert model.hparams["seq_mix"] == "none" and model.seq_mixer is None
    assert not any(k.startswith("seq_mixer") for k in model.state_dict())
    m0 = SlotKneeS(**dict(model.hparams, seq_mix="none"), pretrained=False)
    missing, unexpected = m0.load_state_dict(model.state_dict(), strict=True)
    assert not missing and not unexpected
    x, mask = _inputs(B=2, G=2, seed=31)
    mask[1, 2] = 0
    model.eval()
    m0.eval()
    with torch.no_grad():
        assert torch.equal(model(x, mask), m0(x, mask))


@pytest.mark.parametrize("mix", ["gru", "conv"])
def test_seq_mix_changes_logits_not_shapes(model, mix):
    m = _mil_clone(model, seq_mix=mix)          # strict=False: seq_mixer keeps its init
    assert m.hparams["seq_mix"] == mix
    _, tot_new = m.trainable_parameters()
    _, tot_old = model.trainable_parameters()
    d = model.d
    expected = 2 * 3 * ((d // 2) * d + (d // 2) ** 2 + 2 * (d // 2)) if mix == "gru" else 4 * d
    assert tot_new - tot_old == expected        # 296,448 gru / 1,024 conv at d=256
    assert expected <= 800_000
    x, mask = _inputs(B=2, G=3, seed=32)
    mask[0, [1, 3]] = 0
    model.eval()
    with torch.no_grad():
        a, b = model(x, mask), m(x, mask)
    assert a.shape == b.shape == (2, N_LABELS)
    assert torch.isfinite(b).all()
    assert not torch.allclose(a, b, atol=1e-4), f"seq_mix={mix} path is dead"
    head_ids = {id(p) for p in m.param_groups(1e-4, 1e-3)[-1]["params"]}
    assert all(id(p) in head_ids for p in m.seq_mixer.parameters())


@pytest.mark.parametrize("mix,sensitive", [("none", False), ("gru", True), ("conv", True)])
def test_anchor_order_sensitivity(model, mix, sensitive):
    """slot mode pools by mean over G, which is exactly permutation-invariant -- unless
    a sequence mixer runs along the anchor axis first.  (In group mode the learned
    group-position embedding already makes anchor order matter for every seq_mix, so
    the clean contrast lives in slot mode.)"""
    m = _mil_clone(model, seq_mix=mix, token_level="slot")
    x, mask = _inputs(B=1, G=3, seed=33)
    x_rev = x.flip(dims=[2])                    # reverse the anchor order in every slot
    with torch.no_grad():
        a, b = m(x, mask), m(x_rev, mask)
    if sensitive:
        assert not torch.allclose(a, b, atol=1e-4), f"seq_mix={mix} ignores anchor order"
    else:
        assert torch.allclose(a, b, atol=1e-5)  # only fp summation order differs


def test_seq_mix_absent_slots_get_exactly_zero_delta(model):
    m = _mil_clone(model, seq_mix="gru")
    x, mask = _inputs(B=1, G=3, seed=34)
    mask[0, [0, 5]] = 0
    per = torch.randn(1, N_SLOTS, 3, model.d)
    delta = m._seq_mix_delta(per, mask > 0)
    assert delta.shape == per.shape
    assert (delta[0, [0, 5]] == 0).all()
    assert (delta[0, [1, 2, 3, 4]] != 0).any()
    with torch.no_grad():
        out = m(x, mask)
    assert torch.isfinite(out).all()
    assert (m.slot_attention()[0, :, [0, 5]] == 0).all()


# --------------------------------------------------------------------------- #
# memory probe
# --------------------------------------------------------------------------- #

_PROBE = r"""
import resource, sys, torch
sys.path.insert(0, sys.argv[1])
from src.slotknee import SlotKneeS
m = SlotKneeS().eval()
x = torch.randint(0, 256, (1, 6, 3, 3, 224, 224), dtype=torch.uint8)
mask = torch.ones(1, 6)
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with torch.no_grad():
    out = m(x, mask)
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
scale = 1 if sys.platform == "darwin" else 1024   # ru_maxrss: bytes on macOS, KiB on Linux
print("RSS_DELTA_BYTES", (after - before) * scale)
print("RSS_PEAK_BYTES", after * scale)
"""


def test_peak_rss_delta_of_a_forward_is_small():
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    proc = subprocess.run([sys.executable, "-c", _PROBE, ROOT], capture_output=True, text=True,
                          env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    vals = dict(line.split() for line in proc.stdout.splitlines() if line.startswith("RSS_"))
    delta_gb = int(vals["RSS_DELTA_BYTES"]) / 2 ** 30
    peak_gb = int(vals["RSS_PEAK_BYTES"]) / 2 ** 30
    assert delta_gb < 1.5, f"forward added {delta_gb:.2f} GB RSS"
    assert peak_gb < 4.0, f"process peaked at {peak_gb:.2f} GB RSS"
