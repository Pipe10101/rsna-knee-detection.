"""Tests for src/model.py -- backbone selection, offline weights, patch-14 safety.

Run:  python3 -m pytest tests/test_model.py -q

Every test here runs with ``pretrained=False`` or against a locally-copied
checkpoint, so the suite never touches the network.  That is deliberate: the
target is a Kaggle code competition with no internet, and a test that silently
downloads weights would hide the exact failure we care about.
"""

import os
import sys
import warnings

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (  # noqa: E402
    BACKBONE_ALIASES,
    PatchSizeError,
    RSNA25DModel,
    check_patch_compatible,
    depth_container,
    estimate_training_memory_gb,
    freeze_backbone_blocks,
    patch_compatible_sizes,
    plan_batch_size,
    recommend_capacity,
    resolution_schedule,
    resolve_backbone,
    snap_to_patch_multiple,
    valid_image_size,
)

IN_CHANNELS = 3      # matches Config.in_channels / kaggle_data's 2.5D slice stack
NUM_CLASSES = 12     # 12 independent binary knee findings

#: Every alias must build and produce (B, 12).  Kept explicit rather than derived
#: from BACKBONE_ALIASES so that adding an alias without a test is a visible diff.
SMALL_ALIASES = ["mobilenetv3_small", "effnet_lite0", "effnet_b0"]
MID_ALIASES = ["effnetv2_s", "convnext_tiny"]
BIG_ALIASES = ["effnetv2_m", "convnext_small", "convnext_base",
               "dinov2_small", "dinov2_base", "dinov2_large"]


def _build(alias, **kw):
    kw.setdefault("pretrained", False)
    kw.setdefault("in_channels", IN_CHANNELS)
    kw.setdefault("num_classes", NUM_CLASSES)
    kw.setdefault("image_size", 224)
    return RSNA25DModel(backbone_name=alias, **kw)


# --------------------------------------------------------------------------- #
# 1. Every supported backbone builds and returns (B, 12)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("alias", SMALL_ALIASES + MID_ALIASES)
def test_backbone_builds_and_returns_b12(alias):
    model = _build(alias).eval()
    with torch.no_grad():
        out = model(torch.randn(2, IN_CHANNELS, 224, 224))
    assert out.shape == (2, NUM_CLASSES), f"{alias} -> {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"{alias} produced non-finite logits"


@pytest.mark.slow
@pytest.mark.parametrize("alias", BIG_ALIASES)
def test_big_backbone_builds_and_returns_b12(alias):
    model = _build(alias).eval()
    with torch.no_grad():
        out = model(torch.randn(2, IN_CHANNELS, 224, 224))
    assert out.shape == (2, NUM_CLASSES), f"{alias} -> {tuple(out.shape)}"


def test_every_alias_in_registry_is_covered_by_a_test():
    covered = set(SMALL_ALIASES + MID_ALIASES + BIG_ALIASES)
    missing = set(BACKBONE_ALIASES) - covered - {"effnet_b3"}
    assert not missing, f"aliases with no build test: {sorted(missing)}"


def test_batch_of_one_works():
    """BatchNorm blows up on B=1 in train mode; eval must still work."""
    model = _build("effnet_b0").eval()
    with torch.no_grad():
        assert model(torch.randn(1, IN_CHANNELS, 224, 224)).shape == (1, NUM_CLASSES)


@pytest.mark.parametrize("in_ch", [1, 3, 5, 7])
def test_arbitrary_slice_counts(in_ch):
    """kaggle_data stacks cfg.in_channels slices -> (B, in_channels, H, W)."""
    model = _build("effnet_b0", in_channels=in_ch).eval()
    with torch.no_grad():
        out = model(torch.randn(2, in_ch, 224, 224))
    assert out.shape == (2, NUM_CLASSES)


def test_backward_pass_produces_gradients():
    model = _build("effnet_b0").train()
    out = model(torch.randn(2, IN_CHANNELS, 224, 224))
    out.pow(2).mean().backward()
    assert model.fc.weight.grad is not None
    assert torch.isfinite(model.fc.weight.grad).all()


# --------------------------------------------------------------------------- #
# 2. Patch-14: incompatible resolutions are rejected with a clear message
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("size", [384, 512, 300, 225])
def test_patch14_incompatible_size_is_rejected(size):
    with pytest.raises(PatchSizeError) as exc:
        check_patch_compatible("dinov2_small", size)
    msg = str(exc.value)
    # The message must be actionable, not just "invalid".
    assert "14" in msg, msg
    assert "not a multiple" in msg, msg
    assert "silently discards" in msg, msg
    assert str(snap_to_patch_multiple(size, 14)) in msg, msg


@pytest.mark.parametrize("size", [224, 322, 378, 392, 448, 518])
def test_patch14_compatible_sizes_are_accepted(size):
    assert check_patch_compatible("dinov2_small", size) == size
    assert size % 14 == 0


@pytest.mark.parametrize("size", [224, 300, 384, 512])
def test_cnn_accepts_any_size(size):
    """Fully-convolutional backbones have no patch constraint."""
    assert check_patch_compatible("effnet_b0", size) == size


def test_strict_image_size_raises_at_construction():
    with pytest.raises(PatchSizeError) as exc:
        _build("dinov2_small", image_size=384, strict_image_size=True)
    assert "dinov2_small" in str(exc.value)


def test_non_strict_snaps_and_warns_instead_of_cropping():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _build("dinov2_small", image_size=384, strict_image_size=False)
    assert model.image_size == 378, "384 must snap to 27*14 = 378"
    assert any("patch" in str(w.message).lower() for w in caught), \
        "snapping the resolution must warn"
    # And a 384px batch must still work -- resized, not cropped.
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, IN_CHANNELS, 384, 384)).shape == (2, NUM_CLASSES)


def test_timm_really_does_silently_crop_at_384():
    """Regression guard on the finding that motivates PatchSizeError.

    If a future timm starts raising (or padding) at a non-multiple img_size, this
    test fails and the surrounding docs/messages must be revisited.
    """
    import timm
    m = timm.create_model("vit_small_patch14_dinov2", pretrained=False,
                          num_classes=0, global_pool="", img_size=384,
                          in_chans=IN_CHANNELS).eval()
    base = torch.zeros(1, IN_CHANNELS, 384, 384)
    edge = base.clone()
    edge[:, :, 378:, :] = 5.0          # only the trailing 6 rows
    edge[:, :, :, 378:] = 5.0          # only the trailing 6 cols
    inside = base.clone()
    inside[:, :, 370:378, :] = 5.0     # control: inside the covered region
    with torch.no_grad():
        o0, o_edge, o_in = m(base), m(edge), m(inside)
    assert torch.equal(o0, o_edge), \
        "timm no longer silently ignores the trailing pixels -- update model.py docs"
    assert not torch.equal(o0, o_in), "control perturbation had no effect; test is broken"


def test_patch_compatible_sizes_helper():
    sizes = patch_compatible_sizes(14, 196, 560)
    assert all(s % 14 == 0 for s in sizes)
    assert 378 in sizes and 384 not in sizes


def test_valid_image_size_and_schedule():
    assert valid_image_size("effnet_b0", 384) == 384
    assert valid_image_size("dinov2_large", 384) == 378
    assert resolution_schedule("convnext_base", [224, 384]) == [224, 384]
    assert resolution_schedule("dinov2_large", [224, 384]) == [224, 378]
    # consecutive duplicates collapse: 380 and 384 both snap to 378
    assert resolution_schedule("dinov2_small", [380, 384]) == [378]


def test_resolve_backbone_legacy_and_alias():
    assert resolve_backbone("tf_efficientnet_b0_ns")[0] == "tf_efficientnet_b0.ns_jft_in1k"
    assert resolve_backbone("effnet_b0")[0] == "tf_efficientnet_b0.ns_jft_in1k"
    name, is_vit, patch, _ = resolve_backbone("dinov2_large")
    assert is_vit and patch == 14 and name == "vit_large_patch14_dinov2.lvd142m"
    assert resolve_backbone("effnet_b0")[2] is None


def test_windowed_architectures_are_refused_clearly():
    with pytest.raises(NotImplementedError) as exc:
        _build("swin_base_patch4_window7_224")
    assert "swin" in str(exc.value).lower() or "window" in str(exc.value).lower()


# --------------------------------------------------------------------------- #
# 3. Offline / local-weights loading
# --------------------------------------------------------------------------- #

def _cached_checkpoint():
    """A real timm checkpoint already on this machine, or None."""
    import glob
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--timm--tf_efficientnet_b0.ns_jft_in1k/"
        "snapshots/*/model.safetensors"))
    return hits[0] if hits else None


@pytest.mark.skipif(_cached_checkpoint() is None, reason="no local timm checkpoint")
def test_local_weights_file_loads_and_matches(monkeypatch):
    from safetensors.torch import load_file
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    ckpt = _cached_checkpoint()
    ref = load_file(ckpt)["conv_stem.weight"]

    model = _build("effnet_b0", pretrained=False, pretrained_path=ckpt)
    got = model.backbone.conv_stem.weight.detach()
    assert got.shape == ref.shape
    assert torch.allclose(got, ref), "local checkpoint was not actually loaded"


@pytest.mark.skipif(_cached_checkpoint() is None, reason="no local timm checkpoint")
def test_local_weights_directory_loads(tmp_path, monkeypatch):
    import shutil
    from safetensors.torch import load_file
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    ckpt = _cached_checkpoint()
    # Mimic an attached Kaggle Dataset: a directory holding the checkpoint.
    dst = tmp_path / "tf_efficientnet_b0.safetensors"
    shutil.copy(ckpt, dst)
    model = _build("effnet_b0", pretrained=False, pretrained_path=str(tmp_path))
    ref = load_file(ckpt)["conv_stem.weight"]
    assert torch.allclose(model.backbone.conv_stem.weight.detach(), ref)


@pytest.mark.skipif(_cached_checkpoint() is None, reason="no local timm checkpoint")
def test_local_weights_adapt_to_slice_count(tmp_path, monkeypatch):
    """A 3-channel checkpoint must still initialise a 5-slice 2.5D stem."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = _build("effnet_b0", in_channels=5, pretrained=False,
                   pretrained_path=_cached_checkpoint()).eval()
    assert model.backbone.conv_stem.weight.shape[1] == 5
    with torch.no_grad():
        assert model(torch.randn(2, 5, 224, 224)).shape == (2, NUM_CLASSES)


def test_missing_weights_path_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        _build("effnet_b0", pretrained_path="/nonexistent/kaggle/input/weights")


def test_empty_weights_directory_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        _build("effnet_b0", pretrained_path=str(tmp_path))
    assert "no weight file" in str(exc.value)


def test_ambiguous_weights_directory_raises_valueerror(tmp_path):
    (tmp_path / "alpha.safetensors").write_bytes(b"x")
    (tmp_path / "beta.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError) as exc:
        _build("convnext_tiny", pretrained_path=str(tmp_path))
    assert "exact file" in str(exc.value)


def test_pretrained_false_never_needs_network(monkeypatch):
    """The inference path (infer.py builds with pretrained=False) must be offline-safe."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    model = _build("dinov2_small", pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(1, IN_CHANNELS, 224, 224)).shape == (1, NUM_CLASSES)


# --------------------------------------------------------------------------- #
# 4. Freezing / regularisation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("alias", ["effnet_b0", "convnext_tiny", "dinov2_small"])
def test_depth_container_is_found(alias):
    model = _build(alias)
    attr, stack = depth_container(model.backbone)
    assert attr in ("blocks", "stages"), f"{alias}: {attr}"
    assert len(stack) >= 4


def test_freeze_backbone_leaves_only_the_head_trainable():
    model = _build("effnet_b0", freeze_backbone=True)
    trainable, total = model.trainable_parameters()
    assert trainable < total * 0.01, (trainable, total)
    # Head must still be trainable, or there is nothing to learn.
    assert model.fc.weight.requires_grad
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_freeze_blocks_is_monotonic():
    prev = None
    for n in range(0, 8):
        model = _build("effnet_b0", freeze_blocks=n)
        trainable, _ = model.trainable_parameters()
        if prev is not None:
            assert trainable <= prev, f"freeze_blocks={n} increased trainable params"
        prev = trainable


def test_frozen_batchnorm_stats_do_not_drift():
    """requires_grad=False does NOT stop running_mean/var from being overwritten."""
    model = _build("effnet_b0", freeze_blocks=3, freeze_norm_stats=True)
    model.train()
    before = model.backbone.bn1.running_mean.clone()
    model(torch.randn(4, IN_CHANNELS, 224, 224))
    assert torch.equal(before, model.backbone.bn1.running_mean), \
        "frozen BatchNorm statistics drifted during a training forward pass"


def test_freeze_norm_stats_false_allows_drift():
    """Control for the test above -- proves the guard is what stops the drift."""
    model = _build("effnet_b0", freeze_blocks=3, freeze_norm_stats=False)
    model.train()
    before = model.backbone.bn1.running_mean.clone()
    model(torch.randn(4, IN_CHANNELS, 224, 224))
    assert not torch.equal(before, model.backbone.bn1.running_mean)


def test_frozen_model_still_trains_the_head():
    model = _build("effnet_b0", freeze_backbone=True).train()
    out = model(torch.randn(2, IN_CHANNELS, 224, 224))
    out.pow(2).mean().backward()
    assert model.fc.weight.grad is not None
    assert torch.isfinite(model.fc.weight.grad).all()


def test_drop_path_is_actually_wired():
    """drop_path_rate must reach the backbone, not be silently dropped."""
    model = _build("convnext_tiny", drop_path_rate=0.3)
    rates = [m.drop_prob for m in model.backbone.modules()
             if type(m).__name__ == "DropPath"]
    assert rates, "no DropPath modules created"
    assert max(rates) > 0.0


def test_head_dropout_rates_follow_the_base_rate():
    model = _build("effnet_b0", n_dropout=5, head_dropout=0.1)
    assert [round(d.p, 3) for d in model.dropouts] == [0.1, 0.2, 0.3, 0.4, 0.5]
    strong = _build("effnet_b0", n_dropout=3, head_dropout=0.2)
    assert [round(d.p, 3) for d in strong.dropouts] == [0.2, 0.4, 0.6]


def test_multi_sample_dropout_is_deterministic_in_eval():
    model = _build("effnet_b0", n_dropout=5).eval()
    x = torch.randn(2, IN_CHANNELS, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))


# --------------------------------------------------------------------------- #
# 5. MPS safety (these two bugs previously broke this project)
# --------------------------------------------------------------------------- #

def test_no_transformer_encoder_without_nested_tensor_disabled():
    """aten::_nested_tensor_from_mask_left_aligned is unimplemented on MPS."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src", "model.py")).read()
    if "nn.TransformerEncoder(" in src:
        assert "enable_nested_tensor=False" in src, \
            "nn.TransformerEncoder must be built with enable_nested_tensor=False on MPS"


def test_no_bare_view_calls():
    """.view() fails on non-contiguous tensors; .reshape() is the safe form."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src", "model.py")).read()
    assert ".view(" not in src, "use .reshape() instead of .view() (MPS/permute safety)"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
@pytest.mark.parametrize("alias", ["effnet_b0", "dinov2_small"])
def test_forward_backward_on_mps(alias):
    model = _build(alias).to("mps").train()
    out = model(torch.randn(2, IN_CHANNELS, 224, 224, device="mps"))
    assert out.shape == (2, NUM_CLASSES)
    out.float().pow(2).mean().backward()
    assert torch.isfinite(model.fc.weight.grad).all()


# --------------------------------------------------------------------------- #
# 6. Memory planning arithmetic
# --------------------------------------------------------------------------- #

def test_t4_vram_is_15_gib_not_16():
    """A T4 reports 15360 MiB. Plans that assume 16+ GiB are arithmetically wrong."""
    from src.model import DEVICE_VRAM_GIB
    assert DEVICE_VRAM_GIB["t4"] == 15.0


def test_optimizer_state_alone_blocks_dinov2_large():
    """303M params x 18 bytes = 5.1 GiB before a single activation."""
    fixed = estimate_training_memory_gb(303_000_000, act_mib_per_sample=0, batch_size=0)
    assert fixed > 5.0, fixed


def test_plan_batch_size_shrinks_with_size_and_capacity():
    small = plan_batch_size("effnet_b0", 224, device="t4")
    big = plan_batch_size("dinov2_large", 224, device="t4")
    assert small["fits"]
    assert small["max_batch_that_fits"] >= big.get("max_batch_that_fits", 0)


def test_plan_batch_size_grad_accum_reaches_effective_batch():
    p = plan_batch_size("effnet_b0", 224, device="t4", effective_batch=16)
    assert p["fits"]
    assert p["batch_size"] * p["grad_accum_steps"] == 16


def test_plan_reports_patch_snapped_size():
    p = plan_batch_size("dinov2_small", 384, device="t4")
    assert p["image_size"] == 378 and p["requested_size"] == 384


def test_recommend_capacity_scales_with_data():
    tiny = recommend_capacity(58, rarest_positives=9)
    bigger = recommend_capacity(649, rarest_positives=9)
    assert tiny["params_per_study"] > bigger["params_per_study"]
    assert tiny["freeze_blocks"] >= bigger["freeze_blocks"]
    # 9 positives / 5 folds = 1.8 per fold -> AUC is not measurable
    assert tiny["cv_reliable"] is False
    assert "WARNING" in tiny["note"]


def test_activation_lookup_is_name_canonical():
    """alias, legacy spelling and full timm tag must hit the same measured entry."""
    from src.model import activation_mib_per_sample as act
    vals = {act(n, 224) for n in ("effnet_b0", "tf_efficientnet_b0_ns",
                                  "tf_efficientnet_b0.ns_jft_in1k")}
    assert len(vals) == 1, vals


def test_dinov2_large_activation_is_measured_not_a_flat_constant():
    """The depth-blind constant under-estimates DINOv2-Large ~5x -- the direction
    that turns into a mid-run OOM."""
    from src.model import activation_mib_per_sample as act
    large = act("dinov2_large", 224)
    flat_constant = 0.9 * 224 ** 2 / 1000.0   # ~45.2
    assert large > 4 * flat_constant, f"fell back to the depth-blind constant: {large}"
    assert large == pytest.approx(242.8, rel=0.01), "measured value changed; re-benchmark"


def test_depth_width_extrapolation_predicts_a_held_out_vit():
    """Validate the fallback formula against a backbone temporarily removed from
    the measured table, so the test exercises extrapolation rather than lookup."""
    import src.model as M
    name = "vit_large_patch14_dinov2.lvd142m"
    measured = M.MEASURED_ACT_MIB_PER_SAMPLE[(name, 224)]
    # Remove EVERY size for this backbone, otherwise the lookup falls back to
    # pixel-scaling from a sibling size and never reaches the depth*width path.
    saved = {k: v for k, v in M.MEASURED_ACT_MIB_PER_SAMPLE.items() if k[0] == name}
    for k in saved:
        del M.MEASURED_ACT_MIB_PER_SAMPLE[k]
    try:
        predicted = M.activation_mib_per_sample("dinov2_large", 224)
    finally:
        M.MEASURED_ACT_MIB_PER_SAMPLE.update(saved)
    # depth x width from dinov2_small: 46.0 * (24/12) * (1024/384) = 245.3
    assert predicted == pytest.approx(measured, rel=0.05), (predicted, measured)


def test_unknown_backbone_warns_about_its_own_guess():
    from src.model import activation_mib_per_sample as act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        act("resnet50", 224)
    assert any("depth-blind" in str(w.message) for w in caught)


def test_dinov2_large_fixed_cost_is_40_percent_of_a_t4():
    p = plan_batch_size("dinov2_large", 224, device="t4")
    assert p["fixed_gib"] > 5.5, p["fixed_gib"]
    assert p["fixed_gib"] / 15.0 > 0.35


def test_freeze_report_shape():
    rep = _build("dinov2_small", freeze_blocks=6).freeze_report
    assert rep["n_blocks_total"] == 12
    assert rep["n_blocks_frozen"] == 6
    assert rep["frozen_params"] > 0 and rep["trainable_params"] > 0
