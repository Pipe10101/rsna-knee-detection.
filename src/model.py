"""Selectable backbones for the RSNA 2.5D knee-MRI multi-label model.

Input contract (see ``src/kaggle_data.RSNADataset.__getitem__``): the dataset
stacks ``cfg.in_channels`` single-channel slices along dim 0, so a batch is
``(B, in_channels, H, W)`` with ``H == W == cfg.image_size``.  Every backbone
here is built with timm's ``in_chans`` so the pretrained stem is *adapted*
(summed / repeated) rather than thrown away.

Backbone menu (``Config.backbone`` accepts either the short alias or any raw
timm model name).  All numbers MEASURED on timm 1.0.28 at 224px with
``in_channels=3, num_classes=12`` -- i.e. the whole model, head included, which
is why they differ slightly from timm's published backbone-only counts:

    alias              timm name                            params   feat dim
    ---------------    ---------------------------------   -------   --------
    mobilenetv3_small  mobilenetv3_small_100.lamb_in1k        1.5M       1024
    effnet_lite0       tf_efficientnet_lite0.in1k             3.4M       1280
    effnet_b0          tf_efficientnet_b0.ns_jft_in1k         4.0M       1280
    effnetv2_s         tf_efficientnetv2_s.in21k_ft_in1k     20.2M       1280
    convnext_tiny      convnext_tiny.fb_in22k_ft_in1k        27.8M        768
    convnext_small     convnext_small.fb_in22k_ft_in1k       49.5M        768
    effnetv2_m         tf_efficientnetv2_m.in21k_ft_in1k     52.9M       1280
    convnext_base      convnext_base.fb_in22k_ft_in1k        87.6M       1024
    dinov2_small       vit_small_patch14_dinov2.lvd142m      21.6M        768
    dinov2_base        vit_base_patch14_dinov2.lvd142m       85.8M       1536
    dinov2_large       vit_large_patch14_dinov2.lvd142m     303.3M       2048

(A ViT's parameter count depends on ``image_size``: the position embedding is
resampled to the requested grid, so ``dinov2_base`` is 85.8M at 224 but larger at
518, its native resolution.  A CNN's count is resolution-independent.)

PATCH-14 (DINOv2) -- MEASURED, not assumed.  ``vit_*_patch14_dinov2`` uses a
14x14 patch.  timm does **not** raise on a bad size:
``timm.create_model('vit_small_patch14_dinov2', img_size=384)`` builds a 27x27
grid (27*14 = 378 px), ``strict_img_size`` accepts a 384x384 input because
384 == img_size, and the stride-14 / kernel-14 ``patch_embed`` conv drops the
trailing 6 px.  Verified directly: perturbing only rows/cols 378..383 of the
input changes the logits by exactly 0.0, while perturbing rows 370..377 changes
them by 5.41.  ``dynamic_img_size=True`` *does* assert divisibility, but it is
not the default.

This module handles it two ways:
  * default (``strict_image_size=False``) -- snap to the nearest multiple of 14
    and bilinearly resize the incoming tensor, so the whole field of view
    survives at slightly lower resolution, with a ``RuntimeWarning``;
  * ``strict_image_size=True`` -- raise :class:`PatchSizeError` naming the valid
    sizes, for when a silent 384 -> 378 would invalidate an experiment.
Valid sizes: 224, 238, ..., 322, 378, 392, 448, 518 (any multiple of 14).

CAPACITY.  This competition subset has 58 labelled studies.  ``dinov2_large``
is 303M parameters, i.e. ~5.2M parameters per labelled example; even
``effnet_b0`` is ~69k.  See :func:`recommend_capacity` and
:func:`freeze_backbone_blocks` -- at this data scale, how much of the backbone
is *trainable* matters more than which backbone it is.

OFFLINE WEIGHTS.  Kaggle code competitions have no internet, so timm cannot
download anything at runtime.  Pass ``pretrained_path=`` (or set
``Config.backbone_weights``) to a file or directory from an attached Kaggle
Dataset; it is routed through timm's ``pretrained_cfg_overlay={'file': ...}``,
which also handles position-embedding resampling and ``in_chans`` adaptation.
"""

import glob
import os
import re
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# --------------------------------------------------------------------------- #
# Backbone registry
# --------------------------------------------------------------------------- #

#: alias -> (timm model name, default image size)
#:
#: Ordered smallest-first.  With 58 labelled studies the top of this table is
#: the interesting end, not the bottom: ``dinov2_large`` carries 303M parameters
#: for 58 examples (~5.2M params per labelled study), which is a memorisation
#: machine, while ``mobilenetv3_small`` carries 1.5M.  See
#: :func:`recommend_capacity`.
BACKBONE_ALIASES = {
    # -- deliberately small: viable when only tens of studies are labelled ---
    "mobilenetv3_small": ("mobilenetv3_small_100.lamb_in1k", 224),
    "effnet_lite0": ("tf_efficientnet_lite0.in1k", 224),
    # -- current default ----------------------------------------------------
    "effnet_b0": ("tf_efficientnet_b0.ns_jft_in1k", 224),
    "effnet_b3": ("tf_efficientnet_b3.ns_jft_in1k", 300),
    # -- mid --------------------------------------------------------------
    "effnetv2_s": ("tf_efficientnetv2_s.in21k_ft_in1k", 384),
    "effnetv2_m": ("tf_efficientnetv2_m.in21k_ft_in1k", 384),
    "convnext_tiny": ("convnext_tiny.fb_in22k_ft_in1k", 288),
    "convnext_small": ("convnext_small.fb_in22k_ft_in1k", 288),
    "convnext_base": ("convnext_base.fb_in22k_ft_in1k", 384),
    # -- large: only with far more labels than this competition subset has ---
    "dinov2_small": ("vit_small_patch14_dinov2.lvd142m", 224),
    "dinov2_base": ("vit_base_patch14_dinov2.lvd142m", 224),
    "dinov2_large": ("vit_large_patch14_dinov2.lvd142m", 224),
}

#: ``tf_efficientnet_b0_ns`` is a timm-0.x name kept alive only by timm's
#: deprecation shim (``timm.models.get_pretrained_cfg`` returns None for it),
#: so map the legacy spellings onto their current tags explicitly.
LEGACY_NAME_MAP = {
    "tf_efficientnet_b0_ns": "tf_efficientnet_b0.ns_jft_in1k",
    "tf_efficientnet_b3_ns": "tf_efficientnet_b3.ns_jft_in1k",
    "tf_efficientnet_b4_ns": "tf_efficientnet_b4.ns_jft_in1k",
    "tf_efficientnet_b5_ns": "tf_efficientnet_b5.ns_jft_in1k",
}

_VIT_KEYWORDS = ("vit_", "dinov2", "deit", "beit", "eva02", "eva_", "flexivit", "siglip")
#: architectures whose token/window layout this wrapper does not model correctly
_UNSUPPORTED_KEYWORDS = ("swin", "maxvit", "coatnet", "nest_")

WEIGHT_FILE_PATTERNS = ("*.safetensors", "*.bin", "*.pth", "*.pt", "*.ckpt")


def resolve_backbone(name):
    """Map an alias / legacy name to ``(timm_name, is_vit, patch_size, default_size)``.

    ``patch_size`` is ``None`` for non-patch backbones.
    """
    key = (name or "").strip()
    default_size = None
    if key in BACKBONE_ALIASES:
        timm_name, default_size = BACKBONE_ALIASES[key]
    else:
        timm_name = LEGACY_NAME_MAP.get(key, key)

    lowered = timm_name.lower()
    for bad in _UNSUPPORTED_KEYWORDS:
        if bad in lowered:
            raise NotImplementedError(
                f"Backbone '{timm_name}' has a windowed / hybrid token layout that "
                "RSNA25DModel does not pool correctly. Use a CNN (convnext_base, "
                "effnetv2_m) or a plain ViT (dinov2_*) instead."
            )

    is_vit = any(kw in lowered for kw in _VIT_KEYWORDS)
    patch_match = re.search(r"patch(\d+)", lowered)
    patch_size = int(patch_match.group(1)) if (is_vit and patch_match) else None

    if default_size is None:
        try:
            cfg = timm.models.get_pretrained_cfg(timm_name)
            default_size = cfg.input_size[-1] if cfg is not None else 224
        except Exception:
            default_size = 224

    return timm_name, is_vit, patch_size, default_size


def snap_to_patch_multiple(image_size, patch_size):
    """Round ``image_size`` to the nearest positive multiple of ``patch_size``."""
    if not patch_size:
        return int(image_size)
    n = max(1, int(round(image_size / patch_size)))
    return n * patch_size


class PatchSizeError(ValueError):
    """A resolution that a patch-based ViT cannot represent exactly.

    Subclasses ``ValueError`` so ``except ValueError`` in callers still catches it.
    """


def patch_compatible_sizes(patch_size, lo=196, hi=560):
    """Every resolution in ``[lo, hi]`` that this patch size covers exactly."""
    if not patch_size:
        return list(range(lo, hi + 1))
    return [s for s in range(lo, hi + 1) if s % patch_size == 0]


def check_patch_compatible(backbone_name, image_size):
    """Raise :class:`PatchSizeError` if ``image_size`` is not exact for this backbone.

    MEASURED against timm 1.0.28: ``timm.create_model('vit_small_patch14_dinov2',
    img_size=384)`` does **not** raise.  It builds a 27x27 patch grid (27*14 =
    378 px), ``strict_img_size`` then happily accepts a 384x384 input because
    384 == img_size, and the stride-14 / kernel-14 ``patch_embed`` conv drops the
    trailing 6 px.  Perturbing only rows/cols 378..383 of the input changes the
    output by exactly 0.0, while perturbing rows 370..377 changes it by 5.41 --
    i.e. the last 6 px of every image are silently discarded.

    (``dynamic_img_size=True`` *does* assert divisibility -- "Input height (384)
    should be divisible by patch size (14)" -- but it is not the default, and it
    resamples the position embedding on every forward pass.)

    Non-patch backbones are fully convolutional and always pass.
    """
    _, _, patch_size, _ = resolve_backbone(backbone_name)
    size = int(image_size)
    if not patch_size or size % patch_size == 0:
        return size

    grid = size // patch_size
    covered = grid * patch_size
    nearby = [s for s in patch_compatible_sizes(patch_size, max(patch_size, size - 3 * patch_size),
                                                size + 3 * patch_size)]
    raise PatchSizeError(
        f"image_size={size} is invalid for '{backbone_name}': it uses {patch_size}x{patch_size} "
        f"patches and {size} is not a multiple of {patch_size}. timm will NOT raise -- it builds "
        f"a {grid}x{grid} grid covering {covered}px and silently discards the trailing "
        f"{size - covered}px of every image. "
        f"Use one of {nearby} (nearest exact size: {snap_to_patch_multiple(size, patch_size)}), "
        f"or pass strict_image_size=False to snap and resize instead of failing."
    )


def valid_image_size(backbone_name, image_size):
    """Public helper: the resolution this backbone will actually run at.

    The data pipeline can call this to produce correctly sized tensors up-front
    and avoid the in-model resize entirely.
    """
    _, _, patch_size, _ = resolve_backbone(backbone_name)
    return snap_to_patch_multiple(image_size, patch_size)


def _resolve_weights_file(path, timm_name):
    """Turn a Kaggle-Dataset path into a single checkpoint file."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"backbone weights path does not exist: {path}")

    candidates = []
    for pattern in WEIGHT_FILE_PATTERNS:
        candidates.extend(sorted(glob.glob(os.path.join(path, pattern))))
        candidates.extend(sorted(glob.glob(os.path.join(path, "*", pattern))))
    # de-duplicate, preserving order
    seen, unique = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    if not unique:
        raise FileNotFoundError(
            f"no weight file ({', '.join(WEIGHT_FILE_PATTERNS)}) found under {path}"
        )
    if len(unique) == 1:
        return unique[0]

    stem = timm_name.split(".")[0]
    preferred = [c for c in unique if stem in os.path.basename(c)]
    if len(preferred) == 1:
        return preferred[0]
    raise ValueError(
        f"{len(unique)} candidate weight files under {path} "
        f"({[os.path.basename(c) for c in unique[:5]]}...). "
        "Point backbone_weights at the exact file instead of the directory."
    )


# --------------------------------------------------------------------------- #
# Freezing (capacity control)
# --------------------------------------------------------------------------- #

#: Attribute names timm uses for the ordered depth axis of a backbone.
#: MEASURED via ``named_children()`` on timm 1.0.28:
#:   tf_efficientnet_b0     conv_stem, bn1, [blocks x7], conv_head, bn2
#:   tf_efficientnetv2_s    conv_stem, bn1, [blocks x6], conv_head, bn2
#:   mobilenetv3_small_100  conv_stem, bn1, [blocks x6], conv_head
#:   convnext_tiny          stem,           [stages x4], norm_pre, head
#:   vit_*_patch14_dinov2   patch_embed, pos_drop, patch_drop, norm_pre,
#:                          [blocks x12/24], norm
_DEPTH_ATTRS = ("blocks", "stages", "layers")

#: Bare ``nn.Parameter`` attributes that belong to the stem but are not children.
_STEM_PARAM_ATTRS = ("cls_token", "pos_embed", "reg_token", "mask_token", "storage_tokens")


def depth_container(backbone):
    """Return ``(attr_name, sequential)`` for the backbone's ordered depth axis.

    Returns ``(None, None)`` when the architecture has no recognisable stack.
    """
    for attr in _DEPTH_ATTRS:
        mod = getattr(backbone, attr, None)
        if isinstance(mod, (nn.Sequential, nn.ModuleList)) and len(mod) > 0:
            return attr, mod
    return None, None


def freeze_backbone_blocks(backbone, n_blocks=0, freeze_all=False, freeze_norm_stats=True):
    """Freeze the stem plus the first ``n_blocks`` depth stages, in place.

    Why this exists: with 58 labelled studies, *trainable* parameter count is the
    number that decides whether the model memorises.  Freezing is the only knob
    that reduces it without changing the architecture, and it keeps the ImageNet
    /LVD-142M features that were learned from millions of images instead of
    letting 58 knees overwrite them.

    ``freeze_norm_stats`` additionally puts frozen BatchNorm modules into eval
    mode and keeps them there (see :meth:`RSNA25DModel.train`).  This matters:
    ``requires_grad_(False)`` stops the *affine* weights from being learned but
    does **not** stop BatchNorm's ``running_mean`` / ``running_var`` buffers from
    being overwritten by every forward pass, so a "frozen" stem still drifts
    towards 58 studies unless it is also in eval mode.

    Returns a dict describing exactly what happened, so callers can assert on it.
    """
    attr, stack = depth_container(backbone)
    frozen_modules = []

    if freeze_all:
        frozen_modules.append(backbone)
        n_frozen_blocks = len(stack) if stack is not None else 0
    else:
        n_blocks = int(n_blocks)
        if n_blocks <= 0:
            return dict(attr=attr, n_blocks_total=(len(stack) if stack is not None else 0),
                        n_blocks_frozen=0, frozen_params=0, trainable_params=None,
                        froze_stem=False, norm_stats_frozen=False)
        if stack is None:
            raise NotImplementedError(
                f"{type(backbone).__name__} exposes none of {_DEPTH_ATTRS} as a "
                "Sequential/ModuleList, so freeze_blocks cannot address it. Use "
                "freeze_backbone=True (whole-backbone linear probe) or 0."
            )
        n_frozen_blocks = min(n_blocks, len(stack))
        # Everything declared before the depth container is the stem.
        for name, child in backbone.named_children():
            if name == attr:
                break
            frozen_modules.append(child)
        for p_attr in _STEM_PARAM_ATTRS:
            p = getattr(backbone, p_attr, None)
            if isinstance(p, nn.Parameter):
                p.requires_grad_(False)
        frozen_modules.extend(list(stack)[:n_frozen_blocks])

    frozen_norms = []
    for mod in frozen_modules:
        for p in mod.parameters():
            p.requires_grad_(False)
        if freeze_norm_stats:
            for sub in mod.modules():
                if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                    sub.eval()
                    frozen_norms.append(sub)

    frozen = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    return dict(attr=attr, n_blocks_total=(len(stack) if stack is not None else 0),
                n_blocks_frozen=n_frozen_blocks, frozen_params=frozen,
                trainable_params=trainable, froze_stem=True,
                norm_stats_frozen=bool(frozen_norms),
                frozen_batchnorms=len(frozen_norms),
                # Exact module list, so RSNA25DModel.train() re-applies eval() to
                # precisely the modules that were frozen -- inferring the set from
                # requires_grad would also catch affine=False norms elsewhere.
                frozen_norm_modules=frozen_norms)


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #

class GeMPool(nn.Module):
    """Generalized Mean Pooling over an ``(B, C, H, W)`` map.

    Computed in float32: under fp16/bf16 autocast ``x ** p`` overflows easily and
    this pipeline has a history of ``loss=nan``.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float().clamp(min=self.eps)
        pooled = F.avg_pool2d(x.pow(self.p.float()), (x.size(-2), x.size(-1)))
        return pooled.pow(1.0 / self.p.float()).to(dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class AvgMaxPool(nn.Module):
    """Concat of global average and global max pooling -> 2*C features."""

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return torch.cat([avg, mx], dim=1)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class RSNA25DModel(nn.Module):
    """Backbone-agnostic multi-label head for 2.5D knee MRI stacks.

    Args:
        backbone_name: alias from :data:`BACKBONE_ALIASES` or any timm name.
        pretrained: load pretrained weights (needs internet unless
            ``pretrained_path`` is given).
        in_channels: number of stacked slices; matches ``Config.in_channels``.
        num_classes: 12 for this competition.
        image_size: resolution the dataloader emits. For patch ViTs this is
            snapped to the nearest multiple of the patch size and inputs are
            resized to match.
        pretrained_path: local file *or* directory holding timm weights
            (Kaggle offline).
        vit_pool: ``cls`` | ``avg`` | ``cls_avg`` (DINOv2 linear-probe recipe).
        cnn_pool: ``gem`` | ``avg`` | ``avgmax``.
        drop_path_rate: stochastic depth, applied when > 0.
        grad_checkpointing: trade ~30-40% speed for a large activation saving.
        strict_image_size: for a patch ViT, raise :class:`PatchSizeError` instead
            of snapping ``image_size`` to a multiple of the patch size. Use this
            when a silent 384 -> 378 change would invalidate an experiment.
        freeze_blocks: freeze the stem plus the first N backbone stages.
        freeze_backbone: freeze the whole backbone (linear probe); overrides
            ``freeze_blocks``.
        freeze_norm_stats: keep frozen BatchNorm modules in eval mode so their
            running statistics do not drift (see :func:`freeze_backbone_blocks`).
        head_dropout: base rate for multi-sample dropout; pass ``i`` uses
            ``head_dropout * (i + 1)``, so the default 0.1 gives 0.1 .. 0.5.
    """

    def __init__(
        self,
        backbone_name="tf_efficientnet_b0_ns",
        pretrained=True,
        in_channels=3,
        num_classes=12,
        image_size=224,
        pretrained_path=None,
        vit_pool="cls_avg",
        cnn_pool="gem",
        drop_path_rate=0.0,
        grad_checkpointing=False,
        n_dropout=5,
        head_norm=True,
        probe=True,
        resize_inputs="auto",
        strict_image_size=False,
        freeze_blocks=0,
        freeze_backbone=False,
        freeze_norm_stats=True,
        head_dropout=0.1,
        mil_mode=False,
        mil_pool="max",
    ):
        super().__init__()

        timm_name, is_vit, patch_size, _ = resolve_backbone(backbone_name)
        self.backbone_name = backbone_name
        self.timm_name = timm_name
        self.is_vit = is_vit
        self.patch_size = patch_size
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.vit_pool = vit_pool
        self.cnn_pool = cnn_pool
        self.mil_mode = bool(mil_mode)
        self.mil_pool = str(mil_pool).lower()

        # -- input-resize policy ----------------------------------------------
        # A patch ViT bakes its position-embedding grid at build time and timm
        # asserts ``strict_img_size``, so a ViT *must* be fed exactly
        # ``self.image_size``.  A CNN is fully convolutional and the poolers here
        # (GeM / avg / avgmax) collapse any spatial extent, so a CNN can and
        # should run at the resolution the dataloader actually emits.
        #
        #   "auto"   resize only for backbones that require a fixed input (ViTs)
        #   "always" resize every input to self.image_size (legacy behaviour)
        #   "never"  never resize; the caller guarantees the size
        #
        # This matters: callers that build the model without ``image_size`` get
        # the 224 default, and under "always" a 384px progressive-resolution
        # phase is silently bilinearly downsampled back to 224 -- i.e. the whole
        # phase becomes a no-op.  "auto" makes the CNN phases real.
        self.resize_inputs = str(resize_inputs).lower()
        if self.resize_inputs not in ("auto", "always", "never"):
            raise ValueError(
                f"unknown resize_inputs={resize_inputs!r} (auto|always|never)")
        self.requires_fixed_input = bool(is_vit)
        self._warned_input_size = False

        # -- resolution -------------------------------------------------------
        requested = int(image_size)
        if strict_image_size:
            # Raises PatchSizeError with the exact valid sizes; a no-op for CNNs.
            check_patch_compatible(backbone_name, requested)
        self.strict_image_size = bool(strict_image_size)
        self.image_size = snap_to_patch_multiple(requested, patch_size) if is_vit else requested
        if is_vit and self.image_size != requested:
            warnings.warn(
                f"{timm_name} uses patch {patch_size}; image_size={requested} is not a "
                f"multiple of {patch_size} (timm would silently crop "
                f"{requested - (requested // patch_size) * patch_size}px). Running at "
                f"{self.image_size} and resizing inputs.",
                RuntimeWarning,
            )

        # -- backbone ---------------------------------------------------------
        kwargs = dict(num_classes=0, global_pool="", in_chans=1 if self.mil_mode else self.in_channels)
        if drop_path_rate and drop_path_rate > 0:
            kwargs["drop_path_rate"] = float(drop_path_rate)
        if is_vit:
            # global_pool='' returns the full token sequence (B, N, C); we pool
            # it ourselves so cls/avg/cls_avg are all reachable.
            kwargs["img_size"] = self.image_size

        overlay = None
        if pretrained_path:
            overlay = dict(file=_resolve_weights_file(str(pretrained_path), timm_name))
            pretrained = True
        if overlay is not None:
            kwargs["pretrained_cfg_overlay"] = overlay

        try:
            self.backbone = timm.create_model(timm_name, pretrained=bool(pretrained), **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raise with Kaggle guidance
            if pretrained and overlay is None:
                raise RuntimeError(
                    f"Failed to create '{timm_name}' with pretrained=True: {exc}. "
                    "Kaggle code competitions are offline, so timm cannot download "
                    "weights. Attach the weights as a Kaggle Dataset and set "
                    "Config.backbone_weights=/kaggle/input/<dataset>/<file>."
                ) from exc
            raise

        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        self.grad_checkpointing = bool(grad_checkpointing)

        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1 if is_vit else 0))

        # -- probe the real output layout (never assume) -----------------------
        self.layout, backbone_dim = self._probe_output(probe)

        # -- pooling ----------------------------------------------------------
        if self.layout in ("nchw", "nhwc"):
            if cnn_pool == "gem":
                self.pooling = GeMPool()
                feat_dim = backbone_dim
            elif cnn_pool == "avgmax":
                self.pooling = AvgMaxPool()
                feat_dim = backbone_dim * 2
            elif cnn_pool == "avg":
                self.pooling = nn.AdaptiveAvgPool2d(1)
                feat_dim = backbone_dim
            else:
                raise ValueError(f"unknown cnn_pool={cnn_pool!r} (gem|avg|avgmax)")
        elif self.layout == "tokens":
            self.pooling = nn.Identity()
            if vit_pool == "cls_avg":
                feat_dim = backbone_dim * 2
            elif vit_pool in ("cls", "avg"):
                feat_dim = backbone_dim
            else:
                raise ValueError(f"unknown vit_pool={vit_pool!r} (cls|avg|cls_avg)")
        else:  # already a pooled vector
            self.pooling = nn.Identity()
            feat_dim = backbone_dim

        self.backbone_dim = backbone_dim
        self.num_features = feat_dim
        
        if self.mil_mode and self.mil_pool == "gem":
            self.mil_gem = GeMPool()

        # -- head -------------------------------------------------------------
        self.head_norm = nn.LayerNorm(feat_dim) if head_norm else nn.Identity()
        n_dropout = max(1, int(n_dropout))
        self.head_dropout = float(head_dropout)
        self.dropouts = nn.ModuleList([
            nn.Dropout(min(0.9, self.head_dropout * (i + 1))) for i in range(n_dropout)
        ])
        self.fc = nn.Linear(feat_dim, self.num_classes)

        # -- freezing (capacity control; must come after the backbone exists) --
        self.freeze_report = freeze_backbone_blocks(
            self.backbone,
            n_blocks=freeze_blocks,
            freeze_all=bool(freeze_backbone),
            freeze_norm_stats=bool(freeze_norm_stats),
        )
        self.freeze_norm_stats = bool(freeze_norm_stats)
        # Plain list (not a ModuleList): these modules are already registered
        # under self.backbone, so re-registering would duplicate them in
        # state_dict(). deepcopy (used by SWA/EMA) shares the memo, so the copies
        # stay linked to the copied backbone.
        self._frozen_norms = list(self.freeze_report.pop("frozen_norm_modules", []))

    # ------------------------------------------------------------------ #

    def train(self, mode=True):
        """Standard ``train()``, except frozen BatchNorms stay in eval mode.

        ``requires_grad_(False)`` freezes the affine parameters but NOT the
        ``running_mean`` / ``running_var`` buffers, which every training forward
        pass overwrites.  Without this override a "frozen" stem still adapts its
        normalisation statistics to 58 studies.
        """
        super().train(mode)
        for m in getattr(self, "_frozen_norms", ()):   # may run before __init__ finishes
            m.eval()
        return self

    def trainable_parameters(self):
        """``(trainable, total)`` parameter counts for the whole model."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    # ------------------------------------------------------------------ #

    def _probe_output(self, probe):
        """Run one dummy forward to learn the backbone's output layout + width.

        Returns ``(layout, channels)`` where layout is one of
        ``nchw`` | ``nhwc`` | ``tokens`` | ``vector``.
        """
        declared = int(getattr(self.backbone, "num_features", 0)) or None

        if not probe:
            if declared is None:
                raise RuntimeError("probe=False requires backbone.num_features")
            return ("tokens" if self.is_vit else "nchw"), declared

        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                probe_ch = 1 if self.mil_mode else self.in_channels
                out = self.backbone(
                    torch.zeros(1, probe_ch, self.image_size, self.image_size)
                )
        finally:
            self.backbone.train(was_training)

        if out.dim() == 2:
            return "vector", int(out.shape[1])
        if out.dim() == 3:
            return "tokens", int(out.shape[2])
        if out.dim() == 4:
            if declared is not None and out.shape[1] != declared and out.shape[-1] == declared:
                return "nhwc", int(out.shape[-1])
            return "nchw", int(out.shape[1])
        raise RuntimeError(f"unsupported backbone output rank {out.dim()} for {self.timm_name}")

    def _match_input_size(self, x):
        """Apply the ``resize_inputs`` policy to an incoming batch."""
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if h == self.image_size and w == self.image_size:
            return x

        if self.resize_inputs == "never":
            return x

        if self.resize_inputs == "auto" and not self.requires_fixed_input:
            # Fully-convolutional backbone: run natively at the incoming size.
            if not self._warned_input_size:
                self._warned_input_size = True
                warnings.warn(
                    f"{self.timm_name} was built with image_size={self.image_size} but is "
                    f"receiving {h}x{w}. It is fully convolutional, so it runs natively at "
                    f"{h}x{w} (resize_inputs='auto'). Build the model with "
                    f"image_size={h} to silence this and to make the memory planner correct; "
                    "pass resize_inputs='always' for the old downsample-to-image_size "
                    "behaviour.",
                    RuntimeWarning,
                )
            return x

        if not self._warned_input_size:
            self._warned_input_size = True
            warnings.warn(
                f"resizing input {h}x{w} -> {self.image_size}x{self.image_size} for "
                f"{self.timm_name}"
                + (" (patch ViT: the position-embedding grid is fixed at build time)"
                   if self.requires_fixed_input else " (resize_inputs='always')")
                + ". If this was a progressive-resolution phase, the phase is a no-op: "
                "build the model with the phase's image_size.",
                RuntimeWarning,
            )
        return F.interpolate(
            x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )

    def forward_features(self, x):
        """``(B, in_channels, H, W)`` -> pooled feature vector ``(B, num_features)``."""
        x = self._match_input_size(x)
        
        if self.mil_mode:
            B, C, H, W = x.shape
            x = x.reshape(B * C, 1, H, W)

        feats = self.backbone(x)

        if self.layout == "vector":
            pooled = feats
        elif self.layout == "tokens":
            prefix = self.num_prefix_tokens
            patch_tokens = feats[:, prefix:] if prefix else feats
            if self.vit_pool == "avg":
                pooled = patch_tokens.mean(dim=1)
            else:
                cls = feats[:, 0] if prefix else patch_tokens.mean(dim=1)
                if self.vit_pool == "cls":
                    pooled = cls
                else:
                    pooled = torch.cat([cls, patch_tokens.mean(dim=1)], dim=1)
        elif self.layout == "nhwc":
            feats = feats.permute(0, 3, 1, 2).contiguous()
            pooled = self.pooling(feats).flatten(1)
        else:
            pooled = self.pooling(feats).flatten(1)

        if self.mil_mode:
            F_dim = pooled.shape[-1]
            pooled = pooled.reshape(B, C, F_dim)
            if self.mil_pool == "max":
                pooled = pooled.max(dim=1)[0]
            elif self.mil_pool == "avg":
                pooled = pooled.mean(dim=1)
            elif self.mil_pool == "gem":
                pooled = self.mil_gem(pooled.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
                
        return pooled

    def forward(self, x):
        pooled = self.head_norm(self.forward_features(x))
        # Multi-sample dropout: average the logits over several dropout rates.
        logits = None
        for i, dropout in enumerate(self.dropouts):
            out = self.fc(dropout(pooled))
            logits = out if i == 0 else logits + out
        return logits / len(self.dropouts)


# --------------------------------------------------------------------------- #
# Config factory
# --------------------------------------------------------------------------- #

def build_model(cfg):
    """Construct :class:`RSNA25DModel` from a ``Config``.

    Uses ``getattr`` defaults throughout so it keeps working if ``config.py``
    has not yet grown the optional fields.
    """
    weights = getattr(cfg, "backbone_weights", "") or None
    return RSNA25DModel(
        backbone_name=cfg.backbone,
        pretrained=getattr(cfg, "pretrained", True),
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        image_size=getattr(cfg, "image_size", 224),
        pretrained_path=weights,
        vit_pool=getattr(cfg, "vit_pool", "cls_avg"),
        cnn_pool=getattr(cfg, "cnn_pool", "gem"),
        drop_path_rate=getattr(cfg, "drop_path_rate", 0.0),
        grad_checkpointing=getattr(cfg, "grad_checkpointing", False),
        resize_inputs=getattr(cfg, "resize_inputs", "auto"),
        # Regularisation / capacity knobs. All read via getattr so build_model
        # still works against a Config that predates them.
        n_dropout=getattr(cfg, "n_dropout", 5),
        head_dropout=getattr(cfg, "head_dropout", 0.1),
        head_norm=getattr(cfg, "head_norm", True),
        strict_image_size=getattr(cfg, "strict_image_size", False),
        freeze_blocks=getattr(cfg, "freeze_blocks", 0),
        freeze_backbone=getattr(cfg, "freeze_backbone", False),
        freeze_norm_stats=getattr(cfg, "freeze_norm_stats", True),
        mil_mode=getattr(cfg, "mil_mode", False),
        mil_pool=getattr(cfg, "mil_pool", "max"),
    )


# --------------------------------------------------------------------------- #
# Resolution schedule
# --------------------------------------------------------------------------- #

def resolution_schedule(backbone_name, sizes):
    """Snap a progressive-resolution schedule to what the backbone can run.

    ``sizes`` is an iterable of requested sides (e.g. ``[224, 384]``).  For a
    patch ViT each entry is snapped to a multiple of the patch size and
    consecutive duplicates are collapsed -- there is no point running two
    "phases" that are the same resolution.

    >>> resolution_schedule("convnext_base", [224, 384])
    [224, 384]
    >>> resolution_schedule("dinov2_large", [224, 384])   # patch 14
    [224, 378]
    """
    _, _, patch_size, _ = resolve_backbone(backbone_name)
    out = []
    for s in sizes:
        snapped = snap_to_patch_multiple(int(s), patch_size)
        if not out or snapped != out[-1]:
            out.append(snapped)
    return out


# --------------------------------------------------------------------------- #
# Memory planning
# --------------------------------------------------------------------------- #

#: MEASURED activation footprint: MiB of saved-for-backward tensors per sample,
#: under 16-bit autocast, ``in_channels=3``, no gradient checkpointing.
#:
#: Measured with ``torch.autograd.graph.saved_tensors_hooks`` (see
#: ``scripts/bench_backbones.py``) as ``bytes(B=2) - bytes(B=1)`` over unique
#: storages, so parameter-sized saved tensors cancel out and this is a true
#: per-sample slope.  Tensor shapes and dtypes are device-independent, so these
#: numbers transfer from this machine to a T4 -- what does *not* transfer is
#: cuDNN workspace and allocator fragmentation, which
#: :func:`estimate_training_memory_gb` covers with ``workspace_gb``.
#:
#: Key is ``(timm_name, effective_image_size)`` -- the *resolved* timm name, not
#: the alias, so "effnet_b0", "tf_efficientnet_b0_ns" and the full tag all hit
#: the same entry.  Regenerate with:
#:     python3 scripts/bench_backbones.py --emit-table
#:
#: Measured 2026-08-21 on torch 2.8 / timm 1.0.28.
MEASURED_ACT_MIB_PER_SAMPLE = {
    ("mobilenetv3_small_100", 224): 8.5,
    ("mobilenetv3_small_100", 384): 24.8,
    ("tf_efficientnet_lite0", 224): 38.7,
    ("tf_efficientnet_lite0", 384): 113.2,
    ("tf_efficientnet_b0.ns_jft_in1k", 224): 43.1,
    ("tf_efficientnet_b0.ns_jft_in1k", 384): 126.1,
    ("tf_efficientnetv2_s.in21k_ft_in1k", 224): 74.3,
    ("tf_efficientnetv2_s.in21k_ft_in1k", 384): 218.0,
    ("convnext_tiny.fb_in22k_ft_in1k", 224): 53.7,
    ("convnext_tiny.fb_in22k_ft_in1k", 384): 157.9,
    ("tf_efficientnetv2_m.in21k_ft_in1k", 224): 120.4,
    ("tf_efficientnetv2_m.in21k_ft_in1k", 384): 353.1,
    ("convnext_base.fb_in22k_ft_in1k", 224): 112.9,
    ("convnext_base.fb_in22k_ft_in1k", 384): 331.6,
    ("vit_small_patch14_dinov2.lvd142m", 224): 46.0,
    ("vit_small_patch14_dinov2.lvd142m", 378): 130.6,
    ("vit_base_patch14_dinov2.lvd142m", 224): 91.6,
    ("vit_base_patch14_dinov2.lvd142m", 378): 260.2,
    ("vit_large_patch14_dinov2.lvd142m", 224): 242.8,
    ("vit_large_patch14_dinov2.lvd142m", 378): 689.5,
}

#: Ratio of activation memory with ``grad_checkpointing=True`` to without.
#: Measured, same method.  Note the spread: EfficientNet-V2-M saves 93% of its
#: activations, ConvNeXt only 80%, MobileNetV3 only 71% -- checkpointing is not
#: a uniform 4x, it depends on how much of the graph sits inside checkpointed
#: blocks versus in the stem/head.
MEASURED_GRAD_CKPT_RATIO = {
    "mobilenetv3_small_100": 0.291,
    "tf_efficientnet_lite0": 0.121,
    "tf_efficientnet_b0.ns_jft_in1k": 0.109,
    "tf_efficientnetv2_s.in21k_ft_in1k": 0.092,
    "convnext_tiny.fb_in22k_ft_in1k": 0.216,
    "tf_efficientnetv2_m.in21k_ft_in1k": 0.073,
    "convnext_base.fb_in22k_ft_in1k": 0.197,
    "vit_small_patch14_dinov2.lvd142m": 0.113,
    "vit_base_patch14_dinov2.lvd142m": 0.111,
    "vit_large_patch14_dinov2.lvd142m": 0.105,
}

#: Nominal *total* device memory, GiB.  A T4 advertises "16GB" but reports
#: 15360 MiB = 15.0 GiB; a P100 reports 16280 MiB = 15.9 GiB.  Any plan that
#: claims "~18GB fits on a T4" is self-contradictory.
DEVICE_VRAM_GIB = {"t4": 15.0, "p100": 15.9, "v100": 15.8, "a100": 39.5, "l4": 22.0}


def activation_mib_per_sample(backbone_name, image_size, grad_checkpointing=False):
    """Per-sample activation MiB: measured if we have it, else pixel-scaled.

    Activation memory is very close to linear in pixel count for both the CNNs
    here and for timm ViTs (which use ``scaled_dot_product_attention``, so the
    attention matrix is never materialised).  When an exact (alias, size) pair
    is missing we scale the nearest measured size for the same backbone by the
    pixel ratio, and only fall back to a crude constant if the backbone was
    never measured at all.
    """
    # Resolve to the canonical timm name so aliases, legacy spellings and full
    # tags all hit the same measured entry.
    timm_name = resolve_backbone(backbone_name)[0]
    size = int(valid_image_size(backbone_name, image_size))

    exact = MEASURED_ACT_MIB_PER_SAMPLE.get((timm_name, size))
    if exact is None:
        same = {k[1]: v for k, v in MEASURED_ACT_MIB_PER_SAMPLE.items() if k[0] == timm_name}
        if same:
            # Pixel-count scaling. VALIDATED: across every backbone measured
            # here, act(384)/act(224) came out 2.92-2.94 against a theoretical
            # (384/224)^2 = 2.939, i.e. within 0.5%.
            near = min(same, key=lambda s: abs(s - size))
            exact = same[near] * (size ** 2) / (near ** 2)
    if exact is None:
        # Never measured for this backbone at any size.  A flat "MiB per pixel"
        # constant is NOT safe here: it is independent of depth and width, so it
        # under-estimates a big model badly (for vit_large_patch14_dinov2 at 224
        # it gives 45 MiB against a true ~245 MiB -- a 5x optimistic error, which
        # is the direction that turns into a mid-run CUDA OOM).
        #
        # For a plain ViT, activation memory is very close to
        # ``depth * width * tokens``.  VALIDATED THREE TIMES against later
        # measurements: dinov2_base@224 predicted from dinov2_small@224 by
        # (12/12)*(768/384) -> 92.0 MiB vs 91.6 measured (+0.4%);
        # dinov2_large@224 by (24/12)*(1024/384) -> 245.3 vs 242.8 (+1.0%);
        # dinov2_large@378 pixel-scaled from its own 224 -> 691.4 vs 689.5 (+0.3%).
        exact = _extrapolate_activation(timm_name, size)

    if grad_checkpointing:
        exact *= MEASURED_GRAD_CKPT_RATIO.get(timm_name, 0.25)
    return float(exact)


#: ``timm_name -> (depth, width)`` for the plain ViTs in the registry, so an
#: unmeasured ViT can be extrapolated from a measured sibling instead of falling
#: back to a depth-blind constant.  MEASURED from the built models.
_VIT_GEOMETRY = {
    "vit_small_patch14_dinov2.lvd142m": (12, 384),
    "vit_base_patch14_dinov2.lvd142m": (12, 768),
    "vit_large_patch14_dinov2.lvd142m": (24, 1024),
}


def _extrapolate_activation(timm_name, size):
    """Best-effort per-sample activation MiB for a backbone we never measured."""
    geom = _VIT_GEOMETRY.get(timm_name)
    if geom:
        depth, width = geom
        # Find any measured ViT sibling and scale by depth * width * pixels.
        for (name, msize), mib in MEASURED_ACT_MIB_PER_SAMPLE.items():
            sib = _VIT_GEOMETRY.get(name)
            if sib:
                sd, sw = sib
                return mib * (depth / sd) * (width / sw) * (size ** 2) / (msize ** 2)

    warnings.warn(
        f"No measured activation footprint for '{timm_name}'. Falling back to a "
        "depth-blind 0.9 MiB/1000px constant, which under-estimates large models "
        "several-fold. Measure it before trusting any batch-size plan: "
        "python3 scripts/bench_backbones.py --backbones <name> --emit-table",
        RuntimeWarning,
    )
    return 0.9 * (size ** 2) / 1000.0


def recommend_capacity(n_studies, n_labels=12, rarest_positives=None):
    """Right-size the model for ``n_studies`` labelled examples.

    This is a HEURISTIC, not a measurement, and it says so in the returned
    ``basis`` field -- but the arithmetic it is built on is exact and is the
    part worth checking:

    * ``params_per_study = trainable_params / n_studies``.  At 58 studies,
      DINOv2-Large's 303M parameters are 5.2M parameters *per labelled example*.
      Even a 4M-parameter EfficientNet-B0 is 69k per example.  Nothing in this
      range is "supported" by the data in a classical sense; the only thing
      keeping it honest is that the backbone is pretrained and mostly reused,
      which is exactly why FREEZING (which cuts trainable params by 10-100x) is
      a more effective lever here than picking a different architecture.
    * ``supervision_cells = n_studies * n_labels`` is the optimistic count of
      independent supervised scalars.  58 x 12 = 696, and they are strongly
      correlated within a study, so the true figure is lower.
    * ``rarest_positives`` decides whether the CV score is even measurable.
      With k-fold CV, a fold holds ``rarest_positives / k`` positives of the
      rarest class; below ~5 the per-class ROC-AUC is dominated by which
      individual study landed in which fold, and macro-AUC inherits that noise.
    """
    n = max(1, int(n_studies))
    if n <= 120:
        tier = "tiny"
        backbone, image_size = "effnet_b0", 224
        # MEASURED: EfficientNet-B0 is back-loaded -- freezing 5 of its 7 blocks
        # still leaves 78.8% of parameters trainable, and only freeze_blocks>=6
        # (28.5% trainable) is a real reduction. A shallow freeze on this family
        # is close to a no-op, which is why the number here is 6 and not 2 or 3.
        freeze_blocks, drop_path, head_dropout = 6, 0.1, 0.2
        note = ("Fine-tune only the last stage. The early stages are generic edge/"
                "texture filters that ~58 knees cannot improve on. Also evaluate "
                "freeze_backbone=True (pure linear probe, 17.9k trainable params) as "
                "the floor, and mobilenetv3_small / effnet_lite0 as smaller "
                "alternatives -- at this scale the ranking between them is inside CV "
                "noise, so pick by validation, not by datasheet. If you want *graded* "
                "capacity control, note that a ViT freezes linearly (each dinov2_small "
                "block is ~8.2% of its parameters) whereas EfficientNet's freeze curve "
                "is effectively an on/off switch.")
    elif n <= 800:
        tier = "small"
        backbone, image_size = "effnet_b0", 288
        freeze_blocks, drop_path, head_dropout = 2, 0.1, 0.15
        note = ("Enough data to move the whole backbone, but not enough for a "
                "100M+ parameter one. effnetv2_s / convnext_tiny become defensible "
                "as ensemble members; validate the jump rather than assuming it.")
    elif n <= 5000:
        tier = "medium"
        backbone, image_size = "effnetv2_s", 384
        freeze_blocks, drop_path, head_dropout = 0, 0.2, 0.1
        note = "Full fine-tuning of a 20-50M backbone is now reasonable."
    else:
        tier = "large"
        backbone, image_size = "convnext_base", 384
        freeze_blocks, drop_path, head_dropout = 0, 0.3, 0.1
        note = "Large backbones (including dinov2_base) are finally data-supported."

    params = count_parameters(backbone, image_size)
    out = dict(
        n_studies=n, tier=tier, backbone=backbone, image_size=image_size,
        freeze_blocks=freeze_blocks, drop_path_rate=drop_path,
        head_dropout=head_dropout,
        params_total=params, params_per_study=params / n,
        supervision_cells=n * int(n_labels),
        basis="HEURISTIC tier + MEASURED parameter count",
        note=note,
    )
    if rarest_positives is not None:
        r = int(rarest_positives)
        out["rarest_positives"] = r
        out["rarest_positives_per_fold_k5"] = r / 5.0
        out["cv_reliable"] = (r / 5.0) >= 5.0
        if not out["cv_reliable"]:
            out["note"] += (
                f" WARNING: the rarest class has {r} positives, i.e. {r / 5.0:.1f} per "
                "fold at k=5. Per-class ROC-AUC on <5 positives is near-noise, so a "
                "macro-AUC difference between two backbones is probably not real. "
                "Prefer repeated / stratified CV and compare on the pooled out-of-fold "
                "predictions, not per-fold means."
            )
    return out


def count_parameters(backbone_name, image_size=224, in_channels=3, num_classes=12):
    """Exact parameter count of the full model (backbone + head), built on CPU."""
    m = RSNA25DModel(backbone_name=backbone_name, pretrained=False,
                     in_channels=in_channels, num_classes=num_classes,
                     image_size=image_size)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


def estimate_training_memory_gb(
    n_params,
    act_mib_per_sample,
    batch_size,
    optimizer_bytes_per_param=16,
    autocast_cache=True,
    workspace_gb=0.9,
):
    """Estimate peak GPU memory (GiB) for AdamW + 16-bit autocast training.

    ``optimizer_bytes_per_param=16`` = fp32 params (4) + fp32 grads (4) +
    AdamW exp_avg/exp_avg_sq (8). ``autocast_cache`` adds the 2-byte weight copy
    autocast keeps for the duration of the forward pass. ``workspace_gb`` covers
    the CUDA context, cuDNN workspaces and allocator fragmentation.
    """
    gib = 1024 ** 3
    state = n_params * optimizer_bytes_per_param
    if autocast_cache:
        state += n_params * 2
    activations = act_mib_per_sample * (1024 ** 2) * batch_size
    return (state + activations) / gib + workspace_gb


def plan_batch_size(
    backbone_name,
    image_size,
    device="t4",
    n_params=None,
    effective_batch=16,
    max_batch=32,
    grad_checkpointing=False,
    in_channels=3,
    headroom=0.92,
):
    """Largest micro-batch that fits, plus the grad-accum steps to reach
    ``effective_batch``.

    Returns a dict; ``fits`` is False when even ``batch_size=1`` overflows, in
    which case ``advice`` says what to change.  All memory numbers are GiB.

    The optimiser-state term alone is ``n_params * 18 bytes`` (fp32 master +
    fp32 grad + AdamW m/v + the 2-byte autocast weight cache), which for
    DINOv2-Large (303M) is 5.1 GiB *before a single activation* -- that is the
    term the "just use a smaller batch" instinct cannot reduce.
    """
    vram = DEVICE_VRAM_GIB.get(str(device).lower(), float(device)
                               if str(device).replace(".", "").isdigit() else 15.0)
    eff_size = int(valid_image_size(backbone_name, image_size))
    if n_params is None:
        n_params = count_parameters(backbone_name, eff_size, in_channels)
    act = activation_mib_per_sample(backbone_name, eff_size, grad_checkpointing)

    budget = vram * float(headroom)
    fixed = estimate_training_memory_gb(n_params, act, 0)   # state + workspace
    best_bs = 0
    for bs in range(1, int(max_batch) + 1):
        if estimate_training_memory_gb(n_params, act, bs) <= budget:
            best_bs = bs
        else:
            break

    if best_bs == 0:
        need = estimate_training_memory_gb(n_params, act, 1)
        advice = (
            f"does not fit at batch_size=1 ({need:.1f} GiB > {budget:.1f} GiB usable of "
            f"{vram:.1f} GiB). Fixed optimiser+workspace cost alone is {fixed:.1f} GiB. "
            "Options, cheapest first: (1) lower the resolution, "
            "(2) grad_checkpointing=True, (3) a smaller backbone, "
            "(4) an 8-bit optimiser or freeze the backbone (linear probe)."
        )
        return dict(backbone=backbone_name, requested_size=int(image_size),
                    image_size=eff_size, device=str(device), vram_gib=vram,
                    params_m=n_params / 1e6, act_mib_per_sample=act,
                    grad_checkpointing=bool(grad_checkpointing),
                    batch_size=0, grad_accum_steps=0, est_peak_gib=need,
                    fixed_gib=fixed, fits=False, advice=advice)

    bs = min(best_bs, int(effective_batch))
    # Use a micro-batch that divides effective_batch, so accumulation lands on
    # exactly the intended effective batch rather than overshooting it.
    while bs > 1 and int(effective_batch) % bs:
        bs -= 1
    accum = max(1, int(round(float(effective_batch) / bs)))
    peak = estimate_training_memory_gb(n_params, act, bs)
    return dict(backbone=backbone_name, requested_size=int(image_size),
                image_size=eff_size, device=str(device), vram_gib=vram,
                params_m=n_params / 1e6, act_mib_per_sample=act,
                grad_checkpointing=bool(grad_checkpointing),
                batch_size=bs, max_batch_that_fits=best_bs, grad_accum_steps=accum,
                effective_batch=bs * accum, est_peak_gib=peak, fixed_gib=fixed,
                fits=True, advice="")
