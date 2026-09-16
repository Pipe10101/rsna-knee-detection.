"""SlotKnee-S: a per-finding slot-attention reader for multi-sequence knee MRI.

Contract: ``docs/slotknee_spec.md`` §4 plus the head extensions
(group-level tokens, token mixer, aux slot logits).  Rationale:
``docs/research_pivot_efficient_model.md`` §4; head compute measured at ~23 s fp16
for the whole 1,300-study test set (docs/slotknee_review.md), so head-side capacity
is effectively free next to the encoder.

Input layout (what ``src/slots.build_study_tensor`` / ``SlotCache`` emit)::

    x    [B, 6, G, T, P, P]   uint8 0..255  (or float already scaled to 0..1)
    mask [B, 6]               1 = sequence slot present, 0 = missing (x is zeros there)

Pipeline::

    B*6*G triplets ──(uint8 -> /255)──(x - mean) / std, buffers from the timm pretrained cfg
      ──► DINOv2-S/14 encoder, ``encoder_chunk`` images at a time
            stem + first (depth - trainable_blocks) blocks: frozen, eval mode, torch.no_grad()
            last ``trainable_blocks`` blocks + final norm: trainable (optionally checkpointed)
      ──► concat(CLS, mean of patch tokens) 768 ──► Linear(768->d) + LayerNorm + GELU
      ──► tokens + learned slot embedding:
            token_level="group" (default): one token per (slot, group)  -> [B, 6*G, d],
                plus a learned group-position embedding; mask expanded across G
            token_level="slot": mean over the slot's G triplets         -> [B, 6, d]
      ──► ``mixer_layers`` pre-norm self-attention layers over the present tokens
            (built from nn.MultiheadAttention directly -- see _MixerLayer)
      ──► n_labels learned queries, softmax over *present* tokens only
      ──► per-label linear d->1 (einsum) ──► logits [B, n_labels]

Aux head (``aux_slot_logits=True``): ``forward`` returns ``(logits, aux)`` with
``aux [B, 6, n_labels]`` from a shared linear on the *pre-mixer* slot-mean tokens, so
each slot must be independently predictive.  Intended use: masked soft-target BCE on
``aux`` at weight ~0.2 next to the main loss, with absent slots masked out by the
caller via ``mask`` (RSNA 2023 winners' aux losses bought +0.01-0.03 AUC and training
stability).  Default ``False``: the return type stays a plain logits tensor.

MIL pooling (``mil_pool="lse"`` | ``"max"``, default ``"none"``): findings that live on one
or two slices (ACL, MCL, lateral meniscus) are diluted by mean-like attention, so, as in
MRNet and the RSNA 2023 winners, an instance head ``nn.Linear(d, n_labels)`` scores every
(slot, group) token (post-mixer when there is one) and the per-label study logit is pooled
over *present* tokens: ``lse`` = ``tau_mil * (logsumexp(z / tau_mil) - log n_present)``
with a learnable scalar ``tau_mil`` (init 1.0; mean at high tau, max at low tau), ``max``
= hard max (ablation reference).  Final logits = ``(1 - mil_alpha) * attention_logits +
mil_alpha * mil_logits``.  ``last_mil_logits`` / ``last_token_logits`` are kept for
diagnostics.  ``"none"`` adds no parameters, so the state_dict layout is unchanged.

Sequence mixing (``seq_mix="gru"`` | ``"conv"``, default ``"none"``): anchors within a
slot are geometrically ordered and tears are continuous across neighbouring slices, but
tokens are otherwise treated as a bag.  ``"gru"`` runs one bidirectional GRU (hidden d/2
per direction, so output d) along the anchor axis of every slot (sequence length G,
batch B*n_slots); ``"conv"`` is a depthwise 1D conv (k=3) as the cheap alternative.  The
output is residual-added to the per-triplet features BEFORE slot means / tokens are
built, so the attention head, the MIL head and the aux head all see sequence-aware
features.  Absent slots receive exactly zero delta.  Runs internally in fp32
(autocast-safe).  ``"none"`` adds no parameters and keeps today's numerics.

Memory notes (these decide whether the model trains on a 16 GB T4 / 24 GB M-series):

* The frozen prefix runs under ``torch.no_grad()`` **and** stays in eval mode, so it
  retains no activations and is deterministic (its output could be cached per epoch).
* The encoder sees ``encoder_chunk`` images per call.  Under ``no_grad`` (inference)
  that bounds peak activation memory to one chunk.  During training the trainable
  tail's activations for *all* chunks are kept until backward -- that is what
  ``grad_checkpointing=True`` (timm ``set_grad_checkpointing``) addresses.
* Images of absent slots are not encoded at all (``encode_absent=False``); their
  tokens are just the learned embeddings and receive exactly zero query attention.
* The whole new head (group embedding + 1 mixer layer) adds ~0.53M parameters.

fp16 safety: attention masking uses ``torch.finfo(dtype).min`` rather than ``-inf``,
so autocast fp16/bf16 forwards stay finite; masked tokens still receive exactly zero
attention because ``exp(finfo.min - rowmax)`` underflows to 0 in every float format.

Backbone families: any plain timm ``VisionTransformer`` (DINOv2 ``vit_*_patch14_dinov2``,
with or without ``reg4``) and timm's ``Eva`` class, which is what backs the DINOv3 ViTs
(``vit_small_patch16_dinov3.lvd1689m``: patch 16, so 196 patch tokens at P=224 against
DINOv2's 256, plus 5 prefix tokens = 1 CLS + 4 storage/registers).  DINOv3 carries *no*
position-embedding parameter -- position lives entirely in a RoPE table that every block
takes as ``rope=``; ``_rope()`` reconstructs it (see there).  ConvNeXt-DINOv3
(``convnext_small.dinov3_lvd1689m``) is **not** supported: it has no ``blocks`` /
``patch_embed`` / token axis at all, so it needs a separate encoder adapter.

Offline weights: pass ``pretrained_path`` (file or Kaggle-dataset directory) and the
checkpoint is routed through :func:`src.model._resolve_weights_file` into timm's
``pretrained_cfg_overlay``; otherwise timm resolves ``pretrained=True`` from the local
HF cache (``HF_HUB_OFFLINE=1`` is honoured).  Build with ``pretrained=False`` when a
full ``state_dict`` is about to be loaded anyway.

Checkpoint compatibility: every constructor argument is recorded in ``model.hparams``
(saved next to the state_dict), so checkpoints are self-describing.  A model built
with ``token_level="slot", mixer_layers=0, aux_slot_logits=False`` has exactly the
pre-extension parameter set, so old checkpoints still load with ``strict=True``; the
later ``attn_tau`` parameter is filled with ``attn_tau_init`` when a checkpoint lacks it.
"""

import math
import warnings
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:  # timm >= 0.9
    from timm.models._manipulate import checkpoint_seq
except ImportError:  # pragma: no cover - older timm
    from timm.models.helpers import checkpoint_seq

from src.model import _resolve_weights_file, check_patch_compatible

__all__ = ["SlotKneeS", "DEFAULT_BACKBONE", "POOLS", "TOKEN_LEVELS"]

# DELIBERATELY NOT the adopted recipe backbone.  The ADOPTED production encoder is
# `vit_small_patch14_reg4_dinov2.lvd142m` (registers; +0.0220 pooled OOF, ACL +0.0617,
# 2026-08-24) and the production kernels pass it explicitly.  This library default stays on
# plain DINOv2-S because the ablation driver's arms do NOT pass --backbone: flipping it
# mid-campaign would silently retrain every future arm on a different encoder and make each
# one incomparable to the baselines already measured.  Flip it only once the arm queue has
# drained (recipe freeze), and re-baseline when you do.
DEFAULT_BACKBONE = "vit_small_patch14_dinov2.lvd142m"
POOLS = ("cls_mean", "cls", "mean", "cls_mean_max", "cls_mean_topk")
TOKEN_LEVELS = ("group", "slot")
MIL_POOLS = ("none", "lse", "max")
SEQ_MIXES = ("none", "gru", "conv")

#: mixer geometry (per the spec; head compute is ~free at 18-36 tokens)
MIXER_MLP_RATIO = 2.0
MIXER_DROPOUT = 0.1
MIXER_HEADS = 4

#: timm ``VisionTransformer`` attributes this module drives directly.
_REQUIRED_ENCODER_ATTRS = ("patch_embed", "_pos_embed", "patch_drop", "norm_pre", "blocks", "norm")


def _channel_stats(values: Sequence[float], T: int) -> List[float]:
    """Spread 3-channel mean/std over ``T`` input channels.

    Mirrors timm's ``in_chans`` weight adaptation: ``T == 1`` sums the RGB filters
    (so use the average statistic); any other ``T`` tiles the RGB filters cyclically.
    """
    vals = [float(v) for v in values]
    if len(vals) == T:
        return vals
    if T == 1:
        return [sum(vals) / len(vals)]
    return [vals[i % len(vals)] for i in range(T)]


class LoRALinear(nn.Module):
    """``base(x) + B(A(drop(x))) * alpha/r`` with ``base`` frozen (Hu et al. 2021).

    Evidence for using it HERE (docs/research_architecture_20260907.md, S1): fine-tuning only the
    last N blocks is the worst regime in every published head-to-head, while LoRA on all blocks
    matches or beats full fine-tuning below ~10k studies (AnyMC3D: +0.11 AUC over frozen features
    on 3D medical classification; Veasey 2025: LoRA 0.85-0.89 vs partial "far outperformed").
    ``lora_B`` starts at zero so step 0 is exactly the pretrained model.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = int(r), float(alpha) / float(r)
        self.lora_A = nn.Parameter(torch.zeros(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scale


class LoRAConv2d(nn.Module):
    """LoRA for the patch embedding: ``base(x) + B(A(drop(x))) * alpha/r`` where ``A`` is a conv with the
    base kernel/stride (in -> r) and ``B`` a 1x1 conv (r -> out), zero-initialised.  AnyMC3D adapts the
    patch embedding too; it costs ~8k parameters on ViT-S/14."""

    def __init__(self, base: nn.Conv2d, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = int(r), float(alpha) / float(r)
        self.lora_A = nn.Conv2d(base.in_channels, self.r, kernel_size=base.kernel_size, stride=base.stride,
                                padding=base.padding, bias=False)
        self.lora_B = nn.Conv2d(self.r, base.out_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.drop(x))) * self.scale


def _apply_lora(enc, r: int, alpha: float, dropout: float, mlp: bool, patch: bool) -> int:
    """Wrap every attention projection (and optionally the MLP) of every block, plus the patch embedding.
    Returns the number of LoRA parameters added.  Handles timm's fused ``qkv`` and split q/k/v attention."""
    n = 0
    for blk in enc.blocks:
        attn = blk.attn
        for name in ("qkv", "q_proj", "k_proj", "v_proj", "proj"):
            m = getattr(attn, name, None)
            if isinstance(m, nn.Linear):
                w = LoRALinear(m, r, alpha, dropout); setattr(attn, name, w); n += w.lora_A.numel() + w.lora_B.numel()
        if mlp:
            for name in ("fc1", "fc2"):
                m = getattr(blk.mlp, name, None)
                if isinstance(m, nn.Linear):
                    w = LoRALinear(m, r, alpha, dropout); setattr(blk.mlp, name, w); n += w.lora_A.numel() + w.lora_B.numel()
    if patch and isinstance(getattr(enc.patch_embed, "proj", None), nn.Conv2d):
        w = LoRAConv2d(enc.patch_embed.proj, r, alpha, dropout); enc.patch_embed.proj = w
        n += sum(p.numel() for p in (w.lora_A.weight, w.lora_B.weight))
    return n


class _MixerLayer(nn.Module):
    """One pre-norm transformer-encoder layer over the slot/group tokens.

    Deliberately built from ``nn.MultiheadAttention`` instead of
    ``nn.TransformerEncoder``: the latter silently switches to a nested-tensor fast
    path exactly when ``src_key_padding_mask`` is supplied, and that path calls
    ``aten::_nested_tensor_from_mask_left_aligned`` which is unimplemented on MPS
    (docs/known-defects.md §5c).  Building from MHA removes the second code path.
    """

    def __init__(self, d: int, n_heads: int, mlp_ratio: float = MIXER_MLP_RATIO,
                 dropout: float = MIXER_DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d), nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        q = self.norm1(tokens)
        attn_out, _ = self.attn(q, q, q, key_padding_mask=key_padding_mask, need_weights=False)
        tokens = tokens + self.drop(attn_out)
        return tokens + self.mlp(self.norm2(tokens))


class SlotKneeS(nn.Module):
    """DINOv2-S encoder + per-label attention over sequence-slot / group tokens.

    Args:
        P: side of the square input crop; must be a multiple of the ViT patch size (14).
        T: slices per triplet = encoder input channels (``in_chans``).
        d: width of the token / query space.
        n_labels: number of findings (12 for this competition).
        n_slots: number of sequence slots (6: SAG/COR/AX x FS/T1).
        backbone: timm name of a plain ``VisionTransformer``.
        pretrained_path: local weight file or directory (Kaggle offline); implies
            ``pretrained=True``.
        trainable_blocks: number of final transformer blocks left trainable (the final
            norm is always trainable; the stem is always frozen).
        pool: ``cls_mean`` (concat CLS and mean patch token), ``cls`` or ``mean``.
        encoder_chunk: images per encoder call (``<= 0`` = all at once).
        grad_checkpointing: checkpoint the trainable tail (training only).
        drop_path: stochastic-depth rate passed to timm (linearly increasing over depth).
        head_dropout: dropout on the per-label context vectors.
        pretrained: load ImageNet/LVD-142M weights (``False`` when a state_dict follows).
        encode_absent: also run the encoder on zero images of absent slots (off by
            default; only changes the degenerate "no slot present" output).
        token_level: ``"group"`` (default) = per-finding attention over all present
            (slot, group) tokens, keeping within-stack localisation (a meniscal tear
            lives on specific slices); ``"slot"`` = pre-extension behaviour, mean over
            G before attention.
        token_pooling: ``True`` applies 2x2 average pooling to patches before the tail.
        mixer_layers: pre-norm self-attention layers over the tokens before query
            pooling (0 = off = pre-extension behaviour).
        aux_slot_logits: return ``(logits, aux)`` with per-slot logits ``[B, n_slots,
            n_labels]`` for an auxiliary loss (see module docstring).  Default False
            keeps the return type a plain tensor.
        max_groups: capacity of the learned group-position embedding (forward accepts
            any ``G <= max_groups`` when ``token_level="group"``).
        attn_tau_init: initial value of the learnable per-label attention temperature
            ``attn_tau [n_labels]`` that scales the query·key/√d logits before the masked
            softmax.  1.0 reproduces the pre-temperature numerics exactly; a larger value
            sharpens attention from the start (the trained fold-0 model's attention had
            collapsed to uniform, i.e. mean pooling).
        attn_entropy_weight: when > 0, ``forward`` also stores the differentiable
            ``last_attention_entropy`` (mean normalised entropy over present tokens, in
            [0, 1]) so the training script can add ``attn_entropy_weight *
            last_attention_entropy`` to its loss.  The forward return type is unchanged.
            Note: entropy is stationary at exactly-uniform attention, so its gradient
            vanishes there -- the temperature is the stronger lever.
        attn_cosine: ``True`` computes the per-finding attention as a SCALED COSINE
            (both query and token L2-normalised) instead of a raw dot product, so the
            logits are bounded in [-1, 1] and only ``attn_tau`` can sharpen them.  The
            trained default-path model flattens its attention to uniform (entropy 0.9994)
            partly because a dot product lets it do so for free by shrinking norms.
        mil_pool: ``"none"`` (default, no extra parameters), ``"lse"`` (instance head on
            every token, log-sum-exp pooling over present tokens with learnable
            ``tau_mil``) or ``"max"`` (hard max; ablation reference).  See module docstring.
        mil_alpha: weight of the MIL logits in the final mix (attention logits get
            ``1 - mil_alpha``); only used when ``mil_pool != "none"``.
        seq_mix: ``"none"`` (default, no extra parameters), ``"gru"`` (bidirectional
            GRU along each slot's anchor axis, residual) or ``"conv"`` (depthwise 1D
            conv, k=3, residual).  See module docstring.
    """

    def __init__(
        self,
        P: int = 224,
        T: int = 3,
        d: int = 256,
        n_labels: int = 12,
        n_slots: int = 6,
        backbone: str = DEFAULT_BACKBONE,
        pretrained_path: Optional[str] = None,
        trainable_blocks: int = 4,
        pool: str = "cls_mean",
        encoder_chunk: int = 64,
        grad_checkpointing: bool = False,
        drop_path: float = 0.05,
        head_dropout: float = 0.1,
        pretrained: bool = True,
        encode_absent: bool = False,
        token_level: str = "group",
        token_pooling: bool = False,
        mixer_layers: int = 1,
        aux_slot_logits: bool = False,
        max_groups: int = 8,
        attn_tau_init: float = 1.0,
        attn_entropy_weight: float = 0.0,
        attn_cosine: bool = False,
        pool_topk: int = 8,
        mil_pool: str = "none",
        mil_alpha: float = 0.5,
        seq_mix: str = "none",
        aux_slotid: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_mlp: bool = False,
        lora_patch: bool = True,
        slot_tok_embed: bool = False,
    ):
        super().__init__()
        # LoRA (all blocks) replaces the last-N-blocks regime: every block then runs in the "tail" with
        # gradients, the base weights stay frozen, and only LoRA + LayerNorm affines train.
        self.lora_rank = int(lora_rank)
        if self.lora_rank > 0:
            trainable_blocks = 10**6   # clamped to depth below: no frozen prefix under LoRA
        P, T, d = int(P), int(T), int(d)
        n_labels, n_slots = int(n_labels), int(n_slots)
        mixer_layers, max_groups = int(mixer_layers), int(max_groups)
        if min(P, T, d, n_labels, n_slots, max_groups) <= 0:
            raise ValueError("P, T, d, n_labels, n_slots and max_groups must all be positive")
        if pool not in POOLS:
            raise ValueError(f"unknown pool={pool!r} (expected one of {POOLS})")
        if token_level not in TOKEN_LEVELS:
            raise ValueError(f"unknown token_level={token_level!r} (expected one of {TOKEN_LEVELS})")
        if mixer_layers < 0:
            raise ValueError(f"mixer_layers={mixer_layers} must be >= 0")
        if not float(attn_tau_init) > 0:
            raise ValueError(f"attn_tau_init={attn_tau_init} must be > 0")
        if float(attn_entropy_weight) < 0:
            raise ValueError(f"attn_entropy_weight={attn_entropy_weight} must be >= 0")
        if mil_pool not in MIL_POOLS:
            raise ValueError(f"unknown mil_pool={mil_pool!r} (expected one of {MIL_POOLS})")
        if not 0.0 <= float(mil_alpha) <= 1.0:
            raise ValueError(f"mil_alpha={mil_alpha} must be in [0, 1]")
        if seq_mix not in SEQ_MIXES:
            raise ValueError(f"unknown seq_mix={seq_mix!r} (expected one of {SEQ_MIXES})")
        if seq_mix == "gru" and d % 2:
            raise ValueError(f"seq_mix='gru' needs an even d (got d={d})")
        # Raises PatchSizeError (a ValueError) for e.g. P=230: timm would silently
        # crop the trailing pixels instead of failing.
        check_patch_compatible(backbone, P)

        self.P, self.T, self.d = P, T, d
        self.n_labels, self.n_slots = n_labels, n_slots
        self.backbone_name = str(backbone)
        self.pool = pool
        self.encoder_chunk = int(encoder_chunk)
        self.encode_absent = bool(encode_absent)
        self.token_level = token_level
        self.token_pooling = bool(token_pooling)
        self.mixer_layers = mixer_layers
        self.aux_slot_logits = bool(aux_slot_logits)
        self.max_groups = max_groups
        self.attn_tau_init = float(attn_tau_init)
        self.attn_entropy_weight = float(attn_entropy_weight)
        self.attn_cosine = bool(attn_cosine)
        self.pool_topk = int(pool_topk)
        self.mil_pool = str(mil_pool)
        self.mil_alpha = float(mil_alpha)
        self.seq_mix = str(seq_mix)
        self.hparams = dict(
            P=P, T=T, d=d, n_labels=n_labels, n_slots=n_slots, backbone=self.backbone_name,
            trainable_blocks=int(trainable_blocks), pool=pool, encoder_chunk=self.encoder_chunk,
            grad_checkpointing=bool(grad_checkpointing), drop_path=float(drop_path),
            head_dropout=float(head_dropout), encode_absent=self.encode_absent,
            token_level=token_level, mixer_layers=mixer_layers,
            aux_slot_logits=self.aux_slot_logits, max_groups=max_groups,
            token_pooling=self.token_pooling,
            attn_tau_init=self.attn_tau_init, attn_entropy_weight=self.attn_entropy_weight, attn_cosine=self.attn_cosine, pool_topk=self.pool_topk,
            mil_pool=self.mil_pool, mil_alpha=self.mil_alpha, seq_mix=self.seq_mix,
            aux_slotid=bool(aux_slotid),
            lora_rank=self.lora_rank, lora_alpha=float(lora_alpha), lora_dropout=float(lora_dropout),
            lora_mlp=bool(lora_mlp), lora_patch=bool(lora_patch), slot_tok_embed=bool(slot_tok_embed),
        )
        # AUXILIARY FREE TARGET (2026-09-07): predict each image's SLOT (plane x sequence) from its own
        # feature.  Auxiliary losses on free targets were the largest measured lever in three RSNA
        # competitions (+0.01-0.03); this one forces per-image features to carry plane/sequence
        # identity, which the shared encoder otherwise never sees.  Zero inference cost; the head
        # is ignored at inference.  Stored on `last_slotid_logits` (differentiable) for the trainer,
        # like `last_attention_entropy`, so the forward return type is unchanged.
        self.aux_slotid = bool(aux_slotid)
        self.slotid_head = nn.Linear(d, n_slots) if self.aux_slotid else None
        self.last_slotid_logits = None

        # -- encoder ----------------------------------------------------------
        kwargs = dict(img_size=P, num_classes=0, global_pool="", in_chans=T)
        if drop_path and float(drop_path) > 0:
            kwargs["drop_path_rate"] = float(drop_path)
        if pretrained_path:
            kwargs["pretrained_cfg_overlay"] = dict(
                file=_resolve_weights_file(str(pretrained_path), self.backbone_name))
            pretrained = True
        try:
            self.encoder = timm.create_model(self.backbone_name, pretrained=bool(pretrained), **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raise with offline guidance
            if pretrained and not pretrained_path:
                raise RuntimeError(
                    f"Failed to create '{self.backbone_name}' with pretrained=True: {exc}. "
                    "No network is available at runtime: either keep the weights in the "
                    "local HF cache or pass pretrained_path=<file or Kaggle dataset dir>; "
                    "pass pretrained=False when a state_dict is loaded afterwards."
                ) from exc
            raise
        self.pretrained = bool(pretrained)
        enc = self.encoder
        missing = [a for a in _REQUIRED_ENCODER_ATTRS if not hasattr(enc, a)]
        if missing:
            raise TypeError(
                f"{self.backbone_name} ({type(enc).__name__}) is not a plain timm "
                f"VisionTransformer (missing {missing}); SlotKneeS drives the block stack "
                "directly so it needs one.")
        self.embed_dim = int(enc.embed_dim)
        self.num_prefix_tokens = int(getattr(enc, "num_prefix_tokens", 1))
        self.depth = len(enc.blocks)
        k = int(trainable_blocks)
        if self.lora_rank > 0:
            k = self.depth
        if k < 0 or k > self.depth:
            raise ValueError(f"trainable_blocks={k} must be in [0, {self.depth}]")
        self.trainable_blocks = k
        self.n_frozen_blocks = self.depth - k

        # -- freezing: stem + first (depth - k) blocks -------------------------
        # Plain lists (not ModuleLists): the modules are already registered under
        # self.encoder, so re-registering would duplicate them in state_dict().
        self._frozen_blocks = list(enc.blocks)[: self.n_frozen_blocks]
        self._tail_blocks = list(enc.blocks)[self.n_frozen_blocks:]
        # ``patch_drop`` is ``None`` (not Identity) on timm's Eva class, which is what
        # backs the DINOv3 ViTs -- filter Nones instead of crashing on ``.parameters()``.
        self._frozen_modules = [m for m in (enc.patch_embed, enc.pos_drop, enc.patch_drop,
                                            enc.norm_pre) if m is not None]
        self._frozen_modules.extend(self._frozen_blocks)
        for m in self._frozen_modules:
            for p in m.parameters():
                p.requires_grad_(False)
        for name in ("cls_token", "pos_embed", "reg_token", "mask_token"):
            p = getattr(enc, name, None)
            if isinstance(p, nn.Parameter):
                p.requires_grad_(False)
        # -- LoRA: freeze every base encoder weight, wrap projections, train LoRA + LayerNorm affines --
        self._stem_grad = False
        self.lora_params = 0
        if self.lora_rank > 0:
            for p in enc.parameters():
                p.requires_grad_(False)
            self.lora_params = _apply_lora(enc, self.lora_rank, float(lora_alpha), float(lora_dropout),
                                           bool(lora_mlp), bool(lora_patch))
            for blk in enc.blocks:                      # LayerNorm affines train (ExPLoRA)
                for name in ("norm1", "norm2"):
                    ln = getattr(blk, name, None)
                    if ln is not None:
                        for p in ln.parameters():
                            p.requires_grad_(True)
            for p in enc.norm.parameters():
                p.requires_grad_(True)
            # the patch-embedding LoRA lives in the stem, which must then run WITH gradients
            self._stem_grad = bool(lora_patch)
        # -- SLOT IDENTITY INSIDE THE ENCODER (MM-DINOv2, research S4): a zero-initialised per-slot
        # vector added to every token of an image at the input of the first TRAINABLE block (block 0
        # under LoRA, block depth-k otherwise), so the encoder knows which sequence it is looking at.
        self.slot_tok_embed = nn.Embedding(n_slots, self.embed_dim) if slot_tok_embed else None
        if self.slot_tok_embed is not None:
            nn.init.zeros_(self.slot_tok_embed.weight)
        self.grad_checkpointing = False
        self.set_grad_checkpointing(grad_checkpointing)

        # -- normalisation buffers (from the pretrained cfg) ---------------------
        cfg = getattr(enc, "pretrained_cfg", None) or {}
        mean, std = cfg.get("mean"), cfg.get("std")
        if not mean or not std:
            warnings.warn(
                f"{self.backbone_name} has no mean/std in its pretrained_cfg; "
                "falling back to the ImageNet constants.", RuntimeWarning)
            mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        self.register_buffer("norm_mean", torch.tensor(_channel_stats(mean, T)).view(1, T, 1, 1))
        self.register_buffer("norm_std", torch.tensor(_channel_stats(std, T)).view(1, T, 1, 1))

        # -- projection + head ------------------------------------------------
        feat_dim = self.embed_dim * (3 if pool in ("cls_mean_max", "cls_mean_topk")
                                     else 2 if pool == "cls_mean" else 1)
        self.feat_dim = feat_dim
        self.proj = nn.Sequential(nn.Linear(feat_dim, d), nn.LayerNorm(d), nn.GELU())
        self.slot_embed = nn.Parameter(torch.zeros(n_slots, d))
        if token_level == "group":
            self.group_embed = nn.Parameter(torch.zeros(max_groups, d))
            nn.init.trunc_normal_(self.group_embed, std=0.02)
        else:
            self.group_embed = None
        if mixer_layers > 0:
            heads = MIXER_HEADS
            while heads > 1 and d % heads != 0:
                heads //= 2
            self.mixer = nn.ModuleList(
                [_MixerLayer(d, heads, MIXER_MLP_RATIO, MIXER_DROPOUT) for _ in range(mixer_layers)])
        else:
            self.mixer = None
        self.aux_head = nn.Linear(d, n_labels) if self.aux_slot_logits else None
        if self.mil_pool != "none":
            self.mil_head = nn.Linear(d, n_labels)                 # instance (per-token) head
            self.tau_mil = nn.Parameter(torch.ones(()))             # LSE temperature
        else:
            self.mil_head = None
            self.tau_mil = None
        if self.seq_mix == "gru":
            self.seq_mixer = nn.GRU(d, d // 2, num_layers=1, batch_first=True, bidirectional=True)
        elif self.seq_mix == "conv":
            self.seq_mixer = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d)
        else:
            self.seq_mixer = None
        
        # GQA: Group the 12 labels into 5 anatomical groups
        # Ligaments(2), Meniscus(2), OA(3), Fluid/Inflamm(3), Trauma(2)
        self.query_groups = 5
        self.base_queries = nn.Parameter(torch.zeros(self.query_groups, d))
        self.register_buffer("group_map", torch.tensor(
            [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4], dtype=torch.long
        ))
        
        self.label_weight = nn.Parameter(torch.empty(n_labels, d))
        self.label_bias = nn.Parameter(torch.zeros(n_labels))
        # Learnable per-label attention temperature.  Multiplying by 1.0 is exact, so
        # the default reproduces the pre-temperature numerics bit for bit.
        self.attn_tau = nn.Parameter(torch.full((n_labels,), self.attn_tau_init))
        self.head_dropout = nn.Dropout(float(head_dropout))
        nn.init.trunc_normal_(self.slot_embed, std=0.02)
        nn.init.trunc_normal_(self.base_queries, std=0.02)
        bound = 1.0 / math.sqrt(d)  # what nn.Linear(d, 1) would use
        nn.init.uniform_(self.label_weight, -bound, bound)

        #: cached rotary position embedding for RoPE encoders (DINOv3); ``None`` for
        #: DINOv2, which has no ``rope`` module and never touches this path.
        self._rope_cache = None
        self._last_attention = None
        self._last_group_attention = None
        self._last_token_attention = None
        self._last_token_present = None
        self.last_attention_entropy = None
        self.last_mil_logits = None
        self.last_token_logits = None
        self.last_n_encoded = 0
        self.train()  # puts the frozen prefix into eval mode

    # ------------------------------------------------------------------ #
    # mode / bookkeeping
    # ------------------------------------------------------------------ #

    def train(self, mode: bool = True):
        """Standard ``train()``, except the frozen prefix always stays in eval mode.

        Frozen blocks have nothing to learn, so their stochastic depth would only
        inject noise into features the tail then has to learn around; keeping them
        in eval also makes the prefix output deterministic (cacheable).
        """
        super().train(mode)
        for m in getattr(self, "_frozen_modules", ()):  # may run before __init__ finishes
            m.eval()
        return self

    def set_grad_checkpointing(self, enable: bool = True):
        """Checkpoint the trainable tail (delegates to timm ``set_grad_checkpointing``)."""
        self.grad_checkpointing = bool(enable)
        if hasattr(self.encoder, "set_grad_checkpointing"):
            self.encoder.set_grad_checkpointing(self.grad_checkpointing)
        return self

    def trainable_parameters(self):
        """``(trainable, total)`` parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def slot_attention(self) -> torch.Tensor:
        """Attention of the last forward: ``[B, n_labels, n_slots]`` (detached).

        With ``token_level="group"`` this is the group attention summed within each
        slot, so its meaning is stable across both token levels: how much each
        finding read each sequence slot.  Exactly 0 on absent slots.
        """
        if self._last_attention is None:
            raise RuntimeError("slot_attention() called before any forward pass")
        return self._last_attention

    def group_attention(self) -> torch.Tensor:
        """Group-level attention of the last forward: ``[B, n_labels, n_slots, G]``.

        Exactly 0 on absent slots (all their groups).  Only meaningful with
        ``token_level="group"``; raises otherwise.
        """
        if self._last_group_attention is None:
            raise RuntimeError(
                "group_attention() requires token_level='group' and a prior forward pass")
        return self._last_group_attention

    @staticmethod
    def _normalised_entropy(attn: torch.Tensor, tok_present: torch.Tensor) -> torch.Tensor:
        """Mean over batch and labels of the attention entropy over present tokens,
        divided by ``log(n_present)`` so it lies in [0, 1] (1 = uniform = mean pooling).

        Absent tokens have attention exactly 0 and contribute nothing; rows with fewer
        than two present tokens have zero entropy and are normalised by log 2.
        """
        p = attn.float()
        h = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)              # [B, L]
        n_present = tok_present.sum(dim=-1).clamp_min(2).float()           # [B]
        return (h / torch.log(n_present)[:, None]).clamp(0.0, 1.0).mean()

    def attention_report(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                         is_cached: bool = False) -> Dict:
        """Diagnostic for the training log: where each finding looks, averaged over a batch.

        Runs one forward in eval mode under ``no_grad`` (mode restored afterwards) and
        returns CPU tensors::

            slot         [n_labels, n_slots]  mean slot attention over studies (absent = 0)
            slot_present [n_labels, n_slots]  mean over the studies where the slot is present
            anchor       [n_labels, G]        group attention summed over slots, mean over
                                              studies (None with token_level="slot")
            entropy      float                normalised attention entropy in [0, 1]
            tau          [n_labels]           current per-label temperature
            n_studies    int

        Uniform rows (≈1/G anchors, ≈1/n_present slots, entropy ≈ 1) mean the query
        attention has collapsed to mean pooling.
        """
        was_training = self.training
        self.eval()
        try:
            device = self.norm_mean.device
            x = x.to(device)
            mask_dev = None if mask is None else mask.to(device)
            with torch.no_grad():
                self(x, mask_dev, is_cached=is_cached)
                B, S = int(x.shape[0]), int(x.shape[1])
                present = (torch.ones(B, S, dtype=torch.bool, device=device)
                           if mask_dev is None else mask_dev > 0)
                slot = self._last_attention.float()                                  # [B, L, S]
                slot_present = slot.sum(dim=0) / present.float().sum(dim=0).clamp_min(1.0)[None, :]
                anchor = None
                if self._last_group_attention is not None:
                    anchor = self._last_group_attention.float().sum(dim=2).mean(dim=0)   # [L, G]
                entropy = self._normalised_entropy(self._last_token_attention, self._last_token_present)
        finally:
            self.train(was_training)
        return dict(
            slot=slot.mean(dim=0).cpu(),
            slot_present=slot_present.cpu(),
            anchor=None if anchor is None else anchor.cpu(),
            entropy=float(entropy),
            tau=self.attn_tau.detach().float().cpu(),
            n_studies=int(slot.shape[0]),
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """``nn.Module.load_state_dict`` that also accepts checkpoints saved before
        ``attn_tau`` existed: the missing temperature is filled with ``attn_tau_init``
        (1.0 reproduces the old numerics exactly), so old checkpoints load ``strict``.
        """
        if "attn_tau" not in state_dict:
            warnings.warn(
                f"checkpoint has no 'attn_tau'; filling it with attn_tau_init={self.attn_tau_init:g}",
                RuntimeWarning)
            state_dict = dict(state_dict)
            state_dict["attn_tau"] = torch.full((self.n_labels,), self.attn_tau_init)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def extra_repr(self) -> str:
        return (f"P={self.P}, T={self.T}, d={self.d}, n_labels={self.n_labels}, "
                f"n_slots={self.n_slots}, pool={self.pool}, trainable_blocks="
                f"{self.trainable_blocks}/{self.depth}, encoder_chunk={self.encoder_chunk}, "
                f"grad_checkpointing={self.grad_checkpointing}, encode_absent={self.encode_absent}, "
                f"token_level={self.token_level}, mixer_layers={self.mixer_layers}, "
                f"aux_slot_logits={self.aux_slot_logits}, token_pooling={self.token_pooling}, "
                f"attn_tau_init={self.attn_tau_init}, attn_entropy_weight={self.attn_entropy_weight}, "
                f"mil_pool={self.mil_pool}, mil_alpha={self.mil_alpha}, seq_mix={self.seq_mix}")

    # ------------------------------------------------------------------ #
    # optimiser groups
    # ------------------------------------------------------------------ #

    def param_groups(self, lr_backbone: float, lr_head: float, llrd: float = 0.8) -> List[Dict]:
        """Layer-wise LR decay: deepest trainable block at ``lr_backbone``, each earlier
        block multiplied by ``llrd``; final norm at ``lr_backbone``; everything outside
        the encoder (projection, slot/group embeddings, mixer, queries, per-label
        linear, aux head) at ``lr_head``.

        Every trainable parameter appears in exactly one group; frozen ones in none.
        Each group carries a ``name`` for logging (torch optimisers keep extra keys).
        """
        if self.lora_rank > 0:
            llrd = 1.0        # LoRA recipes use one uniform LR (AnyMC3D, MIDOG-25); decay is a full-FT device
        groups: List[Dict] = []
        for i in range(self.n_frozen_blocks, self.depth):
            params = [p for p in self.encoder.blocks[i].parameters() if p.requires_grad]
            if params:
                groups.append(dict(params=params, lr=float(lr_backbone) * float(llrd) ** (self.depth - 1 - i),
                                   name=f"blocks.{i}"))
        norm_params = [p for p in self.encoder.norm.parameters() if p.requires_grad]
        if norm_params:
            groups.append(dict(params=norm_params, lr=float(lr_backbone), name="norm"))
        covered = {id(p) for g in groups for p in g["params"]}
        rest = [p for p in self.encoder.parameters() if p.requires_grad and id(p) not in covered]
        if rest:  # defensive: any other trainable encoder parameter (none for DINOv2)
            groups.append(dict(params=rest, lr=float(lr_backbone), name="encoder.other"))
        head = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
        groups.append(dict(params=head, lr=float(lr_head), name="head"))
        n_grouped = sum(len(g["params"]) for g in groups)
        n_trainable = sum(1 for p in self.parameters() if p.requires_grad)
        if n_grouped != n_trainable:
            raise RuntimeError(f"param_groups covers {n_grouped} tensors but {n_trainable} are trainable")
        return groups

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def _normalise(self, imgs: torch.Tensor) -> torch.Tensor:
        """uint8 0..255 -> float 0..1 -> (x - mean) / std.  Float input is assumed 0..1."""
        if not imgs.is_floating_point():
            imgs = imgs.to(self.norm_mean.dtype).div_(255.0)
        return (imgs - self.norm_mean) / self.norm_std

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Squash one image's patch tokens into a single vector.

        THE DILUTION THIS ADDRESSES (measured 2026-08-25).  A 224 px image is 256 patches; a
        meniscus tear or an MCL sprain occupies perhaps 3-8 of them.  ``cls_mean`` averages all
        256, so a small lesion's evidence is attenuated ~30-80x BEFORE any slot, attention or
        MIL machinery ever sees it.  That is almost certainly why MIL pooling over (slot,
        anchor) tokens REGRESSED (-0.0176): it attacked the second dilution while the first had
        already destroyed the signal.  ``cls_mean_max`` and ``cls_mean_topk`` keep a peak
        statistic alongside the mean, so focal evidence survives the encoder.

        Note these change ``feat_dim`` and therefore the projection shape: a checkpoint trained
        with one pool cannot be loaded into another.  ``hparams`` records it, so inference
        rebuilds correctly.
        """
        prefix = self.num_prefix_tokens
        patches = tokens[:, prefix:] if prefix else tokens
        if self.pool == "mean":
            return patches.mean(dim=1)
        cls = tokens[:, 0] if prefix else patches.mean(dim=1)
        if self.pool == "cls":
            return cls
        if self.pool == "cls_mean_max":
            return torch.cat([cls, patches.mean(dim=1), patches.max(dim=1).values], dim=1)
        if self.pool == "cls_mean_topk":
            # mean of the k most-activated patches: a softer max, less noise-sensitive than a
            # single argmax while still preserving a focal response.
            k = max(1, min(self.pool_topk, patches.shape[1]))
            top = patches.topk(k, dim=1).values.mean(dim=1)
            return torch.cat([cls, patches.mean(dim=1), top], dim=1)
        return torch.cat([cls, patches.mean(dim=1)], dim=1)

    def _rope(self, ref: torch.Tensor) -> Optional[torch.Tensor]:
        """Rotary position embedding for a RoPE encoder, or ``None`` for DINOv2.

        timm's Eva class (which backs ``vit_*_dinov3``) has no position-embedding
        parameter: every block needs ``rope=`` or the encoder silently loses all
        positional information (measured: cos 0.887 against the correct output).
        ``img_size`` is fixed at ``P``, so the embedding is a constant of the build
        and is cached here -- ``rope.get_embed(shape=(P/patch, P/patch))`` is bitwise
        equal to what ``Eva._pos_embed`` computes.  Caching it also lets the
        cached-activation path (which never runs ``patch_embed``) reconstruct it.
        """
        rope_mod = getattr(self.encoder, "rope", None)
        if rope_mod is None:
            return None
        cached = self._rope_cache
        if cached is None or cached.device != ref.device:
            patch = self.encoder.patch_embed.patch_size[0]
            g = self.P // int(patch)
            with torch.no_grad():
                emb = rope_mod.get_embed(shape=(g, g))
            self._rope_cache = cached = emb.to(ref.device)
        return cached

    def _encode_frozen_chunk(self, imgs: torch.Tensor) -> torch.Tensor:
        enc = self.encoder
        with torch.set_grad_enabled(torch.is_grad_enabled() and self._stem_grad):  # frozen prefix unless patch-LoRA
            t = enc.patch_embed(imgs)
            t = enc._pos_embed(t)
            if isinstance(t, tuple):  # Eva/DINOv3 returns (tokens, rotary embedding)
                t = t[0]
            if enc.patch_drop is not None:  # None on Eva (already applied in _pos_embed)
                t = enc.patch_drop(t)
            t = enc.norm_pre(t)
            rope = self._rope(t)
            for blk in self._frozen_blocks:
                t = blk(t) if rope is None else blk(t, rope=rope)
        return t

    def _encode_tail_chunk(self, t: torch.Tensor) -> torch.Tensor:
        enc = self.encoder
        if self._tail_blocks:
            if getattr(self, "token_pooling", False):
                B_t, N_t, C_t = t.shape
                H_t = int(math.sqrt(N_t - self.num_prefix_tokens))
                if H_t * H_t == N_t - self.num_prefix_tokens:
                    cls_toks = t[:, :self.num_prefix_tokens, :]
                    patches = t[:, self.num_prefix_tokens:, :].reshape(B_t, H_t, H_t, C_t).permute(0, 3, 1, 2)
                    patches = F.avg_pool2d(patches, kernel_size=2)
                    patches = patches.flatten(2).transpose(1, 2)
                    t = torch.cat([cls_toks, patches], dim=1)

            rope = self._rope(t)
            ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()
            if ckpt and rope is None:
                t = checkpoint_seq(self._tail_blocks, t, use_reentrant=False)
            elif rope is None:
                for blk in self._tail_blocks:
                    t = blk(t)
            else:
                # checkpoint_seq() forwards no kwargs; Eva blocks take rope positionally.
                for blk in self._tail_blocks:
                    t = (torch.utils.checkpoint.checkpoint(blk, t, rope, use_reentrant=False)
                         if ckpt else blk(t, rope=rope))
        t = enc.norm(t)
        return self._pool_tokens(t)

    def encode_frozen(self, imgs: torch.Tensor) -> torch.Tensor:
        """Runs only the frozen prefix and returns the tokens (for offline caching)."""
        n = int(imgs.shape[0])
        chunk = self.encoder_chunk if self.encoder_chunk > 0 else n
        pooled = [self._encode_frozen_chunk(self._normalise(imgs[s:s + chunk])) for s in range(0, n, max(1, chunk))]
        return pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=0)

    def _add_slot_tok(self, t: torch.Tensor, slot_ids) -> torch.Tensor:
        if self.slot_tok_embed is None or slot_ids is None:
            return t
        return t + self.slot_tok_embed(slot_ids.to(t.device))[:, None, :].to(t.dtype)

    def encode_images(self, imgs: torch.Tensor, slot_ids=None) -> torch.Tensor:
        """``[n, T, P, P]`` uint8/float -> projected ``[n, d]``, ``encoder_chunk`` images at a time.
        ``slot_ids [n]`` (optional) feeds the in-encoder slot embedding."""
        n = int(imgs.shape[0])
        chunk = self.encoder_chunk if self.encoder_chunk > 0 else n
        pooled = []
        for s in range(0, n, max(1, chunk)):
            t = self._encode_frozen_chunk(self._normalise(imgs[s:s + chunk]))
            t = self._add_slot_tok(t, None if slot_ids is None else slot_ids[s:s + chunk])
            pooled.append(self._encode_tail_chunk(t))
        pooled = pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=0)
        return self.proj(pooled)

    def encode_tail(self, cached_t: torch.Tensor, slot_ids=None) -> torch.Tensor:
        """``[n, N, C]`` cached frozen tokens -> projected ``[n, d]``, ``encoder_chunk`` images at a time."""
        n = int(cached_t.shape[0])
        chunk = self.encoder_chunk if self.encoder_chunk > 0 else n
        pooled = []
        for s in range(0, n, max(1, chunk)):
            t = cached_t[s:s + chunk].to(self.slot_embed.device)
            t = self._add_slot_tok(t, None if slot_ids is None else slot_ids[s:s + chunk])
            pooled.append(self._encode_tail_chunk(t))
        pooled = pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=0)
        return self.proj(pooled)

    def _build_feats(self, x: torch.Tensor, present: torch.Tensor, B: int, S: int, G: int, T: int, H: int, W: int, is_cached: bool = False):
        N = B * S * G
        if is_cached:
            imgs = x.reshape(N, x.shape[-2], x.shape[-1]) # [N, 257, 384]
        else:
            imgs = x.reshape(N, T, H, W)
            
        # slot index of every image, in the (B, S, G) flattening order
        slot_ids = (torch.arange(N, device=imgs.device) // G) % S if self.slot_tok_embed is not None else None
        if self.encode_absent:
            feats = self.encode_tail(imgs, slot_ids) if is_cached else self.encode_images(imgs, slot_ids)
            self.last_n_encoded = N
        else:
            keep = present[:, :, None].expand(B, S, G).reshape(N)
            idx = keep.nonzero(as_tuple=True)[0]
            self.last_n_encoded = int(idx.numel())
            feats = torch.zeros(N, self.d, dtype=self.norm_mean.dtype, device=x.device if not is_cached else self.slot_embed.device)
            if idx.numel():
                sid = None if slot_ids is None else slot_ids[idx]
                enc = self.encode_tail(imgs[idx], sid) if is_cached else self.encode_images(imgs[idx], sid)
                feats = feats.to(enc.dtype).index_copy(0, idx.to(enc.device), enc)
        return feats

    def _seq_mix_delta(self, per_triplet: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Sequence mixing along the anchor axis of each slot, as a residual delta.

        ``per_triplet [B, S, G, d]`` -> delta of the same shape.  Each slot is one
        sequence of length G (batch B*S), so slots never mix with each other here.
        Runs in fp32 (GRU/conv under low-precision autocast is not reliable on every
        backend) and returns exactly zero for absent slots -- their tokens stay the
        learned embeddings, and the degenerate all-absent fallback stays deterministic.
        """
        B, S, G, d = per_triplet.shape
        seq = per_triplet.reshape(B * S, G, d).float()
        if self.seq_mix == "gru":
            out, _ = self.seq_mixer(seq)                                   # [B*S, G, d]
        else:
            out = self.seq_mixer(seq.transpose(1, 2)).transpose(1, 2)      # depthwise conv
        out = out.reshape(B, S, G, d) * present[:, :, None, None].to(out.dtype)
        return out.to(per_triplet.dtype)

    def _mil_pool_tokens(self, tokens: torch.Tensor, tok_present: torch.Tensor):
        """Instance head on every token, pooled per label over the present tokens.

        Returns ``(mil_logits [B, n_labels] fp32, token_logits [B, n_tokens, n_labels])``.
        Pooling runs in fp32 (autocast-safe).  Absent tokens are filled with
        ``finfo.min`` *after* the division by ``tau_mil`` so they are constants with
        respect to the temperature (no ``-z / tau**2`` overflow in its gradient) and
        contribute exactly 0 to the log-sum-exp.  A study with no present token pools
        over all its tokens, matching the attention fallback.
        """
        token_logits = self.mil_head(tokens)                                     # [B, N, L]
        z = token_logits.float()
        n_tokens = z.shape[1]
        has_any = tok_present.any(dim=1, keepdim=True)
        absent = (~tok_present & has_any)[:, :, None]
        neg = torch.finfo(z.dtype).min
        if self.mil_pool == "max":
            pooled = z.masked_fill(absent, neg).max(dim=1).values
        else:
            tau = self.tau_mil.float().clamp_min(1e-3)
            n_present = torch.where(has_any[:, 0], tok_present.sum(dim=1),
                                    torch.full_like(tok_present.sum(dim=1), n_tokens)).float()
            lse = torch.logsumexp((z / tau).masked_fill(absent, neg), dim=1)      # [B, L]
            pooled = tau * (lse - torch.log(n_present)[:, None])
        return pooled, token_logits

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, is_cached: bool = False):
        if is_cached:
            # x is [B, 6, G, 257, 384] (the frozen token activations)
            B, S, G, N_tok, C_tok = x.shape
            T, H, W = self.T, self.P, self.P # unused for cached
        else:
            if x.dim() != 6:
                raise ValueError(f"expected x [B, {self.n_slots}, G, {self.T}, {self.P}, {self.P}], got {tuple(x.shape)}")
            B, S, G, T, H, W = x.shape
            if S != self.n_slots or T != self.T or H != self.P or W != self.P:
                raise ValueError(
                    f"expected x [B, {self.n_slots}, G, {self.T}, {self.P}, {self.P}], got {tuple(x.shape)}")
                    
        if self.token_level == "group" and G > self.max_groups:
            raise ValueError(
                f"G={G} exceeds max_groups={self.max_groups}; rebuild with max_groups>={G}")
        if mask is None:
            present = torch.ones(B, S, dtype=torch.bool, device=x.device if not is_cached else self.slot_embed.device)
        else:
            if tuple(mask.shape) != (B, S):
                raise ValueError(f"expected mask [{B}, {S}], got {tuple(mask.shape)}")
            present = mask.to(x.device if not is_cached else self.slot_embed.device) > 0

        feats = self._build_feats(x, present, B, S, G, T, H, W, is_cached)

        # -- build tokens ------------------------------------------------------
        per_triplet = feats.reshape(B, S, G, self.d)
        self.last_slotid_logits = (self.slotid_head(self.head_dropout(per_triplet)) if self.slotid_head is not None else None)   # [B, S, G, n_slots]
        if self.seq_mixer is not None:
            # BEFORE the mixer/attention: every head sees sequence-aware features.
            per_triplet = per_triplet + self._seq_mix_delta(per_triplet, present)
        slot_means = per_triplet.mean(dim=2) + self.slot_embed.to(feats.dtype)   # [B, S, d]
        if self.token_level == "group":
            tokens = (per_triplet
                      + self.slot_embed.to(feats.dtype)[None, :, None, :]
                      + self.group_embed[:G].to(feats.dtype)[None, None, :, :])
            tokens = tokens.reshape(B, S * G, self.d)                             # [B, S*G, d]
            tok_present = present[:, :, None].expand(B, S, G).reshape(B, S * G)
        else:
            tokens = slot_means                                                   # [B, S, d]
            tok_present = present

        # -- mixer -------------------------------------------------------------
        if self.mixer is not None:
            kpm = ~tok_present
            # A study with no present token would make every key masked -> NaN out
            # of MHA; let those (degenerate) rows attend everywhere instead.
            kpm = kpm & ~kpm.all(dim=1, keepdim=True)
            for layer in self.mixer:
                tokens = layer(tokens, key_padding_mask=kpm)

        # -- per-label query attention over present tokens ---------------------
        queries = self.base_queries[self.group_map]
        if self.attn_cosine:
            # Scaled COSINE attention (cf. Swin-V2, introduced there for exactly this failure).
            # With a raw dot product the network can flatten attention for free by shrinking
            # ||q|| or ||token||, and measurement showed it does: the trained logits span only
            # 0.177 and the learnable temperature DRIFTED DOWN from 1.0 to ~0.96.  Normalising
            # both sides bounds the logits to [-1, 1], so sharpness can only come from
            # attn_tau -- which is explicit, inspectable and regularisable.
            q = F.normalize(queries.to(tokens.dtype), dim=-1)
            k = F.normalize(tokens, dim=-1)
            att = torch.einsum("ld,bnd->bln", q, k)
        else:
            att = torch.einsum("ld,bnd->bln", queries.to(tokens.dtype), tokens) / math.sqrt(self.d)
        att = att * self.attn_tau.to(att.dtype)[None, :, None]          # per-label temperature
        # finfo.min instead of -inf: fp16/bf16-safe, still exp-underflows to exactly
        # 0 after the softmax's row-max subtraction.
        att = att.masked_fill(~tok_present[:, None, :], torch.finfo(att.dtype).min)
        # A study with no token at all: uniform attention instead of NaN.
        att = att.masked_fill(~tok_present.any(dim=1)[:, None, None], 0.0)
        attn = att.softmax(dim=-1)                            # exactly 0 on absent tokens
        ctx = torch.einsum("bln,bnd->bld", attn, tokens)      # [B, L, d]
        ctx = self.head_dropout(ctx)
        logits = torch.einsum("bld,ld->bl", ctx, self.label_weight.to(ctx.dtype)) + self.label_bias

        # -- MIL pooling of per-token logits (slice-level findings) --------------
        if self.mil_head is not None:
            mil_logits, token_logits = self._mil_pool_tokens(tokens, tok_present)
            self.last_mil_logits = mil_logits.detach()
            self.last_token_logits = token_logits.detach()
            logits = (1.0 - self.mil_alpha) * logits + self.mil_alpha * mil_logits.to(logits.dtype)
        else:
            self.last_mil_logits = None
            self.last_token_logits = None

        self._last_token_attention = attn.detach()
        self._last_token_present = tok_present
        # Differentiable (not detached): the training script adds
        # attn_entropy_weight * last_attention_entropy to its loss.
        self.last_attention_entropy = (
            self._normalised_entropy(attn, tok_present) if self.attn_entropy_weight > 0 else None)
        if self.token_level == "group":
            ga = attn.reshape(B, self.n_labels, S, G)
            self._last_group_attention = ga.detach()
            self._last_attention = ga.sum(dim=-1).detach()
        else:
            self._last_group_attention = None
            self._last_attention = attn.detach()

        if self.aux_head is not None:
            # Pre-mixer slot means: each slot has to be independently predictive.
            # Absent slots produce logits too -- the caller masks them via `mask`.
            aux = self.aux_head(self.head_dropout(slot_means))   # [B, S, n_labels]
            return logits, aux
        return logits
