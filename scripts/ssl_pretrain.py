#!/usr/bin/env python3
"""Continued self-supervised pretraining (MAE) of the SlotKnee-S DINOv2-S/14 encoder.

Masked Autoencoder (He et al., CVPR 2022) on the cached knee slices written by
``scripts/build_slot_cache.py``: every *present* (study, slot, group) triplet is
one T-channel image (T=3 -> RGB-like; T=1 is repeated to 3).  The encoder is the
very same timm encoder that ``src.slotknee.SlotKneeS``
builds (same ``img_size``, same ImageNet mean/std normalisation buffers), so the
exported ``encoder.safetensors`` is a drop-in for

    python3 scripts/train_slotknee.py --pretrained-path <out>/encoder.safetensors ...

Nothing changes downstream: same architecture, same inference cost.
Backbone: pass ``--backbone vit_small_patch14_reg4_dinov2.lvd142m`` (the adopted recipe; the Kaggle
kernel does).  The library default stays the plain ``vit_small_patch14_dinov2.lvd142m`` so
tests/test_ssl_pretrain.py keeps loading its export into a plain SlotKneeS; a plain export cannot
load into the reg4 model (4 register tokens) and vice versa.

Why: the encoder has only ever seen natural images (LVD-142M).  OAI-DINO
(ISMRM 2025) and MRI-CORE report that self-supervised adaptation on knee / MRI
slices improves downstream classification.  MAE is the cheapest such recipe:
75 % of the patch tokens are dropped before the encoder, so one SSL step costs
about a quarter of a supervised forward on the same image.

Usage
-----
    python3 scripts/ssl_pretrain.py --cache cache/slots_P224_full/slots_P224 \\
        --out models/ssl_mae [--epochs 30] [--bs 128] [--init-path weights/] \\
        [--limit-minutes 470] [--max-images 2000] [--workers 2]

Several ``--cache`` directories may be given (same P); ``--splits train test``
adds the test cache of each directory when it exists.

Outputs (``--out``)
-------------------
    encoder.safetensors   encoder state_dict ONLY, timm key names (strict load)
    mae_full.pt           encoder + decoder + optimiser + schedule (``--resume``)
    log.json              args, dataset stats, per-epoch loss / lr / images/s,
                          export verification

Runs fp16 autocast + GradScaler on CUDA, fp32 on MPS / CPU.  Resumes from
``<out>/mae_full.pt`` automatically (``--no-resume`` to start over).  Python 3.9.
"""
import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# scripts/ is not a package: make ``src`` importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import Block

from src.model import _resolve_weights_file, check_patch_compatible
from src.slotknee import DEFAULT_BACKBONE
from src.slots import SlotCache

try:  # cv2 is a project dependency; torch interpolation is the fallback.
    import cv2
    cv2.setNumThreads(0)   # one image per call; let the DataLoader workers parallelise
except ImportError:  # pragma: no cover
    cv2 = None

ENCODER_FILE = "encoder.safetensors"
FULL_FILE = "mae_full.pt"
LOG_FILE = "log.json"
T0 = time.time()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def to_rgb(img: np.ndarray) -> np.ndarray:
    """[T, P, P] uint8 -> [3, P, P] uint8 (T=1 repeated; T>3 evenly subsampled)."""
    T = img.shape[0]
    if T == 3:
        return img
    if T == 1:
        return np.repeat(img, 3, axis=0)
    idx = np.linspace(0, T - 1, 3).round().astype(int)
    return np.ascontiguousarray(img[idx])


def _resize_chw(img: np.ndarray, P: int) -> np.ndarray:
    """Bilinear resize of a [3, h, w] uint8 image to [3, P, P]."""
    if cv2 is not None:
        hwc = np.ascontiguousarray(img.transpose(1, 2, 0))
        hwc = cv2.resize(hwc, (P, P), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(hwc.transpose(2, 0, 1))
    t = torch.from_numpy(img)[None].float()
    t = F.interpolate(t, size=(P, P), mode="bilinear", align_corners=False)
    return t[0].round().clamp_(0, 255).to(torch.uint8).numpy()


class SlotImageDataset(Dataset):
    """Every present (study, slot, group) triplet of one or more slot caches.

    Caches are opened lazily inside each DataLoader worker (a memmap must never
    be pickled), rows are read zero-copy and only the [T, P, P] triplet (150 KB
    at P=224) is copied.  Absent slots (``mask == 0``) and unfinished rows
    (``done == 0``) are skipped at index-build time.

    Augmentation (train only): random resized crop, scale ``aug_scale`` of the
    area, aspect 3/4..4/3, resized back to P x P; mild brightness / contrast.
    NO flips of any kind -- laterality is a signal the cache builder preserved.
    """

    def __init__(self, cache_dirs: Sequence[str], splits: Sequence[str] = ("train",),
                 max_images: Optional[int] = None, seed: int = 0, aug: bool = True,
                 aug_scale: Tuple[float, float] = (0.6, 1.0), brightness: float = 0.1,
                 contrast: float = 0.1):
        self.cache_dirs = [str(d) for d in cache_dirs]
        self.splits = tuple(splits)
        self.aug = bool(aug)
        self.aug_scale = (float(aug_scale[0]), float(aug_scale[1]))
        self.brightness, self.contrast = float(brightness), float(contrast)
        entries: List[np.ndarray] = []
        self.P: Optional[int] = None
        self.sources: List[dict] = []
        self.n_absent = 0
        for ci, d in enumerate(self.cache_dirs):
            for si, split in enumerate(self.splits):
                try:
                    c = SlotCache(d, split)
                except FileNotFoundError:
                    continue
                if c.N == 0:
                    continue
                if self.P is None:
                    self.P = c.P
                elif c.P != self.P:
                    raise ValueError("all caches must share P: %s has P=%d, expected %d" % (d, c.P, self.P))
                present = np.asarray(c.mask).astype(bool) & c.done[:, None]    # [N, S]
                self.n_absent += int((~present & c.done[:, None]).sum()) * c.G
                rows, slots = np.nonzero(present)
                for g in range(c.G):
                    entries.append(np.stack([np.full_like(rows, ci), np.full_like(rows, si), rows, slots,
                                             np.full_like(rows, g)], axis=1))
                self.sources.append(dict(dir=d, split=split, N=int(c.N), n_done=int(c.done.sum()), S=c.S,
                                         G=c.G, T=c.T, P=c.P, n_images=int(present.sum()) * c.G))
                del c
        if not entries:
            raise FileNotFoundError("no cache rows found under %s (splits %s)" % (self.cache_dirs, self.splits))
        index = np.concatenate(entries).astype(np.int64)
        self.n_total = int(len(index))
        if max_images is not None and 0 < int(max_images) < len(index):
            keep = np.random.RandomState(seed).permutation(len(index))[: int(max_images)]
            index = index[np.sort(keep)]
        self.index = index
        self._caches: Dict[Tuple[int, int], SlotCache] = {}

    def __len__(self) -> int:
        return int(len(self.index))

    def _cache(self, ci: int, si: int) -> SlotCache:
        key = (ci, si)
        c = self._caches.get(key)
        if c is None:
            c = SlotCache(self.cache_dirs[ci], self.splits[si])
            self._caches[key] = c
        return c

    def raw(self, i: int) -> np.ndarray:
        """Un-augmented [3, P, P] uint8 copy of item ``i``."""
        ci, si, row, slot, g = (int(v) for v in self.index[i])
        x, _ = self._cache(ci, si)[row]
        return to_rgb(np.array(x[slot, g], dtype=np.uint8, copy=True))   # the memmap view is read-only

    def _augment(self, img: np.ndarray) -> np.ndarray:
        P = img.shape[-1]
        area = float(P * P)
        h = w = P
        y0 = x0 = 0
        for _ in range(10):                              # torchvision RandomResizedCrop sampling
            target = area * random.uniform(*self.aug_scale)
            aspect = math.exp(random.uniform(math.log(3.0 / 4.0), math.log(4.0 / 3.0)))
            cw = int(round(math.sqrt(target * aspect)))
            ch = int(round(math.sqrt(target / aspect)))
            if 0 < cw <= P and 0 < ch <= P:
                h, w = ch, cw
                y0, x0 = random.randint(0, P - h), random.randint(0, P - w)
                break
        crop = img[:, y0:y0 + h, x0:x0 + w]
        if (h, w) != (P, P):
            crop = _resize_chw(crop, P)
        out = crop.astype(np.float32)
        if self.contrast > 0 or self.brightness > 0:
            c = 1.0 + random.uniform(-self.contrast, self.contrast)
            b = random.uniform(-self.brightness, self.brightness) * 255.0
            m = float(out.mean())
            out = (out - m) * c + m + b
        return np.clip(out, 0, 255).astype(np.uint8)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = self.raw(i)
        if self.aug:
            img = self._augment(img)
        return torch.from_numpy(np.ascontiguousarray(img))      # uint8 [3, P, P]


class EpochSampler(Sampler):
    """Seeded per-epoch permutation that can skip the first ``skip`` items (resume)."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed = int(n), int(seed)
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch, self.skip = int(epoch), int(skip)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        return iter(perm[self.skip:])

    def __len__(self) -> int:
        return max(0, self.n - self.skip)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample random token masking (MAE).

    Returns ``(x_keep [B, L_keep, D], mask [B, L] float (1 = masked), ids_restore [B, L])``.
    ``L_keep = int(L * (1 - mask_ratio))`` so the masked fraction is exact.
    """
    B, L, D = x.shape
    len_keep = int(L * (1.0 - float(mask_ratio)))
    noise = torch.rand(B, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, L, device=x.device, dtype=x.dtype)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_keep, mask, ids_restore


def sincos_pos_embed_2d(dim: int, grid: Tuple[int, int], n_prefix: int) -> torch.Tensor:
    """[1, n_prefix + h*w, dim] fixed 2-D sin-cos table (MAE); prefix rows are zero."""
    h, w = grid
    gh, gw = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    half = dim // 2

    def _1d(pos: np.ndarray) -> np.ndarray:
        omega = 1.0 / (10000 ** (np.arange(half // 2, dtype=np.float64) / (half / 2.0)))
        out = pos.reshape(-1)[:, None] * omega[None, :]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    table = np.concatenate([_1d(gh), _1d(gw)], axis=1).astype(np.float32)     # [h*w, dim]
    table = np.concatenate([np.zeros((n_prefix, dim), np.float32), table], axis=0)
    return torch.from_numpy(table)[None]


class MAE(nn.Module):
    """MAE around a plain timm ``VisionTransformer`` (CLS / register tokens kept).

    Inputs are uint8 [B, 3, P, P]; they are scaled to 0..1 and normalised with the
    encoder's pretrained-cfg mean/std exactly as ``SlotKneeS._normalise`` does, so
    the fine-tuned weights see the same input distribution downstream.
    """

    def __init__(self, backbone: str = DEFAULT_BACKBONE, P: int = 224, init_path: Optional[str] = None,
                 pretrained: bool = True, mask_ratio: float = 0.75, dec_dim: int = 256,
                 dec_depth: int = 2, dec_heads: int = 8, norm_pix: bool = True,
                 pix_eps: float = 1e-6, drop_path: float = 0.0):
        super().__init__()
        P = int(P)
        check_patch_compatible(backbone, P)
        kwargs = dict(img_size=P, num_classes=0, global_pool="", in_chans=3)
        if drop_path and float(drop_path) > 0:
            kwargs["drop_path_rate"] = float(drop_path)
        if init_path:
            kwargs["pretrained_cfg_overlay"] = dict(file=_resolve_weights_file(str(init_path), backbone))
            pretrained = True
        try:
            self.encoder = timm.create_model(backbone, pretrained=bool(pretrained), **kwargs)
        except Exception as exc:  # noqa: BLE001
            if pretrained and not init_path:
                raise RuntimeError(
                    "Failed to create '%s' with pretrained=True: %s. No network is available at "
                    "runtime: keep the weights in the local HF cache or pass --init-path <file or "
                    "Kaggle dataset dir>." % (backbone, exc)) from exc
            raise
        enc = self.encoder
        for a in ("patch_embed", "cls_token", "pos_embed", "blocks", "norm", "norm_pre", "pos_drop"):
            if not hasattr(enc, a):
                raise TypeError("%s is not a plain timm VisionTransformer (missing %s)" % (backbone, a))
        self.backbone_name, self.P = str(backbone), P
        self.patch = int(enc.patch_embed.patch_size[0])
        self.grid = tuple(int(v) for v in enc.patch_embed.grid_size)
        self.L = self.grid[0] * self.grid[1]
        self.D = int(enc.embed_dim)
        self.n_prefix = int(getattr(enc, "num_prefix_tokens", 1))
        self.no_embed_class = bool(getattr(enc, "no_embed_class", False))
        self.mask_ratio = float(mask_ratio)
        self.norm_pix, self.pix_eps = bool(norm_pix), float(pix_eps)
        self.hparams = dict(backbone=self.backbone_name, P=P, mask_ratio=self.mask_ratio, dec_dim=int(dec_dim),
                            dec_depth=int(dec_depth), dec_heads=int(dec_heads), norm_pix=self.norm_pix,
                            pix_eps=self.pix_eps, drop_path=float(drop_path))

        cfg = getattr(enc, "pretrained_cfg", None) or {}
        mean, std = cfg.get("mean") or IMAGENET_DEFAULT_MEAN, cfg.get("std") or IMAGENET_DEFAULT_STD
        self.register_buffer("norm_mean", torch.tensor([float(v) for v in mean]).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor([float(v) for v in std]).view(1, 3, 1, 1))

        # -- light decoder: mask tokens + learned pos-embed (sin-cos initialised) ----
        self.decoder_embed = nn.Linear(self.D, int(dec_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(dec_dim)))
        self.decoder_pos_embed = nn.Parameter(sincos_pos_embed_2d(int(dec_dim), self.grid, self.n_prefix))
        self.decoder_blocks = nn.ModuleList(
            [Block(int(dec_dim), int(dec_heads), mlp_ratio=4.0, qkv_bias=True) for _ in range(int(dec_depth))])
        self.decoder_norm = nn.LayerNorm(int(dec_dim))
        self.decoder_pred = nn.Linear(int(dec_dim), self.patch * self.patch * 3)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in (self.decoder_embed, self.decoder_pred):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    # -- pixels <-> patches -------------------------------------------------
    def normalise(self, imgs: torch.Tensor) -> torch.Tensor:
        """uint8 0..255 -> float 0..1 -> (x - mean) / std (same as SlotKneeS)."""
        if imgs.dtype == torch.uint8:
            imgs = imgs.to(self.norm_mean.dtype).div_(255.0)
        return (imgs - self.norm_mean) / self.norm_std

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = imgs.shape
        p, h, w = self.patch, self.grid[0], self.grid[1]
        x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, h * w, p * p * C)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p, h, w = self.patch, self.grid[0], self.grid[1]
        x = x.reshape(B, h, w, p, p, 3).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, 3, h * p, w * p)

    # -- encoder / decoder -------------------------------------------------
    def forward_encoder(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.encoder
        B = imgs.shape[0]
        x = enc.patch_embed(imgs)                                   # [B, L, D]
        pos = enc.pos_embed
        if self.no_embed_class:                                     # reg4 variants: no prefix rows
            pos_prefix, pos_patch = None, pos
        else:
            pos_prefix, pos_patch = pos[:, : self.n_prefix], pos[:, self.n_prefix:]
        x = x + pos_patch
        x, mask, ids_restore = random_masking(x, self.mask_ratio)
        prefix = [enc.cls_token.expand(B, -1, -1)]
        if getattr(enc, "reg_token", None) is not None:
            prefix.append(enc.reg_token.expand(B, -1, -1))
        prefix_t = torch.cat(prefix, dim=1)
        if pos_prefix is not None:
            prefix_t = prefix_t + pos_prefix
        x = torch.cat([prefix_t, x], dim=1)
        x = enc.pos_drop(x)
        x = enc.norm_pre(x)
        x = enc.blocks(x)
        x = enc.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(latent)                              # [B, n_prefix + L_keep, d]
        B, n_keep = x.shape[0], x.shape[1] - self.n_prefix
        mask_tokens = self.mask_token.to(x.dtype).expand(B, self.L - n_keep, -1)
        x_ = torch.cat([x[:, self.n_prefix:], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, : self.n_prefix], x_], dim=1) + self.decoder_pos_embed.to(x.dtype)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, self.n_prefix:]                                 # [B, L, p*p*3]

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(loss, mask [B, L], pred [B, L, p*p*3])``; loss is an fp32 scalar."""
        imgs = self.normalise(imgs)
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        target = self.patchify(imgs).float()
        if self.norm_pix:                                           # per-patch normalised targets
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + self.pix_eps).sqrt()
        loss = ((pred.float() - target) ** 2).mean(dim=-1)         # [B, L]
        mask = mask.float()
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, mask, pred

    def encoder_state(self) -> Dict[str, torch.Tensor]:
        """CPU fp32 contiguous copy of the encoder state_dict (timm key names)."""
        return {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in self.encoder.state_dict().items()}


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _layer_id(name: str, depth: int) -> int:
    """BEiT/MAE layer ids: encoder stem 0, encoder.blocks.i -> i+1, encoder.norm and the decoder -> depth+1."""
    if name.startswith("encoder."):
        n = name[len("encoder."):]
        if n.startswith(("patch_embed", "cls_token", "pos_embed", "reg_token")):
            return 0
        if n.startswith("blocks."):
            return int(n.split(".")[1]) + 1
    return depth + 1


def param_groups(model: nn.Module, weight_decay: float, layer_decay: float = 1.0) -> List[dict]:
    """MAE convention (no weight decay on biases, norms, tokens, position tables) plus optional
    layer-wise lr decay: group lr = base_lr * layer_decay ** (depth + 1 - layer_id).
    28 groups for ViT-S (14 layer ids x decay/no-decay); keys are sorted so the optimizer
    state_dict layout is deterministic for --resume."""
    depth = len(model.encoder.blocks)
    groups: Dict[Tuple[int, bool], dict] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or n.endswith(".bias") or any(t in n for t in ("pos_embed", "cls_token", "reg_token", "mask_token"))
        lid = _layer_id(n, depth)
        scale = float(layer_decay) ** (depth + 1 - lid) if 0.0 < float(layer_decay) < 1.0 else 1.0
        g = groups.setdefault((lid, no_decay), dict(params=[], weight_decay=0.0 if no_decay else float(weight_decay), lr_scale=scale))
        g["params"].append(p)
    return [groups[k] for k in sorted(groups)]


def lr_at(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / float(warmup_steps)
    denom = max(1, total_steps - warmup_steps)
    t = min(1.0, max(0.0, (step - warmup_steps) / float(denom)))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def export_encoder(model: MAE, path: str, meta: Dict[str, str]) -> Dict[str, torch.Tensor]:
    from safetensors.torch import save_file
    sd = model.encoder_state()
    save_file(sd, path, metadata={k: str(v) for k, v in meta.items()})
    return sd


def verify_export(path: str, backbone: str, P: int, reference: Dict[str, torch.Tensor]) -> dict:
    """Build SlotKneeS(pretrained_path=path) and check its encoder equals ``reference``."""
    from src.slotknee import SlotKneeS
    model = SlotKneeS(P=P, T=3, backbone=backbone, pretrained_path=path, mixer_layers=0, drop_path=0.0)
    sd = model.encoder.state_dict()
    extra, missing = sorted(set(sd) - set(reference)), sorted(set(reference) - set(sd))
    if extra or missing:
        raise RuntimeError("encoder key mismatch after export: extra=%s missing=%s" % (extra[:5], missing[:5]))
    max_diff = max(float((sd[k].float() - reference[k].float()).abs().max()) for k in reference)
    if max_diff != 0.0:
        raise RuntimeError("exported encoder differs from the trained one (max |diff| = %g)" % max_diff)
    # The adopted cache is T=1 (g10t1): prove the downstream path too.  timm adapts in_chans
    # by summing the RGB patch-embedding kernel; every other tensor must be identical.
    m1 = SlotKneeS(P=P, T=1, backbone=backbone, pretrained_path=path, mixer_layers=0, drop_path=0.0)
    sd1 = m1.encoder.state_dict()
    if set(sd1) != set(reference):
        raise RuntimeError("T=1 encoder key mismatch after export")
    w1, w3 = sd1["patch_embed.proj.weight"].float(), reference["patch_embed.proj.weight"].float()
    if not torch.allclose(w1, w3.sum(1, keepdim=True), atol=1e-6):
        raise RuntimeError("T=1 patch_embed adaptation is not the RGB sum")
    max_diff_t1 = max(float((sd1[k].float() - reference[k].float()).abs().max())
                      for k in reference if k != "patch_embed.proj.weight")
    if max_diff_t1 != 0.0:
        raise RuntimeError("T=1 export differs (max |diff| = %g)" % max_diff_t1)
    return dict(ok=True, n_tensors=len(sd), max_abs_diff=max_diff,
                norm_mean=[float(v) for v in model.norm_mean.flatten()],
                norm_std=[float(v) for v in model.norm_std.flatten()],
                t1_ok=True, t1_norm_mean=[float(v) for v in m1.norm_mean.flatten()])


def save_full(path: str, model: MAE, opt: torch.optim.Optimizer, scaler, epoch: int, batch: int, step: int,
              log: dict, args: argparse.Namespace) -> None:
    state = dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=int(epoch), batch=int(batch),
                 step=int(step), log=log, args=vars(args), hparams=model.hparams,
                 scaler=scaler.state_dict() if scaler is not None else None)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def write_log(path: str, log: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(log, fh, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", nargs="+", default=["cache/slots_P224_full/slots_P224"],
                    help="one or more slot-cache dirs (same P)")
    ap.add_argument("--splits", nargs="+", default=["train"], help="cache splits to use, e.g. train test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--init-path", default=None,
                    help="initial DINOv2 weights file/dir (Kaggle runs offline); default: HF cache")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=None, help="default 128 on cuda, 32 elsewhere")
    ap.add_argument("--blr", type=float, default=1e-4,
                    help="base lr; actual lr = blr * bs / 256. S5: 5e-5..1e-4 for continued pretraining "
                         "from DINOv2, NOT the from-scratch MAE 1.5e-4")
    ap.add_argument("--layer-decay", type=float, default=0.8,
                    help="layer-wise lr decay: encoder layer i gets lr * decay**(depth+1-i); decoder full lr (1.0 = off)")
    ap.add_argument("--snapshot-epochs", type=int, nargs="*", default=[10, 20, 30],
                    help="also write <out>/encoder_epNN.safetensors at the end of these epochs (never overwritten)")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument("--clip-grad", type=float, default=0.0, help="max grad norm (0 = off, as in MAE)")
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--decoder-dim", type=int, default=256)
    ap.add_argument("--decoder-depth", type=int, default=2)
    ap.add_argument("--decoder-heads", type=int, default=8)
    ap.add_argument("--no-norm-pix", action="store_true", help="raw-pixel targets instead of per-patch normalised")
    ap.add_argument("--pix-eps", type=float, default=1e-6, help="variance epsilon of the per-patch normalisation")
    ap.add_argument("--drop-path", type=float, default=0.0)
    ap.add_argument("--aug-scale", type=float, nargs=2, default=(0.6, 1.0))
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--fp32", action="store_true", help="disable fp16 autocast on cuda")
    ap.add_argument("--limit-minutes", type=float, default=None, help="stop (and export) after M minutes")
    ap.add_argument("--max-images", type=int, default=None, help="random subset of the cache (smoke runs)")
    ap.add_argument("--ckpt-minutes", type=float, default=20.0, help="periodic mae_full.pt / encoder export")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing <out>/mae_full.pt")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    use_fp16 = device.type == "cuda" and not args.fp32
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    bs = int(args.bs) if args.bs else (128 if device.type == "cuda" else 32)
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_fp16 else contextlib.nullcontext
    print("device: %s  amp: %s  bs: %d" % (device, "fp16" if use_fp16 else "fp32", bs), flush=True)

    # -- data ---------------------------------------------------------------
    ds = SlotImageDataset(args.cache, splits=args.splits, max_images=args.max_images, seed=args.seed,
                          aug=not args.no_aug, aug_scale=tuple(args.aug_scale))
    P = int(ds.P)
    print("dataset: %d images (of %d present; %d absent slot-groups skipped) P=%d from %d cache(s)"
          % (len(ds), ds.n_total, ds.n_absent, P, len(ds.sources)), flush=True)
    sampler = EpochSampler(len(ds), args.seed)
    loader = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=int(args.workers),
                        pin_memory=device.type == "cuda", drop_last=False,
                        persistent_workers=int(args.workers) > 0, prefetch_factor=4 if args.workers > 0 else None)
    steps_per_epoch = int(math.ceil(len(ds) / float(bs)))
    total_steps = int(args.epochs) * steps_per_epoch
    warmup_steps = int(round(float(args.warmup_epochs) * steps_per_epoch))
    base_lr = float(args.blr) * bs / 256.0

    # -- model / optimiser --------------------------------------------------
    model = MAE(backbone=args.backbone, P=P, init_path=args.init_path, mask_ratio=args.mask_ratio,
                dec_dim=args.decoder_dim, dec_depth=args.decoder_depth, dec_heads=args.decoder_heads,
                norm_pix=not args.no_norm_pix, pix_eps=args.pix_eps, drop_path=args.drop_path).to(device)
    opt = torch.optim.AdamW(param_groups(model, args.wd, args.layer_decay), lr=base_lr, betas=(0.9, 0.95))
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.parameters()) - n_enc
    print("model: %s  encoder %.1fM  decoder %.1fM  lr %.2e  steps/epoch %d  warmup %d  total %d"
          % (args.backbone, n_enc / 1e6, n_dec / 1e6, base_lr, steps_per_epoch, warmup_steps, total_steps), flush=True)

    log: dict = dict(args=vars(args), device=str(device), amp="fp16" if use_fp16 else "fp32", bs=bs, P=P,
                     backbone=args.backbone, n_images=len(ds), n_present=ds.n_total, n_absent_skipped=ds.n_absent,
                     sources=ds.sources, steps_per_epoch=steps_per_epoch, base_lr=base_lr,
                     layer_decay=float(args.layer_decay), blr=float(args.blr), mask_ratio=float(args.mask_ratio),
                     n_params_encoder=n_enc, n_params_decoder=n_dec, hparams=model.hparams,
                     norm_mean=[float(v) for v in model.norm_mean.flatten()],
                     norm_std=[float(v) for v in model.norm_std.flatten()], epochs=[], resumed=False)
    full_path = os.path.join(args.out, FULL_FILE)
    enc_path = os.path.join(args.out, ENCODER_FILE)
    log_path = os.path.join(args.out, LOG_FILE)

    start_epoch, start_batch, step = 0, 0, 0
    if not args.no_resume and os.path.isfile(full_path):
        ck = torch.load(full_path, map_location="cpu")
        if ck.get("hparams") != model.hparams:
            raise RuntimeError("%s was trained with %s; current hparams %s. Use --no-resume or a new --out."
                               % (full_path, ck.get("hparams"), model.hparams))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch, start_batch, step = int(ck["epoch"]), int(ck["batch"]), int(ck["step"])
        log["epochs"] = list(ck["log"].get("epochs", []))
        log["resumed"] = True
        print("resumed from %s: epoch %d batch %d step %d" % (full_path, start_epoch, start_batch, step), flush=True)

    def minutes() -> float:
        return (time.time() - T0) / 60.0

    def out_of_time() -> bool:
        return args.limit_minutes is not None and minutes() >= float(args.limit_minutes)

    def checkpoint(epoch: int, batch: int) -> None:
        save_full(full_path, model, opt, scaler, epoch, batch, step, log, args)
        export_encoder(model, enc_path, dict(format="pt", backbone=args.backbone, img_size=P, ssl="mae",
                                             epoch=epoch, step=step, source="scripts/ssl_pretrain.py"))
        write_log(log_path, log)

    # -- train --------------------------------------------------------------
    stopped_early = False
    last_ckpt = time.time()
    epoch = start_epoch
    try:
        for epoch in range(start_epoch, int(args.epochs)):
            if out_of_time():
                stopped_early = True
                break
            model.train()
            sampler.set_epoch(epoch, start_batch * bs)
            t_ep = time.time()
            loss_sum = mask_sum = 0.0
            n_img = n_steps = n_nonfinite = 0
            t_steady, n_steady = None, 0
            batch = start_batch
            for imgs in loader:
                lr = lr_at(step, base_lr, warmup_steps, total_steps)
                for g in opt.param_groups:
                    g["lr"] = lr * float(g.get("lr_scale", 1.0))
                imgs = imgs.to(device, non_blocking=True)
                with amp_ctx():
                    loss, mask, _ = model(imgs)
                if not torch.isfinite(loss):
                    n_nonfinite += 1
                    if n_nonfinite > 20:
                        raise RuntimeError("loss was non-finite for 20 consecutive steps (epoch %d)" % epoch)
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    batch += 1
                    continue
                n_nonfinite = 0
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if args.clip_grad > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    opt.step()
                step += 1
                batch += 1
                n_steps += 1
                n_img += int(imgs.shape[0])
                loss_sum += float(loss.detach()) * int(imgs.shape[0])
                mask_sum += float(mask.mean()) * int(imgs.shape[0])
                if n_steps == 5:                                   # steady-state throughput window
                    t_steady, n_steady = time.time(), n_img
                if args.log_every > 0 and step % args.log_every == 0:
                    print("ep %d step %d/%d loss %.4f lr %.2e %.1f img/s %.1f min"
                          % (epoch, step, total_steps, loss_sum / max(1, n_img), lr,
                             n_img / max(1e-6, time.time() - t_ep), minutes()), flush=True)
                if args.ckpt_minutes > 0 and (time.time() - last_ckpt) / 60.0 >= float(args.ckpt_minutes):
                    checkpoint(epoch, batch)
                    last_ckpt = time.time()
                if out_of_time():
                    stopped_early = True
                    break
            elapsed = time.time() - t_ep
            steady = ((n_img - n_steady) / max(1e-6, time.time() - t_steady)) if t_steady else n_img / max(1e-6, elapsed)
            entry = dict(epoch=epoch, loss=loss_sum / max(1, n_img), mask_frac=mask_sum / max(1, n_img),
                         lr=lr_at(step - 1, base_lr, warmup_steps, total_steps), n_images=n_img, steps=n_steps,
                         seconds=round(elapsed, 1), img_s=round(n_img / max(1e-6, elapsed), 2),
                         img_s_steady=round(steady, 2), partial=bool(stopped_early), elapsed_min=round(minutes(), 2))
            log["epochs"].append(entry)
            print("epoch %d: loss %.4f mask %.3f %d images %.1fs (%.1f img/s, steady %.1f) total %.1f min%s"
                  % (epoch, entry["loss"], entry["mask_frac"], n_img, elapsed, entry["img_s"], steady, minutes(),
                     "  [time limit]" if stopped_early else ""), flush=True)
            start_batch = 0
            if stopped_early:
                checkpoint(epoch, batch)
                break
            checkpoint(epoch + 1, 0)
            if (epoch + 1) in {int(e) for e in (args.snapshot_epochs or [])}:
                snap = os.path.join(args.out, "encoder_ep%02d.safetensors" % (epoch + 1))
                export_encoder(model, snap, dict(format="pt", backbone=args.backbone, img_size=P, ssl="mae",
                                                 epoch=epoch + 1, step=step, mask_ratio=args.mask_ratio,
                                                 blr=args.blr, layer_decay=args.layer_decay,
                                                 source="scripts/ssl_pretrain.py"))
                log.setdefault("snapshots", []).append(dict(epoch=epoch + 1, step=step, path=os.path.basename(snap)))
                write_log(log_path, log)
                print("snapshot %s" % snap, flush=True)
            last_ckpt = time.time()
    except KeyboardInterrupt:
        print("interrupted: saving checkpoint", flush=True)
        checkpoint(epoch, 0)
        raise

    # -- export + verify ----------------------------------------------------
    final_epoch = log["epochs"][-1]["epoch"] + 1 if log["epochs"] else start_epoch
    save_full(full_path, model, opt, scaler, final_epoch, 0, step, log, args)
    reference = export_encoder(model, enc_path, dict(format="pt", backbone=args.backbone, img_size=P, ssl="mae",
                                                     epoch=final_epoch, step=step, source="scripts/ssl_pretrain.py"))
    log["verify"] = verify_export(enc_path, args.backbone, P, reference)
    log["stopped_early"] = bool(stopped_early)
    log["epochs_done"] = final_epoch
    log["elapsed_min"] = round(minutes(), 2)
    write_log(log_path, log)
    print("wrote %s (%d tensors, verified strict load into SlotKneeS) and %s; %.1f min total"
          % (enc_path, log["verify"]["n_tensors"], full_path, minutes()), flush=True)
    return log


if __name__ == "__main__":
    main()
