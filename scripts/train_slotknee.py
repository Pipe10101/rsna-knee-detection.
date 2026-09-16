#!/usr/bin/env python3
"""SlotKnee-S training orchestrator.

Implements the training loop specified in docs/slotknee_spec.md §6.
"""

import argparse
import json
import math
import os
import random
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

from src.config import Config
from src.folds import FoldIntegrityError, assign_grouped_folds, build_groups
from src.llm_labels import (build_targets, cell_weights, regex_fallback,
                            ID_COL, LABELS, Y_COLS, W_COLS, y_col, w_col, load_llm_labels)
from src.slotknee import SlotKneeS, DEFAULT_BACKBONE
from src.slots import SlotCache, N_SLOTS, is_medial_slot
from src.train import MaskedBCEWithLogitsLoss, MaskedFocalLoss, AsymmetricLoss, MaskedAUCMLoss

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _is_yw_csv(path):
    """True if `path` is a ready-made y_/w_ schema CSV (e.g. scripts/pseudo_fill.py
    output).  Those go to build_targets directly: load_llm_labels would match the
    y_ columns as probabilities and silently drop the baked-in w_ weights."""
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return all(c in cols for c in Y_COLS) and all(c in cols for c in W_COLS)


CANONICAL_FOLD_LABELS = os.path.join("labels_external", "stevenleehans",
                                     "llm_labels_v4_blend.csv")


def fold_reference_frame(targets_df, groups, args, gold_df):
    """Frame whose y_ columns drive the fold split — INVARIANT to --labels.

    assign_grouped_folds stratifies on the y_ columns, so deriving folds from the
    training labels moved ~66% of studies across folds the moment a different labels
    CSV was passed (relabel arm, 2026-08-24: uid-ovl 0.34) and broke every
    arm-vs-baseline comparison.  Folds therefore always come from the canonical v4
    labels when they can be found; training targets are untouched.  Runs whose
    --labels IS the canonical file skip this (bit-identical to the historical split).
    """
    cands = [os.path.join(args.data_dir, CANONICAL_FOLD_LABELS)]
    for lab in args.labels[:1]:
        d = os.path.dirname(os.path.abspath(lab))
        cands += [os.path.join(d, "stevenleehans", "llm_labels_v4_blend.csv"),
                  os.path.join(d, "llm_labels_v4_blend.csv"),
                  os.path.join(os.path.dirname(d), "stevenleehans", "llm_labels_v4_blend.csv")]
    canon = next((c for c in cands if os.path.exists(c)), None)
    if canon is None or [os.path.abspath(p) for p in args.labels] == [os.path.abspath(canon)]:
        return targets_df
    try:
        ref = build_targets(load_llm_labels([canon]), gold_df=gold_df,
                            gold_weight=args.gold_weight)
        ref = ref.set_index(ref[ID_COL].astype(str))
        uids = targets_df[ID_COL].astype(str)
        if not uids.isin(ref.index).all():
            print("fold reference: canonical labels miss some studies; folds fall back to --labels")
            return targets_df
        fold_df = ref.loc[uids].reset_index(drop=True)
        fold_df["group"] = list(groups)
        print(f"fold split derived from canonical labels: {canon}")
        return fold_df
    except Exception as e:
        print(f"fold reference labels unusable ({e}); folds fall back to --labels")
        return targets_df


def _cached_activation_meta(path):
    """Sidecar meta written by scripts/cache_activations.py, or None (legacy file)."""
    meta_path = str(path) + ".json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def open_cached_activations(path, cache):
    """np.memmap for a scripts/cache_activations.py file.

    The shape comes from the sidecar meta (<path>.json).  Without one (legacy
    files) it falls back to the old hardcoded [N, S, G, 257, 384] fp16 layout,
    which is only correct for P=224 ViT-S without register tokens (257 = 16x16
    patches + CLS; P=252 gives 325 tokens, reg4 backbones add 4).
    """
    N = len(cache.uids)
    S, G = int(getattr(cache, "S", 6)), int(cache.G)
    meta = _cached_activation_meta(path)
    if meta is not None:
        shape = tuple(int(v) for v in meta["shape"])
        if shape[:3] != (N, S, G):
            raise SystemExit(
                f"cached activations {path}: shape {shape[:3]} does not match the "
                f"slot cache ({(N, S, G)}); rebuild with scripts/cache_activations.py")
    else:
        shape = (N, S, G, 257, 384)
    return np.memmap(path, dtype=np.float16, mode="r", shape=shape)


def check_cached_activations(path, cache, trainable_blocks, backbone):
    """Refuse a cached-activation file whose recorded settings disagree with this run.

    The cached tokens are the output of the first (depth - trainable_blocks)
    encoder blocks: training with a different --trainable-blocks would silently
    skip blocks in the middle of the encoder, and a different backbone/P/T is a
    different feature space entirely.
    """
    if not os.path.exists(path):
        raise SystemExit(f"--use-cached-activations {path}: file not found "
                         "(build it with scripts/cache_activations.py)")
    meta = _cached_activation_meta(path)
    if meta is None:
        print(f"WARNING: {path} has no sidecar meta ({path}.json); assuming the legacy "
              "[N, S, G, 257, 384] fp16 layout built with trainable_blocks=4 "
              "(only valid for P=224 ViT-S without register tokens).")
        return
    problems = []
    if meta.get("trainable_blocks") is not None and \
            int(meta["trainable_blocks"]) != int(trainable_blocks):
        problems.append(f"trainable_blocks {meta['trainable_blocks']} != --trainable-blocks "
                        f"{trainable_blocks} (would silently skip encoder blocks)")
    if meta.get("backbone") and str(meta["backbone"]) != str(backbone):
        problems.append(f"backbone {meta['backbone']!r} != --backbone {backbone!r}")
    for k, want in (("P", int(cache.P)), ("T", int(cache.T))):
        if meta.get(k) is not None and int(meta[k]) != want:
            problems.append(f"{k} {meta[k]} != cache {k} {want}")
    if problems:
        raise SystemExit("cached activations " + path + " do not match this run:\n  - "
                         + "\n  - ".join(problems))


def mixup_batch(x, mask, y, w, distill_y, lam, idx):
    """Blend a batch with its idx-permutation (mixup), made safe (2026-08-24 audit):

    * uint8 pixels are scaled to [0, 1] BEFORE blending: SlotKneeS._normalise
      divides by 255 only for integer inputs, so a float blend of raw uint8
      pixels would reach the encoder 255x too large.
    * slot masks are OR-ed, not AND-ed: the blended pixels contain every slot
      that was present in EITHER study.
    * distill targets blend NaN-safely: a study without teacher probabilities
      (NaN row) keeps its partner's targets instead of poisoning both mixed
      rows; NaN survives only where both sides are NaN (the loss masks those).
    """
    if not x.is_floating_point():           # raw uint8 pixels (cached activations are float)
        x = x.float().div_(255.0)
    x = lam * x + (1 - lam) * x[idx]
    y = lam * y + (1 - lam) * y[idx]
    w = lam * w + (1 - lam) * w[idx]
    mask = mask | mask[idx]
    if distill_y is not None:
        other = distill_y[idx]
        blend = lam * distill_y + (1 - lam) * other
        blend = torch.where(torch.isnan(distill_y) & ~torch.isnan(other), other, blend)
        blend = torch.where(torch.isnan(other) & ~torch.isnan(distill_y), distill_y, blend)
        distill_y = blend
    return x, mask, y, w, distill_y

class SlotKneeDataset(Dataset):
    def __init__(self, cache: SlotCache, targets_df: pd.DataFrame, is_train: bool,
                 slot_dropout: float = 0.1, cached_activations_path: str = None, group_subsample: int = 0, cutout: bool = False, rot_aug: bool = False, anchor_bag: int = 0, slot_keep=None, aug_bc: bool = True, aug_shift: bool = True, aug_gdrop: bool = True, flip_swap: float = 0.0):
        self.cache = cache
        # LATERALITY MIRROR augmentation (2026-09-07).  The cache normalises every study to a LEFT knee
        # (src.slots.apply_laterality: COR/AX columns flipped, SAG anchor order reversed for right knees).
        # Applying that same transform again yields a valid RIGHT-knee-like study whose medial and
        # lateral structures have swapped sides, so the four medial/lateral labels swap too.  A plain
        # horizontal flip WITHOUT the swap would be label-corrupting; this one is exact.  Competition
        # evidence: +0.01 as augmentation, up to +0.027 as TTA (RSNA aneurysm 2nd); doubled the data for
        # RSNA lumbar 3rd.  Off by default (0.0) so no earlier measurement changes.
        self.flip_swap = float(flip_swap)
        names = list(getattr(cache, "slot_names", []))
        self.slot_names_kept = ([names[i] for i in slot_keep] if (slot_keep is not None and names) else names)
        self.lat_perm = list(range(len(LABELS)))
        for a_, b_ in (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA")):
            if a_ in LABELS and b_ in LABELS:
                ia, ib = LABELS.index(a_), LABELS.index(b_); self.lat_perm[ia], self.lat_perm[ib] = ib, ia
        self.group_subsample = int(group_subsample)   # train-time: keep K random anchors per slot, zero the rest
        self.anchor_bag = int(anchor_bag)             # train-time: SELECT K random anchors (see __getitem__)
        self.slot_keep = None if slot_keep is None else list(slot_keep)   # per-plane specialists
        # These three augmentations were HARDCODED and always on, never gated (audit 2026-08-25).
        self.aug_bc, self.aug_shift, self.aug_gdrop = bool(aug_bc), bool(aug_shift), bool(aug_gdrop)
        self.cutout = bool(cutout)
        self.rot_aug = bool(rot_aug)
        self.cached_activations = None
        self.cached_activations_path = cached_activations_path
        if cached_activations_path and os.path.exists(cached_activations_path):
            self.cached_activations = open_cached_activations(cached_activations_path, cache)


        self.slot_dropout = float(slot_dropout)
        self.uids = []
        self.indices = []
        self.y = []
        self.w = []
        self.is_gold = []
        
        # Build mapping from UID to cache row
        uid_to_row = {uid: i for i, uid in enumerate(cache.uids) if cache.done[i]}
        
        for r in targets_df.to_dict('records'):
            uid = str(r["StudyInstanceUID"])
            if uid in uid_to_row:
                self.uids.append(uid)
                self.indices.append(uid_to_row[uid])
                self.y.append([r[y_col(l)] for l in LABELS])
                self.w.append([r[w_col(l)] for l in LABELS])
                self.is_gold.append(bool(r["is_gold"]))
                
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.w = torch.tensor(self.w, dtype=torch.float32)
        
        self.distill_y = None
        if "distill_targets" in targets_df.columns:
            # Assume distill_targets is a list/array of 12 floats for each row
            self.distill_y = torch.tensor(targets_df["distill_targets"].tolist(), dtype=torch.float32)

        self.is_train = is_train

    def __len__(self):
        return len(self.uids)

    def _mirror(self, x_t):
        """x_t [S, G, T, P, P] -> laterality-mirrored copy (see __init__).  A medial-centred zoom
        slot (``is_medial_slot``) is zeroed instead of flipped: its mirror is not lateral anatomy,
        so under the swapped labels it must be absent (``_mirror_mask`` clears its mask)."""
        out = x_t.clone()
        for s_i, name in enumerate(self.slot_names_kept[: x_t.shape[0]]):
            plane = name.split("_")[0]
            if is_medial_slot(name):
                out[s_i] = 0
            elif plane in ("COR", "AX"):
                out[s_i] = x_t[s_i].flip(-1)
            elif plane == "SAG":
                out[s_i] = x_t[s_i].flip(0)
            else:
                raise ValueError(f"cannot infer plane from slot name {name!r}")
        return out

    def _mirror_mask(self, mask_t):
        """Slot mask for the mirrored study: medial-centred zoom slots become absent."""
        out = mask_t.clone()
        for s_i, name in enumerate(self.slot_names_kept[: mask_t.shape[0]]):
            if is_medial_slot(name):
                out[s_i] = 0
        return out

    def __getitem__(self, idx):
        cache_row = self.indices[idx]
        x, mask = self.cache[cache_row]
        
        # x is uint8 [6, G, T, P, P] view, mask is uint8 [6]
        # Always copy to tensor so we don't modify memmap and it's writable
        x_t = torch.from_numpy(np.array(x, copy=True))
        mask_t = torch.from_numpy(np.array(mask, copy=True))
        if self.slot_keep is not None:
            # PER-PLANE SPECIALIST.  The per-finding attention has measurably collapsed to a
            # uniform mean and every attempt to force it open failed, so it cannot learn to
            # look at the coronal plane for MCL.  Rather than keep fighting that, impose the
            # specialisation architecturally: train one model on the coronal sequences only,
            # another on the sagittal, and let scripts/select_ensemble.py combine them.  Two
            # plane specialists are far more decorrelated than two backbones on identical
            # input, which is what an ensemble actually needs.
            x_t = x_t[self.slot_keep]
            mask_t = mask_t[self.slot_keep]
        
        # If we have cached activations, we return the cached vectors instead of raw images
        if self.cached_activations is not None:
            # We don't augment the cached activations (since they are deep features),
            # but we still return the mask and allow MixUp/Curriculum to function normally.
            x_t = torch.from_numpy(np.array(self.cached_activations[cache_row], copy=True)).float()
        else:
            # Augmentation on uint8 tensors, cheap and label-safe (Spec §6)
            if self.is_train:
                # Random group dropout (drop one of G with p=0.2).  NOTE: with --anchor-bag
                # this now fires ON TOP of bagging, so a "bag of 6" is sometimes really 5.
                G = x_t.shape[1]
                if self.aug_gdrop and G > 1 and random.random() < 0.2:
                    g_drop = random.randint(0, G - 1)
                    x_t[:, g_drop] = 0
                # RANDOM ANCHOR BAGGING.  Our anchors are FIXED, so without this the model
                # sees the identical 60 images every epoch.  The strongest public solutions
                # instead sample a fresh bag of slices per study per epoch, which buys three
                # things at once: augmentation, exposure to EVERY cached slice across epochs
                # rather than a fixed subset, and an ensemble-like average at inference.
                # Unlike --group-subsample (which ZEROES the unkept anchors, so the encoder
                # still burns compute on blank images and the head sees dead tokens), this
                # SELECTS K anchors, shrinking the tensor.
                if 0 < self.anchor_bag < G:
                    keep = torch.randperm(G)[: self.anchor_bag].sort().values
                    x_t = x_t[:, keep]
                    G = int(self.anchor_bag)
                if 0 < self.group_subsample < G:
                    for s_i in range(x_t.shape[0]):
                        if mask_t[s_i] == 0:
                            continue
                        keep = set(torch.randperm(G)[: self.group_subsample].tolist())
                        for g in range(G):
                            if g not in keep:
                                x_t[s_i, g] = 0

                # Slot-dropout: zero a whole present slot (pixels AND mask) to match
                # the test-time missing-slot distribution; keep >= 1 slot present.
                if self.slot_dropout > 0:
                    for s in range(x_t.shape[0]):
                        if mask_t[s] != 0 and int(mask_t.sum()) > 1 and random.random() < self.slot_dropout:
                            x_t[s] = 0
                            mask_t[s] = 0

                # Per-slot brightness/contrast ±10%, small translation ±4px
                for s in range(x_t.shape[0]):
                    if mask_t[s] == 0: continue
                    # Brightness/contrast jitter.  SUSPECT ON MRI: absolute intensity is
                    # diagnostic here (fluid is bright on fluid-sensitive sequences), and a
                    # +-25.5 offset is 10% of the dynamic range — exactly the signal that
                    # separates effusion / synovitis / bone oedema.  Never measured.
                    if self.aug_bc and random.random() < 0.5:
                        alpha = random.uniform(0.9, 1.1)
                        beta = random.uniform(-25.5, 25.5)
                        # Convert to float for math, then back to uint8
                        xf = x_t[s].float() * alpha + beta
                        x_t[s] = xf.clamp(0, 255).byte()
                    # Translation +-4 px.  Never measured, but benign in principle.
                    if self.aug_shift and random.random() < 0.5:
                        dy = random.randint(-4, 4)
                        dx = random.randint(-4, 4)
                        if dy != 0 or dx != 0:
                            x_s = x_t[s] # [G, T, P, P]
                            pad = (max(0, -dx), max(0, dx), max(0, -dy), max(0, dy))
                            x_s = F.pad(x_s, pad)
                            y0 = max(0, dy)
                            x0 = max(0, dx)
                            x_t[s] = x_s[..., y0:y0+x_t.shape[-2], x0:x0+x_t.shape[-1]]

                    # Cutout / Random Erasing (16x16 block) — opt-in: unconditional
                    # augmentation changes silently break arm comparability
                    if self.cutout and random.random() < 0.5:
                        P = x_t.shape[-1]
                        cy = random.randint(0, max(0, P - 16))
                        cx = random.randint(0, max(0, P - 16))
                        x_t[s, ..., cy:cy+16, cx:cx+16] = 0
                        
                    # Small Random Rotation (±5 degrees) — opt-in, same reason
                    if self.rot_aug and random.random() < 0.5:
                        angle = random.uniform(-5.0, 5.0)
                        # Reshape to [C, H, W] for TF.rotate compatibility
                        G, T, H, W = x_t[s].shape
                        x_flat = x_t[s].view(G * T, H, W)
                        x_rot = TF.rotate(x_flat, angle)
                        x_t[s] = x_rot.view(G, T, H, W)

        y_i, w_i = self.y[idx], self.w[idx]
        d_i = self.distill_y[idx] if self.distill_y is not None else None
        if self.is_train and self.cached_activations is None and self.flip_swap > 0 and random.random() < self.flip_swap:
            x_t = self._mirror(x_t)
            mask_t = self._mirror_mask(mask_t)
            y_i, w_i = y_i[self.lat_perm], w_i[self.lat_perm]
            if d_i is not None:
                d_i = d_i[self.lat_perm]
        if d_i is not None:
            return x_t, mask_t, y_i, w_i, self.is_gold[idx], d_i
        return x_t, mask_t, y_i, w_i, self.is_gold[idx]

    # DataLoader workers on macOS use spawn, which would pickle (i.e. copy) the
    # memmapped cache; ship (dir, split) instead and reopen it per worker.  The
    # cached-activation memmap would pickle as a FULL in-memory array (np.memmap
    # is an ndarray subclass), so it is dropped and reopened from its path too.
    def __getstate__(self):
        state = dict(self.__dict__)
        state["cache"] = (self.cache.dir, self.cache.split)
        state["cached_activations"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if isinstance(self.cache, tuple):
            self.cache = SlotCache(self.cache[0], split=self.cache[1])
        p = getattr(self, "cached_activations_path", None)
        if self.cached_activations is None and p and os.path.exists(p):
            self.cached_activations = open_cached_activations(p, self.cache)


class ModelEMA:
    """Per-step exponential moving average, fp32 shadow on the model's device."""

    def __init__(self, model, decay=0.998):
        self.decay = float(decay)
        self.n = 0
        self.shadow = {
            k: v.detach().clone().float() if torch.is_floating_point(v) else v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        # Ramp-in: a flat 0.998 would keep the shadow at the init for hundreds of
        # steps; short folds would then evaluate a nearly untrained head.
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if torch.is_floating_point(s):
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                s.copy_(v)

    @torch.no_grad()
    def copy_to(self, model):
        sd = model.state_dict()
        model.load_state_dict({k: v.to(sd[k].dtype) for k, v in self.shadow.items()})


def peak_rss_gb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024 ** 3 if sys.platform == "darwin" else ru / 1024 ** 2  # bytes vs KB


def device_mem_gb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    if device.type == "mps":
        # torch.mps has no peak counter; driver-allocated is the usable proxy.
        return torch.mps.driver_allocated_memory() / 1024 ** 3
    return 0.0


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_amp_device(amp_arg):
    amp_arg = (amp_arg or "auto").lower()
    if amp_arg == "cpu":
        return torch.device("cpu"), "fp32", None
    if torch.cuda.is_available():
        if amp_arg == "fp32":
            return torch.device("cuda"), "fp32", None
        return torch.device("cuda"), "fp16", torch.amp.GradScaler("cuda")
    if torch.backends.mps.is_available():
        if amp_arg == "fp16":
            print("--amp fp16: MPS training stays fp32 (fp16 is inference-only here).")
        return torch.device("mps"), "fp32", None
    return torch.device("cpu"), "fp32", None


@torch.no_grad()
def _sam_ascend(params, rho):
    """Move params to the local worst case (first-order SAM); returns the deltas to undo."""
    norm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in params if p.grad is not None]), 2)
    if not torch.isfinite(norm) or float(norm) == 0.0:
        return None
    scale = rho / (norm + 1e-12)
    eps = []
    for p in params:
        if p.grad is None:
            eps.append(None)
            continue
        e = p.grad.detach() * scale.to(p)
        p.add_(e)
        eps.append(e)
    return eps


@torch.no_grad()
def _sam_descend(params, eps):
    for p, e in zip(params, eps):
        if e is not None:
            p.sub_(e)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, amp_dtype,
                ema=None, mixup_alpha=0.0, loss_fn=None, aux_weight=0.2, is_cached=False, distill_weight=None, rank_loss_weight=0.0,
                sam_rho=0.0, attn_entropy_weight=0.0, aux_slotid_weight=0.0):
    model.train()
    loss_meter = AverageMeter()

    use_amp = (amp_dtype == "fp16")
    if sam_rho > 0 and scaler is not None:
        raise SystemExit("--sam-rho needs a scaler-free AMP mode (--amp bf16 or cpu); "
                         "fp16's GradScaler cannot unscale twice in one step.")
    sam_params = [p for p in model.parameters() if p.requires_grad] if sam_rho > 0 else []

    for batch in loader:
        x, mask, y, w = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        distill_y = batch[5].to(device) if len(batch) == 6 else None

        if mixup_alpha > 0.0:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(x.size(0))
            x, mask, y, w, distill_y = mixup_batch(x, mask, y, w, distill_y, lam, idx)

        optimizer.zero_grad()

        def _compute_loss():
          with torch.autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
            out = model(x, mask, is_cached=is_cached)
            logits = out[0] if isinstance(out, tuple) else out
            aux = out[1] if isinstance(out, tuple) else None
            loss = loss_fn(logits, y, weight=w)

            if rank_loss_weight > 0:
                # Pairwise AUC surrogate: within the batch, per label, every (pos, neg) pair
                # (binarised at 0.5 over cells with weight > 0) contributes softplus(s_n - s_p).
                r_terms = []
                for li in range(logits.shape[1]):
                    mcol = w[:, li] > 0
                    yb = y[:, li] > 0.5
                    pos = logits[mcol & yb, li]
                    neg = logits[mcol & ~yb, li]
                    if pos.numel() and neg.numel():
                        r_terms.append(F.softplus(neg.unsqueeze(0) - pos.unsqueeze(1)).mean())
                if r_terms:
                    loss = loss + rank_loss_weight * torch.stack(r_terms).mean()
            if distill_y is not None:
                # Multi-label distillation: per-finding soft-target BCE against the teacher's
                # probabilities. We blend the base loss and distillation loss per label.
                tm = ~torch.isnan(distill_y)
                if tm.any():
                    dwl = list(distill_weight) if distill_weight is not None else [0.5] * 12
                    if len(set(round(float(v), 6) for v in dwl)) == 1:
                        # UNIFORM weight: keep the single joint blend.  The per-label form
                        # below re-weights the base loss to one-vote-per-label, which is a
                        # second difference from the no-distill baseline — a distill arm must
                        # differ from its baseline by distillation ALONE or its gate is
                        # unreadable.  Identical to the pre-per-label behaviour.
                        dwu = float(dwl[0])
                        dl = F.binary_cross_entropy_with_logits(
                            logits[tm], distill_y[tm].clamp(1e-4, 1 - 1e-4))
                        loss = (1.0 - dwu) * loss + dwu * dl
                    else:
                        # PER-LABEL weights (the point of the feature): the model out-teaches
                        # its labels on Fracture/Effusion/Contusion but trails them badly on
                        # MCL/lateral meniscus, so one uniform weight helps half the findings
                        # and corrupts the other half.  Note a label with no supervised cell
                        # in the batch contributes dw_j * teacher only — that IS the intent
                        # (the teacher fills the radiologist's silence), not a bug.
                        dw = torch.tensor(dwl, device=device, dtype=torch.float32)
                        losses = []
                        for j in range(12):
                            l_base_j = loss_fn(logits[:, j:j+1], y[:, j:j+1], weight=w[:, j:j+1])
                            tm_j = tm[:, j]
                            if tm_j.any() and dw[j] > 0:
                                l_dist_j = F.binary_cross_entropy_with_logits(
                                    logits[tm_j, j], distill_y[tm_j, j].clamp(1e-4, 1 - 1e-4))
                                l_j = (1.0 - dw[j]) * l_base_j + dw[j] * l_dist_j
                            else:
                                l_j = l_base_j
                            losses.append(l_j)
                        loss = torch.stack(losses).mean()

            if aux is not None and aux_weight > 0:
                aux_terms = []
                for s in range(aux.shape[1]):
                    rows = mask[:, s] > 0
                    if rows.any():
                        aux_terms.append(loss_fn(aux[rows, s, :], y[rows], weight=w[rows]))
                if aux_terms:
                    loss = loss + aux_weight * torch.stack(aux_terms).mean()

            # Attention-collapse penalty: `last_attention_entropy` is the mean NORMALISED
            # entropy of the per-finding attention over present tokens (1.0 = uniform =
            # mean pooling). Adding it to the loss pushes the head to actually choose
            # sequences/anchors instead of averaging them.
            ent = getattr(model, "last_attention_entropy", None)
            if ent is not None and attn_entropy_weight > 0:
                loss = loss + attn_entropy_weight * ent
            # Auxiliary free target: slot identity per image (see SlotKneeS.aux_slotid).  CE over
            # PRESENT slots only; absent slots carry zero features and would teach nothing.
            sl = getattr(model, "last_slotid_logits", None)
            if sl is not None and aux_slotid_weight > 0:
                Bq, Sq, Gq, _ = sl.shape
                tgt = torch.arange(Sq, device=sl.device)[None, :, None].expand(Bq, Sq, Gq)
                pm = (mask > 0)[:, :, None].expand(Bq, Sq, Gq)
                if pm.any():
                    ce = F.cross_entropy(sl[pm].float(), tgt[pm], reduction="mean")
                    loss = loss + aux_slotid_weight * ce
            return loss

        loss = _compute_loss()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if sam_rho > 0:
                # SAM: step to the local worst case, re-measure the gradient there, step back,
                # then let the optimizer descend with the sharpness-aware gradient.
                eps = _sam_ascend(sam_params, sam_rho)
                if eps is not None:
                    optimizer.zero_grad(set_to_none=True)
                    _compute_loss().backward()
                    _sam_descend(sam_params, eps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        if ema: ema.update(model)
        loss_meter.update(loss.item())
        
    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, device, amp_dtype, loss_fn=None, aux_weight=0.2, is_cached=False):
    model.eval()
    loss_meter = AverageMeter()
    
    all_logits = []
    all_y = []
    all_w = []
    all_mask = []
    all_is_gold = []
    all_aux = []        # per-slot aux-head logits [B, S, 12] when the model returns (logits, aux)

    use_amp = (amp_dtype == "fp16")

    for batch in loader:
        x, mask, y, w, is_gold = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4]
        
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16 if use_amp else torch.float32):
            out = model(x, mask, is_cached=is_cached)
            logits = out[0] if isinstance(out, tuple) else out
            aux = out[1] if (isinstance(out, tuple) and len(out) > 1 and out[1] is not None) else None
            loss = loss_fn(logits, y, weight=w)

        loss_meter.update(loss.item())
        all_logits.append(logits.float().cpu())
        if aux is not None:
            all_aux.append(aux.float().cpu())
        all_y.append(y.cpu())
        all_w.append(w.cpu())
        all_mask.append(mask.cpu())
        all_is_gold.append(is_gold)

    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    w = torch.cat(all_w)
    slot_mask = torch.cat(all_mask)
    is_gold = torch.cat(all_is_gold)
    aux_all = torch.cat(all_aux) if all_aux else None

    def calc_auc(mask):
        if not mask.any(): return float("nan")
        m_y, m_logits, m_w = y[mask], logits[mask], w[mask]
        aucs = []
        for i in range(12):
            active = m_w[:, i] > 0
            if active.sum() > 0:
                y_true = (m_y[active, i].numpy() > 0.5).astype(int)
                y_pred = m_logits[active, i].numpy()
                if len(np.unique(y_true)) > 1:
                    aucs.append(roc_auc_score(y_true, y_pred))
        return float(np.mean(aucs)) if aucs else float("nan")

    # Threshold Tuning (F1 / Sensitivity maximization)
    best_thresholds = np.full(12, 0.5)
    best_f1s = np.zeros(12)
    for i in range(12):
        active = w[:, i] > 0
        if active.sum() > 0:
            y_true = (y[active, i].numpy() > 0.5).astype(int)
            p_pred = torch.sigmoid(logits[active, i]).numpy()
            if len(np.unique(y_true)) > 1:
                thresholds = np.linspace(0.1, 0.9, 50)
                f1s = []
                for th in thresholds:
                    y_pred = (p_pred >= th).astype(int)
                    tp = (y_pred * y_true).sum()
                    fp = (y_pred * (1 - y_true)).sum()
                    fn = ((1 - y_pred) * y_true).sum()
                    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                    f1s.append(f1)
                best_idx = np.argmax(f1s)
                best_thresholds[i] = thresholds[best_idx]
                best_f1s[i] = f1s[best_idx]

    return {
        "val_loss": loss_meter.avg,
        "val_auc_derived": calc_auc(~is_gold),
        "val_auc_gold": calc_auc(is_gold),
        "best_thresholds": best_thresholds,
        "best_f1s": best_f1s,
        "logits": logits.numpy(),
        "y": y.numpy(),
        "w": w.numpy(),
        "mask": slot_mask.numpy(),
        "aux": aux_all.numpy() if aux_all is not None else None,   # [N, S, 12] or None
    }


def _fold_fallback_banner(reason):
    """Announce that grouped folds were NOT used. Never let this happen quietly."""
    print("!" * 78,
          f"\nWARNING: grouped fold assignment failed: {reason}\n"
          "WARNING: using ROUND-ROBIN folds. CV is inflated ~0.053 and is NOT comparable\n"
          "WARNING: to any grouped-fold run. Do not gate on this number.\n" + "!" * 78,
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", nargs="+", default=[])
    parser.add_argument("--weights-from", default=None)
    parser.add_argument("--gold-weight", type=float, default=8.0)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-backbone", type=float, default=5e-5)
    parser.add_argument("--trainable-blocks", type=int, default=4)
    parser.add_argument("--out", default="models/slotknee")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-seed", type=int, default=42,
                        help="fold-split seed, decoupled from --seed so seed arms stay fold-matched")
    parser.add_argument("--amp", default="auto", choices=["auto", "cpu", "fp32", "fp16"])
    parser.add_argument("--max-studies", type=int)
    parser.add_argument("--limit-minutes", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--slot-dropout", type=float, default=0.1)
    parser.add_argument("--cutout", action="store_true", help="16x16 random erasing per slot (opt-in)")
    parser.add_argument("--rot-aug", action="store_true", help="±5° random rotation per slot (opt-in)")
    parser.add_argument("--no-aug-bc", action="store_true",
                        help="disable the always-on brightness/contrast jitter. On MRI, absolute "
                             "intensity is diagnostic, so a +-25.5 offset may be destroying the "
                             "very signal that marks effusion/synovitis/oedema. Never gated until now.")
    parser.add_argument("--no-aug-shift", action="store_true", help="disable the always-on +-4px translation")
    parser.add_argument("--no-aug-gdrop", action="store_true",
                        help="disable the always-on p=0.2 anchor dropout (redundant with --anchor-bag)")
    parser.add_argument("--slots", default=None,
                        help="comma-separated sequence slots to train on, e.g. COR_FS,COR_T1 "
                             "(default: all). Builds a PER-PLANE SPECIALIST — cheaper to train "
                             "and run, and far more decorrelated from other specialists than "
                             "two backbones on the same input.")
    parser.add_argument("--allow-ungrouped-folds", action="store_true",
                        help="fall back to round-robin folds if grouped assignment fails "
                             "(inflates CV ~0.053; off by default so the failure is loud)")
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="LoRA rank on every block's attention projections + the patch embedding; base weights frozen, "
                             "LayerNorm affines trainable; 0 = off (last-N-blocks fine-tuning as before)")
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-mlp", action="store_true", help="also adapt the MLP fc1/fc2 (0.59M vs 0.22M params at r=8)")
    parser.add_argument("--no-lora-patch", action="store_true", help="do not adapt the patch embedding")
    parser.add_argument("--slot-tok-embed", action="store_true",
                        help="zero-initialised per-slot embedding added to every token at the first trainable block (MM-DINOv2)")
    parser.add_argument("--aux-slotid", type=float, default=0.0,
                        help="weight of the auxiliary slot-identity (plane x sequence) head on per-image features; 0 = off")
    parser.add_argument("--flip-swap", type=float, default=0.0,
                        help="p of the exact laterality-mirror augmentation (COR/AX flip + SAG anchor reversal + "
                             "medial/lateral label swap). 0 = off (default, keeps prior arms comparable)")
    parser.add_argument("--anchor-bag", type=int, default=0,
                        help="TRAIN-time random slice bagging: select K of the cache's G anchors "
                             "fresh each epoch (0 = off, use all G every epoch as before). Our "
                             "fixed anchors show the model the same images every epoch; the "
                             "strongest public solutions resample a bag per study per epoch. "
                             "Validation and inference still use all G.")
    parser.add_argument("--group-subsample", type=int, default=0,
                        help="train-time: keep this many random anchors per slot (0 = all); infer uses all")
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument("--loss", default="bce", choices=["bce", "focal", "asl", "aucm"])
    parser.add_argument("--aucm-epochs", type=int, default=0,
                        help="train the LAST N epochs with the AUC-margin loss (0 = off). Optimises the "
                             "competition metric directly after BCE has done the feature learning; "
                             "--loss aucm uses it for every epoch instead.")
    parser.add_argument("--aucm-margin", type=float, default=1.0)
    parser.add_argument("--sam-rho", type=float, default=0.0,
                        help="Sharpness-Aware Minimisation radius over the TRAINABLE params (0 = off). "
                             "Costs a second forward+backward per step. Needs a scaler-free AMP mode "
                             "(--amp bf16/cpu); fp16's GradScaler cannot unscale twice per step.")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=1.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--curriculum-epochs", type=int, default=0)
    parser.add_argument("--distill-targets", default=None, help="CSV of teacher probabilities (StudyInstanceUID + 12 labels)")
    parser.add_argument("--distill-weight", type=str, default="0.5", help="Single float or comma-separated list of 12 floats")
    parser.add_argument("--full-fit", action="store_true", help="train on ALL rows (no holdout); the last EMA epoch is saved as fold_<k>_best.pt")
    parser.add_argument("--token-pooling", action="store_true")
    parser.add_argument("--use-cached-activations", type=str, default=None)
    parser.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m",
                        help="timm ViT name, e.g. vit_small_patch14_reg4_dinov2.lvd142m")
    parser.add_argument("--mixer-layers", type=int, default=1)
    parser.add_argument("--attn-tau-init", type=float, default=1.0)
    parser.add_argument("--pool", default="cls_mean",
                        choices=["cls_mean", "cls", "mean", "cls_mean_max", "cls_mean_topk"],
                        help="how one image's patch tokens become one vector. The default "
                             "cls_mean AVERAGES all 256 patches, attenuating a 3-8 patch lesion "
                             "~30-80x before the head sees it; cls_mean_max/topk keep a peak "
                             "statistic too (+0.1M head params, ZERO extra encoder time).")
    parser.add_argument("--pool-topk", type=int, default=8,
                        help="k for --pool cls_mean_topk (mean of the k most-activated patches)")
    parser.add_argument("--attn-cosine", action="store_true",
                        help="scaled COSINE per-finding attention (Swin-V2 style): L2-normalise "
                             "query and token so the logits are bounded and only --attn-tau-init "
                             "can sharpen them. A raw dot product lets the net flatten attention "
                             "for free by shrinking norms, which is measurably what it does.")
    parser.add_argument("--attn-entropy-weight", type=float, default=0.0,
                        help="penalise UNIFORM per-finding attention (0 = off). Diagnosed "
                             "2026-08-24: the trained head collapses to mean pooling "
                             "(entropy 0.9994, learned tau 0.96-1.00 from a 1.0 init), so the "
                             "per-finding query attention contributes nothing. Entropy is "
                             "stationary at exactly-uniform attention, so pair this with "
                             "--attn-tau-init > 1 rather than using it alone.")
    parser.add_argument("--rank-loss-weight", type=float, default=0.0,
                        help="pairwise AUC-surrogate loss weight (metric is a rank statistic; BCE alone optimizes calibration)")
    parser.add_argument("--seq-mix", default="none", choices=["none", "gru", "conv"],
                        help="ordered-sequence modeling along the anchor axis (tears are cross-slice continuous)")
    parser.add_argument("--mil-pool", default="none", choices=["none", "lse", "max"],
                        help="instance-level pooling branch over (slot, anchor) tokens (MIL), mixed 50/50 with the attention head")
    parser.add_argument("--init-from-dir", default=None,
                        help="warm-start: load fold_{k}_best.pt from this dir before training (same hparams required)")
    parser.add_argument("--pretrained-path", default=None,
                        help="local DINOv2 weights file/dir (Kaggle runs offline: timm cannot download)")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    
    device, amp_dtype, scaler = get_amp_device(args.amp)
    cache = SlotCache(args.cache, split="train")
    if args.use_cached_activations:
        check_cached_activations(args.use_cached_activations, cache,
                                 args.trainable_blocks, args.backbone)

    gold_df = pd.read_csv(os.path.join(args.data_dir, "train_gold.csv"))
    if args.labels and len(args.labels) == 1 and not args.weights_from \
            and _is_yw_csv(args.labels[0]):
        # Ready-made y_/w_ schema (e.g. scripts/pseudo_fill.py output): hand it to
        # build_targets directly.  load_llm_labels would match the y_ columns as
        # probabilities and silently DROP the baked-in w_ weights.
        llm_df = pd.read_csv(args.labels[0])
    elif args.labels:
        llm_df = load_llm_labels(args.labels)
        if args.weights_from:
            wsrc = load_llm_labels([args.weights_from]).set_index(ID_COL)
            p = llm_df[LABELS].to_numpy(dtype=float)
            pw = wsrc.reindex(llm_df[ID_COL].astype(str))[LABELS].to_numpy(dtype=float)
            w = cell_weights(np.where(np.isfinite(pw), pw, 0.5))
            w = np.where(np.isfinite(pw), w, cell_weights(p))
            yw = pd.DataFrame({ID_COL: llm_df[ID_COL].astype(str).to_numpy()})
            for j, lab in enumerate(LABELS): yw[y_col(lab)] = p[:, j]
            for j, lab in enumerate(LABELS): yw[w_col(lab)] = w[:, j]
            llm_df = yw
    else:
        llm_df = regex_fallback(os.path.join(args.data_dir, "train.csv"), args.data_dir)

    targets_df = build_targets(llm_df, gold_df=gold_df, gold_weight=args.gold_weight)
    cached = {str(u) for u, d in zip(cache.uids, cache.done) if d}
    targets_df = targets_df[targets_df[ID_COL].astype(str).isin(cached)].reset_index(drop=True)

    if args.max_studies: targets_df = targets_df.head(args.max_studies)
    
    if args.distill_targets:
        distill_df = pd.read_csv(args.distill_targets).set_index(ID_COL)
        targets_df["distill_targets"] = targets_df[ID_COL].apply(lambda uid: distill_df.loc[uid][LABELS].astype(float).tolist() if uid in distill_df.index else [float("nan")]*12)
        dw = [float(x.strip()) for x in args.distill_weight.split(",")]
        if len(dw) == 1:
            dw = dw * 12
        elif len(dw) != 12:
            raise ValueError("--distill-weight must be a single float or exactly 12 comma-separated floats")
        args.distill_weight = dw
    else:
        args.distill_weight = None

    groups = build_groups(targets_df, args.data_dir)
    targets_df["group"] = groups
    n_folds = min(5, len(targets_df))
    fold_df = fold_reference_frame(targets_df, groups, args, gold_df)
    try:
        targets_df["fold"] = assign_grouped_folds(fold_df, n_folds=n_folds, seed=args.fold_seed)
    except FoldIntegrityError:
        # The grouping itself is broken (no group column, corrupt kept folds).  Substituting
        # round-robin folds here lets scanner identity straddle the split and inflates CV by
        # ~0.053 -- a number that looks BETTER than the truth with nothing in the log to say
        # so.  That silent substitution is the bug this guard replaced; never absorb it.
        if not args.allow_ungrouped_folds:
            raise
        _fold_fallback_banner("integrity failure overridden by --allow-ungrouped-folds")
        targets_df["fold"] = np.arange(len(targets_df)) % n_folds
    except Exception as exc:
        # Capacity: too few studies or groups for n_folds stratified splits (unit tests,
        # --max-studies smoke runs).  Round-robin is then the only option -- but it is still
        # announced loudly, because the original bug here was the SILENCE, not the fallback.
        _fold_fallback_banner(f"{type(exc).__name__}: {exc}")
        targets_df["fold"] = np.arange(len(targets_df)) % n_folds
    
    if "fold" in gold_df.columns:
        gold_folds = gold_df.set_index("StudyInstanceUID")["fold"].to_dict()
        targets_df["fold"] = targets_df.apply(lambda r: gold_folds.get(r.StudyInstanceUID, r.fold), axis=1)

    t_start = time.time()
    for fold in args.folds:
        print(f"\n=== Fold {fold} ===")
        log_path = os.path.join(args.out, f"fold_{fold}_log.json")
        # Resume: skip a fold whose OOF exists and whose log ends with the trailing
        # {"completed": true} marker after >= args.epochs epochs (the final write).  The
        # driver's .arm_done is bash-side and is lost when the driver is killed mid-arm;
        # without this check a relaunch re-trains finished folds from scratch.
        if not args.full_fit and os.path.isfile(os.path.join(args.out, f"oof_fold_{fold}.npz")) \
                and os.path.isfile(log_path):
            try:
                with open(log_path) as f:
                    hist = json.load(f)
                n_ep = sum(1 for e in hist if "epoch" in e)
                if hist and hist[-1].get("completed") and n_ep >= args.epochs:
                    print(f"fold {fold}: already completed ({n_ep} epochs, oof present); "
                          f"skipping — delete {log_path} to retrain")
                    continue
            except Exception:
                pass
        train_df, valid_df = targets_df[targets_df["fold"] != fold], targets_df[targets_df["fold"] == fold]
        if args.full_fit:
            train_df = targets_df
            valid_df = targets_df.sample(n=min(200, len(targets_df)), random_state=args.seed)   # logging only (in-train)
        
        shuffle = True
        slot_keep = None
        if args.slots:
            names = list(getattr(cache, "slot_names", []))
            want = [w.strip() for w in args.slots.split(",") if w.strip()]
            missing = [w for w in want if w not in names]
            if missing:
                raise SystemExit(f"--slots {missing} not in this cache's slots {names}")
            slot_keep = [names.index(w) for w in want]
            print(f"per-plane specialist: training on slots {want} (indices {slot_keep})", flush=True)
        train_ds = SlotKneeDataset(cache, train_df, is_train=True, slot_dropout=args.slot_dropout, cached_activations_path=args.use_cached_activations, group_subsample=args.group_subsample, cutout=args.cutout, rot_aug=args.rot_aug, anchor_bag=args.anchor_bag, slot_keep=slot_keep, aug_bc=not args.no_aug_bc, aug_shift=not args.no_aug_shift, aug_gdrop=not args.no_aug_gdrop, flip_swap=args.flip_swap)
        valid_ds = SlotKneeDataset(cache, valid_df, is_train=False, cached_activations_path=args.use_cached_activations, slot_keep=slot_keep)
        train_dl = DataLoader(train_ds, batch_size=args.bs, sampler=None, shuffle=shuffle, drop_last=True, num_workers=args.workers)
        valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers)

        # Provenance in the run log: an SSL-continued encoder (ablate ssl_* arms) must be visibly different
        # from the timm default, and the path is also kept in the checkpoint under args.pretrained_path.
        print(f"[train] encoder init: {args.pretrained_path or 'timm pretrained ' + str(args.backbone)}", flush=True)
        model = SlotKneeS(P=cache.P, T=cache.T, n_slots=(len(slot_keep) if slot_keep else getattr(cache, 'S', 6)), max_groups=max(8, cache.G), trainable_blocks=args.trainable_blocks, pretrained_path=args.pretrained_path, backbone=args.backbone, mixer_layers=args.mixer_layers, **({'attn_tau_init': args.attn_tau_init} if args.attn_tau_init != 1.0 else {}), **({'attn_entropy_weight': args.attn_entropy_weight} if args.attn_entropy_weight > 0 else {}), **({'attn_cosine': True} if args.attn_cosine else {}), **({'pool': args.pool} if args.pool != 'cls_mean' else {}), **({'pool_topk': args.pool_topk} if args.pool_topk != 8 else {}), **({'mil_pool': args.mil_pool} if args.mil_pool != 'none' else {}), **({'seq_mix': args.seq_mix} if args.seq_mix != 'none' else {}), **({'aux_slotid': True} if args.aux_slotid > 0 else {}), **({'lora_rank': args.lora_rank, 'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout, 'lora_mlp': args.lora_mlp, 'lora_patch': not args.no_lora_patch} if args.lora_rank > 0 else {}), **({'slot_tok_embed': True} if args.slot_tok_embed else {}),
                          grad_checkpointing=args.bs >= 8 and not args.no_grad_checkpointing,
                          aux_slot_logits=args.aux_weight > 0, token_pooling=args.token_pooling).to(device)
        
        if args.init_from_dir:
            _ck = torch.load(os.path.join(args.init_from_dir, f'fold_{fold}_best.pt'), map_location='cpu')
            for _k in ('P', 'T', 'n_slots'):
                assert _ck['hparams'].get(_k) == model.hparams.get(_k), f'init-from hparam mismatch: {_k}'
            model.load_state_dict(_ck['state_dict'], strict=False)   # new heads (tau/mil/seq) may not exist in the old ckpt
            print(f'warm-started fold {fold} from {args.init_from_dir}', flush=True)
            del _ck
        if getattr(model, 'mil_head', None) is not None:   # start from the attention logits, not half-weight noise
            torch.nn.init.zeros_(model.mil_head.weight); torch.nn.init.zeros_(model.mil_head.bias)
        optimizer = torch.optim.AdamW(model.param_groups(args.lr_backbone, args.lr_head), weight_decay=0.05)
        scheduler = get_lr_scheduler(optimizer, len(train_dl), len(train_dl) * args.epochs)
        ema = ModelEMA(model, decay=0.998)
        
        if args.loss == "asl":
            loss_fn = AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos, clip=args.asl_clip)
        elif args.loss == "focal":
            loss_fn = MaskedFocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
        elif args.loss == "aucm":
            loss_fn = MaskedAUCMLoss(margin=args.aucm_margin)
        else:
            loss_fn = MaskedBCEWithLogitsLoss()
        # Final-phase AUC maximisation: BCE learns the features, the last N epochs optimise the
        # metric itself.  val_loss stays on the base loss so the epoch curve remains comparable.
        aucm_fn = MaskedAUCMLoss(margin=args.aucm_margin) if args.aucm_epochs > 0 else None

        best_val_auc, log_history = -1.0, []
        for epoch in range(1, args.epochs + 1):
            t_epoch_start = time.time()
            if args.limit_minutes and (time.time() - t_start) / 60.0 > args.limit_minutes: break
            
            if args.loss == "asl":
                # Curriculum ASL: Epoch 1-2 = 0.0 (BCE equivalent), then linearly scale to gamma_neg
                if epoch <= 2:
                    loss_fn.gamma_neg = 0.0
                else:
                    progress = min(1.0, (epoch - 2) / 8.0)
                    loss_fn.gamma_neg = progress * args.asl_gamma_neg
            
            curr_dl = DataLoader(torch.utils.data.Subset(train_ds, [i for i in range(len(train_ds)) if train_ds.cache[train_ds.indices[i]][1].sum() == getattr(train_ds.cache, 'S', 6)]), 
                                 batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers) if epoch <= args.curriculum_epochs else train_dl
            
            epoch_loss_fn = (aucm_fn if (aucm_fn is not None and epoch > args.epochs - args.aucm_epochs)
                             else loss_fn)
            train_loss = train_epoch(model, curr_dl, optimizer, scheduler, scaler, device, amp_dtype, ema, args.mixup_alpha, epoch_loss_fn, args.aux_weight, bool(args.use_cached_activations), args.distill_weight, args.rank_loss_weight, args.sam_rho, args.attn_entropy_weight, args.aux_slotid)
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            val_metrics = validate(model, valid_dl, device, amp_dtype, loss_fn, args.aux_weight, bool(args.use_cached_activations))
            
            # Attention health, measured on the EMA weights while they are loaded.  The
            # per-finding query attention CAN collapse to uniform (== mean pooling), which
            # silently throws away the whole point of the head — it did exactly that for the
            # entire campaign before anyone looked (entropy 0.9994, tau stuck at its init).
            # One extra forward on one batch per epoch is cheap insurance against repeating it.
            attn_entropy = float("nan")
            try:
                _vb = next(iter(valid_dl))
                _rep = model.attention_report(_vb[0], _vb[1],
                                              is_cached=bool(args.use_cached_activations))
                attn_entropy = float(_rep["entropy"])
            except Exception as _e:
                print(f"attention_report unavailable this epoch: {type(_e).__name__}", flush=True)

            val_auc = val_metrics["val_auc_derived"]
            is_best = val_auc > best_val_auc
            if args.full_fit:
                is_best = True   # no holdout: keep the last EMA epoch
            if (is_best and math.isfinite(val_auc)) or epoch == 1:
                if is_best and math.isfinite(val_auc):
                    best_val_auc = val_auc
                torch.save({
                    "state_dict": model.state_dict(),   # EMA weights
                    "hparams": model.hparams,
                    "args": vars(args),
                    "slot_layout": {"G": int(cache.G), "T": int(cache.T), "slot_names": ([list(getattr(cache, "slot_names", []))[i] for i in slot_keep] if slot_keep else list(getattr(cache, "slot_names", []))), "zoom_center": getattr(cache, "zoom_center", "image"), "zoom_spec": getattr(cache, "zoom_spec", None),
                                    "zoom_mm": getattr(cache, "zoom_mm", None),
                                    "zoom_slots": list(getattr(cache, "zoom_slots", ()) or ()),
                                    # anchor selection + crop size of the TRAINING cache: inference must
                                    # rebuild the same slices (a t35 "central" cache trained model read
                                    # with the default 0.15 trim is a silent train/test mismatch).
                                    "trim_frac": float(getattr(cache, "trim_frac", 0.15)),
                                    "crop_mm": float(getattr(cache, "crop_mm", 140.0))},
                }, os.path.join(args.out, f"fold_{fold}_best.pt"))
                
                oof = dict(uids=valid_ds.uids, logits=val_metrics["logits"], y=val_metrics["y"],
                           w=val_metrics["w"], mask=val_metrics["mask"],
                           best_thresholds=val_metrics["best_thresholds"])
                if val_metrics.get("aux") is not None:
                    # per-slot aux-head logits [N, S, 12] + the kept slot names (same order as
                    # the checkpoint's slot_layout) so scripts/aux_as_run.py can pick a slot.
                    _all_names = list(getattr(cache, "slot_names", []))
                    _kept = [_all_names[i] for i in slot_keep] if slot_keep else _all_names
                    oof["aux"] = val_metrics["aux"]
                    oof["slot_names"] = np.array([str(n) for n in _kept])
                np.savez_compressed(os.path.join(args.out, f"oof_fold_{fold}.npz"), **oof)

            model.load_state_dict(backup)
            del backup

            epoch_mins = (time.time() - t_epoch_start) / 60.0
            log_str = (f"Epoch {epoch}/{args.epochs} [{epoch_mins:.1f}m]: "
                       f"train_loss={train_loss:.4f} val_loss={val_metrics['val_loss']:.4f} "
                       f"val_auc(derived)={val_auc:.4f} val_auc(gold)={val_metrics['val_auc_gold']:.4f} "
                       f"peak_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024 / 1024:.2f}GB")
            if device.type == "cuda":
                log_str += f" dev_mem={torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f}GB"
            elif device.type == "mps":
                log_str += f" dev_mem={torch.mps.current_allocated_memory() / 1024 / 1024 / 1024:.2f}GB"
            if math.isfinite(attn_entropy):
                log_str += f" attn_ent={attn_entropy:.3f}"
                if attn_entropy > 0.98:
                    log_str += " [ATTENTION COLLAPSED -> mean pooling]"
            if is_best: log_str += " [BEST]"
            print(log_str)
            
            if epoch == args.epochs:
                mean_f1 = val_metrics['best_f1s'].mean()
                print(f"Optimal Thresholds for Sensitivity: {np.round(val_metrics['best_thresholds'], 3)}")
                print(f"Optimal Macro-F1 (Sensitivity proxy) via tuned thresholds: {mean_f1:.4f}")

            log_history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["val_loss"]),
                "val_auc_derived": float(val_auc),
                "val_auc_gold": float(val_metrics["val_auc_gold"]),
                "peak_rss_gb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024 / 1024),
                "time_mins": epoch_mins,
                "attn_entropy": attn_entropy,
            })
            with open(log_path, "w") as f:
                json.dump(log_history, f, indent=2)

            log_history.append({"completed": True, "best_val_auc": best_val_auc})
        with open(log_path, "w") as f:
            json.dump(log_history, f, indent=2)

if __name__ == "__main__":
    main()
