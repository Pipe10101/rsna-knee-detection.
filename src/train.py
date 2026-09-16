"""Training entry point for the RSNA Knee Abnormality Detection pipeline.

Design notes that are easy to get wrong here and cost real AUC:

1. LABEL INTEGRITY.  data_subset/train.csv has 4407 rows, of which only 58 carry
   labels; the remaining 4349 are NaN across all 12 target columns.  RSNADataset
   (src/kaggle_data.py) does
   ``labels = np.nan_to_num(labels, nan=0.0)`` in ``__getitem__``, so those rows
   do NOT blow up as NaN loss -- they arrive at the criterion as *confident
   all-negative* studies.  3758 of them additionally have no DICOMs on disk, so
   they are all-black images labelled "no pathology".  Feeding them in poisons
   98.7% of every batch.  We therefore select the labelled rows in the dataframe
   *before* the Dataset ever sees them (``labelled_only``), and additionally use
   NaN-masked losses so that any future path which does deliver NaN targets
   (pseudo-labels, partially-annotated CSVs) degrades gracefully instead of
   producing a NaN gradient.

2. RESOLUTION-AWARE CACHE.  The DICOM cache key in kaggle_data.py is
   ``f"{study_id}_{in_channels}.npy"`` -- it does not include image_size.  A
   224px phase followed by a 512px phase would silently reuse the 224px cache
   and upsample it, so the "progressive resolution" would be fake.  We pass a
   per-resolution ``cache_dir`` to sidestep that without touching that file.

3. AMP DTYPE.  bfloat16 has no hardware support below sm_80.  The Kaggle target
   is T4 (sm_75) and P100 (sm_60), so a hard-coded bfloat16 autocast is the wrong
   default there.  ``amp_dtype=auto`` picks bf16 on Ampere+ and fp16 + a live
   GradScaler otherwise.
"""

import os
import json
import math
import time
import random
import re
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from timm.utils import ModelEmaV2

from src.config import Config
from src.model import RSNA25DModel, build_model
from src.kaggle_data import (RSNADataset, KNEE_TARGETS, precache_dataset,
                             build_lateral_swap)


# ══════════════════════════════════════════════════════════════════════════
# Losses
# ══════════════════════════════════════════════════════════════════════════

def _finite_mask(targets, mask=None):
    """Elements that carry a usable supervision signal."""
    m = torch.isfinite(targets)
    if mask is not None:
        m = m & mask.bool()
    return m


def _elementwise_weights(m, weight):
    """Non-negative per-element loss weights, broadcast to ``m``'s shape.

    ``weight`` is how gold and pseudo rows are told apart at the loss: a gold
    cell weighs 1.0, a surviving pseudo cell weighs ``cfg.pseudo_weight``, and
    a pseudo cell that failed the confidence filter weighs exactly 0.0.  Passing
    ``None`` reproduces the unweighted behaviour bit for bit.
    """
    w = m.float()
    if weight is None:
        return w
    weight = torch.as_tensor(weight, dtype=w.dtype, device=w.device)
    if weight.dim() == 1 and w.dim() == 2 and weight.shape[0] == w.shape[0]:
        weight = weight.unsqueeze(1)          # per-row scalar
    return w * weight.clamp(min=0.0).expand_as(w)


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE that ignores NaN / masked-out targets instead of returning NaN.

    Reduction is 'mean over supervised elements', so the loss magnitude does not
    depend on how many labels happen to be missing in the batch.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32),
        )

    def forward(self, logits, targets, mask=None, weight=None):
        m = _finite_mask(targets, mask)
        if not m.any():
            return logits.sum() * 0.0
        safe = torch.where(m, targets, torch.zeros_like(targets))
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), safe.float(), reduction="none", pos_weight=pw
        )
        w = _elementwise_weights(m, weight)
        denom = w.sum()
        if denom <= 0:
            return logits.sum() * 0.0
        return (loss * w).sum() / denom


class MaskedAUCMLoss(nn.Module):
    """Per-label AUC surrogate: squared hinge over in-batch (positive, negative) pairs.

    The competition metric is macro ROC-AUC, which BCE only optimises indirectly.  This is
    the pairwise square-hinge surrogate that LibAUC's AUC-margin loss (Yuan et al., ICCV
    2021 — 1st on CheXpert) rewrites as a min-max problem to avoid the pairwise cost; at
    our batch sizes the pairs are cheap, so they are formed directly and no auxiliary
    (a, b, alpha) variables are needed.

    Soft targets are binarised at 0.5 over supervised cells (weight > 0).  Each pair is
    weighted by the product of its two cells' weights, so the silence weighting carries
    over.  A label with no pos/neg pair in the batch contributes nothing — with bs 8 that
    is common for rare findings, which is why this is meant as a FINAL-PHASE loss on top
    of BCE pretraining (``--aucm-epochs``), not a from-scratch objective.

    Related but different: ``--rank-loss-weight`` ADDS a softplus pairwise term to BCE;
    this REPLACES the objective with the margin form.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = float(margin)

    def forward(self, logits, targets, mask=None, weight=None):
        m = _finite_mask(targets, mask)
        w = _elementwise_weights(m, weight)
        num, den = [], []
        for li in range(logits.shape[1]):
            sel = w[:, li] > 0
            if not bool(sel.any()):
                continue
            yb = targets[:, li] > 0.5
            pos, neg = sel & yb, sel & ~yb
            if not (bool(pos.any()) and bool(neg.any())):
                continue
            sp = logits[pos, li].float().unsqueeze(1)
            sn = logits[neg, li].float().unsqueeze(0)
            pw = w[pos, li].float().unsqueeze(1) * w[neg, li].float().unsqueeze(0)
            d = torch.clamp(self.margin - (sp - sn), min=0.0) ** 2
            num.append((d * pw).sum())
            den.append(pw.sum())
        if not num:
            return logits.sum() * 0.0
        return torch.stack(num).sum() / torch.stack(den).sum().clamp(min=1e-8)


class MaskedFocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017), NaN/mask aware.

    ``alpha < 0`` disables the alpha term.  Note that ``alpha`` weights the
    POSITIVE class and ``1 - alpha`` the negative one, so the RetinaNet default
    of 0.25 down-weights positives -- appropriate at 0.1% prevalence, not at the
    15-60% prevalence of these 12 knee labels.  See the module docstring in
    scripts/ab_loss.md-style notes and the README of this change for why the
    default loss is plain BCE.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits, targets, mask=None, weight=None):
        m = _finite_mask(targets, mask)
        if not m.any():
            return logits.sum() * 0.0
        safe = torch.where(m, targets, torch.zeros_like(targets))
        logits = logits.float()
        safe = safe.float().clamp(0.0, 1.0)

        # SOFT TARGETS.  Written as the EXPECTATION of the hard-label focal loss
        # under Bernoulli(y) -- i.e. linear in y -- rather than by substituting a
        # fractional y into p_t.  Both forms agree exactly for y in {0, 1}, but
        # only this one is the loss a soft label actually means, and only this
        # one stays bounded by the two hard-label losses.  Substituting y=0.75
        # into (1 - p_t)^gamma produced a value BELOW both L(0) and L(1).
        # logsigmoid gives stable log(p) / log(1-p) for |logit| > 20, which is
        # exactly where focal weighting matters most.
        p = torch.sigmoid(logits)
        log_p = F.logsigmoid(logits)
        log_1mp = F.logsigmoid(-logits)
        pos = (1.0 - p).clamp(0.0, 1.0).pow(self.gamma) * (-log_p)
        neg = p.clamp(0.0, 1.0).pow(self.gamma) * (-log_1mp)
        if self.alpha >= 0.0:
            pos = self.alpha * pos
            neg = (1.0 - self.alpha) * neg
        focal = safe * pos + (1.0 - safe) * neg
        w = _elementwise_weights(m, weight)
        denom = w.sum()
        if denom <= 0:
            return logits.sum() * 0.0
        return (focal * w).sum() / denom


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ben-Baruch et al., 2021).

    The key insight: in multi-label problems with many classes, most cells are
    *negative* (zero).  Standard focal loss uses the same gamma for positives
    and negatives; ASL uses a LARGER gamma_neg to aggressively down-weight easy
    negatives (confident zeros) while keeping all the gradient from positives.

    Default gamma_neg=4, gamma_pos=1 is the paper's recommendation and beats
    focal loss on every multi-label benchmark including NUS-WIDE and MS-COCO.
    The optional probability margin `m` shifts negative probabilities down by m
    before computing the focus weight so that very confident negatives get zero
    weight -- equivalent to hard thresholding.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0,
                 clip: float = 0.05, label_smoothing: float = 0.0,
                 ohem_ratio: float = 1.0):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)   # probability margin; 0.05 is the paper default
        self.label_smoothing = float(label_smoothing)
        self.ohem_ratio = float(ohem_ratio)  # 1.0 = no OHEM, 0.7 = keep hardest 70%

    def forward(self, logits, targets, mask=None, weight=None):
        m = _finite_mask(targets, mask)
        if not m.any():
            return logits.sum() * 0.0
        safe = torch.where(m, targets, torch.zeros_like(targets))
        
        # Apply label smoothing to the targets
        if self.label_smoothing > 0:
            safe = safe * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            
        logits = logits.float()
        safe = safe.float().clamp(0.0, 1.0)

        # ── SOFT TARGETS ────────────────────────────────────────────────
        # The previous implementation branched on ``safe.bool()``, i.e. "is the
        # target != 0".  That is a HARD threshold, and it is wrong twice over:
        #   * with label smoothing (cfg.label_smoothing defaults to 0.05, and
        #     train_one_epoch already applies it) every target becomes 0.05 or
        #     0.95, so ``safe.bool()`` is TRUE for all of them -- every negative
        #     cell was scored with the POSITIVE branch and gamma_pos, which
        #     silently turned ASL into plain focal loss with gamma=1 and threw
        #     away the entire asymmetry the loss exists for;
        #   * report-derived soft targets in (0, 1) would be treated as fully
        #     confident positives.
        # The fix is the natural continuous form, which reduces EXACTLY to the
        # original for y in {0, 1}:
        #     L = y * (1 - p)^gamma_pos * (-log p)
        #       + (1 - y) * (p_m)^gamma_neg * (-log(1 - p_m)),   p_m = max(p - clip, 0)
        p = torch.sigmoid(logits)
        p_m = (p - self.clip).clamp(min=0.0)     # margin-shifted, for negatives

        log_p = F.logsigmoid(logits)                       # stable log(p)
        loss_pos = (1.0 - p).clamp(0, 1).pow(self.gamma_pos) * (-log_p)
        loss_neg = p_m.pow(self.gamma_neg) * (-torch.log((1.0 - p_m).clamp(min=1e-8)))

        asl = safe * loss_pos + (1.0 - safe) * loss_neg
        w = _elementwise_weights(m, weight)
        
        loss = asl * w
        
        # Online Hard Example Mining (OHEM): zero out the loss for the easiest examples
        if self.ohem_ratio < 1.0:
            flat_loss = loss[m]
            if flat_loss.numel() > 0:
                k = max(1, int(flat_loss.numel() * self.ohem_ratio))
                # Use topk instead of kthvalue because kthvalue is not supported on MPS (Apple Silicon)
                threshold = torch.topk(flat_loss, k).values[-1]
                loss = torch.where(loss >= threshold, loss, torch.zeros_like(loss))

        denom = w.sum()
        if denom <= 0:
            return logits.sum() * 0.0
        return loss.sum() / denom


def build_criterion(cfg):
    """Resolve cfg.loss / cfg.focal_loss into a criterion + a human label."""
    choice = str(getattr(cfg, "loss", "bce")).lower()
    if choice == "auto":
        choice = "focal" if getattr(cfg, "focal_loss", False) else "bce"
    if choice == "asl":
        gn = float(getattr(cfg, "asl_gamma_neg", 4.0))
        gp = float(getattr(cfg, "asl_gamma_pos", 1.0))
        clip = float(getattr(cfg, "asl_clip", 0.05))
        # Label smoothing is applied ONCE, in train_one_epoch, which clamps the
        # targets into [eps, 1-eps] and therefore leaves genuine soft targets in
        # the interior untouched.  Passing cfg.label_smoothing here as well
        # smoothed twice (0 -> 0.05 -> 0.025) for no benefit; loss_label_smoothing
        # defaults to 0 and exists only for an explicit loss-side A/B.
        ls = float(getattr(cfg, "loss_label_smoothing", 0.0))
        ohem = float(getattr(cfg, "ohem_ratio", 1.0))
        return AsymmetricLoss(gamma_neg=gn, gamma_pos=gp, clip=clip, label_smoothing=ls, ohem_ratio=ohem), \
               f"AsymmetricLoss(gamma_neg={gn}, gamma_pos={gp}, clip={clip}, ls={ls}, ohem={ohem})"
    if choice == "focal":
        alpha = float(getattr(cfg, "focal_alpha", 0.25))
        crit = MaskedFocalLoss(gamma=float(cfg.focal_gamma), alpha=alpha)
        return crit, f"MaskedFocalLoss(gamma={cfg.focal_gamma}, alpha={alpha})"
    if choice not in ("bce", "auto"):
        print(f"WARNING: unknown loss '{choice}', falling back to bce")
    return MaskedBCEWithLogitsLoss(), "MaskedBCEWithLogitsLoss"


# ══════════════════════════════════════════════════════════════════════════
# Runtime helpers
# ══════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp(cfg, device):
    """Return (use_amp, autocast_dtype, needs_grad_scaler).

    bfloat16 is only natively supported from sm_80 (Ampere).  Kaggle's free
    accelerators are T4 (sm_75) and P100 (sm_60): on those, bf16 autocast is
    either rejected or emulated at a large slowdown, so fp16 + GradScaler is the
    correct choice.  fp16 can overflow, which is what the GradScaler is for --
    combined with the masked losses and grad clipping this is NaN-safe.
    """
    want = str(getattr(cfg, "amp_dtype", "auto")).lower()
    if device.type != "cuda" or want == "fp32":
        return False, torch.float32, False

    bf16_ok = False
    cap = None
    try:
        cap = torch.cuda.get_device_capability()
        bf16_ok = cap[0] >= 8 and torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False

    if want == "bf16":
        if bf16_ok:
            return True, torch.bfloat16, False
        print(f"WARNING: bf16 requested but device capability {cap} has no native "
              f"bfloat16 (needs sm_80+). Falling back to fp16 + GradScaler.")
        return True, torch.float16, True
    if want == "fp16":
        return True, torch.float16, True

    if bf16_ok:
        return True, torch.bfloat16, False
    print(f"INFO: device capability {cap} lacks native bfloat16 -> using fp16 + GradScaler.")
    return True, torch.float16, True


class WallClock:
    """Wall-clock budget guard.

    Uses ``time.time()`` on purpose.  ``time.monotonic()`` / ``perf_counter()``
    map to CLOCK_MONOTONIC, which does NOT advance while the machine is
    suspended -- a guard built on it under-counts elapsed time and can overrun a
    Kaggle session limit (Kaggle bills real elapsed time).  ``time.time()``
    tracks real elapsed time including any suspension.  We take the max of the
    two so that a backwards NTP step cannot make the budget look larger than it
    is.

    Verified default: ``elapsed()`` is sleep-INCLUSIVE, which is correct for the
    Kaggle target (the 9-hour limit is billed in real time and a Kaggle VM never
    suspends).  ``mode="monotonic"`` is available for local development, where a
    laptop that sleeps overnight would otherwise burn the whole budget while
    doing no work; it maps to CLOCK_MONOTONIC and does NOT count suspend.
    ``--set time_budget_clock=monotonic`` selects it.
    """

    def __init__(self, budget_min: float, mode: str = "wall"):
        self.budget_s = float(budget_min) * 60.0
        self.mode = "monotonic" if str(mode).lower().startswith("mono") else "wall"
        self.t0_wall = time.time()
        self.t0_mono = time.monotonic()

    def elapsed(self) -> float:
        if self.mode == "monotonic":
            return time.monotonic() - self.t0_mono
        return max(time.time() - self.t0_wall, time.monotonic() - self.t0_mono)

    def expired(self, reserve_s: float = 0.0) -> bool:
        if self.budget_s <= 0:
            return False
        return self.elapsed() + reserve_s >= self.budget_s

    def remaining(self) -> float:
        return math.inf if self.budget_s <= 0 else self.budget_s - self.elapsed()


def strip_compile_prefix(state):
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def base_module(model):
    """The uncompiled nn.Module behind a possible torch.compile wrapper."""
    return getattr(model, "_orig_mod", model)


class Lookahead:
    """Lookahead optimizer wrapper (Zhang et al. 2019).

    Maintains a set of 'slow weights' (theta) that are updated every k steps
    by linearly interpolating toward the fast weights (phi):
        theta = theta + alpha * (phi - theta)

    This smoothes the optimization landscape and reliably improves
    generalization on transformers.  k=6, alpha=0.5 are the paper defaults.
    Particularly effective combined with AdamW on DINOv2.
    """

    def __init__(self, optimizer, k: int = 6, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self._step = 0
        self.param_groups = optimizer.param_groups
        # Snapshot slow weights
        self._slow = [
            [p.data.clone() for p in group["params"]]
            for group in optimizer.param_groups
        ]

    def step(self):
        self.optimizer.step()
        self._step += 1
        if self._step % self.k == 0:
            for group, slow_group in zip(self.optimizer.param_groups, self._slow):
                for p, slow in zip(group["params"], slow_group):
                    if p.grad is None and slow.shape != p.data.shape:
                        continue
                    slow.add_(self.alpha * (p.data - slow))
                    p.data.copy_(slow)

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"fast": self.optimizer.state_dict(), "step": self._step,
                "slow": [[s.cpu() for s in sg] for sg in self._slow]}

    def load_state_dict(self, d):
        self.optimizer.load_state_dict(d["fast"])
        self._step = d.get("step", 0)
        if "slow" in d:
            for sg, saved in zip(self._slow, d["slow"]):
                for s, sv in zip(sg, saved):
                    s.copy_(sv)

    def get_last_lr(self):
        return self.optimizer.get_last_lr() if hasattr(self.optimizer, "get_last_lr") else [self.optimizer.param_groups[0]["lr"]]


# ══════════════════════════════════════════════════════════════════════════
# Folds
# ══════════════════════════════════════════════════════════════════════════

def multilabel_stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Greedy iterative stratification (Sechidis et al. 2011), no extra deps.

    Assigns each sample to the fold that is most 'starved' of the rarest label
    the sample carries.  Much better than StratifiedKFold on a single column
    when there are 12 correlated targets and only ~58 samples.
    """
    n, L = y.shape
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)

    desired = np.full(n_folds, n / n_folds, dtype=float)
    desired_lbl = np.tile((y.sum(0) / n_folds).astype(float), (n_folds, 1))  # (F, L)
    count = np.zeros(n_folds, dtype=float)
    count_lbl = np.zeros((n_folds, L), dtype=float)
    folds = np.full(n, -1, dtype=int)

    # Rarest labels first so the scarce positives get spread out.
    label_order = np.argsort(y.sum(0))

    for idx in order:
        row = y[idx]
        active = [l for l in label_order if row[l] > 0]
        if active:
            l = active[0]
            need = desired_lbl[:, l] - count_lbl[:, l]
        else:
            need = desired - count
        best = np.flatnonzero(need >= need.max() - 1e-12)
        if len(best) > 1:
            slack = desired[best] - count[best]
            best = best[np.flatnonzero(slack >= slack.max() - 1e-12)]
        f = int(best[rng.randint(len(best))]) if len(best) > 1 else int(best[0])
        folds[idx] = f
        count[f] += 1
        count_lbl[f] += row
    return folds


def assign_folds(df: pd.DataFrame, target_cols, cfg) -> pd.DataFrame:
    """Attach a 'fold' column, preferring one that already ships with the CSV."""
    df = df.reset_index(drop=True)

    if "fold" in df.columns and df["fold"].notna().all() and df["fold"].min() >= 0:
        sizes = df["fold"].astype(int).value_counts().sort_index().to_dict()
        print(f"Using pre-computed 'fold' column from CSV. Sizes: {sizes}")
        df["fold"] = df["fold"].astype(int)
        return df

    usable = [c for c in target_cols if c in df.columns]
    if usable and df[usable].notna().all(axis=1).all():
        y = df[usable].values.astype(int)
        df["fold"] = multilabel_stratified_folds(y, cfg.n_folds, cfg.seed)
        sizes = df["fold"].value_counts().sort_index().to_dict()
        print(f"Multilabel-stratified folds built over {len(usable)} targets. Sizes: {sizes}")
        return df

    # Last resort: single-column stratification (what the old code did).
    if usable:
        strat = df[usable[0]].fillna(0).astype(int)
    else:
        strat = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    df["fold"] = -1
    try:
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        for i, (_, val_idx) in enumerate(skf.split(df, strat)):
            df.loc[val_idx, "fold"] = i
    except ValueError:
        df["fold"] = np.arange(len(df)) % cfg.n_folds
    print("WARNING: fell back to single-column StratifiedKFold "
          f"(fold sizes {df['fold'].value_counts().sort_index().to_dict()})")
    return df


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_dataframe(cfg):
    """Load the training table and return (df, target_cols)."""
    candidates = []
    if getattr(cfg, "train_csv", ""):
        candidates.append(os.path.join(cfg.data_dir, cfg.train_csv))
    candidates.append(os.path.join(cfg.data_dir, "train.csv"))

    df, path = None, None
    for c in candidates:
        if os.path.exists(c):
            df, path = pd.read_csv(c), c
            break

    if df is None:
        print("WARNING: no training CSV found, generating dummy data for testing")
        df = pd.DataFrame({"StudyInstanceUID": [str(i) for i in range(20)]})
        for t in KNEE_TARGETS:
            df[t] = [i % 2 for i in range(20)]
        return df, list(KNEE_TARGETS)

    print(f"Loaded {os.path.basename(path)} with {len(df)} rows")

    # NOTE: pseudo-labels are deliberately NOT merged here any more.  The old
    # code did ``df.update(pseudo_df)`` on the gold frame, which (a) adds no
    # rows -- DataFrame.update only overwrites the index/column intersection, so
    # with the default train_csv=train_gold.csv it was a silent no-op that still
    # printed "Injected 4349 pseudo-labels"; (b) had no fold awareness, so a
    # study's pseudo-label came from an ensemble that had trained on the very
    # gold fold the student would validate on; and (c) put pseudo rows into the
    # validation split.  Pseudo rows are now attached per training fold, after
    # the gold folds are fixed -- see attach_pseudo_rows() and main().
    non_target = {"StudyInstanceUID", "fold", "study_id", "patient_id",
                  "Report", "has_images", "is_gold", "split",
                  "is_pseudo", "teacher_fold", "sample_weight"}
    target_cols = [c for c in df.columns
                   if c not in non_target and df[c].dtype in ("float64", "int64", "float32", "int32")]
    if not target_cols:
        print(f"WARNING: no numeric target columns found. Columns: {list(df.columns)}")
        target_cols = [c for c in KNEE_TARGETS if c in df.columns]
    return df, target_cols


def filter_labelled(df, target_cols, cfg):
    """Drop rows that carry no supervision.

    This is the load-bearing half of the label-integrity fix: RSNADataset turns
    NaN targets into 0.0, so a NaN row that reaches the loader is indistinguishable
    from a genuinely all-negative study.  The mask has to be applied here, on the
    dataframe, while the NaNs still exist.
    """
    if not target_cols:
        return df
    present = df[target_cols].notna()
    any_lab = present.any(axis=1)
    all_lab = present.all(axis=1)

    n_total, n_any, n_all = len(df), int(any_lab.sum()), int(all_lab.sum())
    print(f"Label audit: {n_total} rows | {n_all} fully labelled | "
          f"{n_any - n_all} partially labelled | {n_total - n_any} entirely NaN")

    if not cfg.labelled_only:
        if n_total - n_any > 0:
            print(f"WARNING: labelled_only=False -- {n_total - n_any} unlabelled rows "
                  "will reach the loss as all-negative studies (RSNADataset does "
                  "nan_to_num on labels). This is almost certainly not what you want.")
        return df

    if n_any == 0:
        print("WARNING: no labelled rows at all; keeping the frame as-is.")
        return df

    kept = df[any_lab].reset_index(drop=True)
    if n_total - n_any:
        print(f"Dropped {n_total - n_any} unlabelled rows -> training on {len(kept)} studies")
    return kept


def drop_missing_images(df, image_dir):
    """Drop studies with no directory on disk (they would train on black frames)."""
    if not os.path.isdir(image_dir):
        return df
    on_disk = {d for d in os.listdir(image_dir) if not d.startswith(".")}
    have = df["StudyInstanceUID"].astype(str).isin(on_disk)
    n_missing = int((~have).sum())
    if n_missing:
        print(f"Dropped {n_missing} studies with no DICOMs under {image_dir} "
              f"(they would have been all-black images)")
        df = df[have].reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════
# Pseudo-labels (semi-supervised / self-training path)
#
# Contract with src/pseudo_label.py:
#   * train_pseudo.csv is LONG: one row per (StudyInstanceUID, teacher_fold).
#     ``teacher_fold = f`` means "predicted by models that never trained on gold
#     fold f", so those rows -- and only those -- are legal training material
#     for student fold f.  Mixing folds would distil the student's own
#     validation labels into its training targets.
#   * Cells that failed the confidence filter are NaN and get weight 0.
#   * train_pseudo_meta.json pins the gold fold map the teacher assumed.  If the
#     student's fold map differs, every out-of-fold guarantee is void, so we
#     refuse to train (or warn loudly when pseudo_strict_oof=false).
#   * Gold rows always keep weight 1.0 and are never overwritten by a
#     pseudo-label; validation is gold-only, always.
# ══════════════════════════════════════════════════════════════════════════

PSEUDO_FOLD_COL = "teacher_fold"


class PseudoBundle:
    """A loaded train_pseudo.csv + its sidecar metadata."""

    def __init__(self, frame, meta, path):
        self.frame = frame
        self.meta = meta or {}
        self.path = path

    @property
    def teacher_folds(self):
        if self.frame is None or len(self.frame) == 0:
            return []
        return sorted(int(f) for f in self.frame[PSEUDO_FOLD_COL].unique())

    def for_fold(self, fold, target_cols, exclude_ids=()):
        """Pseudo rows usable when training student fold ``fold``."""
        if self.frame is None or len(self.frame) == 0:
            return self.frame.iloc[:0] if self.frame is not None else None
        sub = self.frame[self.frame[PSEUDO_FOLD_COL].astype(int) == int(fold)]
        if len(exclude_ids):
            # A gold study must never arrive as a pseudo row: its real label wins.
            sub = sub[~sub["StudyInstanceUID"].astype(str).isin(set(map(str, exclude_ids)))]
        cols = [c for c in target_cols if c in sub.columns]
        sub = sub[sub[cols].notna().any(axis=1)] if cols else sub.iloc[:0]
        return sub.reset_index(drop=True)


def load_pseudo_bundle(cfg, target_cols, gold_folds=None):
    """Load the pseudo-label bundle and verify its out-of-fold provenance.

    ``gold_folds`` is the student's {StudyInstanceUID: fold} map.  Returns None
    when pseudo-labelling is off or the file is absent.
    """
    if not getattr(cfg, "use_pseudo_labels", False):
        return None

    path = os.path.join(cfg.data_dir, getattr(cfg, "pseudo_csv", "train_pseudo.csv"))
    strict = bool(getattr(cfg, "pseudo_strict_oof", True))
    if not os.path.exists(path):
        print(f"WARNING: use_pseudo_labels=True but {path} not found -- "
              "training on gold rows only.")
        return None

    frame = pd.read_csv(path)
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
        except Exception as e:
            print(f"WARNING: could not read {meta_path} ({e})")

    def _fail(msg):
        if strict:
            raise SystemExit(
                f"REFUSING to train with leaky pseudo-labels: {msg}\n"
                f"Regenerate with `python3 -m src.pseudo_label` (it writes the "
                f"long, per-teacher-fold format), or set pseudo_strict_oof=false "
                f"to proceed anyway and treat the CV score as meaningless.")
        print(f"WARNING (pseudo_strict_oof=false): {msg}")

    if PSEUDO_FOLD_COL not in frame.columns:
        _fail(f"{os.path.basename(path)} has no '{PSEUDO_FOLD_COL}' column, so its "
              "rows cannot be proven out-of-fold. This is the old wide format, "
              "where every study's label came from an ensemble that had trained "
              "on every gold fold.")
        return None

    missing = [c for c in target_cols if c not in frame.columns]
    if missing:
        _fail(f"{os.path.basename(path)} is missing target columns {missing}")
        return None

    # The fold map the teacher assumed must be the one the student uses.
    if gold_folds is not None:
        recorded = (meta or {}).get("gold_folds")
        if recorded is None:
            _fail(f"no gold fold map recorded in {os.path.basename(meta_path)}; "
                  "cannot verify that teacher fold f really excluded the studies "
                  "this run validates in fold f.")
        else:
            shared = set(recorded) & set(map(str, gold_folds))
            drift = [s for s in sorted(shared)
                     if int(recorded[s]) != int(gold_folds[s])]
            if not shared:
                _fail("the recorded gold fold map shares no study with this run's "
                      "gold set.")
            elif drift:
                _fail(f"{len(drift)} gold studies changed fold since the "
                      f"pseudo-labels were generated (e.g. {drift[:3]}). Every "
                      "out-of-fold guarantee is void.")
            else:
                print(f"Pseudo-label fold provenance verified against "
                      f"{len(shared)} gold studies.")

    n_folds = int(getattr(cfg, "n_folds", 5))
    bad = sorted({int(f) for f in frame[PSEUDO_FOLD_COL].unique()
                  if not (0 <= int(f) < n_folds)})
    if bad:
        _fail(f"teacher_fold values {bad} are outside [0, {n_folds}).")

    bundle = PseudoBundle(frame, meta, path)
    print(f"Loaded {len(frame)} pseudo rows from {os.path.basename(path)} "
          f"covering teacher folds {bundle.teacher_folds} "
          f"({frame['StudyInstanceUID'].nunique()} distinct studies)")
    if meta:
        print(f"  filter: margin={meta.get('pseudo_conf_margin')} "
              f"cap={meta.get('pseudo_max_per_label')} "
              f"soft_labels={meta.get('soft_labels')}")
    return bundle


def attach_pseudo_rows(gold_train_df, bundle, fold, target_cols, cfg, image_dir=None):
    """Return (combined_train_df, weights) for one student fold.

    ``weights`` is an (N, L) float32 array of per-cell loss weights aligned with
    ``combined_train_df``: 1.0 for every gold cell, ``cfg.pseudo_weight`` for a
    surviving pseudo cell, 0.0 for a filtered-out one.  The gold rows come
    first and are returned unmodified -- a pseudo-label can never overwrite a
    real one because the two live in different rows.
    """
    gold = gold_train_df.reset_index(drop=True).copy()
    gold["is_pseudo"] = False
    gold_w = np.ones((len(gold), len(target_cols)), dtype=np.float32)

    if bundle is None:
        return gold, gold_w

    sub = bundle.for_fold(fold, target_cols, exclude_ids=[])
    if sub is None or len(sub) == 0:
        print(f"  no pseudo rows for teacher fold {fold} -- gold only")
        return gold, gold_w

    if image_dir:
        sub = drop_missing_images(sub, image_dir)
    if len(sub) == 0:
        return gold, gold_w

    w_pseudo = float(getattr(cfg, "pseudo_weight", 0.30))
    vals = sub[list(target_cols)].to_numpy(dtype=np.float64)
    keep = np.isfinite(vals)
    pseudo_w = np.where(keep, w_pseudo, 0.0).astype(np.float32)

    rows = pd.DataFrame({"StudyInstanceUID": sub["StudyInstanceUID"].astype(str).values})
    for j, c in enumerate(target_cols):
        # NaN cells become 0.0 targets with weight 0: RSNADataset does
        # nan_to_num on labels, so a NaN would otherwise arrive as a confident
        # negative. The weight, not the value, is what silences them.
        rows[c] = np.where(keep[:, j], vals[:, j], 0.0)
    rows["is_pseudo"] = True
    rows["fold"] = -1                  # -1 never equals a student fold -> train only

    for col in gold.columns:
        if col not in rows.columns:
            rows[col] = np.nan if col != "is_pseudo" else True
    rows = rows[gold.columns]

    combined = pd.concat([gold, rows], ignore_index=True)
    weights = np.concatenate([gold_w, pseudo_w], axis=0)

    kept_cells = int(keep.sum())
    print(f"  + {len(rows)} pseudo studies from teacher fold {fold} "
          f"({kept_cells}/{keep.size} cells survived the confidence filter, "
          f"weight {w_pseudo}); gold rows keep weight 1.0")
    return combined, weights


# ══════════════════════════════════════════════════════════════════════════
# Report-derived (weak) labels — src/labels.py
#
# The gold panel is 58 studies.  All 649 image-bearing studies carry a radiology
# report, so a text-derived label table takes training from 58 to 649 rows: an
# 11x larger training set at the cost of a noisier target.  Three properties
# make that safe, and all three are enforced HERE rather than trusted:
#
#   1. PER-SAMPLE WEIGHTS.  A gold cell weighs cfg.gold_weight (1.0); a derived
#      cell weighs cfg.labels_weight (0.30); a derived cell that fails the
#      confidence gate weighs exactly 0.0.  The weight, not the value, is what
#      silences a cell -- RSNADataset nan_to_num's labels, so a NaN target would
#      otherwise arrive at the loss as a *confident negative*.
#   2. SOFT TARGETS.  Derived targets live anywhere in [0, 1] and every
#      criterion here consumes them directly (see AsymmetricLoss.forward for the
#      hard-threshold bug this replaced).
#   3. GOLD-ONLY VALIDATION.  Derived rows are appended with fold = -1, which
#      never equals a student fold, so they can only ever be trained on.
#      enforce_gold_only_validation() then checks the split rather than assuming
#      it.  Validating against derived labels measures agreement with a keyword
#      matcher, not with ground truth, and would make every reported number a
#      measure of the wrong thing.
#
# Nothing below activates until cfg.labels_csv names a file that exists, so this
# is inert until src/labels.py lands.
# ══════════════════════════════════════════════════════════════════════════

def _truthy(series):
    """Robust truthiness for a flag column that may be bool / int / float / str."""
    if series.dtype == bool:
        return series
    if series.dtype == object:
        return series.map(
            lambda v: str(v).strip().lower() not in
            ("", "false", "0", "0.0", "no", "n", "nan", "none", "null")).astype(bool)
    return series.fillna(0).astype(float) > 0.5


def load_derived_labels(cfg, target_cols):
    """Load the report-derived label table, or None if it is not there yet."""
    rel = str(getattr(cfg, "labels_csv", "") or "")
    if not rel:
        return None
    path = rel if os.path.isabs(rel) else os.path.join(cfg.data_dir, rel)
    if not os.path.exists(path):
        print(f"NOTE: labels_csv='{rel}' not found under {cfg.data_dir} -- "
              "training on gold rows only.")
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as e:
        print(f"WARNING: could not read {path} ({e}); gold rows only.")
        return None
    if "StudyInstanceUID" not in frame.columns:
        print(f"WARNING: {os.path.basename(path)} has no StudyInstanceUID column; ignoring.")
        return None
    have = [c for c in target_cols if c in frame.columns]
    if not have:
        print(f"WARNING: {os.path.basename(path)} carries none of the {len(target_cols)} "
              "target columns; ignoring.")
        return None
    if len(have) < len(target_cols):
        print(f"NOTE: {os.path.basename(path)} covers {len(have)}/{len(target_cols)} "
              f"targets; the rest are weighted 0 on those rows.")
    print(f"Loaded {len(frame)} report-derived rows from {os.path.basename(path)} "
          f"({frame['StudyInstanceUID'].nunique()} distinct studies)")
    return frame


def build_derived_from_reports(cfg, target_cols, gold_df=None):
    """Bridge ``cfg.use_report_labels`` -> ``src.labels`` -> the derived-row path.

    THE GAP THIS CLOSES
    -------------------
    ``load_derived_labels()`` only fires when ``cfg.labels_csv`` names a file that
    already exists on disk.  ``cfg.use_report_labels`` and the whole ``report_*``
    block were declared in Config and documented here, but nothing read them, so
    a run with ``use_report_labels=true`` and no pre-baked CSV attached exactly
    0 derived rows and silently trained on the 58 gold studies alone.  This
    builds the same frame in-process from src/labels.py so the flag means what
    it says, and returns it in the shape ``attach_derived_rows`` already expects.

    CALIBRATION LEAKAGE
    -------------------
    ``report_label_calibrate_on_folds`` exists so the prior and the per-verdict
    reliability table can be refit WITHOUT the fold being validated.  Both
    calibrators already take ``folds=``; this is what finally passes it.  Leave
    the setting empty to use the constants shipped in src/labels.py.
    """
    if not bool(getattr(cfg, "use_report_labels", False)):
        return None
    try:
        import src.labels as L
    except Exception as e:                                  # pragma: no cover
        print(f"WARNING: use_report_labels=true but src/labels.py is unusable ({e}); "
              "training on gold rows only.")
        return None

    rel = str(getattr(cfg, "report_csv", "train.csv") or "train.csv")
    path = rel if os.path.isabs(rel) else os.path.join(cfg.data_dir, rel)
    if not os.path.exists(path):
        print(f"WARNING: report_csv='{rel}' not found under {cfg.data_dir}; "
              "training on gold rows only.")
        return None
    try:
        frame = pd.read_csv(path, engine="python")
    except Exception as e:
        print(f"WARNING: could not read {path} ({e}); gold rows only.")
        return None

    text_col = str(getattr(cfg, "report_text_col", "Report"))
    if text_col not in frame.columns or "StudyInstanceUID" not in frame.columns:
        print(f"WARNING: {os.path.basename(path)} lacks '{text_col}' or "
              "'StudyInstanceUID'; gold rows only.")
        return None

    # A report without pixels is not a training sample.
    if bool(getattr(cfg, "report_labels_require_images", True)):
        image_dir = os.path.join(cfg.data_dir, "train_series")
        if os.path.isdir(image_dir):
            on_disk = {d for d in os.listdir(image_dir) if not d.startswith(".")}
            before = len(frame)
            frame = frame[frame["StudyInstanceUID"].astype(str).isin(on_disk)]
            print(f"  report labels: {before} reports -> {len(frame)} with DICOMs on disk")

    cap = int(getattr(cfg, "report_label_max_studies", 0) or 0)
    if cap > 0 and len(frame) > cap:
        frame = frame.head(cap)
        print(f"  report labels: capped at {cap} studies (debugging aid)")
    if frame.empty:
        return None

    # Refit the calibration on gold folds that this run will NOT validate on.
    priors = calib = None
    folds_spec = str(getattr(cfg, "report_label_calibrate_on_folds", "") or "").strip()
    if folds_spec and gold_df is not None:
        try:
            keep = [int(x) for x in re.split(r"[,\s]+", folds_spec) if x != ""]
            priors = L.calibrate_priors(gold_df, labels=target_cols, folds=keep)
            calib = L.calibrate_states(gold_df, labels=target_cols, folds=keep,
                                       priors=priors)
            print(f"  report labels: calibration refit on gold folds {keep} "
                  "(the validated fold contributed nothing to it)")
        except Exception as e:
            priors = calib = None
            print(f"  WARNING: calibration refit failed ({e}); using the "
                  "constants shipped in src/labels.py.")

    try:
        derived = L.label_dataframe(frame, text_col=text_col, labels=list(target_cols),
                                    priors=priors, calibration=calib)
    except Exception as e:
        print(f"WARNING: src.labels.label_dataframe failed ({e}); gold rows only.")
        return None

    keep_cols = [c for c in derived.columns
                 if c in set(target_cols) or c.endswith("__conf")]
    out = pd.concat([frame[["StudyInstanceUID"]].reset_index(drop=True),
                     derived[keep_cols].reset_index(drop=True)], axis=1)

    # The report_* knobs are the report path's spelling of the derived path's
    # knobs; map them across so a single attach implementation serves both.
    cfg.labels_weight = float(getattr(cfg, "report_label_weight", 0.35))
    cfg.labels_min_conf = float(getattr(cfg, "report_min_confidence", 0.10))
    lang = derived["report_lang"] if "report_lang" in derived else None
    lang_note = f" | languages: {lang.value_counts().to_dict()}" if lang is not None else ""
    print(f"Built {len(out)} report-derived rows from {os.path.basename(path)} "
          f"via src/labels.py (weight {cfg.labels_weight}, "
          f"min_conf {cfg.labels_min_conf}){lang_note}")
    return out


def derived_cell_weights(sub, target_cols, cfg):
    """(values, weights) for the derived block.  Pure; unit-testable.

    values  -- (N, L) float32 targets, NaN cells replaced by 0.0 (their weight
               is 0, so the value is never read by the loss).
    weights -- (N, L) float32, 0.0 wherever the cell is missing or fails the
               confidence gate, else labels_weight [* the row's own confidence].
    """
    n, L = len(sub), len(target_cols)
    vals = np.full((n, L), np.nan, dtype=np.float64)
    for j, c in enumerate(target_cols):
        if c in sub.columns:
            vals[:, j] = pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=np.float64)

    keep = np.isfinite(vals)
    vals = np.where(keep, np.clip(vals, 0.0, 1.0), np.nan)
    if not bool(getattr(cfg, "soft_targets", True)):
        vals = np.where(keep, (vals >= 0.5).astype(np.float64), vals)

    # PER-CELL confidence.  src/labels.py emits one "<label>__conf" column per
    # target alongside the soft probability, and its "unmentioned" state comes
    # back as (prob = class prior, confidence = 0.0).  Multiplying that
    # confidence into the weight is what makes an unmentioned finding cost
    # exactly nothing instead of training the net on a prior -- and it does so
    # without a threshold, so a hedged finding still contributes, proportionally.
    # Falls back to |p - 0.5| * 2 when the file carries no confidences.
    suffix = str(getattr(cfg, "labels_conf_suffix", "__conf") or "")
    conf = None
    if suffix:
        cols = [c + suffix for c in target_cols]
        if any(c in sub.columns for c in cols):
            conf = np.zeros((n, L), dtype=np.float64)
            for j, c in enumerate(cols):
                if c in sub.columns:
                    conf[:, j] = pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=np.float64)
            conf = np.nan_to_num(conf, nan=0.0).clip(0.0, 1.0)
    derived_from_prob = conf is None
    if derived_from_prob:
        conf = np.abs(np.where(keep, vals, 0.5) - 0.5) * 2.0

    min_conf = float(getattr(cfg, "labels_min_conf", 0.0))
    if min_conf > 0:
        keep = keep & (conf >= min_conf)

    w = np.where(keep, float(getattr(cfg, "labels_weight", 0.30)), 0.0)
    # Scale by the explicit confidence only.  Deriving a scale from |p - 0.5|
    # would silently down-weight every genuinely uncertain SOFT target, which is
    # the opposite of the point: a 0.6 target means "60% likely", not "40% less
    # important".  Without a confidence column the gate above is the only filter.
    if not derived_from_prob:
        w = w * conf

    wcol = str(getattr(cfg, "labels_weight_column", "") or "")
    if wcol and wcol in sub.columns:
        row_w = pd.to_numeric(sub[wcol], errors="coerce").to_numpy(dtype=np.float64)
        row_w = np.nan_to_num(row_w, nan=1.0).clip(min=0.0)
        w = w * row_w[:, None]

    return np.nan_to_num(vals, nan=0.0).astype(np.float32), w.astype(np.float32)


def attach_derived_rows(train_df, weights, derived, target_cols, cfg,
                        gold_ids=(), image_dir=None):
    """Append report-derived rows (fold = -1) and their per-cell weights."""
    if derived is None or len(derived) == 0:
        return train_df, weights

    sub = derived.copy()
    sub["StudyInstanceUID"] = sub["StudyInstanceUID"].astype(str)
    # A gold study is NEVER re-labelled from its report: ground truth wins.
    if len(gold_ids):
        before = len(sub)
        sub = sub[~sub["StudyInstanceUID"].isin(set(map(str, gold_ids)))]
        if before - len(sub):
            print(f"  derived: {before - len(sub)} rows dropped (study already gold)")
    sub = sub.drop_duplicates(subset="StudyInstanceUID").reset_index(drop=True)
    if image_dir:
        sub = drop_missing_images(sub, image_dir)
    if len(sub) == 0:
        return train_df, weights

    vals, w = derived_cell_weights(sub, target_cols, cfg)

    rows = pd.DataFrame({"StudyInstanceUID": sub["StudyInstanceUID"].values})
    for j, c in enumerate(target_cols):
        rows[c] = vals[:, j]
    rows["fold"] = -1                      # never equals a student fold
    rows["is_pseudo"] = False
    gcol = str(getattr(cfg, "gold_column", "is_gold"))
    rows[gcol] = False
    for col in train_df.columns:
        if col not in rows.columns:
            rows[col] = np.nan
    rows = rows[train_df.columns]

    combined = pd.concat([train_df, rows], ignore_index=True)
    weights = np.concatenate([np.asarray(weights, dtype=np.float32), w], axis=0)

    live = int((w > 0).sum())
    soft = int(((vals > 1e-6) & (vals < 1 - 1e-6) & (w > 0)).sum())
    print(f"  + {len(rows)} report-derived studies (fold=-1, train only): "
          f"{live}/{w.size} cells supervised, {soft} of them SOFT, "
          f"weight {float(getattr(cfg, 'labels_weight', 0.30))} vs gold "
          f"{float(getattr(cfg, 'gold_weight', 1.0))}")
    return combined, weights


def apply_gold_weight(train_df, weights, cfg):
    """Scale the gold block by cfg.gold_weight (1.0 = unchanged)."""
    gw = float(getattr(cfg, "gold_weight", 1.0))
    if gw == 1.0:
        return weights
    w = np.asarray(weights, dtype=np.float32).copy()
    gcol = str(getattr(cfg, "gold_column", "is_gold"))
    if gcol in train_df.columns:
        mask = _truthy(train_df[gcol]).to_numpy()
    elif "is_pseudo" in train_df.columns:
        mask = ~_truthy(train_df["is_pseudo"]).to_numpy()
    else:
        mask = np.ones(len(train_df), dtype=bool)
    w[mask] *= gw
    print(f"  gold cells scaled by gold_weight={gw}")
    return w


def enforce_gold_only_validation(valid_df, cfg, fold):
    """Validation is scored against GROUND TRUTH only -- checked, not assumed.

    A pseudo row in the validation split scores the student against its own
    teacher; a report-derived row scores it against a keyword matcher.  Both
    move val AUC without moving generalisation, which is worse than useless
    because it is the number that drives checkpoint selection.
    """
    if "is_pseudo" in valid_df.columns:
        n_pseudo = int(_truthy(valid_df["is_pseudo"]).sum())
        if n_pseudo:
            raise SystemExit(
                f"FATAL: {n_pseudo} pseudo-labelled rows reached the validation "
                f"split of fold {fold}. Pseudo/derived rows must carry fold=-1.")

    gcol = str(getattr(cfg, "gold_column", "is_gold"))
    if gcol not in valid_df.columns:
        return valid_df                    # no provenance column: all rows gold
    is_gold = _truthy(valid_df[gcol])
    n_bad = int((~is_gold).sum())
    if not n_bad:
        return valid_df
    if not bool(getattr(cfg, "val_gold_only", True)):
        print(f"  WARNING: val_gold_only=False -- {n_bad} NON-GOLD rows are in "
              f"the fold {fold} validation split. Every metric below measures "
              f"agreement with derived labels, not with ground truth.")
        return valid_df
    print(f"  validation: dropped {n_bad} non-gold rows from fold {fold} "
          f"-> {int(is_gold.sum())} gold studies")
    return valid_df[is_gold].reset_index(drop=True)


class WeightedDataset(torch.utils.data.Dataset):
    """Wrap RSNADataset so each item carries its per-cell loss weights.

    RSNADataset (owned by another module) yields ``(image, labels)`` and cannot
    be changed from here, so the weights ride alongside as a third element.

    Laterality caveat: RSNADataset may mirror a sample and permute its labels
    internally, and we cannot observe that draw from outside.  If mirroring is
    live we symmetrise the weights across each medial/lateral pair so the
    permutation leaves them invariant -- otherwise a swapped label would be
    scored against the other column's weight.
    """

    def __init__(self, base, weights, warn=True):
        self.base = base
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != len(base):
            raise ValueError(f"weights has {w.shape[0]} rows for {len(base)} samples")
        perm = getattr(base, "lateral_swap_perm", None)
        if (float(getattr(base, "hflip_p", 0.0)) > 0 and perm is not None
                and w.shape[1] == len(perm)):
            sym = np.minimum(w, w[:, list(perm)])
            if warn and not np.allclose(sym, w):
                print("NOTE: laterality mirror is active; per-cell pseudo weights "
                      "symmetrised across medial/lateral pairs so the label "
                      "permutation cannot desynchronise them.")
            w = sym
        self.weights = w

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        image, labels = item[0], item[1]
        return image, labels, torch.from_numpy(self.weights[idx].copy())


# ══════════════════════════════════════════════════════════════════════════
# GPU utilisation: batch sizing, LR scaling, warmup
#
# THE ARITHMETIC (verified against src/model.py's measured tables, which this
# module only READS).  effnet_b0 @224, in_channels=3, 4,025,481 params:
#
#   fixed  = params * 18 B (fp32 master + fp32 grad + AdamW m/v + fp16 autocast
#            cache) + 0.9 GiB workspace                        = 0.968 GiB
#   act    = 43.10 MiB/sample without grad checkpointing
#            4.70 MiB/sample WITH it (MEASURED ratio 0.109)
#
#   batch    8 -> 1.30 GiB (  8.7% of a T4's 15.0 GiB usable)  <-- shipped
#   batch   64 -> 3.66 GiB ( 24.4%)
#   batch  128 -> 6.35 GiB ( 42.4%)
#   batch  256 -> 11.74 GiB( 78.3%)
#   largest batch at 85% of the card: 279
#
# So memory is not the constraint; the constraint is the TRAINING SPLIT.  Both
# caps are computed below and the smaller wins.
# ══════════════════════════════════════════════════════════════════════════

def optimiser_extras_gib(cfg, n_params):
    """VRAM held by train.py's OWN wrappers, which src/model.py cannot know about.

    ``estimate_training_memory_gb`` prices params + grads + AdamW moments +
    autocast cache + workspace.  It does NOT price the two full fp32 parameter
    copies this file adds and keeps ON THE MODEL'S DEVICE:

    * ``timm.utils.ModelEmaV2`` deep-copies the whole model (``use_ema``).
    * :class:`Lookahead` clones every parameter as its slow weights
      (``use_lookahead``); ``_slow`` is built from ``p.data.clone()``, so it
      lives wherever the model lives.

    SWA is excluded on purpose: train.py keeps that running mean on the CPU.

    For effnet_b0 (4.0M params) the two copies are 0.03 GiB and round away.
    For convnext_base (87.6M) they are 0.65 GiB -- enough to turn an 84.9%
    plan into an 89.3% one and OOM a T4 mid-run, which is exactly the failure
    this term exists to prevent.
    """
    copies = 0
    if bool(getattr(cfg, "use_ema", False)):
        copies += 1
    if bool(getattr(cfg, "use_lookahead", True)):
        copies += 1
    return copies * int(n_params) * 4 / (1024 ** 3)


def memory_cap_batch(cfg, device_key="t4", headroom=0.85, hard_max=1024):
    """Largest micro-batch that fits ``headroom`` of ``device_key``'s VRAM.

    Delegates every per-sample and per-parameter number to ``src.model``'s
    measured tables so this module never becomes a second, drifting source of
    truth, then adds :func:`optimiser_extras_gib` for the state that lives in
    this file.  Returns ``(batch, info_dict)``; ``batch`` is 0 if the planner
    could not be reached.
    """
    try:
        from src.model import (activation_mib_per_sample, count_parameters,
                               estimate_training_memory_gb, DEVICE_VRAM_GIB)
    except Exception as e:                                   # pragma: no cover
        return 0, {"error": f"src.model memory planner unavailable: {e}"}

    bb = getattr(cfg, "backbone", "tf_efficientnet_b0_ns")
    size = int(getattr(cfg, "image_size", 224))
    ckpt = bool(getattr(cfg, "grad_checkpointing", False) or
                getattr(cfg, "gradient_checkpointing", False))
    try:
        n_params = count_parameters(bb, size, int(getattr(cfg, "in_channels", 3)),
                                    int(getattr(cfg, "num_classes", 12)))
        act = activation_mib_per_sample(bb, size, ckpt)
    except Exception as e:                                   # pragma: no cover
        return 0, {"error": f"memory plan failed for {bb}@{size}: {e}"}

    vram = DEVICE_VRAM_GIB.get(str(device_key).lower(), 15.0)
    budget = vram * float(headroom)
    extras = optimiser_extras_gib(cfg, n_params)
    fixed = estimate_training_memory_gb(n_params, act, 0) + extras
    if act <= 0:
        cap = int(hard_max)
    else:
        cap = int(max(0.0, (budget - fixed)) * 1024.0 / act)
    cap = max(0, min(int(hard_max), cap))
    return cap, {
        "device": str(device_key), "vram_gib": vram, "budget_gib": budget,
        "fixed_gib": fixed, "act_mib_per_sample": act,
        "model_fixed_gib": fixed - extras, "extras_gib": extras,
        "grad_checkpointing": ckpt, "params_m": n_params / 1e6,
        "peak_gib_at_cap": fixed + act * cap / 1024.0,
        "n_params": n_params,
    }


class _CfgOverride:
    """Read-only view of ``cfg`` with a few attributes overridden.

    Used to re-price a memory plan under different assumptions (e.g. "what
    would this batch cost WITHOUT gradient checkpointing?") without mutating
    the real Config, which other code is reading concurrently.
    """

    def __init__(self, cfg, **overrides):
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_ov", overrides)

    def __getattr__(self, name):
        ov = object.__getattribute__(self, "_ov")
        if name in ov:
            return ov[name]
        return getattr(object.__getattribute__(self, "_cfg"), name)


def _cfg_without_ckpt(cfg):
    return _CfgOverride(cfg, grad_checkpointing=False, gradient_checkpointing=False)


def peak_gib_for_batch(info, batch):
    """Predicted peak GiB at ``batch`` from a :func:`memory_cap_batch` info dict."""
    if not info or "fixed_gib" not in info:
        return float("nan")
    return info["fixed_gib"] + info["act_mib_per_sample"] * int(batch) / 1024.0


def plan_gpu_utilisation(cfg, n_train, base_batch, base_accum):
    """Choose (batch_size, grad_accum_steps) for a training split of ``n_train``.

    Returns a dict with the decision and every intermediate, so the caller can
    print exactly why it landed where it did.

    TWO CAPS, MINIMUM WINS.

    * ``mem_cap`` -- what the card holds.  For effnet_b0@224 on a T4 this is
      279 at 85% headroom, i.e. it never binds.
    * ``data_cap`` -- ``n_train // auto_batch_target_steps``.  This is the one
      that matters here.  A batch larger than the split turns an epoch into a
      single full-batch step: zero gradient noise, BatchNorm estimating its
      statistics from one fixed set, and only ``epochs`` optimiser updates in
      the entire fold.  At 46 gold training studies and 8 target steps the cap
      is 5, floored to ``auto_batch_min`` = 8 -- which is precisely the batch
      size already shipped.  The 58-study regime is ALREADY correctly sized;
      the wasted 91% of the card is unrecoverable there because there is no
      data to put in it.

    Accumulation is folded away (steps -> 1) whenever the chosen batch fits,
    because accumulation only ever existed to fake a large batch under memory
    pressure and each extra micro-batch costs a measured ~8.8 ms of fixed
    overhead for nothing.
    """
    base_batch = max(1, int(base_batch))
    base_accum = max(1, int(base_accum))
    out = {
        "enabled": bool(getattr(cfg, "auto_batch_size", True)),
        "n_train": int(n_train),
        "base_batch": base_batch, "base_accum": base_accum,
        "base_effective": base_batch * base_accum,
    }
    if not out["enabled"]:
        out.update(batch_size=base_batch, grad_accum_steps=base_accum,
                   effective_batch=base_batch * base_accum,
                   reason="auto_batch_size=False; using the configured values")
        return out

    target_steps = max(1, int(getattr(cfg, "auto_batch_target_steps", 16)))
    bmin = max(1, int(getattr(cfg, "auto_batch_min", 8)))
    bmax = max(bmin, int(getattr(cfg, "auto_batch_max", 128)))
    headroom = float(getattr(cfg, "auto_batch_headroom", 0.85))
    dev_key = str(getattr(cfg, "auto_batch_device", "t4"))

    mem_cap, mem_info = memory_cap_batch(cfg, dev_key, headroom, hard_max=4096)
    out["mem_info"] = mem_info
    out["mem_cap"] = mem_cap

    # Data cap: never fewer than `target_steps` optimiser updates per epoch.
    data_cap = int(n_train) // target_steps
    out["data_cap_raw"] = data_cap
    data_cap = max(bmin, data_cap)
    out["data_cap"] = data_cap

    caps = [bmax, data_cap]
    if mem_cap > 0:
        caps.append(mem_cap)
    batch = max(1, min(caps))
    # Never exceed the split itself; a batch larger than the data is a lie.
    batch = min(batch, max(1, int(n_train)))

    binding = "auto_batch_max"
    if batch == data_cap and data_cap <= bmax and (mem_cap <= 0 or data_cap <= mem_cap):
        binding = ("training-split size (auto_batch_min floor)"
                   if data_cap == bmin and out["data_cap_raw"] < bmin
                   else "training-split size")
    elif mem_cap > 0 and batch == mem_cap:
        binding = f"VRAM on {dev_key} at {headroom:.0%} headroom"
    if batch == int(n_train) and int(n_train) < min(caps):
        binding = "training-split size (batch == whole split)"

    if bool(getattr(cfg, "auto_grad_accum", True)):
        accum = 1
    else:
        accum = base_accum
    out.update(batch_size=int(batch), grad_accum_steps=int(accum),
               effective_batch=int(batch) * int(accum),
               binding_constraint=binding,
               peak_gib=peak_gib_for_batch(mem_info, batch),
               util_pct=(100.0 * peak_gib_for_batch(mem_info, batch)
                         / max(1e-9, mem_info.get("vram_gib", 15.0))),
               updates_per_epoch=max(1, math.ceil(
                   math.ceil(int(n_train) / max(1, int(batch))) / max(1, int(accum)))),
               reason=f"min(auto_batch_max={bmax}, data_cap={data_cap}, "
                      f"mem_cap={mem_cap}) -> {batch}; binding = {binding}")
    return out


def scale_lr_for_batch(cfg, base_lr, effective_batch):
    """Apply the configured batch->LR scaling rule.  Returns ``(lr, info)``.

    THIS IS THE PART MOST LIKELY TO SILENTLY HURT AUC.  Raising the effective
    batch without raising the LR is not a neutral change: the schedule is fixed
    at ``cfg.epochs``, so k times the batch is 1/k the optimiser updates and the
    model simply travels less far.  Raising it too much is worse -- a pretrained
    backbone with single-digit update counts does not survive an 8x LR.

    ``sqrt`` is the default; see config.py for the derivation.  Every run prints
    the rule, the ratio and the resulting LR so this can never be an implicit
    change again.
    """
    rule = str(getattr(cfg, "lr_scale_rule", "sqrt")).lower()
    base_batch = max(1, int(getattr(cfg, "lr_base_batch", 16)))
    ratio = float(effective_batch) / float(base_batch)
    if rule == "linear":
        mult = ratio
    elif rule == "sqrt":
        mult = math.sqrt(ratio)
    elif rule in ("none", "off", ""):
        mult = 1.0
    else:
        print(f"  WARNING: unknown lr_scale_rule={rule!r}; treating as 'none'")
        rule, mult = "none", 1.0
    cap = float(getattr(cfg, "lr_scale_max", 4.0))
    capped = cap > 0 and mult > cap
    if capped:
        mult = cap
    return base_lr * mult, {
        "rule": rule, "base_lr": base_lr, "base_batch": base_batch,
        "effective_batch": int(effective_batch), "ratio": ratio,
        "multiplier": mult, "capped_at": cap if capped else None,
        "lr": base_lr * mult,
    }


#: Fewest optimiser updates OneCycleLR can be built for.  Its two phases must
#: both have strictly positive length, which needs 1 < pct_start*total < total;
#: no pct_start satisfies that below 3.  Under this, train.py runs a constant LR.
ONECYCLE_MIN_UPDATES = 3


def onecycle_pct_start(cfg, updates_per_epoch, total_updates):
    """Warmup fraction for OneCycleLR, derived from ``cfg.warmup_epochs``.

    ``warmup_epochs`` was declared in config.py and read by nothing: the
    scheduler hard-coded ``pct_start=0.1``.  Worse, a fraction is the wrong
    parameterisation when the batch grows -- k times the batch is 1/k the total
    updates, so a fixed 10% silently SHORTENS warmup exactly when a larger batch
    needs more of it.  ``warmup_min_steps`` is the absolute floor that fixes it.
    """
    total_updates = max(1, int(total_updates))
    want = float(getattr(cfg, "warmup_epochs", 1.0)) * max(1, int(updates_per_epoch))
    want = max(want, float(getattr(cfg, "warmup_min_steps", 3)))
    pct = want / total_updates
    # OneCycleLR builds phases with end_step = pct_start * total_steps - 1 for
    # the warmup and total_steps - 1 for the anneal, then divides by
    # (end_step - start_step).  Both phases must therefore have STRICTLY
    # positive length, i.e.
    #       1 < pct_start * total_steps < total_steps
    # Anything else raises ZeroDivisionError inside the constructor (which
    # steps once).  There is no pct_start that satisfies this for total_steps
    # < 3, so :data:`ONECYCLE_MIN_UPDATES` is the caller's guard.
    lo = 1.5 / total_updates                 # > 1 warmup step, with slack
    hi = min(0.5, (total_updates - 1.0) / total_updates)
    if lo <= hi:
        pct = min(max(pct, lo), hi)
    else:
        pct = hi                             # degenerate; caller must not use it
    return float(pct), int(round(pct * total_updates))


def should_drop_last(n_train, batch_size, device_type="cpu", max_drop_frac=0.05):
    """Whether the training loader should discard a ragged final batch.

    Two independent grounds, and they want different things:

    1. BatchNorm.  A trailing batch of 1 raises outright; 2 gives statistics
       that are pure noise.  Applies on every device, always.
    2. Shape stability.  On CUDA a ragged final batch is a second input shape,
       so ``torch.compile`` re-specialises for it and ``cudnn.benchmark``
       re-runs its algorithm search.  Worth avoiding -- but only when the
       remainder is a small slice of the epoch.  With 46 gold studies at batch
       8 the remainder is 13% of the fold and dropping it every epoch to save a
       one-off compile is a bad trade; with 637 report-labelled studies at
       batch 39 it is 2%, which is free.  ``shuffle=True`` means a different
       remainder is dropped each epoch, so nothing is permanently unseen.
    """
    n_train, batch_size = int(n_train), max(1, int(batch_size))
    n_full, rem = n_train // batch_size, n_train % batch_size
    if n_full < 1 or rem == 0:
        return False
    if rem <= 2:
        return True
    return (str(device_type) == "cuda" and n_full >= 4
            and rem / max(1, n_train) <= float(max_drop_frac))


def make_loader(dataset, cfg, shuffle, drop_last):
    nw = int(getattr(cfg, "num_workers", 4))
    try:
        nw = max(0, min(nw, (os.cpu_count() or 1)))
    except Exception:
        nw = 0
    kwargs = dict(batch_size=cfg.batch_size, shuffle=shuffle, drop_last=drop_last,
                  num_workers=nw, pin_memory=torch.cuda.is_available())
    if nw > 0:
        # prefetch_factor=4: keep 4 batches queued per worker so the GPU is
        # never starved waiting for DICOMs to be decoded and augmented.
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(dataset, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
# Train / validate
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, ema_model, loader, optimizer, scheduler, criterion,
                    scaler, device, cfg, use_amp, amp_dtype, clock, smoke):
    model.train()
    accum = max(1, int(getattr(cfg, "grad_accum_steps", 1)))
    # channels_last is applied to the model on CUDA only (see the fold setup);
    # the input has to follow or cuDNN transposes on every conv.
    channels_last = (device.type == "cuda")
    autocast_dev = "cuda" if use_amp else "cpu"
    use_mixup = float(getattr(cfg, "mixup_alpha", 0.0)) > 0
    mixup_alpha = float(getattr(cfg, "mixup_alpha", 0.4))
    label_smoothing = float(getattr(cfg, "label_smoothing", 0.0))

    total, n_batches, skipped = 0.0, 0, 0
    max_grad_norm_seen = 0.0
    pending = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="Training", leave=False)
    interrupted = False

    for step, batch in enumerate(pbar):
        if smoke and step >= 2:
            break
        if clock.expired(reserve_s=90):
            print("\nWall-clock budget reached mid-epoch -- stopping early.")
            interrupted = True
            break

        # A WeightedDataset (pseudo-label path) yields a third element holding
        # the per-cell loss weights; a plain RSNADataset yields two.
        images, labels = batch[0], batch[1]
        weights = batch[2] if len(batch) > 2 else None
        images = images.to(device, non_blocking=True)
        # The MODEL is converted to channels_last on CUDA, but the input was
        # left NCHW, so cuDNN inserted a layout transform in front of every
        # convolution and the NHWC tensor-core kernels were never selected.
        # Converting the input is what makes the model-side conversion pay.
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        if weights is not None:
            weights = weights.to(device, non_blocking=True)

        # Mixup / CutMix: alternate between the two strategies every batch.
        # CutMix pastes a rectangle (good for local structure); Mixup blends
        # globally (good for texture). Together they beat either alone.
        if use_mixup and torch.rand(1).item() > 0.5:
            if torch.rand(1).item() > 0.5:
                images, labels, weights = mixup_batch(images, labels, weights, mixup_alpha)
            else:
                images, labels, weights = cutmix_batch(images, labels, weights, alpha=mixup_alpha)

        # Label smoothing: pull hard 0/1 targets toward 0+eps / 1-eps.
        # Prevents overconfident predictions, especially on the 58-study set.
        if label_smoothing > 0:
            finite = torch.isfinite(labels)
            labels = torch.where(finite,
                                 labels.clamp(label_smoothing, 1.0 - label_smoothing),
                                 labels)

        with torch.amp.autocast(autocast_dev, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
        loss = (criterion(logits.float(), labels) if weights is None
                else criterion(logits.float(), labels, weight=weights))

        # ONE host<->device sync per micro-batch, not two.  `torch.isfinite(loss)`
        # in a bool context and `loss.item()` each force the GPU to drain before
        # the CPU can queue the next batch; at batch 8 that stall is a large
        # fraction of a MEASURED ~117 ms step.  Reading the scalar once and
        # testing it with math.isfinite keeps identical semantics for half the
        # stalls.
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            continue

        scaler.scale(loss / accum).backward()
        pending += 1
        total += loss_value
        n_batches += 1

        if pending == accum:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            max_grad_norm_seen = max(max_grad_norm_seen, float(grad_norm))
            
            # Gradient Centralization: smooths loss landscape before optimizer step
            centralize_gradient(model)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema_model is not None:
                ema_model.update(model)
            # OneCycleLR must step every optimizer update, not every epoch.
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass
            pending = 0

        # loss_value is already on the host -- calling loss.item() here would be
        # a THIRD sync per micro-batch purely to redraw a progress bar.
        pbar.set_postfix({"loss": f"{loss_value:.4f}",
                          "lr": f"{scheduler.get_last_lr()[0]:.2e}" if scheduler else ""})

    # Flush a partial accumulation window so its gradients are not thrown away.
    if pending > 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        max_grad_norm_seen = max(max_grad_norm_seen, float(grad_norm))

        # Gradient Centralization: smooths loss landscape before optimizer step
        centralize_gradient(model)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema_model is not None:
            ema_model.update(model)
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

    if skipped:
        print(f"  WARNING: skipped {skipped} non-finite loss batches")
    if max_grad_norm_seen > 0.8:
        print(f"  INFO: peak grad norm = {max_grad_norm_seen:.3f} (clipped at 1.0)")
    return total / max(n_batches, 1), interrupted


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, amp_dtype, smoke, val_tta=False,
             lateral_swap_perm=None, collect=None):
    """Return (val_loss, val_crit, val_auc, per_column_auc, n_rows).

    val_loss  -- masked BCE, always, regardless of the training criterion, so it
                 stays comparable across a focal-vs-bce A/B.
    val_crit  -- the training criterion's own value.
    val_auc   -- macro ROC-AUC over columns that have both classes present.
    val_tta   -- if True, average original + horizontal-flip predictions for a
                 ~+0.003-0.008 more stable val_auc estimate that leads to better
                 checkpoint selection.  Only 2x cost, worth it every epoch.
    """
    model.eval()
    autocast_dev = "cuda" if use_amp else "cpu"
    bce = MaskedBCEWithLogitsLoss()

    preds, labs = [], []
    for step, batch in enumerate(tqdm(loader, desc="Validating", leave=False)):
        if smoke and step >= 2:
            break
        # Validation is gold-only by construction, so there are no sample
        # weights to honour here; tolerate a 3-tuple anyway and ignore them.
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(autocast_dev, dtype=amp_dtype, enabled=use_amp):
            logits = model(images).float()
            if val_tta:
                # Original + horizontal flip.  A left-right mirror of a knee maps
                # medial <-> lateral, so the flipped logits MUST be permuted back
                # before averaging.  Averaging them unpermuted (the previous
                # behaviour) blends the Medial/Lateral Meniscus and Medial/Lateral
                # OA columns into each other, which corrupts 4 of the 12 val AUCs
                # and therefore corrupts checkpoint selection itself.  src/infer.py
                # already permutes; validation now matches it.
                logits_flip = model(torch.flip(images, dims=[-1])).float()
                if lateral_swap_perm is not None:
                    logits_flip = logits_flip[:, lateral_swap_perm]
                    logits = (logits + logits_flip) * 0.5
                else:
                    pass  # no permutation available -> do not mix mirrored logits
        preds.append(logits.cpu())
        labs.append(labels.float().cpu())

    if not preds:
        return float("inf"), float("inf"), float("nan"), {}, 0

    logits = torch.cat(preds, 0)
    targets = torch.cat(labs, 0)

    val_loss = float(bce(logits, targets).item())
    val_crit = float(criterion(logits, targets).item())

    if collect is not None:
        collect["logits"] = logits.numpy().copy()
        collect["targets"] = targets.numpy().copy()

    p = torch.sigmoid(logits).numpy()
    y = targets.numpy()
    per_col, aucs = {}, []
    for i in range(y.shape[1]):
        col_y, col_p = y[:, i], p[:, i]
        keep = np.isfinite(col_y)
        col_y, col_p = col_y[keep], col_p[keep]
        if col_y.size == 0 or len(np.unique(col_y)) < 2:
            per_col[i] = None
            continue
        try:
            a = float(roc_auc_score(col_y, col_p))
        except ValueError:
            per_col[i] = None
            continue
        if np.isnan(a):
            per_col[i] = None
            continue
        per_col[i] = a
        aucs.append(a)

    val_auc = float(np.mean(aucs)) if aucs else float("nan")
    return val_loss, val_crit, val_auc, per_col, int(y.shape[0])


# ══════════════════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════════════════

class RunningWeightAverage:
    """Stochastic Weight Averaging with O(1) memory.

    torch.optim.swa_utils.AveragedModel keeps a second *live* model, which on a
    16GB T4 is a full extra copy of the weights in VRAM for the whole fold.  We
    only ever need the running mean, so we keep ONE fp32 copy on the CPU and
    fold each new epoch into it in place:  mu_n = mu_{n-1} + (w_n - mu_{n-1})/n.

    VRAM cost: zero.  Host RAM cost: one fp32 state_dict (21M params = 84 MB for
    EfficientNetV2-S).  Compute cost: one CPU copy per epoch, not per step.
    """

    def __init__(self):
        self.state = None
        self.n = 0

    def update(self, module):
        sd = base_module(module).state_dict()
        if self.state is None:
            self.state = {k: v.detach().to("cpu", copy=True).float()
                          if torch.is_floating_point(v) else v.detach().to("cpu", copy=True)
                          for k, v in sd.items()}
            self.n = 1
            return
        self.n += 1
        for k, v in sd.items():
            cur = self.state.get(k)
            v = v.detach().to("cpu")
            if cur is None or not torch.is_floating_point(v):
                # ints such as BatchNorm.num_batches_tracked: take the latest.
                self.state[k] = v.clone()
                continue
            cur.add_((v.float() - cur) / self.n)

    def load_into(self, module):
        """Copy the averaged weights into `module`, preserving its dtypes."""
        if self.state is None:
            return False
        target = base_module(module)
        tgt_sd = target.state_dict()
        casted = {k: (v.to(tgt_sd[k].dtype) if k in tgt_sd and torch.is_floating_point(v) else v)
                  for k, v in self.state.items()}
        target.load_state_dict(casted, strict=True)
        return True


def _has_batchnorm(module) -> bool:
    """SWA only needs a BN-statistics pass if the net actually has BatchNorm.

    EfficientNet* does (BatchNorm2d).  ConvNeXt / ViT use LayerNorm, whose
    statistics are computed per-sample at forward time, so the expensive
    update_bn pass is a no-op for them and is skipped.
    """
    return any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.SyncBatchNorm))
               for m in base_module(module).modules())


def dump_oof(path, ids, logits, targets, target_cols):
    """Persist out-of-fold predictions so the ensemble/soup can be *measured*.

    Without this the only way to combine folds is to assume rank-averaging
    helps.  With it, src/soup.py can compare a weight-averaged soup against the
    logit ensemble on real held-out labels before committing to either.
    """
    try:
        np.savez(path,
                 ids=np.asarray(ids, dtype=object),
                 logits=np.asarray(logits, dtype=np.float32),
                 targets=np.asarray(targets, dtype=np.float32),
                 target_cols=np.asarray(list(target_cols), dtype=object))
        print(f"  OOF predictions -> {path}")
    except Exception as e:
        print(f"  WARNING: could not write OOF predictions ({e})")


def save_best(model, path):
    """Save the pure module state (no wrapper prefixes) with high compression."""
    state = base_module(model).state_dict()
    torch.save(strip_compile_prefix(state), path)


def centralize_gradient(model):
    """Gradient Centralization (GC) - Yong et al. 2020.
    Operates directly on gradients by centralizing them to have zero mean.
    Acts as a powerful regularizer and improves generalization for Conv/Linear layers.
    """
    for p in model.parameters():
        if p.grad is None:
            continue
        if p.grad.dim() > 1:
            # Conv/Linear weights have dim > 1. Biases have dim = 1.
            p.grad.data.add_(-p.grad.data.mean(dim=tuple(range(1, p.grad.dim())), keepdim=True))


def save_last(path, model, ema_model, optimizer, scheduler, scaler, epoch, best, history, cfg_snapshot):
    # Omit optimizer/scheduler states to stay within Kaggle's 20 GB disk limit.
    state = {
        "model": strip_compile_prefix(base_module(model).state_dict()),
        "epoch": epoch,
        "best": best,
        "history": history,
        "cfg": cfg_snapshot,
    }
    if ema_model is not None:
        state["ema_model"] = strip_compile_prefix(base_module(ema_model.module).state_dict())
    # Use legacy pickle format: Kaggle's PyTorch has a known ZIP64 2 GB overflow
    # bug in the new zipfile serializer that corrupts saves of >2 GB dicts such as
    # DINOv2 (model=1.22 GB) + EMA (1.22 GB) = 2.44 GB.  The legacy format has no
    # size limit and is load-compatible with all torch >= 1.6.
    torch.save(state, path, _use_new_zipfile_serialization=False)


def try_load_last(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                       # torch < 2.0 has no weights_only kwarg
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"WARNING: could not read resume checkpoint {path}: {e}")
        return None


def warm_start(model, path):
    if not path:
        return False
    if not os.path.exists(path):
        print(f"WARNING: init_from='{path}' not found -- training from ImageNet weights.")
        return False
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = strip_compile_prefix(state)
    missing, unexpected = base_module(model).load_state_dict(state, strict=False)
    print(f"Warm-started from {path} "
          f"({len(missing)} missing / {len(unexpected)} unexpected tensors)")
    return True


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def better(metric_name, new, old):
    """LEGACY two-outcome comparison, kept for callers and tests that predate
    the resolution-aware policy below.  Equivalent to
    ``classify_score(metric_name, new, old, min_delta=0.0) is IMPROVED``."""
    if not np.isfinite(new):
        return False
    if not np.isfinite(old):
        return True
    return new > old if metric_name == "val_auc" else new < old


# ══════════════════════════════════════════════════════════════════════════
# Model selection: metric resolution, min_delta, and a THREE-outcome policy
#
# WHY THIS IS A SEPARATE BLOCK OF PURE FUNCTIONS
# ----------------------------------------------
# Selection used to be four lines buried in the epoch loop, which made it
# impossible to test without a training run and hid the defect that killed an
# earlier run in this project: macro ROC-AUC over 10-13 validation studies is a
# RANK statistic.  For a label with p positives and n negatives its value moves
# only in steps of 1/(p*n); macro-averaged over the K computable labels the
# smallest move a single swapped pair can produce is step/K.  Measured on the
# real data_subset/train_gold.csv folds (12 / 10 / 12 / 11 / 13 studies, all 12
# labels computable in every fold):
#
#     fold 0: 0.00296   fold 1: 0.00412   fold 2: 0.00307
#     fold 3: 0.00345   fold 4: 0.00267        (mean 1/(p*n) / K)
#
# The run that died early stopped on a 0.0003 "non-improvement" -- 9x to 14x
# BELOW the finest thing this metric can express -- while train and val loss
# were still falling monotonically.  It was stopping on numerical noise.
#
# Everything below is a pure function of a handful of floats plus one label
# matrix, so tests/test_train_selection.py drives it with synthetic metric
# sequences and never builds a model.  See docs/known-defects.md section 3.
# ══════════════════════════════════════════════════════════════════════════

IMPROVED = "improved"
WITHIN_RESOLUTION = "within_resolution"
REGRESSED = "regressed"
UNDECIDABLE = "undecidable"

# +1 = larger is better, -1 = smaller is better.
_METRIC_DIRECTION = {"val_auc": +1, "val_loss": -1, "val_crit": -1,
                     "train_loss": -1}


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def metric_direction(metric_name) -> int:
    """+1 if the metric is maximised, -1 if minimised.

    Anything whose name contains "auc" is maximised; everything else is treated
    as a loss.  The legacy ``better()`` special-cased only the exact string
    "val_auc", so e.g. a future "gold_auc" would have been *minimised* there.
    """
    name = str(metric_name)
    if name in _METRIC_DIRECTION:
        return _METRIC_DIRECTION[name]
    return +1 if "auc" in name.lower() else -1


def auc_macro_resolution(y, reduce="mean"):
    """Resolution of macro ROC-AUC on a validation label matrix.

    ``y`` is (N, L); NaN cells are ignored, cells >= 0.5 count as positive so
    soft targets degrade sensibly.  Returns ``(macro_step, info)``.

    For column c with p positives and n negatives, ROC-AUC = (#concordant
    pairs)/(p*n), so the value lives on a lattice of spacing 1/(p*n) and NO
    genuine change can be smaller than that.  The macro average over the K
    columns that are computable at all divides each step by K.

    ``reduce="mean"`` uses the mean per-column step (the formula prescribed in
    docs/known-defects.md section 3); ``reduce="min"`` uses the finest single
    step, i.e. the most permissive floor that still excludes sub-swap noise.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    steps, usable = [], []
    for j in range(y.shape[1]):
        col = y[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        p = int((col >= 0.5).sum())
        n = int(col.size - p)
        if p == 0 or n == 0:
            continue                      # single-class column: AUC undefined
        steps.append(1.0 / (p * n))
        usable.append(int(j))

    k = len(steps)
    info = {"per_column_step": {j: float(s) for j, s in zip(usable, steps)},
            "usable_columns": usable, "n_usable": k,
            "n_rows": int(y.shape[0]), "reduce": str(reduce)}
    if k == 0:
        # Every column single-class => macro AUC is NaN every epoch.  A zero
        # floor here is correct: the NaN fallback in CheckpointSelector, not a
        # min_delta, is what keeps a checkpoint on disk in that case.
        info.update(macro_step=float("nan"), finest_macro_step=float("nan"),
                    coarsest_macro_step=float("nan"))
        return 0.0, info
    agg = min(steps) if str(reduce) == "min" else float(np.mean(steps))
    info.update(macro_step=float(agg / k),
                finest_macro_step=float(min(steps) / k),
                coarsest_macro_step=float(max(steps) / k))
    return float(agg / k), info


def gold_studies_for_auc_resolution(y, target=0.005, n_folds=5):
    """How many GOLD studies would be needed for macro AUC to resolve `target`.

    Printed in the startup log so "the metric is too coarse" is answered with a
    number instead of a shrug.  Uses the observed prevalences: for N validation
    studies, p ~ N*rho and n ~ N*(1-rho), so step_c ~ 1/(N^2 rho(1-rho)) and the
    macro step is mean_c(step_c)/K.  Solving for N and scaling by n_folds gives
    the total gold panel size.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    inv = []
    for j in range(y.shape[1]):
        col = y[:, j][np.isfinite(y[:, j])]
        if col.size == 0:
            continue
        rho = float((col >= 0.5).mean())
        if 0.0 < rho < 1.0:
            inv.append(1.0 / (rho * (1.0 - rho)))
    if not inv or target <= 0:
        return None
    k = len(inv)
    n_val = math.sqrt(float(np.mean(inv)) / (k * float(target)))
    return int(math.ceil(n_val * max(1, int(n_folds))))


def loss_resolution(current_rows, reference_rows=None, k=1.0):
    """Resolution of a val-LOSS comparison: the SE of the *difference*.

    Between two epochs the validation set is IDENTICAL, so "is epoch t better
    than the incumbent?" is a PAIRED question.  The right noise scale is
    ``std(d_i)/sqrt(N)`` over the per-study loss differences d_i, not the
    standard error of either epoch's mean -- the study-to-study spread, which
    dominates with 11 studies, cancels in the pairing.

    With no reference vector yet (the first scored epoch) we fall back to HALF
    the marginal SEM: a deliberately loose stand-in that is never used for a
    decision that matters, because the first finite score always wins anyway.
    Returns 0.0 when there is nothing to measure, which reduces the policy to
    exact float comparison -- never to a silent "everything is a tie".
    """
    cur = np.asarray(current_rows, dtype=float).ravel() if current_rows is not None else None
    if cur is None or cur.size < 2 or not np.all(np.isfinite(cur)):
        return 0.0
    ref = np.asarray(reference_rows, dtype=float).ravel() if reference_rows is not None else None
    if ref is not None and ref.size == cur.size and np.all(np.isfinite(ref)):
        d = cur - ref
        se = float(np.std(d, ddof=1)) / math.sqrt(d.size)
    else:
        se = 0.5 * float(np.std(cur, ddof=1)) / math.sqrt(cur.size)
    if not np.isfinite(se) or se <= 0:
        return 0.0
    return float(max(0.0, float(k)) * se)


def classify_score(metric_name, new, best, min_delta=0.0):
    """One of IMPROVED / WITHIN_RESOLUTION / REGRESSED / UNDECIDABLE.

    Three outcomes instead of two is the whole point.  "Not better" used to
    cover both "measurably worse" and "the metric cannot tell these two models
    apart", and charging patience for the second is how a run gets killed by a
    0.0003 wobble.  UNDECIDABLE is the fourth: the metric was NaN, which is a
    statement about the metric, not about the model.

    With ``min_delta=0.0`` this is exactly the legacy ``better()``:
    non-finite new -> not IMPROVED; non-finite best -> IMPROVED; otherwise a
    strict inequality in the metric's own direction.
    """
    d = _as_float(min_delta)
    d = max(d, 0.0) if np.isfinite(d) else 0.0
    new = _as_float(new)
    if not np.isfinite(new):
        return UNDECIDABLE
    best = _as_float(best)
    if not np.isfinite(best):
        return IMPROVED                   # first finite score always wins
    gain = (new - best) * metric_direction(metric_name)
    if gain > d:
        return IMPROVED
    if gain < -d:
        return REGRESSED
    return WITHIN_RESOLUTION


def resolve_selection_metric(cfg, val_y=None):
    """Pure resolver: config + validation labels -> the metric actually used.

    ``cfg.selection_metric``:
      "auto"     -- use val_auc only when BOTH a single rank swap moves it by no
                    more than ``cfg.auc_resolution_target`` AND the fold has at
                    least ``cfg.auc_min_val_rows`` validation studies.  The
                    second gate matters: on 12 studies the SAMPLING error of an
                    AUC is ~0.1, two orders of magnitude above its 0.003
                    quantisation, so passing the resolution test alone would be
                    a false negative for "this metric is usable".  Nothing is
                    hard-coded to 58, so this starts preferring the competition
                    metric by itself once the panel is big enough (58 gold
                    studies -> val_loss; a 649-study panel -> val_auc).
      "val_loss" -- masked BCE over EVERY validation cell.  Continuous, defined
                    for every study, and the safe default at any panel size.
      "val_auc"  -- the competition metric, guarded by its own resolution.
      "val_crit" -- the training criterion's own value.

    Returns a dict; nothing here touches global state, so it is testable.
    """
    want = str(getattr(cfg, "selection_metric", "auto") or "auto").lower()
    reduce = str(getattr(cfg, "selection_min_delta_reduce", "mean"))
    target = float(getattr(cfg, "auc_resolution_target", 0.005))
    override = float(getattr(cfg, "selection_min_delta", -1.0))
    override = None if override < 0 else override

    auc_delta, info = (0.0, {"n_usable": 0, "macro_step": float("nan")})
    if val_y is not None:
        auc_delta, info = auc_macro_resolution(val_y, reduce=reduce)

    reason = "explicitly configured"
    if want == "auto":
        step = info.get("macro_step", float("nan"))
        n_rows = int(info.get("n_rows", 0))
        min_rows = int(getattr(cfg, "auc_min_val_rows", 100))
        fine_enough = bool(np.isfinite(step) and step <= target
                           and info.get("n_usable", 0) > 0)
        big_enough = n_rows >= min_rows
        if fine_enough and big_enough:
            metric, reason = "val_auc", (
                f"auto: macro-AUC step {step:.5f} <= target {target:g} on "
                f"{n_rows} validation studies (>= {min_rows}), so the "
                f"competition metric is resolved well enough to drive control flow")
        else:
            need = gold_studies_for_auc_resolution(
                val_y, target, int(getattr(cfg, "n_folds", 5))) if val_y is not None else None
            shown = "n/a" if not np.isfinite(step) else f"{step:.5f}"
            why = []
            if not fine_enough:
                why.append(f"macro-AUC step {shown} > target {target:g}"
                           + (f" (~{need} gold studies would reach it)" if need else ""))
            if not big_enough:
                why.append(f"only {n_rows} validation studies, below the "
                           f"{min_rows} needed for an AUC whose sampling error "
                           f"is smaller than the effects being compared")
            metric, reason = "val_loss", (
                "auto: " + "; ".join(why) +
                " -- selecting on masked BCE over every validation cell instead")
    else:
        metric = want

    min_delta = (override if override is not None
                 else (auc_delta if metric_direction(metric) > 0 else 0.0))
    return {"metric": metric, "reason": reason, "min_delta": float(min_delta),
            "auc_min_delta": float(auc_delta), "auc_info": info,
            "override": override, "direction": metric_direction(metric)}


class CheckpointSelector:
    """Three-outcome model selection + early stopping, resolution-aware.

    Per epoch:
      IMPROVED          -- beat the incumbent by MORE than the metric's own
                           resolution: save the checkpoint, reset patience.
      WITHIN_RESOLUTION -- the metric cannot distinguish the two models.  No
                           vote is cast: the incumbent is kept and patience is
                           NOT charged.  This is the direct fix for the 0.0003
                           kill; a sub-resolution wobble is not evidence.
      REGRESSED         -- measurably worse: charge patience.
      UNDECIDABLE       -- the metric was NaN this epoch.

    Because ties are patience-neutral, RAISING min_delta can only ever DELAY
    early stopping, never hasten it.  A permanent plateau is still bounded by
    ``stale_patience`` (ties + regressions) and by cfg.epochs.

    NaN SAFETY (the "no checkpoint was ever written" defect).  ``NaN > x`` is
    False for every x, so a selector that only asks "is the new score better?"
    writes nothing at all when the metric is NaN every epoch -- exactly what a
    degenerate gold subset produces (every validation column single-class =>
    macro AUC over an empty list => NaN).  Three independent guards:
      1. UNDECIDABLE is its own outcome, so NaN never reads as "not improved".
      2. If the primary metric has NEVER been finite on this fold, selection
         falls back to ``fallback_metric`` (val_loss: masked BCE over every
         validation cell, finite whenever any label exists).  The fallback is
         only ever entered before the first finite primary score, so the two
         criteria can never be mixed mid-fold.
      3. ``checkpoint_exists=False`` forces a save whatever the metric did.
    Guard 3 alone already guarantees "there is always a checkpoint"; guards 1
    and 2 make that checkpoint a *chosen* one rather than merely the first.

    ``legacy=True`` reproduces the pre-fix arithmetic bit for bit: min_delta 0,
    no fallback, and ties charged to patience like regressions.
    """

    def __init__(self, metric="val_loss", auc_min_delta=0.0, loss_delta_k=1.0,
                 override_min_delta=None, fallback_metric="val_loss",
                 patience=8, min_epochs=1, stale_patience=None, legacy=False):
        self.metric = str(metric)
        self.legacy = bool(legacy)
        self.auc_min_delta = 0.0 if self.legacy else max(0.0, float(auc_min_delta))
        self.loss_delta_k = 0.0 if self.legacy else max(0.0, float(loss_delta_k))
        self.override = None if self.legacy else override_min_delta
        self.fallback_metric = None if self.legacy else str(fallback_metric)
        self.patience_limit = int(patience)
        self.min_epochs = int(min_epochs)
        self.stale_limit = (int(stale_patience) if stale_patience is not None
                            else max(2 * self.patience_limit, self.patience_limit + 2))

        self.reset()

    def reset(self):
        """Return to the pre-first-epoch state (used when a resume fails)."""
        direction = metric_direction(self.metric)
        self.best = -math.inf if direction > 0 else math.inf
        self.best_fallback = math.inf     # fallback is always a loss
        self.patience = 0
        self.stale = 0
        self.n_epochs = 0
        self.saw_finite_primary = False
        self.ref_rows = None              # per-study loss of the incumbent
        self.last = None

    # -- resolution ------------------------------------------------------
    def delta_for(self, metric_name, per_row_loss=None):
        if self.override is not None:
            return max(0.0, float(self.override))
        if metric_direction(metric_name) > 0:
            return self.auc_min_delta
        return loss_resolution(per_row_loss, self.ref_rows, self.loss_delta_k)

    # -- one epoch -------------------------------------------------------
    def step(self, metrics, checkpoint_exists=True, per_row_loss=None):
        """Decide what to do with this epoch's metrics.  Pure w.r.t. the model.

        ``metrics`` is the same dict the epoch loop logs; ``per_row_loss`` is
        the per-validation-study masked BCE vector (optional, used only to size
        the loss resolution).  Returns a decision dict.
        """
        self.n_epochs += 1
        primary = _as_float(metrics.get(self.metric, float("nan")))
        primary_finite = bool(np.isfinite(primary))

        use_fallback = (not primary_finite and not self.saw_finite_primary
                        and self.fallback_metric is not None
                        and self.fallback_metric != self.metric)
        if use_fallback:
            name = self.fallback_metric
            score = _as_float(metrics.get(name, float("nan")))
            incumbent = self.best_fallback
        else:
            name, score, incumbent = self.metric, primary, self.best

        delta = self.delta_for(name, per_row_loss)
        status = classify_score(name, score, incumbent, delta)
        forced = (not checkpoint_exists) and status != IMPROVED
        save = (status == IMPROVED) or (not checkpoint_exists)

        if status == IMPROVED:
            if use_fallback:
                self.best_fallback = score
            else:
                self.best = score
            self.patience = 0
            self.stale = 0
        elif self.legacy:
            # Two-outcome legacy arithmetic: anything that is not an
            # improvement charges patience, and a forced save clears it.
            if save:
                self.patience = 0
            else:
                self.patience += 1
            self.stale = self.patience
        elif status == REGRESSED:
            self.patience += 1
            self.stale += 1
        else:                              # WITHIN_RESOLUTION / UNDECIDABLE
            self.stale += 1

        if primary_finite:
            self.saw_finite_primary = True
        if save and per_row_loss is not None:
            rows = np.asarray(per_row_loss, dtype=float).ravel()
            self.ref_rows = rows.copy() if rows.size else None

        self.last = {
            "metric": name, "score": score, "incumbent": incumbent,
            "status": status, "min_delta": float(delta), "save": bool(save),
            "forced": bool(forced), "fallback": bool(use_fallback),
            "best": self.best, "best_fallback": self.best_fallback,
            "patience": self.patience, "stale": self.stale,
        }
        return dict(self.last)

    # -- early stopping --------------------------------------------------
    def stop_reason(self, epoch_index):
        """None, or a string naming why training should stop after this epoch."""
        if epoch_index + 1 < self.min_epochs:
            return None
        if self.patience_limit > 0 and self.patience >= self.patience_limit:
            return "early_stopping"
        if not self.legacy and self.stale_limit > 0 and self.stale >= self.stale_limit:
            return "early_stopping_plateau"
        return None

    def describe(self, decision):
        """One line, resolution included, so the log can never hide the scale."""
        d = decision
        gap = (_as_float(d["score"]) - _as_float(d["incumbent"]))
        gap_s = "n/a" if not np.isfinite(gap) else f"{gap:+.5f}"
        score_s = "nan" if not np.isfinite(_as_float(d["score"])) else f"{d['score']:.5f}"
        tag = "FALLBACK " if d["fallback"] else ""
        return (f"  select[{tag}{d['metric']}]: {score_s} vs best "
                f"{_as_float(d['incumbent']):.5f} ({gap_s}, resolution "
                f"{d['min_delta']:.5f}) -> {d['status']}"
                + ("  [forced: no checkpoint on disk]" if d["forced"] else "")
                + f" | patience {d['patience']}/{self.patience_limit}"
                f" stale {d['stale']}/{self.stale_limit}")


def per_row_masked_bce(logits, targets):
    """Per-validation-study mean masked BCE, used to size the loss resolution.

    Rows with no finite target contribute NaN and are dropped by the caller.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if logits.ndim == 1:
        logits, targets = logits[:, None], targets[:, None]
    m = np.isfinite(targets)
    safe = np.where(m, targets, 0.0)
    # log(1+exp(-|z|)) + max(z,0) - z*y  is the stable BCE-with-logits form.
    z = logits
    loss = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0) - z * safe
    loss = np.where(m, loss, 0.0)
    denom = m.sum(axis=1)
    out = np.full(loss.shape[0], np.nan, dtype=np.float64)
    nz = denom > 0
    out[nz] = loss[nz].sum(axis=1) / denom[nz]
    return out[np.isfinite(out)]


def _make_param_groups(model, cfg):
    """Layer-wise Learning Rate Decay (LLRD) + no weight decay on norms/biases.

    Two improvements in one:
    1. LLRD: backbone layers get lr * llrd_factor (default 0.1), the head gets
       the full lr.  Prevents destroying pre-trained ViT weights in epoch 1.
    2. No-WD on norms & biases: LayerNorm/BatchNorm scale+bias and all bias
       parameters are excluded from weight decay.  Weight-decaying these causes
       unnecessary regularisation of 1-D affine transforms and hurts AUC.
    """
    llrd = float(getattr(cfg, "llrd_factor", 0.1))
    raw = base_module(model)

    # --- separate backbone vs head ---
    backbone_wd, backbone_no_wd = [], []
    head_wd, head_no_wd = [], []

    no_wd_names = {"bias"}
    no_wd_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                   nn.GroupNorm, nn.InstanceNorm2d)

    for module_name, module in raw.named_modules():
        is_norm = isinstance(module, no_wd_types)
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            no_wd = is_norm or any(n in param_name for n in no_wd_names)
            is_backbone = any(k in full_name
                              for k in ("backbone", "patch_embed", "blocks", "encoder"))
            if is_backbone:
                (backbone_no_wd if no_wd else backbone_wd).append(param)
            else:
                (head_no_wd if no_wd else head_wd).append(param)

    groups = []
    # cfg.weight_decay was previously ignored here (hard-coded 1e-4), so raising
    # it in the config or on the command line silently did nothing.
    wd = float(getattr(cfg, "weight_decay", 1e-4))
    if backbone_wd:    groups.append({"params": backbone_wd,    "lr": cfg.lr * llrd, "weight_decay": wd,  "name": "backbone_wd"})
    if backbone_no_wd: groups.append({"params": backbone_no_wd, "lr": cfg.lr * llrd, "weight_decay": 0.0, "name": "backbone_no_wd"})
    if head_wd:        groups.append({"params": head_wd,        "lr": cfg.lr,        "weight_decay": wd,  "name": "head_wd"})
    if head_no_wd:     groups.append({"params": head_no_wd,     "lr": cfg.lr,        "weight_decay": 0.0, "name": "head_no_wd"})
    if not groups:
        groups = [{"params": list(raw.parameters()), "lr": cfg.lr}]

    n_params = sum(p.numel() for g in groups for p in g["params"])
    print(f"LLRD: backbone lr={cfg.lr * llrd:.2e}  head lr={cfg.lr:.2e}  wd={wd:g}  "
          f"({len(backbone_wd)+len(backbone_no_wd)} backbone / "
          f"{len(head_wd)+len(head_no_wd)} head params, {n_params/1e6:.1f}M total)")
    return groups


def mixup_batch(images, labels, weights, alpha=0.4):
    """Mixup augmentation (Zhang et al. 2018) applied at the batch level.

    Blends pairs of images and their soft labels.  With alpha=0.4 the mixing
    coefficient Beta(0.4, 0.4) rarely goes below 0.6, so one image always
    dominates -- the model still sees recognisable anatomy.
    Mixup is consistently +0.01-0.03 AUC in medical imaging competitions.
    """
    lam = float(np.random.beta(alpha, alpha))
    bs = images.size(0)
    idx = torch.randperm(bs, device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[idx]
    # Labels: blend finite targets; preserve NaN (masked) status.
    labels_a, labels_b = labels, labels[idx]
    # Where both are finite blend normally; where one is NaN keep the other.
    both_finite = torch.isfinite(labels_a) & torch.isfinite(labels_b)
    only_a = torch.isfinite(labels_a) & ~torch.isfinite(labels_b)
    only_b = ~torch.isfinite(labels_a) & torch.isfinite(labels_b)
    mixed_labels = torch.full_like(labels_a, float("nan"))
    mixed_labels = torch.where(both_finite,
                               lam * labels_a + (1 - lam) * labels_b,
                               mixed_labels)
    mixed_labels = torch.where(only_a, labels_a, mixed_labels)
    mixed_labels = torch.where(only_b, labels_b, mixed_labels)
    # Mix weights if present.
    if weights is not None:
        mixed_weights = lam * weights + (1.0 - lam) * weights[idx]
    else:
        mixed_weights = None
    return mixed_images, mixed_labels, mixed_weights


def cutmix_batch(images, labels, weights, alpha=1.0):
    """CutMix augmentation (Yun et al. 2019) -- paste a random rectangle from
    one image onto another and interpolate labels by the area ratio.

    CutMix and Mixup are complementary: Mixup blends globally (good for texture),
    CutMix blends locally (good for structure). Alternating them 50/50 beats either
    alone by ~+0.01 AUC on medical imaging benchmarks.
    """
    lam = float(np.random.beta(alpha, alpha))
    bs, _, H, W = images.shape
    idx = torch.randperm(bs, device=images.device)

    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = max(1, int(H * cut_ratio))
    cut_w = max(1, int(W * cut_ratio))
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(0, cx - cut_w // 2); x2 = min(W, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2); y2 = min(H, cy + cut_h // 2)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    # Actual lambda from the true cut area
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

    # Label interpolation identical to mixup_batch
    labels_a, labels_b = labels, labels[idx]
    both_finite = torch.isfinite(labels_a) & torch.isfinite(labels_b)
    only_a = torch.isfinite(labels_a) & ~torch.isfinite(labels_b)
    only_b = ~torch.isfinite(labels_a) & torch.isfinite(labels_b)
    mixed_labels = torch.full_like(labels_a, float("nan"))
    mixed_labels = torch.where(both_finite,
                               lam * labels_a + (1 - lam) * labels_b,
                               mixed_labels)
    mixed_labels = torch.where(only_a, labels_a, mixed_labels)
    mixed_labels = torch.where(only_b, labels_b, mixed_labels)
    if weights is not None:
        mixed_weights = lam * weights + (1.0 - lam) * weights[idx]
    else:
        mixed_weights = None
    return mixed, mixed_labels, mixed_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--set", nargs="+", help="Overrides for Config, k=v")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    os.makedirs(cfg.models_dir, exist_ok=True)
    state_dir = cfg.state_dir or cfg.models_dir
    os.makedirs(state_dir, exist_ok=True)

    seed_everything(cfg.seed)
    clock = WallClock(cfg.time_budget_min, mode=getattr(cfg, "time_budget_clock", "wall"))
    if cfg.time_budget_min > 0:
        print(f"Wall-clock budget: {cfg.time_budget_min:.1f} min "
              f"({'monotonic, sleep-EXCLUSIVE' if clock.mode == 'monotonic' else 'real time, sleep-inclusive'})")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp, amp_dtype, needs_scaler = resolve_amp(cfg, device)
    print(f"Device: {device} | AMP: {'off' if not use_amp else str(amp_dtype).split('.')[-1]}"
          f" | GradScaler: {'on' if needs_scaler else 'off'}")

    # TF32: on Ampere (A100) allows matrix multiplications to run at ~2x speed
    # with negligible accuracy loss (TF32 = 10 mantissa bits vs FP32's 23).
    # Safe to enable unconditionally; ignored on older GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device.type == "cuda":
        # Enable Flash Attention / Memory Efficient Attention for scaled dot product
        # (used natively by timm's DINOv2 / ConvNeXt implementations in PyTorch 2.0+).
        # Massive memory savings and speedup for self-attention.
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        except Exception:
            pass

    # Fixed input sizes mean cuDNN can benchmark & cache the fastest conv algorithm
    # per layer -- free 5-10% throughput with zero code changes.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Data ─────────────────────────────────────────────────────────────
    df, target_cols = load_dataframe(cfg)
    if target_cols:
        import src.kaggle_data as kd
        kd.KNEE_TARGETS = target_cols
        cfg.num_classes = len(target_cols)
        print(f"Targets ({len(target_cols)}): {target_cols}")
        if list(target_cols) != list(KNEE_TARGETS) and int(getattr(cfg, "num_workers", 0)) > 0:
            # kaggle_data.KNEE_TARGETS is monkey-patched in this process; a
            # 'spawn' DataLoader worker re-imports the module and would silently
            # revert to the built-in list. Force single-process loading instead.
            print("WARNING: detected targets differ from kaggle_data.KNEE_TARGETS; "
                  "forcing num_workers=0 so worker processes cannot revert them.")
            cfg.num_workers = 0

    df = filter_labelled(df, target_cols, cfg)
    image_dir = os.path.join(cfg.data_dir, "train_series")
    df = drop_missing_images(df, image_dir)
    df = assign_folds(df, target_cols, cfg)

    if len(df) == 0:
        raise SystemExit("No usable training rows after filtering -- check data_dir / train_csv.")

    # Every row that survived the gold pipeline IS gold.  Stamping the column
    # here (after target auto-detection, which ignores it by name) means the
    # gold-only validation guard has provenance to check even when the CSV did
    # not ship one.
    gold_col = str(getattr(cfg, "gold_column", "is_gold"))
    if gold_col not in df.columns:
        df[gold_col] = True
    else:
        df[gold_col] = _truthy(df[gold_col])

    # Report-derived weak labels (src/labels.py).  Loaded here but attached PER
    # FOLD below, after the gold fold map is fixed, so derived rows can never
    # perturb the fold assignment they are supposed to stay out of.
    derived = load_derived_labels(cfg, target_cols)
    if derived is None:
        # cfg.labels_csv named nothing usable; fall back to generating the table
        # in-process from the reports when cfg.use_report_labels asks for it.
        derived = build_derived_from_reports(cfg, target_cols, gold_df=df)

    # Resolution-scoped cache: kaggle_data's cache key omits image_size, so a
    # shared directory would feed a 224px cache into a 512px phase.
    cache_dir = os.path.join(cfg.cache_dir, f"sz{cfg.image_size}_ch{cfg.in_channels}")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"DICOM cache: {cache_dir}")

    criterion, crit_name = build_criterion(cfg)
    criterion = criterion.to(device)
    print(f"Loss: {crit_name} | selection metric: {cfg.selection_metric}")

    # Label permutation induced by a left-right mirror, used by flip-TTA during
    # validation so mirrored logits are un-mirrored before they are averaged.
    val_swap_perm, _swap_pairs, _swap_blockers = build_lateral_swap(
        target_cols if target_cols else KNEE_TARGETS)

    # ── Pseudo-labels ────────────────────────────────────────────────────
    # Loaded AFTER the gold folds are fixed, so the recorded teacher fold map
    # can be checked against the map this run will actually validate under.
    gold_ids = set(df["StudyInstanceUID"].astype(str))
    gold_fold_map = dict(zip(df["StudyInstanceUID"].astype(str), df["fold"].astype(int)))
    pseudo = load_pseudo_bundle(cfg, target_cols, gold_fold_map)

    precache_df = df[["StudyInstanceUID"]].astype(str)
    for _extra_frame in (pseudo.frame if pseudo is not None else None, derived):
        if _extra_frame is None or len(_extra_frame) == 0:
            continue
        extra = (_extra_frame[["StudyInstanceUID"]].astype(str)
                 .drop_duplicates()
                 .query("StudyInstanceUID not in @gold_ids"))
        precache_df = pd.concat([precache_df, extra],
                                ignore_index=True).drop_duplicates()
    precache_dataset(precache_df, image_dir, cfg, cache_dir=cache_dir)

    # ── GPU-utilisation baselines ────────────────────────────────────────
    # Captured ONCE, outside the fold loop.  Auto batch sizing and LR scaling
    # both write back into `cfg` (because _make_param_groups, make_loader and
    # build_model all read it), so without an immutable baseline fold 1 would
    # scale fold 0's already-scaled values and the LR would compound per fold.
    base_batch_size = int(cfg.batch_size)
    base_accum_steps = int(getattr(cfg, "grad_accum_steps", 1))
    base_lr = float(cfg.lr)
    # TWO fields turn checkpointing on and they reach the model by different
    # routes: `grad_checkpointing` is passed to build_model(), while
    # `gradient_checkpointing` drives the explicit set_grad_checkpointing() call
    # below.  Either one alone enables it, so the auto-disable has to clear both
    # or the plan and the model disagree about what is running.
    base_grad_ckpt_a = bool(getattr(cfg, "gradient_checkpointing", False))
    base_grad_ckpt_b = bool(getattr(cfg, "grad_checkpointing", False))
    base_grad_ckpt = base_grad_ckpt_a or base_grad_ckpt_b

    # ── Folds ────────────────────────────────────────────────────────────
    for fold in args.folds:
        print(f"\n=== Fold {fold} ===")
        cfg.batch_size = base_batch_size
        cfg.grad_accum_steps = base_accum_steps
        cfg.lr = base_lr
        cfg.gradient_checkpointing = base_grad_ckpt_a
        cfg.grad_checkpointing = base_grad_ckpt_b
        gold_train_df = df[df["fold"] != fold].reset_index(drop=True)
        # Validation is GOLD ONLY, always. A pseudo row in the validation split
        # would score the student against its own teacher's guesses, which moves
        # val AUC without moving generalisation. `df` holds only gold rows here
        # (pseudo rows are attached to the training frame below and carry
        # fold == -1), so this is true by construction; assert it anyway.
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        valid_df = enforce_gold_only_validation(valid_df, cfg, fold)

        train_df, sample_weights = attach_pseudo_rows(
            gold_train_df, pseudo, fold, target_cols, cfg, image_dir=image_dir)
        n_gold_train = len(gold_train_df)
        n_pseudo_train = len(train_df) - n_gold_train
        # Report-derived rows come last so their weight block lines up with the
        # rows appended to the frame.  fold = -1 keeps them out of every split.
        train_df, sample_weights = attach_derived_rows(
            train_df, sample_weights, derived, target_cols, cfg,
            gold_ids=gold_ids, image_dir=image_dir)
        sample_weights = apply_gold_weight(train_df, sample_weights, cfg)
        n_derived_train = len(train_df) - n_gold_train - n_pseudo_train
        print(f"train={len(train_df)} ({n_gold_train} gold + {n_pseudo_train} pseudo "
              f"+ {n_derived_train} derived) valid={len(valid_df)} (gold only)")
        if len(train_df) == 0 or len(valid_df) == 0:
            print("  skipping (empty split)")
            continue

        # The validation label matrix decides the selection metric's resolution,
        # so it is read from the GOLD validation frame -- never from anything a
        # keyword matcher produced.
        val_y = valid_df[[c for c in target_cols if c in valid_df.columns]].to_numpy(dtype=float)
        sel = resolve_selection_metric(cfg, val_y)
        _ainfo = sel["auc_info"]
        print(f"  selection metric: {sel['metric']}  ({sel['reason']})")
        print(f"  macro-AUC resolution on this fold: "
              f"{_ainfo.get('macro_step', float('nan')):.5f} "
              f"over {_ainfo.get('n_usable', 0)}/{len(target_cols)} computable labels, "
              f"n={len(valid_df)} studies "
              f"[finest {_ainfo.get('finest_macro_step', float('nan')):.5f}, "
              f"coarsest {_ainfo.get('coarsest_macro_step', float('nan')):.5f}]")

        train_ds = RSNADataset(train_df, image_dir, cfg, is_train=True, cache_dir=cache_dir)
        valid_ds = RSNADataset(valid_df, image_dir, cfg, is_train=False, cache_dir=cache_dir)
        if not np.allclose(np.asarray(sample_weights, dtype=np.float32), 1.0):
            train_ds = WeightedDataset(train_ds, sample_weights)
            print(f"  per-sample loss weights active "
                  f"(min {float(np.min(sample_weights)):.3f}, "
                  f"max {float(np.max(sample_weights)):.3f})")
        # ── GPU utilisation: checkpointing, batch size, accumulation, LR ──
        # batch_size=8 on a T4 is 1.30 of 15.0 GiB (8.7%) and spends a MEASURED
        # ~37% of each step on per-step overhead the batch would amortise.  The
        # binding constraint, though, is the split: see plan_gpu_utilisation.
        #
        # Checkpointing is decided FIRST, because it changes the per-sample
        # activation cost (43.10 -> 4.70 MiB) and therefore the memory cap the
        # batch plan is about to use.  Deciding it after the plan would print a
        # peak-VRAM figure for a configuration that is no longer the one running.
        # MEASURED cost of checkpointing: +28-51% wall clock at batch 64
        # (scripts/bench_gpu_util.py).  Saving 0.3 GiB of a 15.0 GiB card for a
        # third of the run time is a bad trade.
        if bool(getattr(cfg, "auto_grad_checkpointing", True)) and base_grad_ckpt:
            _target_steps = max(1, int(getattr(cfg, "auto_batch_target_steps", 16)))
            _want = min(int(getattr(cfg, "auto_batch_max", 128)),
                        max(int(getattr(cfg, "auto_batch_min", 8)),
                            len(train_df) // _target_steps),
                        max(1, len(train_df)))
            if not bool(getattr(cfg, "auto_batch_size", True)):
                _want = int(cfg.batch_size)
            _unck_cap, _uinfo = memory_cap_batch(
                _cfg_without_ckpt(cfg), str(getattr(cfg, "auto_batch_device", "t4")),
                float(getattr(cfg, "auto_batch_headroom", 0.85)), hard_max=4096)
            _vram = float(_uinfo.get("vram_gib", 15.0))
            _peak_unck = peak_gib_for_batch(_uinfo, _want)
            if _unck_cap >= _want:
                cfg.gradient_checkpointing = False
                cfg.grad_checkpointing = False
                print(f"  gradient checkpointing DISABLED: batch {_want} needs "
                      f"{_peak_unck:.2f} / {_vram:.1f} GiB without it "
                      f"({100 * _peak_unck / max(_vram, 1e-9):.1f}%), and "
                      "recomputing activations costs a MEASURED +28-51% wall "
                      "clock to save memory this card does not need. "
                      "Set auto_grad_checkpointing=false to keep it.")
            else:
                print(f"  gradient checkpointing KEPT: batch {_want} would need "
                      f"{_peak_unck:.2f} / {_vram:.1f} GiB without it "
                      f"(cap {_unck_cap}).")

        plan = plan_gpu_utilisation(cfg, len(train_df), base_batch_size, base_accum_steps)
        if plan.get("enabled"):
            cfg.batch_size = plan["batch_size"]
            cfg.grad_accum_steps = plan["grad_accum_steps"]
            _mi = plan.get("mem_info", {})
            if _mi.get("error"):
                print(f"  batch plan: {_mi['error']} -- keeping batch_size="
                      f"{cfg.batch_size}")
            print(f"  batch plan: {base_batch_size}x{base_accum_steps} "
                  f"(eff {plan['base_effective']}) -> {plan['batch_size']}x"
                  f"{plan['grad_accum_steps']} (eff {plan['effective_batch']})  "
                  f"| {plan['reason']}")
            if "peak_gib" in plan and plan["peak_gib"] == plan["peak_gib"]:
                print(f"  predicted peak VRAM on {_mi.get('device', '?')}: "
                      f"{plan['peak_gib']:.2f} / {_mi.get('vram_gib', 0):.1f} GiB "
                      f"({plan['util_pct']:.1f}%)  "
                      f"[{_mi.get('act_mib_per_sample', 0):.1f} MiB/sample x "
                      f"{plan['batch_size']} + {_mi.get('fixed_gib', 0):.2f} GiB fixed]"
                      "  -- COMPUTED from src/model.py's measured table, not "
                      "measured on this host")
                if plan["util_pct"] < 25.0:
                    print(f"  NOTE: only {plan['util_pct']:.1f}% of the card is used. "
                          f"With {len(train_df)} training studies that is not "
                          "recoverable -- a larger batch would exceed the split. "
                          "More labelled data (use_report_labels=true) is the only "
                          "way to fill this GPU.")
                elif plan["util_pct"] > 75.0:
                    print(f"  WARNING: {plan['util_pct']:.1f}% of the card is planned. "
                          "The estimate covers params, grads, AdamW state, the "
                          "autocast weight cache, activations, this file's "
                          f"EMA/Lookahead copies ({_mi.get('extras_gib', 0):.2f} GiB) "
                          "and a 0.9 GiB workspace allowance -- but NOT allocator "
                          "fragmentation or torch.compile's own buffers. If a fold "
                          "OOMs mid-run, lower auto_batch_headroom (currently "
                          f"{float(getattr(cfg, 'auto_batch_headroom', 0.85)):.2f}) "
                          "to 0.75 before reaching for grad checkpointing.")

        # LR must follow the effective batch or the run underfits: same epochs,
        # fewer updates.  Rule + arithmetic are printed, never implicit.
        if plan.get("effective_batch") != plan.get("base_effective") or \
                str(getattr(cfg, "lr_scale_rule", "sqrt")).lower() not in ("none", "off", ""):
            cfg.lr, lr_info = scale_lr_for_batch(cfg, base_lr, plan["effective_batch"])
            print(f"  lr scaling [{lr_info['rule']}]: effective batch "
                  f"{lr_info['base_batch']} -> {lr_info['effective_batch']} "
                  f"(x{lr_info['ratio']:.2f})  =>  lr {lr_info['base_lr']:.2e} -> "
                  f"{lr_info['lr']:.2e} (x{lr_info['multiplier']:.3f}"
                  f"{', CAPPED' if lr_info['capped_at'] else ''})")

        # See should_drop_last() for the BatchNorm / shape-stability trade-off.
        drop_last = should_drop_last(len(train_df), cfg.batch_size, device.type)
        train_loader = make_loader(train_ds, cfg, shuffle=True, drop_last=drop_last)
        valid_loader = make_loader(valid_ds, cfg, shuffle=False, drop_last=False)
        print(f"  loader: {len(train_loader)} train batches/epoch "
              f"(batch {cfg.batch_size}, drop_last={drop_last}), "
              f"{math.ceil(len(train_loader) / max(1, cfg.grad_accum_steps))} "
              f"optimiser updates/epoch")

        # build_model(), not a direct constructor call: this site previously
        # passed neither `pretrained_path` nor `image_size`, so
        # `cfg.backbone_weights` was set by the scripts, documented in config,
        # and read by nothing during training. On Kaggle there is no internet,
        # so `pretrained=True` could neither download nor find the local file
        # and training died at model construction. Dropping `image_size` also
        # bypassed the patch-14 check, where timm silently discards the
        # trailing pixels of a non-conforming ViT input.
        model = build_model(cfg)

        # Auto pos_weight: give the BCE loss a class-frequency-derived positive
        # weight so rare pathologies are not drowned out by abundant negatives.
        # pos_weight[c] = n_neg[c] / n_pos[c].  Clamped at 10 to avoid extreme
        # values on very rare classes (e.g. Fracture at ~2% prevalence).
        if hasattr(criterion, 'pos_weight') and criterion.pos_weight is None:
            try:
                y_train = train_df[list(target_cols)].values.astype(float)
                n_pos = np.nansum(y_train, axis=0).clip(min=1)
                n_neg = np.sum(np.isfinite(y_train), axis=0) - n_pos
                pw = np.clip(n_neg / n_pos, 1.0, 10.0).astype(np.float32)
                criterion.pos_weight = torch.from_numpy(pw).to(device)
                print(f"Auto pos_weight (clamped 1-10): {dict(zip(target_cols, pw.round(2)))}")
            except Exception as e:
                print(f"  WARNING: could not compute auto pos_weight ({e})")

        best_path = os.path.join(cfg.models_dir, f"fold_{fold}_best.pt")
        last_path = os.path.join(state_dir, f"fold_{fold}_last.pt")
        stat_path = os.path.join(state_dir, f"fold_{fold}_state.json")

        resume_ckpt = try_load_last(last_path) if cfg.resume else None
        if resume_ckpt is None:
            warm_start(model, cfg.init_from)

        model.to(device)

        # Channel-last layout: on Volta/Turing/Ampere (T4, A100) NHWC is natively
        # supported by cuDNN and gives a free 5-15% throughput boost for CNNs.
        # ViT patch-embedding also benefits. Ignored on CPU / MPS.
        if device.type == "cuda":
            try:
                model = model.to(memory_format=torch.channels_last)
            except Exception:
                pass  # Some exotic modules don't support channels_last; skip silently.

        ema_model = None
        if getattr(cfg, "use_ema", False):
            ema_decay = getattr(cfg, "ema_decay", 0.999)
            ema_model = ModelEmaV2(model, decay=ema_decay)
            print(f"EMA enabled (decay={ema_decay})")

        # Gradient checkpointing: recomputes activations on the backward pass
        # instead of keeping them in GPU RAM.  Halves VRAM usage for DINOv2
        # at a ~20% compute overhead -- allows batch_size=8 on T4 instead of 4.
        if getattr(cfg, "gradient_checkpointing", False):
            raw = base_module(model)
            backbone = getattr(raw, "backbone", None)
            if backbone is not None and hasattr(backbone, "set_grad_checkpointing"):
                backbone.set_grad_checkpointing(enable=True)
                print("Gradient checkpointing enabled on backbone.")

        if device.type == "cuda":
            try:
                # default: fuses kernels without CUDAGraphs overhead.
                # Required for AMP + Gradient Accumulation on 16GB T4 to prevent OOM.
                model = torch.compile(model, mode="default")
                print("torch.compile enabled (mode=default)")
            except Exception as e:
                print(f"torch.compile unavailable ({e}); using eager model.")
        else:
            print(f"torch.compile skipped on {device.type} (CUDA only).")

        # fused=True: applies the AdamW step via a single fused CUDA kernel
        # instead of many individual operations. Drastically reduces Python/CUDA
        # overhead and optimizer memory fragmentation.
        _adamw = torch.optim.AdamW(
            _make_param_groups(model, cfg),
            lr=cfg.lr, weight_decay=float(getattr(cfg, "weight_decay", 1e-4)),
            fused=(device.type == "cuda"))
        # Lookahead: wraps AdamW with slow-weights that sync every k steps.
        # Smoothes the optimization trajectory, especially helpful for DINOv2
        # whose loss landscape is very sharp.  k=6, alpha=0.5 is the standard.
        use_lookahead = getattr(cfg, "use_lookahead", True)
        optimizer = Lookahead(_adamw, k=6, alpha=0.5) if use_lookahead else _adamw
        if use_lookahead:
            print("Lookahead enabled (k=6, alpha=0.5)")
        start_epoch = 0
        # `selector` owns the incumbent from here on; `best` mirrors it purely so
        # the checkpoint/state-JSON schema stays unchanged for src/infer.py.
        selector = CheckpointSelector(
            metric=sel["metric"],
            auc_min_delta=sel["auc_min_delta"],
            loss_delta_k=float(getattr(cfg, "selection_loss_delta_k", 1.0)),
            override_min_delta=sel["override"],
            fallback_metric="val_loss",
            patience=int(getattr(cfg, "early_stopping_patience", 8)),
            min_epochs=int(getattr(cfg, "min_epochs", 1)),
            stale_patience=(int(getattr(cfg, "stale_patience", 0)) or None),
            legacy=bool(getattr(cfg, "selection_legacy", False)),
        )
        if selector.legacy:
            print("  selection_legacy=true: reproducing the pre-fix arithmetic "
                  "(min_delta=0, no NaN fallback, ties charge patience).")
        best = selector.best
        history = []

        if resume_ckpt is not None:
            try:
                base_module(model).load_state_dict(resume_ckpt["model"], strict=True)
                if ema_model is not None:
                    if "ema_model" in resume_ckpt:
                        base_module(ema_model.module).load_state_dict(resume_ckpt["ema_model"], strict=True)
                    else:
                        base_module(ema_model.module).load_state_dict(resume_ckpt["model"], strict=True)
                if "optimizer" in resume_ckpt: optimizer.load_state_dict(resume_ckpt["optimizer"])
                start_epoch = int(resume_ckpt["epoch"]) + 1
                best = float(resume_ckpt["best"])
                history = list(resume_ckpt.get("history", []))
                # Restore the incumbent so a resumed fold does not re-save on a
                # score it had already beaten before the session was killed.
                resumed_metric = str((resume_ckpt.get("cfg") or {}).get(
                    "selection_metric", selector.metric))
                if resumed_metric == selector.metric and np.isfinite(best):
                    selector.best = best
                    selector.saw_finite_primary = True
                elif np.isfinite(best):
                    print(f"  NOTE: checkpoint was selected on '{resumed_metric}' but this "
                          f"run selects on '{selector.metric}'; the incumbent is reset.")
                print(f"Resumed fold {fold} at epoch {start_epoch} (best={best:.4f})")
            except Exception as e:
                print(f"WARNING: resume failed ({e}); starting this fold from scratch.")
                start_epoch, history = 0, []
                selector.reset()
                best = selector.best

        # OneCycleLR: linear warmup, then cosine decay.
        # This is the most impactful LR schedule change for fine-tuning large
        # ViTs (DINOv2) -- a cold start at full LR destroys pre-trained weights.
        # Note: OneCycleLR must receive the inner AdamW (not the Lookahead wrapper)
        # since it reads/writes param_groups directly.
        #
        # BUG FIXED HERE.  steps_per_epoch used to be len(train_loader), i.e. the
        # number of MICRO-batches, but train_one_epoch calls scheduler.step() once
        # per OPTIMISER UPDATE -- ceil(len(train_loader) / grad_accum_steps) times.
        # With the shipped grad_accum_steps=2 the scheduler therefore received
        # only half the steps it was built for and never finished its cycle: on
        # 46 gold studies at batch 8 it advanced 36 of 72 steps, so training
        # ENDED at ~59% of max_lr instead of max_lr/1e4.  Every "cosine decay"
        # run so far was a run that never decayed -- which also means SWA was
        # averaging weights taken at a high, still-moving LR.
        steps_per_epoch = max(1, math.ceil(len(train_loader)
                                           / max(1, int(getattr(cfg, "grad_accum_steps", 1)))))

        # If we resumed and the fold is already done, skip scheduler init to avoid 0 epochs error
        _total_updates = steps_per_epoch * max(0, cfg.epochs - start_epoch)
        if start_epoch < cfg.epochs and _total_updates < ONECYCLE_MIN_UPDATES:
            # OneCycleLR divides by (total_steps - 1) internally, so a one-update
            # schedule raises ZeroDivisionError. A constant LR is the only honest
            # thing left at that point.
            print(f"  WARNING: {_total_updates} optimiser update(s) in this fold -- "
                  f"no LR schedule (constant lr={cfg.lr:.2e}). The training split "
                  "is too small for the configured batch/epochs.")
            scheduler = None
        elif start_epoch < cfg.epochs:
            _pct_start, _warm_steps = onecycle_pct_start(cfg, steps_per_epoch, _total_updates)
            print(f"  schedule: {steps_per_epoch} updates/epoch x "
                  f"{cfg.epochs - start_epoch} epochs = {_total_updates} updates, "
                  f"warmup {_warm_steps} updates (pct_start={_pct_start:.3f}, "
                  f"warmup_epochs={getattr(cfg, 'warmup_epochs', 1.0)}, "
                  f"floor={getattr(cfg, 'warmup_min_steps', 3)})")
            if _total_updates < 20:
                print(f"  WARNING: only {_total_updates} optimiser updates in this "
                      "fold. Neither the LR schedule nor SWA has room to work; "
                      "prefer more epochs or a smaller batch over a bigger one.")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                _adamw,
                max_lr=[g["lr"] for g in _adamw.param_groups],
                epochs=cfg.epochs - start_epoch,
                steps_per_epoch=steps_per_epoch,
                pct_start=_pct_start,
                anneal_strategy="cos",
                div_factor=10.0,        # start_lr = max_lr / 10
                final_div_factor=1e4,   # end_lr = max_lr / 10000
            )
            if resume_ckpt is not None and "scheduler" in resume_ckpt:
                try:
                    scheduler.load_state_dict(resume_ckpt["scheduler"])
                except Exception:
                    pass
        else:
            scheduler = None
        
        scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)
        if resume_ckpt is not None and "scaler" in resume_ckpt:
            try:
                scaler.load_state_dict(resume_ckpt["scaler"])
            except Exception:
                pass

        if start_epoch >= cfg.epochs:
            print(f"Fold {fold} already complete ({start_epoch}/{cfg.epochs} epochs).")
            continue

        cfg_snapshot = {"image_size": cfg.image_size, "backbone": cfg.backbone,
                        "lr": cfg.lr, "epochs": cfg.epochs, "loss": crit_name,
                        # The EFFECTIVE metric, not cfg.selection_metric, so a
                        # resumed run can tell whether its incumbent is
                        # comparable to what this run is measuring.
                        "selection_metric": selector.metric,
                        "selection_requested": str(cfg.selection_metric),
                        "auc_min_delta": sel["auc_min_delta"]}
        smoke = bool(cfg.smoke_test)
        val_tta = bool(getattr(cfg, "val_tta", True))
        completed = False
        stopped_reason = "epochs_exhausted"
        # EMA of the selection metric: smooths noisy val_auc (only 10-13 val studies)
        # so early stopping doesn't trigger on a fluke bad epoch.
        ema_score = None
        ema_alpha = float(getattr(cfg, "val_ema_alpha", 0.4))  # 0.4 = moderate smoothing

        # ── Stochastic Weight Averaging ──────────────────────────────────
        # Averaging the weights of the last N epochs costs one CPU copy per
        # epoch and produces a model that sits in a flatter minimum than any
        # single epoch.  It is the cheapest ensemble available: N models'
        # worth of variance reduction at ONE model's inference cost and ONE
        # model's VRAM.  Averaging starts only after the LR has decayed past
        # swa_start_frac of the schedule -- averaging across the high-LR phase
        # mixes weights from different basins and destroys the model.
        swa = RunningWeightAverage() if getattr(cfg, "use_swa", False) else None
        swa_start_epoch = int(math.floor(cfg.epochs * float(getattr(cfg, "swa_start_frac", 0.6))))
        if swa is not None:
            print(f"SWA: averaging epochs {swa_start_epoch + 1}..{cfg.epochs} "
                  f"(running mean on CPU, 0 extra VRAM)")

        for epoch in range(start_epoch, cfg.epochs):
            if clock.expired(reserve_s=120):
                stopped_reason = "time_budget"
                print("Wall-clock budget reached before epoch start -- stopping.")
                break

            train_loss, interrupted = train_one_epoch(
                model, ema_model, train_loader, optimizer, scheduler, criterion,
                scaler, device, cfg, use_amp, amp_dtype, clock, smoke)
            
            # EMA weights are what gets validated AND what gets checkpointed --
            # `eval_model` is the single object handed to validate() and to
            # save_best(), so the selected file always matches the scored model.
            eval_model = ema_model.module if ema_model is not None else model
            val_bucket = {}
            val_loss, val_crit, val_auc, per_col, n_val = validate(
                eval_model, valid_loader, criterion, device, use_amp, amp_dtype,
                smoke, val_tta=val_tta, lateral_swap_perm=val_swap_perm,
                collect=val_bucket)
            # Per-study masked BCE: the paired sample used to size the val_loss
            # resolution (see loss_resolution).  Never used as a training signal.
            per_row_loss = None
            if "logits" in val_bucket and "targets" in val_bucket:
                try:
                    per_row_loss = per_row_masked_bce(val_bucket["logits"],
                                                      val_bucket["targets"])
                except Exception:
                    per_row_loss = None
            # OneCycleLR steps per batch, not per epoch -- no scheduler.step() here.

            metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                       "val_crit": val_crit, "val_auc": val_auc, "n_val": n_val,
                       "lr": scheduler.get_last_lr()[0]}
            history.append(metrics)
            auc_str = "n/a" if not np.isfinite(val_auc) else f"{val_auc:.4f}"
            print(f"Epoch {epoch + 1}/{cfg.epochs} | train {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_crit {val_crit:.4f} | "
                  f"val_auc {auc_str} (n={n_val}) | lr {scheduler.get_last_lr()[0]:.2e}")
            # Print named per-label AUCs so it's easy to see which pathology
            # is struggling (indices alone are opaque in long runs).
            if per_col and target_cols:
                named = {target_cols[i]: (f"{v:.3f}" if v is not None else "n/a")
                         for i, v in per_col.items() if i < len(target_cols)}
                print(f"  per-label AUC: {named}")
            # The macro DENOMINATOR must not move between epochs: a 12-label
            # macro and a 6-label macro are different quantities, and the delta
            # between them is meaningless.  It is fixed by the validation labels
            # (which never change), so a mismatch here means predictions went
            # non-finite -- worth shouting about rather than silently averaging.
            n_auc_cols = sum(1 for v in per_col.values() if v is not None)
            metrics["n_auc_cols"] = n_auc_cols
            expected_cols = int(sel["auc_info"].get("n_usable", n_auc_cols))
            # smoke_test truncates validation to 2 batches, so a smaller
            # denominator there is expected rather than alarming.
            if n_auc_cols != expected_cols and not smoke:
                print(f"  WARNING: macro-AUC averaged {n_auc_cols} labels this "
                      f"epoch but {expected_cols} are computable from the "
                      f"validation labels -- epoch-to-epoch val_auc values are "
                      f"NOT comparable quantities.")

            # ── Model selection ──────────────────────────────────────────
            # Three outcomes, not two.  `checkpoint_exists=False` on the first
            # epoch (or after a wiped models_dir) forces a save regardless of
            # what the metric did, so a fold can NEVER finish with no best.pt --
            # not even when the metric is NaN in every single epoch.
            decision = selector.step(metrics,
                                     checkpoint_exists=os.path.exists(best_path),
                                     per_row_loss=per_row_loss)
            print(selector.describe(decision))
            if decision["save"]:
                save_best(eval_model, best_path)
                print(f"  saved best ({decision['metric']}="
                      f"{_as_float(decision['score']):.5f}) -> {best_path}")
            best = selector.best if np.isfinite(selector.best) else selector.best_fallback

            # Diagnostic only: an EMA of the score, logged so a noisy metric is
            # visible in the history.  It drives NO control flow -- smoothing a
            # metric does not raise its resolution, and using a smoothed value
            # for stopping decisions is what hid the problem last time.
            score = _as_float(decision["score"])
            if ema_score is None or not np.isfinite(ema_score):
                ema_score = score
            elif np.isfinite(score):
                ema_score = ema_alpha * score + (1 - ema_alpha) * ema_score
            metrics["ema_score"] = ema_score
            metrics["selection"] = {k: decision[k] for k in
                                    ("metric", "status", "min_delta", "save",
                                     "fallback", "patience", "stale")}

            if swa is not None and epoch >= swa_start_epoch:
                swa.update(eval_model)
                print(f"  SWA: folded epoch {epoch + 1} into the average (n={swa.n})")

            save_last(last_path, model, ema_model, optimizer, scheduler, scaler, epoch, best,
                      history, cfg_snapshot)
            with open(stat_path, "w") as f:
                json.dump({"fold": fold, "epoch": epoch, "epochs": cfg.epochs,
                           "best": best, "selection_metric": selector.metric,
                           "selection_requested": str(cfg.selection_metric),
                           "selection_reason": sel["reason"],
                           "min_delta": decision["min_delta"],
                           "auc_min_delta": sel["auc_min_delta"],
                           "auc_resolution": sel["auc_info"],
                           "completed": epoch + 1 >= cfg.epochs,
                           "history": history}, f, indent=2)

            if interrupted:
                stopped_reason = "time_budget"
                break
            # Early stopping runs through the SAME resolution guard as
            # checkpoint selection: only a REGRESSION larger than the metric's
            # own min_delta charges patience.  A change the metric cannot
            # resolve is a tie and costs nothing -- that is the fix for the
            # 0.0003 kill.  `stale` bounds a permanent plateau so a metric that
            # never moves still terminates.
            reason = selector.stop_reason(epoch)
            if reason:
                stopped_reason = reason
                print(f"  early stopping: {selector.patience} measurable "
                      f"regressions / {selector.stale} epochs without progress "
                      f"(limits {selector.patience_limit} / {selector.stale_limit}, "
                      f"min_delta={decision['min_delta']:.5f} on "
                      f"{decision['metric']})")
                completed = True
                break
        else:
            completed = True

        with open(stat_path, "w") as f:
            json.dump({"fold": fold, "epochs": cfg.epochs, "best": best,
                       "selection_metric": selector.metric,
                       "selection_requested": str(cfg.selection_metric),
                       "selection_reason": sel["reason"],
                       "auc_min_delta": sel["auc_min_delta"],
                       "auc_resolution": sel["auc_info"],
                       "completed": completed, "reason": stopped_reason,
                       "history": history}, f, indent=2)
                       
        if completed and os.path.exists(last_path):
            try: os.remove(last_path)
            except Exception: pass
            
        print(f"Fold {fold} finished: {stopped_reason} "
              f"(best {selector.metric}={best:.4f}, completed={completed})")

        # Stochastic Weight Averaging (SWA): after a completed fold, build an
        # averaged model from the best checkpoint.  SWA finds a flatter minimum
        # than SGD/Adam and reliably gives +0.005-0.02 AUC -- essentially a free
        # ensemble of all the best states the model ever visited.
        # ── Stochastic Weight Averaging: build it, then EARN it ──────────
        # The previous implementation called AveragedModel.update_parameters()
        # exactly once on the final weights, so fold_N_swa.pt was a *copy of the
        # last epoch* -- not an average of anything -- and src/infer.py silently
        # preferred it over the val-selected best checkpoint.  That threw away
        # model selection on every fold.  Now the average is real (it is folded
        # over every post-decay epoch), and it is only written to disk if it
        # actually beats the selected checkpoint on the held-out fold.
        swa_path = os.path.join(cfg.models_dir, f"fold_{fold}_swa.pt")
        swa_won = False
        if swa is not None and swa.n >= 2:
            try:
                print(f"SWA: evaluating the average of {swa.n} epochs...")
                raw = base_module(model)
                swa.load_into(raw)

                # BatchNorm running statistics are NOT a smooth function of the
                # weights, so an average of BN weights carries meaningless
                # running stats and must be re-estimated with one forward pass
                # over the training data.  LayerNorm nets (ConvNeXt, ViT) have
                # no such buffers, so the pass is skipped entirely.
                if _has_batchnorm(raw):
                    from torch.optim.swa_utils import update_bn
                    print("  re-estimating BatchNorm statistics (1 pass)...")
                    with torch.no_grad():
                        update_bn(train_loader, raw, device=device)
                else:
                    print("  no BatchNorm in this backbone -- skipping the BN pass.")

                swa_bucket = {}
                swa_loss, swa_crit, swa_auc, _, _ = validate(
                    raw, valid_loader, criterion, device, use_amp, amp_dtype,
                    smoke, val_tta=val_tta, lateral_swap_perm=val_swap_perm,
                    collect=swa_bucket)
                # Score the average on the metric that ACTUALLY selected the
                # incumbent, not only on val_auc.  With selection_metric=auto
                # resolving to val_loss on a 58-study panel, the old
                # `if cfg.selection_metric == "val_auc"` test made swa_score
                # None on every fold and silently disabled this whole gate.
                swa_score = {"val_auc": swa_auc, "val_loss": swa_loss,
                             "val_crit": swa_crit}.get(selector.metric)
                swa_rows = None
                if "logits" in swa_bucket and "targets" in swa_bucket:
                    try:
                        swa_rows = per_row_masked_bce(swa_bucket["logits"],
                                                      swa_bucket["targets"])
                    except Exception:
                        swa_rows = None

                # The gate is RESOLUTION-AWARE.  Macro ROC-AUC over 10-13
                # validation studies is a rank statistic whose finest possible
                # step is ~0.003 (measured per fold on train_gold.csv), so a
                # plain `>` would let SWA win on a 0.0003 difference the metric
                # cannot actually express -- the same class of defect that
                # early-stopped an earlier run on numerical noise.  Swapping the
                # weights on that evidence is a coin flip dressed as a decision,
                # so the average must clear the metric's own resolution.
                try:
                    swa_delta = selector.delta_for(selector.metric, swa_rows)
                except Exception:
                    swa_delta = 0.0
                status = (classify_score(selector.metric, swa_score, best, swa_delta)
                          if swa_score is not None else UNDECIDABLE)

                if status == IMPROVED:
                    torch.save(strip_compile_prefix(raw.state_dict()), swa_path,
                               _use_new_zipfile_serialization=False)
                    swa_won = True
                    print(f"  SWA WINS: {swa_score:.4f} > {best:.4f} "
                          f"(by more than the metric's resolution {swa_delta:.5f}) -> {swa_path}")
                    best = swa_score
                else:
                    shown = "n/a" if swa_score is None else f"{swa_score:.4f}"
                    print(f"  SWA rejected [{status}] ({shown} vs best {best:.4f}, "
                          f"resolution {swa_delta:.5f}); keeping {best_path}.")
                    if os.path.exists(swa_path):
                        try: os.remove(swa_path)   # never leave a stale loser on disk
                        except Exception: pass
            except Exception as e:
                print(f"  WARNING: SWA pass failed ({e}); best.pt is still valid.")
                if os.path.exists(swa_path):
                    try: os.remove(swa_path)
                    except Exception: pass
        elif swa is not None:
            print(f"SWA: only {swa.n} epoch(s) averaged -- not enough to beat "
                  f"model selection; skipping.")

        # ── Out-of-fold predictions from whichever model actually won ────────
        # These are the only honest evidence available for choosing between a
        # weight-space soup and a logit ensemble later, so they are written for
        # every fold regardless of which checkpoint won.
        try:
            winner = swa_path if swa_won else best_path
            if os.path.exists(winner):
                raw = base_module(model)
                st = torch.load(winner, map_location="cpu", weights_only=False)
                if isinstance(st, dict) and "model" in st:
                    st = st["model"]
                raw.load_state_dict(strip_compile_prefix(st), strict=False)
                bucket = {}
                validate(raw, valid_loader, criterion, device, use_amp, amp_dtype,
                         smoke, val_tta=val_tta, lateral_swap_perm=val_swap_perm,
                         collect=bucket)
                if "logits" in bucket:
                    dump_oof(os.path.join(cfg.models_dir, f"oof_fold_{fold}.npz"),
                             valid_df["StudyInstanceUID"].astype(str).tolist(),
                             bucket["logits"], bucket["targets"], target_cols)
        except Exception as e:
            print(f"  WARNING: could not write OOF predictions ({e})")

        if stopped_reason == "time_budget":
            print("Re-run the same command to resume from fold_%d_last.pt" % fold)
            return


if __name__ == "__main__":
    main()
