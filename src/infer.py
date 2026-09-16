"""Inference / submission builder.

Reliability contract for a Kaggle code competition, in priority order:

1. A VALID submission.csv exists on disk within seconds of this script starting,
   before a single model is loaded.  Everything after that only ever *improves*
   it.  A crash at minute 250 then costs accuracy, not the entire submission.
2. The submission's shape -- row set, row order, column names, column order --
   is copied from sample_submission.csv, never reconstructed from a constant in
   the source.
3. Inference cost is measured on the first batches and the TTA budget is cut to
   fit the remaining wall clock, instead of being assumed to fit.
4. Models are loaded, used, and freed ONE AT A TIME, so peak VRAM is one model
   regardless of ensemble size.

Efficiency contract (this competition also scores inference cost):

5. Exactly ONE forward call carries all TTA views of a batch.  M sequential
   forwards of B images become one forward of B*M images: identical arithmetic,
   1/M the kernel launches.  On a T4 an effnet_b0 forward is launch-bound, not
   FLOP-bound, so this is the single largest lever.
6. The compute batch is chosen from a MEASUREMENT of bytes/image on the real
   device, not from cfg.batch_size (which is a *training* batch size, sized for
   gradients + optimiser state that inference does not have).  An OOM at any
   point halves the batch and retries rather than losing the submission.
7. torch.inference_mode(), not torch.no_grad(): no autograd version counters,
   no view tracking.  Provably identical outputs (measured: 0.0 max abs delta).
8. The DICOM decode happens once per *preprocessing group* (resolution x
   in_channels), never once per model.  Measured 16.2 ms/study cold vs
   0.31 ms/study warm, so this is worth ~52x on every model after the first.
"""

import os
import gc
import copy
import time
import argparse
import contextlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F        # required by the multi-scale TTA zoom
from torch.utils.data import DataLoader
from scipy.stats import rankdata
from tqdm import tqdm

from src.config import Config
from src.model import RSNA25DModel
from src.kaggle_data import RSNADataset, KNEE_TARGETS, build_lateral_swap


# ══════════════════════════════════════════════════════════════════════════
# Cost accounting
#
# The efficiency leaderboard scores compute, so compute is counted rather than
# estimated.  FORWARD_STATS is the ground truth behind every "N forward passes"
# claim in the final log line.
# ══════════════════════════════════════════════════════════════════════════

FORWARD_STATS = {"calls": 0, "images": 0}

# Size of the TTA family in build_tta. The views are a PREFIX family, so
# build_tta(k) is always the first k of these.
MAX_TTA = 5


def reset_forward_stats():
    FORWARD_STATS["calls"] = 0
    FORWARD_STATS["images"] = 0


def _run_model(model, x):
    FORWARD_STATS["calls"] += 1
    FORWARD_STATS["images"] += int(x.shape[0])
    return model(x)


# ══════════════════════════════════════════════════════════════════════════
# Test-time augmentation
# ══════════════════════════════════════════════════════════════════════════

def build_tta(n, H, W):
    """Return `n` (transform, needs_lateral_swap, weight) triples.

    Knees are never acquired upside-down or rotated 90 degrees, so vflip/rot90
    TTA would feed the model orientations it has never seen.  The only valid
    geometric augmentations at test time are the laterality mirror (whose label
    permutation we can undo exactly) and mild zoom.

    n is honoured, so the caller can trade accuracy for wall clock:
        n=1  original only                 1.0x cost
        n=3  + mirror + 5% zoom            3.0x cost
        n=5  + 10% zoom + mirrored zoom    5.0x cost

    The list is a PREFIX family: build_tta(k) is always the first k entries of
    build_tta(5), only the weights change.  scripts/benchmark_inference.py
    relies on that to answer "what is the AUC at every M?" from a single pass
    over the data (see tta_prefix_weights).
    """
    def _zoom(x, scale):
        new_h, new_w = int(H * scale), int(W * scale)
        up = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        top, left = (new_h - H) // 2, (new_w - W) // 2
        return up[..., top:top + H, left:left + W]

    all_augs = [
        (lambda x: x, False),                                       # original
        (lambda x: torch.flip(x, dims=[-1]), True),                 # laterality mirror
        (lambda x: _zoom(x, 1.05), False),
        (lambda x: _zoom(x, 1.10), False),
        (lambda x: torch.flip(_zoom(x, 1.05), dims=[-1]), True),
    ]
    assert len(all_augs) == MAX_TTA
    n = max(1, min(int(n), len(all_augs)))
    augs = all_augs[:n]

    # The un-augmented view is the only one drawn from the true test
    # distribution, so it keeps half the weight; the rest share the remainder.
    return [(fn, swap, w) for (fn, swap), w in zip(augs, tta_prefix_weights(n))]


def tta_prefix_weights(n):
    """The weight vector build_tta(n) uses.  Split out so an offline sweep can
    re-weight cached per-view predictions without re-running the model."""
    n = max(1, min(int(n), MAX_TTA))
    if n == 1:
        return [1.0]
    return [0.5] + [0.5 / (n - 1)] * (n - 1)


def _autocast(use_amp, device_type, amp_dtype):
    """autocast when it is actually enabled, otherwise a true no-op.

    torch.amp.autocast(..., enabled=False) is already a no-op numerically, but
    constructing it with dtype=float32 on a CPU/MPS device emits a spurious
    "CPU Autocast only supports bfloat16/float16" warning on every batch.
    """
    if not use_amp:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type, dtype=amp_dtype, enabled=True)


def _tta_view_logits(model, images, tta, max_images):
    """Logits for every TTA view, in ONE forward per group of views.

    Returns a list of `len(tta)` tensors, each (B, num_classes) -- exactly what
    len(tta) separate `model(view)` calls would return, because in eval() every
    layer here is per-sample (BatchNorm reads frozen running stats, GeM /
    LayerNorm / Linear act row-wise).  Concatenating views along the batch axis
    therefore changes only which CUDA kernels are picked, not the arithmetic.

    `max_images` bounds how many images may be resident in one forward, so a
    large data batch cannot silently multiply into a B*M tensor that OOMs.
    """
    B = int(images.shape[0])
    per_group = len(tta) if not max_images else max(1, int(max_images) // max(1, B))
    outs = []
    for start in range(0, len(tta), per_group):
        group = tta[start:start + per_group]
        if len(group) == 1:
            x = group[0][0](images)
        else:
            x = torch.cat([fn(images) for fn, _, _ in group], dim=0)
        if x.device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        y = _run_model(model, x)
        for i in range(len(group)):
            outs.append(y[i * B:(i + 1) * B])
    return outs


def tta_predict(model, images, use_amp, amp_dtype, tta, lateral_swap_perm=None,
                max_images=None, return_views=False):
    """Weighted average of sigmoid predictions over the TTA views.

    With return_views the per-view (B, C) probability stack is also returned;
    its spread across views is a free, label-free uncertainty estimate and is
    what the cascade routes on.
    """
    device_type = images.device.type
    with _autocast(use_amp, device_type, amp_dtype):
        view_logits = _tta_view_logits(model, images, tta, max_images)
        out, views = None, []
        for (aug_fn, needs_swap, weight), logits in zip(tta, view_logits):
            pred = torch.sigmoid(logits).float()
            # A mirrored knee's "medial" output describes the ORIGINAL image's
            # lateral side, so the columns must be permuted back before they are
            # averaged into an un-mirrored prediction.
            if needs_swap and lateral_swap_perm is not None:
                pred = pred[:, lateral_swap_perm]
            if return_views:
                views.append(pred)
            out = pred * weight if out is None else out + pred * weight
    if return_views:
        return out, torch.stack(views, dim=0)      # (M, B, C)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Ensembling
# ══════════════════════════════════════════════════════════════════════════

def rank_normalise(p):
    """Per-column ranks scaled to (0, 1].

    ROC-AUC is invariant to any strictly increasing transform of a column, so
    ranking discards nothing the metric can see while making models with
    different output calibrations directly comparable.  axis=0 is load-bearing:
    ranking must compare STUDIES within a label, never labels within a study.
    """
    r = rankdata(p, axis=0)
    return r / r.max(axis=0, keepdims=True)


def rank_average(predictions_list):
    if len(predictions_list) == 1:
        return rank_normalise(predictions_list[0])
    return np.mean([rank_normalise(p) for p in predictions_list], axis=0)


def agreement(a, b):
    """Mean per-column Spearman correlation between two prediction matrices.

    This needs no labels, which is exactly why it is usable on the hidden test
    set.  It does not measure which of the two is better; it measures whether
    they would produce the same ranking -- and therefore the same AUC.
    """
    ra, rb = rankdata(a, axis=0), rankdata(b, axis=0)
    rhos = []
    for i in range(ra.shape[1]):
        x, y = ra[:, i], rb[:, i]
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        rhos.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(rhos)) if rhos else float("nan")


def cascade_merge(base_pred, subset_idx, subset_preds):
    """Let an expensive ensemble re-order ONLY the studies the cheap model was
    unsure about, without disturbing anything else.

    The naive cascade -- rank-average the cheap model over all studies with an
    expensive model that only saw a subset -- is a silent correctness bug:
    rank_normalise on the subset ranks within the subset, so those numbers are
    not on the same scale as the full-set ranks and mixing them scrambles the
    global ordering that AUC is computed from.

    This does the only thing that is safe.  Per label:
      * the studies in `subset_idx` collectively keep exactly the rank SLOTS
        they already occupied in the cheap model's full-set ranking;
      * which member of the subset gets which of those slots is decided by the
        rank-average of the cheap and expensive models restricted to the subset.

    So a confident study's rank is bit-identical to the no-cascade result, and
    the ensemble is spent purely on re-ordering the uncertain ones.

    Ties inside the subset are broken by the CHEAP MODEL's own ranking, not by
    array order.  rank_average ties genuinely and often (the mean of two integer
    rank vectors ties whenever they cross), and with a handful of studies those
    ties are common.  Two alternatives were rejected:
      * argsort's implicit index order -- makes the output depend on the order
        studies happen to sit in sample_submission.csv;
      * averaging the tied slots -- keeps the studies tied, but then a tied pair
        can land exactly on a CONFIDENT study's rank, which changes that
        study's pairwise relations and breaks the guarantee above.
    Deferring to model 1 keeps every slot distinct, keeps the guarantee, and
    breaks the tie with the only other evidence available.

    KNOWN LIMITATION.  If model 1 itself tied two subset studies they share one
    rank slot, so the ensemble's preference between them cannot be expressed.
    Model 1 emits continuous sigmoids, so this needs exactly equal predictions
    to trigger, but it is inherent: this re-ORDERS model 1's ranks, it does not
    replace them.  Set cascade_frac=0 (the default) to rank-average instead.
    """
    out = rank_normalise(base_pred)
    idx = np.asarray(subset_idx, dtype=int)
    if idx.size == 0 or not subset_preds:
        return out
    slots = np.sort(out[idx], axis=0)                       # (|U|, C) ascending
    base_sub = out[idx]                                     # (|U|, C)
    fused = rank_average([p for p in subset_preds])         # (|U|, C)

    k, n_cols = fused.shape
    order = np.empty((k, n_cols), dtype=int)
    for j in range(n_cols):
        # lexsort takes the LAST key as primary: fused first, model 1 as the
        # tiebreaker.
        o = np.lexsort((base_sub[:, j], fused[:, j]))
        order[o, j] = np.arange(k)
    out[idx] = np.take_along_axis(slots, order, axis=0)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Submission I/O
# ══════════════════════════════════════════════════════════════════════════

def load_submission_template(cfg):
    """Return (ids, target_columns) taken from sample_submission.csv.

    The row set, row order and column order of the submission are dictated by
    the competition, not by a constant in this repo.  Falling back to the
    hard-coded KNEE_TARGETS list is a last resort and is announced loudly.
    """
    path = os.path.join(cfg.data_dir, "sample_submission.csv")
    if os.path.exists(path):
        sub = pd.read_csv(path)
        id_col = sub.columns[0]
        targets = [c for c in sub.columns[1:]]
        print(f"Submission template: {path} "
              f"({len(sub)} rows, {len(targets)} target columns)")
        return sub[id_col].astype(str).tolist(), targets, id_col

    print(f"WARNING: {path} not found -- falling back to test.csv ids and the "
          f"built-in target list. Verify the column names before submitting.")
    test_path = os.path.join(cfg.data_dir, "test.csv")
    ids = (pd.read_csv(test_path)["StudyInstanceUID"].astype(str).tolist()
           if os.path.exists(test_path) else [])
    return ids, list(KNEE_TARGETS), "StudyInstanceUID"


def write_submission(path, ids, targets, id_col, values):
    """Atomically write the submission, so a crash mid-write cannot truncate it.

    Also enforces the two things the leaderboard will reject outright: no NaN /
    inf anywhere, and every value inside [0, 1].
    """
    values = np.nan_to_num(np.asarray(values, dtype=np.float64),
                           nan=0.5, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    sub = pd.DataFrame(values, columns=targets)
    sub.insert(0, id_col, ids)
    tmp = path + ".tmp"
    sub.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return sub


# ══════════════════════════════════════════════════════════════════════════
# Model loading / prediction
# ══════════════════════════════════════════════════════════════════════════

def resolve_amp(device, force_fp32=False):
    """bfloat16 needs sm_80+.  Kaggle's usual accelerators are T4 (sm_75) and
    P100 (sm_60), where a hard-coded bf16 autocast is either rejected outright or
    emulated at a large slowdown.  Inference has no gradients to overflow, so
    fp16 is the correct choice below Ampere and needs no GradScaler."""
    if force_fp32 or device.type != "cuda":
        return False, torch.float32
    major = torch.cuda.get_device_capability(0)[0]
    if major >= 8:
        return True, torch.bfloat16
    print(f"INFO: device capability sm_{major}x lacks native bfloat16 -> using fp16.")
    return True, torch.float16


def build_inference_model(backbone, image_size, cfg):
    """Construct the architecture the checkpoint was TRAINED with.

    The previous version called RSNA25DModel(backbone, in_channels, num_classes)
    and took the constructor defaults for everything else.  That is wrong in two
    ways that both end in silently or loudly broken weights:

      * image_size was never passed, so a patch ViT was always built with the
        224 position-embedding grid.  A checkpoint trained at 288/378 then fails
        to load (pos_embed shape mismatch is raised even under strict=False).
      * vit_pool / cnn_pool / head_norm were taken from the constructor defaults
        rather than from cfg, so `--set cnn_pool=avgmax` at training time
        produced a feature width the inference head could not accept.

    build_model(cfg) is the one place that knows the full mapping, so use it.
    """
    from src.model import build_model
    c = copy.copy(cfg)
    c.backbone = backbone
    if image_size:
        c.image_size = int(image_size)
    c.pretrained = False                 # Kaggle is offline; weights come from the ckpt
    c.backbone_weights = ""
    c.grad_checkpointing = False         # trades speed for activations we do not need
    c.drop_path_rate = 0.0               # inactive in eval(); skip building the modules
    try:
        return build_model(c)
    except Exception as e:
        print(f"  NOTE: build_model failed ({e}); falling back to the minimal "
              f"constructor. Verify the checkpoint loads cleanly.")
        return RSNA25DModel(backbone_name=backbone, pretrained=False,
                            in_channels=cfg.in_channels,
                            num_classes=cfg.num_classes,
                            image_size=int(image_size or cfg.image_size))


def load_model(ckpt, backbone, cfg, device, image_size=None, compile_model=True):
    model = build_inference_model(backbone, image_size, cfg)
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  NOTE: {len(missing)} parameters were not present in {os.path.basename(ckpt)} "
              f"(first: {missing[0]})")
    if unexpected:
        # Loud, because this is what a backbone/pooling mismatch looks like: the
        # checkpoint carries weights the model has no slot for, so those weights
        # are silently discarded and the model runs on its initialisation.
        print(f"  WARNING: {len(unexpected)} checkpoint tensors did not match any "
              f"parameter (first: {unexpected[0]}) -- architecture mismatch?")
    model.to(device).eval()
    if device.type == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
        if compile_model:
            try:
                # mode="default" fuses kernels without CUDAGraphs.  "reduce-overhead"
                # captures CUDA graphs, which re-capture on every new input shape --
                # the final short batch of the test set triggers exactly that, and on
                # a 16GB T4 the capture can OOM mid-submission.
                #
                # Every batch this file feeds the model is padded to one fixed
                # shape (see predict), so there is exactly one graph to compile
                # and no shape-driven recompilation.
                model = torch.compile(model, mode="default")
            except Exception as e:
                print(f"  torch.compile unavailable ({e}); running eager.")
    return model


# ── batch sizing ──────────────────────────────────────────────────────────

def measure_bytes_per_image(model, size, in_ch, device, use_amp, amp_dtype, probe=8):
    """Peak allocator bytes attributable to ONE image in a forward pass.

    Measured on the device that will actually run the submission, because the
    only defensible source for this number is the card in front of us.  Returns
    None off CUDA, where there is no peak-allocation counter to read.
    """
    if device.type != "cuda":
        return None
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        x = torch.randn(probe, in_ch, size, size, device=device)
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), _autocast(use_amp, "cuda", amp_dtype):
            model(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del x
        torch.cuda.empty_cache()
        return max(1, (peak - before)) / float(probe)
    except Exception as e:
        print(f"  batch probe failed ({e}); using the configured batch size.")
        return None


def autotune_compute_batch(model, size, in_ch, device, use_amp, amp_dtype, cfg):
    """Largest number of IMAGES to put through one forward, with margin.

    cfg.batch_size is a *training* batch size: it was chosen against activations
    kept for backprop plus optimiser state.  Inference keeps neither, so reusing
    it leaves the card almost idle -- on a 15.0 GiB T4 a batch of 8 effnet_b0
    images at 224px is well under 1% of the memory and nowhere near enough work
    to hide kernel-launch latency.
    """
    forced = int(getattr(cfg, "infer_batch_images", 0) or 0)
    cap = int(getattr(cfg, "infer_max_batch_images", 512) or 512)
    if forced > 0:
        return max(1, min(forced, cap))
    if device.type != "cuda":
        # No allocator counter to read; 64 is comfortably past the point where
        # throughput plateaus on CPU/MPS (measured) and is small enough to be safe.
        return min(cap, 64)

    per_img = measure_bytes_per_image(model, size, in_ch, device, use_amp, amp_dtype)
    if not per_img:
        return max(1, min(int(cfg.batch_size), cap))
    free, total = torch.cuda.mem_get_info()
    frac = float(getattr(cfg, "infer_vram_fraction", 0.60))
    budget = free * frac
    n = int(budget // per_img)
    n = max(1, min(n, cap))
    n -= n % 8 if n >= 16 else 0                 # keep tensor-core friendly shapes
    n = max(1, n)
    print(f"  batch probe: {per_img / 2**20:.2f} MiB/image, {free / 2**30:.2f} GiB free "
          f"-> {n} images per forward ({frac:.0%} of free VRAM budgeted)")
    return n


def _is_oom(exc):
    """True for every flavour of out-of-memory this can hit.

    Both checks are needed: cuDNN/cuBLAS workspace failures surface as a plain
    RuntimeError whose message mentions memory, not as OutOfMemoryError.
    """
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


# ── batch source: decode once, reuse across models ────────────────────────

class BatchSource:
    """Iterates a DataLoader, optionally memoising the collated CPU batches.

    The DICOM decode is already shared across models by RSNADataset's on-disk
    .npy cache (measured: 16.2 ms/study cold, 0.31 ms/study warm).  What is NOT
    shared is the .npy read plus the albumentations transform, which every model
    pays again -- along with a fresh DataLoader worker pool if the loader is
    rebuilt.  Holding the collated tensors removes both, and the tensors are the
    exact objects the first pass produced, so this cannot change a prediction.

    Memoisation is budgeted (cfg.infer_ram_cache_gb) and is committed only when
    a pass ran to completion, so an early-stopped calibration pass can never
    install a truncated cache.
    """

    def __init__(self, loader, budget_bytes):
        self.loader = loader
        self.budget = int(budget_bytes)
        self._cached = None
        self.hits = 0

    @property
    def cached(self):
        return self._cached is not None

    @property
    def batch_size(self):
        """The batch size this source actually yields.

        predict()'s `pad_to` MUST come from here, not from a separately computed
        number: the loader is memoised per resolution, so if the TTA budget is
        cut afterwards the recomputed batch would no longer match what the
        loader emits and every batch would be padded far past its real size.
        """
        return getattr(self.loader, "batch_size", None)

    def __len__(self):
        return len(self._cached) if self._cached is not None else len(self.loader)

    def __iter__(self):
        if self._cached is not None:
            self.hits += 1
            for b in self._cached:
                yield b
            return
        buf, total = ([], 0) if self.budget > 0 else (None, 0)
        for images in self.loader:
            if not isinstance(images, torch.Tensor):
                images = images[0]
            if buf is not None:
                total += images.numel() * images.element_size()
                if total <= self.budget:
                    buf.append(images)
                else:
                    buf = None                    # over budget: stop collecting
            yield images
        # Only reached when the consumer exhausted the iterator; a `break`
        # abandons the generator here and leaves _cached as None.
        if buf is not None:
            self._cached = buf


def predict(model, source, device, use_amp, amp_dtype, tta, perm,
            limit=None, desc="", max_images=None, return_view_std=False,
            pad_to=None):
    """Predict over `source`, optionally stopping after `limit` rows.

    All TTA views of a batch travel through ONE forward call.  `max_images`
    bounds how many images that forward may carry; on OOM it is halved and the
    batch retried, so a mis-sized batch costs seconds rather than the submission.

    `pad_to` pads a short final batch back up to the full batch size by
    repeating its last row, then discards the padded rows.  Every layer in
    eval() is per-sample, so the kept rows are unaffected -- but the model now
    only ever sees ONE input shape, which is what stops cudnn.benchmark from
    re-tuning and torch.compile from re-tracing on the last batch of the run.
    """
    chunks, stds, seen = [], [], 0
    budget = int(max_images) if max_images else None
    with torch.inference_mode():
        for images in tqdm(source, desc=desc, leave=False):
            if not isinstance(images, torch.Tensor):
                images = images[0]
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            real_n = int(images.shape[0])
            if pad_to and device.type == "cuda" and 0 < real_n < int(pad_to):
                pad = images[-1:].repeat(int(pad_to) - real_n, 1, 1, 1)
                images = torch.cat([images, pad], dim=0)

            while True:
                try:
                    got = tta_predict(model, images, use_amp, amp_dtype, tta, perm,
                                      max_images=budget,
                                      return_views=return_view_std)
                    break
                except Exception as exc:                        # noqa: BLE001
                    if not _is_oom(exc) or (budget is not None and budget <= images.shape[0]):
                        raise
                    budget = max(int(images.shape[0]),
                                 (budget or images.shape[0] * len(tta)) // 2)
                    print(f"\n  OOM -> retrying with <= {budget} images per forward")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            if return_view_std:
                out, views = got
                # Spread across TTA views: a label-free confidence signal that
                # costs nothing because the views are already in hand.
                stds.append(views[:, :real_n].std(dim=0, unbiased=False)
                            .mean(dim=1).cpu().numpy())
            else:
                out = got
            chunks.append(out[:real_n].cpu().numpy())
            seen += real_n
            if limit is not None and seen >= limit:
                break

    preds = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), np.float32)
    if return_view_std:
        spread = (np.concatenate(stds, axis=0) if stds
                  else np.zeros((0,), np.float32))
        return preds, spread
    return preds


def free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def report_vram(tag=""):
    """Peak allocator high-water since the last reset, as a fraction of a T4."""
    if not torch.cuda.is_available():
        return None
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"  peak VRAM{tag}: {peak / 2**30:.2f} GiB allocated / "
          f"{reserved / 2**30:.2f} GiB reserved of {total / 2**30:.2f} GiB "
          f"({peak / total:.1%} of the card)")
    return peak


# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=[],
                        help="Per-fold checkpoints (the fallback logit ensemble).")
    parser.add_argument("--soup", nargs="+", default=[],
                        help="Souped weights, one per backbone. Preferred when it "
                             "agrees with the ensemble (see --agreement-threshold).")
    parser.add_argument("--backbones", nargs="+", default=None,
                        help="Backbone per --checkpoints entry, same order.")
    parser.add_argument("--soup-backbones", nargs="+", default=None,
                        help="Backbone per --soup entry, same order.")
    parser.add_argument("--image-sizes", nargs="+", type=int, default=None,
                        help="Input resolution per --checkpoints entry. A model "
                             "MUST be evaluated at the resolution it trained at.")
    parser.add_argument("--soup-sizes", nargs="+", type=int, default=None,
                        help="Input resolution per --soup entry.")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--agreement-sample", type=int, default=96,
                        help="Test studies used to compare soup vs ensemble.")
    parser.add_argument("--agreement-threshold", type=float, default=0.98,
                        help="Mean per-column Spearman above which the soup is "
                             "shipped alone.")
    parser.add_argument("--time-budget-min", type=float, default=0.0,
                        help="Wall clock for this script; 0 disables the guard.")
    parser.add_argument("--cascade-frac", type=float, default=None,
                        help="Fraction of studies (0..1) routed to the full "
                             "ensemble; the rest are decided by model 1 alone. "
                             "0 disables the cascade. Defaults to cfg.cascade_frac.")
    parser.add_argument("--fp32", action="store_true",
                        help="Disable autocast. Use to measure the fp16 delta.")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (its warm-up costs ~1 min/model).")
    parser.add_argument("--set", nargs="+", help="Overrides for Config")
    args = parser.parse_args()

    t0 = time.time()
    cfg = Config.from_args(args)
    budget_s = args.time_budget_min * 60.0
    reset_forward_stats()

    def remaining():
        return budget_s - (time.time() - t0) if budget_s > 0 else float("inf")

    # ── 1. A valid submission exists before anything can go wrong ───────────
    ids, targets, id_col = load_submission_template(cfg)
    if not ids:
        raise SystemExit("No test ids found -- cannot build a submission.")
    cfg.num_classes = len(targets)
    import src.kaggle_data as kd
    kd.KNEE_TARGETS = targets                 # keep the dataset's view consistent

    n, c = len(ids), len(targets)
    write_submission(args.output, ids, targets, id_col, np.full((n, c), 0.5, np.float32))
    print(f"Baseline submission written ({n} rows) -- every later stage only "
          f"improves this file.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        print(f"  {props.name}, sm_{props.major}{props.minor}, "
              f"{props.total_memory / 2**30:.1f} GiB")
    use_amp, amp_dtype = resolve_amp(device, force_fp32=args.fp32)
    print(f"  autocast: {'off (fp32)' if not use_amp else str(amp_dtype).split('.')[-1]}")

    # ── 2. Data ─────────────────────────────────────────────────────────────
    test_df = pd.DataFrame({"StudyInstanceUID": ids})
    image_dir = os.path.join(cfg.data_dir, "test_series")
    _sources, _sub_sources = {}, {}
    ram_budget = int(float(getattr(cfg, "infer_ram_cache_gb", 4.0)) * 2**30)

    def _dataset_for(df, size):
        c = copy.copy(cfg)
        c.image_size = int(size)
        # RSNADataset's cache filename hashes image_size into selection_signature,
        # so a shared directory is already safe; keeping one directory per size
        # additionally keeps a 288px run from evicting a 224px run's entries.
        cache = os.path.join(getattr(cfg, "cache_dir", "/tmp/rsna_cache"),
                             f"test_sz{int(size)}_ch{cfg.in_channels}")
        os.makedirs(cache, exist_ok=True)
        return RSNADataset(df, image_dir, c, is_train=False, cache_dir=cache), cache

    def _loader(df, size, batch):
        ds, cache = _dataset_for(df, size)
        kw = dict(batch_size=int(batch), shuffle=False,
                  num_workers=int(getattr(cfg, "num_workers", 4)), pin_memory=True)
        if kw["num_workers"] > 0:
            kw.update(prefetch_factor=4, persistent_workers=True)
        return DataLoader(ds, **kw), cache

    def source_for(size, batch):
        """Memoised BatchSource for the whole test set at `size` pixels.

        Keyed on resolution only -- that IS the preprocessing group.  Every
        model at the same resolution shares one decode and, when it fits the RAM
        budget, one set of collated tensors.
        """
        size = int(size)
        if size not in _sources:
            dl, cache = _loader(test_df, size, batch)
            _sources[size] = BatchSource(dl, ram_budget)
            print(f"  test loader @ {size}px, batch {batch} (cache {cache})")
        return _sources[size]

    def subset_source_for(size, batch, idx):
        """Memoised BatchSource over the cascade's uncertain studies.

        The subset is fixed once model 1 has run, so every later model at the
        same resolution reuses one loader (and one worker pool) instead of
        spawning its own.
        """
        size = int(size)
        if size not in _sub_sources:
            sub_df = test_df.iloc[idx].reset_index(drop=True)
            dl, _ = _loader(sub_df, size, batch)
            _sub_sources[size] = BatchSource(dl, ram_budget)
        return _sub_sources[size]

    perm, pairs, _ = build_lateral_swap(targets)
    if perm is not None:
        print(f"Laterality: flip-TTA will swap {len(pairs)} medial/lateral pair(s).")
    else:
        print("Laterality: no medial/lateral pair found -- flip views will not be "
              "used for those columns.")

    def _align(values, n, default):
        vals = list(values) if values else []
        if len(vals) != n:
            vals = [default] * n
        return vals

    soup_bbs = _align(args.soup_backbones, len(args.soup), cfg.backbone)
    ens_bbs = _align(args.backbones, len(args.checkpoints), cfg.backbone)
    soup_sz = _align(args.soup_sizes, len(args.soup), cfg.image_size)
    ens_sz = _align(args.image_sizes, len(args.checkpoints), cfg.image_size)

    soup_ck = [(p, b, s) for p, b, s in zip(args.soup, soup_bbs, soup_sz)
               if os.path.exists(p)]
    ens_ck = [(p, b, s) for p, b, s in zip(args.checkpoints, ens_bbs, ens_sz)
              if os.path.exists(p)]
    for p in set(args.soup) - {c for c, _, _ in soup_ck}:
        print(f"WARNING: soup checkpoint missing: {p}")
    for p in set(args.checkpoints) - {c for c, _, _ in ens_ck}:
        print(f"WARNING: checkpoint missing: {p}")
    if not soup_ck and not ens_ck:
        print("No usable checkpoints. Keeping the baseline submission.")
        return

    compile_model = (not args.no_compile) and bool(getattr(cfg, "infer_compile", True))

    # ── 3. Size the compute batch from the real device ──────────────────────
    tta_n = int(getattr(cfg, "tta_n", 5))
    n_models = len(soup_ck) if soup_ck else len(ens_ck)
    probe_path, probe_bb, probe_size = (soup_ck[0] if soup_ck else ens_ck[0])

    probe_model = load_model(probe_path, probe_bb, cfg, device, image_size=probe_size,
                             compile_model=False)
    max_images = autotune_compute_batch(probe_model, int(probe_size), cfg.in_channels,
                                        device, use_amp, amp_dtype, cfg)
    # The DATA batch is the compute batch divided by the TTA multiplier, so one
    # data batch expands to exactly one forward.
    data_batch = max(1, min(int(max_images) // max(1, tta_n), int(max_images)))
    print(f"Compute batch: {max_images} images/forward "
          f"-> data batch {data_batch} studies x {tta_n} views")

    # ── 4. Budget the TTA before spending it ────────────────────────────────
    if budget_s > 0:
        probe_tta = build_tta(1, probe_size, probe_size)
        probe_n = min(len(ids), max(data_batch, 16))
        t = time.time()
        predict(probe_model, source_for(probe_size, data_batch), device, use_amp,
                amp_dtype, probe_tta, perm, limit=probe_n, desc="Calibrating",
                max_images=max_images)
        per_study_per_view = (time.time() - t) / max(1, probe_n)

        projected = per_study_per_view * len(ids) * tta_n * n_models
        reserve = 120.0
        print(f"Measured {per_study_per_view * 1000:.0f} ms per study-view "
              f"(includes DICOM decode).")
        print(f"Projected: {projected / 60:.1f} min for {n_models} model(s) x "
              f"{tta_n} views x {len(ids)} studies. "
              f"Remaining budget: {remaining() / 60:.1f} min.")
        while tta_n > 1 and projected > remaining() - reserve:
            tta_n = 3 if tta_n == 5 else 1
            projected = per_study_per_view * len(ids) * tta_n * n_models
            print(f"  over budget -> cutting TTA to {tta_n} views "
                  f"({projected / 60:.1f} min)")
        if projected > remaining() - reserve and len(soup_ck) > 1:
            soup_ck = soup_ck[:1]
            print(f"  still over budget -> shipping a single backbone")
        # data_batch is deliberately NOT recomputed here. The loaders are
        # memoised per resolution and one may already exist, so a new value
        # would disagree with what the loader emits -- and predict()'s pad_to
        # would then pad every batch up to a size it never produces. Cutting the
        # TTA only makes each forward smaller (B*M shrinks), which is safe;
        # max_images still governs how the views are grouped.
    free(probe_model)

    print(f"TTA: {tta_n} view(s) per study (built per model, at its own "
          f"resolution).\n")

    # ── 5. Soup vs ensemble, decided on evidence, not on faith ──────────────
    use_soup = bool(soup_ck)
    if soup_ck and ens_ck and len(ids) >= 8:
        k = min(args.agreement_sample, len(ids))
        print(f"Agreement test on {k} test studies "
              f"(soup vs {len(ens_ck)}-model ensemble)...")

        def _sample(group, tag):
            out = []
            for path, bb, size in group:
                m = load_model(path, bb, cfg, device, image_size=size,
                               compile_model=False)
                out.append(predict(m, source_for(size, data_batch), device, use_amp,
                                   amp_dtype, build_tta(tta_n, size, size), perm,
                                   limit=k, max_images=max_images,
                                   desc=f"{tag} {os.path.basename(path)}"))
                free(m)
            return out

        s_parts = _sample(soup_ck, "soup")
        e_parts = _sample(ens_ck, "ens")
        k = min(min(p.shape[0] for p in s_parts), min(p.shape[0] for p in e_parts))
        rho = agreement(rank_average([p[:k] for p in s_parts]),
                        rank_average([p[:k] for p in e_parts]))
        use_soup = np.isfinite(rho) and rho >= args.agreement_threshold
        print(f"  mean per-column Spearman: {rho:.4f} "
              f"(threshold {args.agreement_threshold})")
        print(f"  -> shipping the {'SOUP' if use_soup else 'ENSEMBLE'}"
              f"{' (1 model, ~%dx cheaper)' % max(1, len(ens_ck)) if use_soup else ' (soup diverged from it)'}\n")

    chosen = soup_ck if use_soup else (ens_ck or soup_ck)

    # ── 6. Cascade: how much of the ensemble does every study really need? ──
    cascade_frac = (args.cascade_frac if args.cascade_frac is not None
                    else float(getattr(cfg, "cascade_frac", 0.0)))
    cascade_frac = float(min(max(cascade_frac, 0.0), 1.0))
    # The routing signal is the spread across TTA views, so a single-view run
    # has nothing to route on and a single model has nowhere to route to.
    cascade_on = cascade_frac > 0.0 and len(chosen) > 1 and tta_n > 1

    # ── 7. Full-set prediction, one model in memory at a time ───────────────
    per_model, subset_preds, subset_idx = [], [], np.array([], dtype=int)
    for i, (path, bb, size) in enumerate(chosen, 1):
        if remaining() < 90 and per_model:
            print(f"Out of budget after {i - 1}/{len(chosen)} models -- "
                  f"submitting what is finished.")
            break
        print(f"Model {i}/{len(chosen)}: {os.path.basename(path)} ({bb} @ {size}px)")
        warm = _sources.get(int(size))
        if warm is not None and warm.cached:
            print(f"  reusing {len(warm)} collated batch(es) from RAM -- no DICOM "
                  f"decode, no .npy read, no transform for this model")
        m = load_model(path, bb, cfg, device, image_size=size,
                       compile_model=compile_model)

        if cascade_on and i > 1:
            # Only the uncertain studies. Same cache directory, so every study
            # here is a .npy cache hit -- no DICOM is decoded twice.
            # Never a batch larger than the subset: pad_to would otherwise
            # inflate the single short batch back up to the full data batch.
            sub_src = subset_source_for(size, min(data_batch, len(subset_idx)),
                                        subset_idx)
            p = predict(m, sub_src, device, use_amp, amp_dtype,
                        build_tta(tta_n, size, size), perm,
                        max_images=max_images, pad_to=sub_src.batch_size,
                        desc=f"  {os.path.basename(path)} [cascade {len(subset_idx)}]")
            free(m)
            if p.shape[0] != len(subset_idx):
                print(f"  WARNING: cascade got {p.shape[0]} rows for "
                      f"{len(subset_idx)} studies -- skipping.")
                continue
            subset_preds.append(p)
            write_submission(args.output, ids, targets, id_col,
                             cascade_merge(per_model[0], subset_idx,
                                           [per_model[0][subset_idx]] + subset_preds))
            print(f"  submission.csv updated: cascade over {len(subset_idx)} "
                  f"study(ies) with {len(subset_preds) + 1} model(s)")
            continue

        want_std = cascade_on and i == 1
        full_src = source_for(size, data_batch)
        got = predict(m, full_src, device, use_amp, amp_dtype,
                      build_tta(tta_n, size, size), perm, max_images=max_images,
                      return_view_std=want_std, pad_to=full_src.batch_size,
                      desc=f"  {os.path.basename(path)}")
        free(m)
        if want_std:
            p, spread = got
        else:
            p, spread = got, None
        if p.shape[0] != len(ids):
            print(f"  WARNING: got {p.shape[0]} rows for {len(ids)} ids -- skipping.")
            cascade_on = False
            continue
        per_model.append(p)

        if want_std and spread is not None and spread.shape[0] == len(ids):
            k = max(1, int(round(cascade_frac * len(ids))))
            subset_idx = np.argsort(-spread)[:k]
            subset_idx.sort()
            print(f"  cascade: {k}/{len(ids)} study(ies) ({k / len(ids):.0%}) have the "
                  f"widest TTA spread and go to the rest of the ensemble; "
                  f"the other {len(ids) - k} are decided here.")
        elif want_std:
            cascade_on = False

        # Rewrite the submission after EVERY model, so the file on disk always
        # reflects the best ensemble completed so far.
        write_submission(args.output, ids, targets, id_col, rank_average(per_model))
        print(f"  submission.csv updated with {len(per_model)} model(s)")

    if not per_model:
        print("No model produced usable predictions. The baseline submission stands.")
        return

    if cascade_on and subset_preds:
        values = cascade_merge(per_model[0], subset_idx,
                               [per_model[0][subset_idx]] + subset_preds)
        detail = (f"1 model on all {len(ids)}, "
                  f"{len(subset_preds) + 1} on {len(subset_idx)}")
    else:
        values = rank_average(per_model)
        detail = f"{len(per_model)} model(s) x {tta_n} view(s)"

    sub = write_submission(args.output, ids, targets, id_col, values)
    calls, imgs = FORWARD_STATS["calls"], FORWARD_STATS["images"]
    print(f"\nFinal: {args.output} -- {len(sub)} rows, {len(targets)} targets, "
          f"{detail}, {(time.time() - t0) / 60:.1f} min elapsed.")
    print(f"Cost: {calls} forward call(s), {imgs} image(s) "
          f"({imgs / max(1, len(ids)):.1f} images and {calls / max(1, len(ids)):.3f} "
          f"forward calls per study).")
    report_vram()
    print(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
