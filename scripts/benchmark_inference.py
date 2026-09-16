#!/usr/bin/env python3
"""Measure -- never assume -- the cost and the correctness of src/infer.py.

Five independent modes; run them all with `--mode all`.

  equivalence  Prove the optimisations did not move a prediction. Compares the
               old inference path (no_grad + one forward per TTA view + batch 8)
               against the new one (inference_mode + one forward for all views +
               a large batch), and reports the max absolute delta AND the
               pairwise-inversion rate, which is the only thing ROC-AUC can see.

  throughput   Per-study latency and forward-call count vs batch size, for the
               sequential-TTA and batched-TTA paths. This is the "before/after"
               table.

  memory       Peak activation bytes per image and, on CUDA, what fraction of
               the card a given batch actually uses -- so the batch can be sized
               from a measurement instead of a guess.

  io           Proves the DICOM decode happens once per preprocessing group:
               times a cold pass (decode) against warm passes (cache hit).

  sweep        AUC vs number of models N and AUC vs TTA views M, on labelled
               out-of-fold studies. Runs every checkpoint ONCE at the full 5
               views, caches the per-view predictions, and then evaluates every
               (N, M) offline -- build_tta is a prefix family, so a single pass
               answers the whole grid exactly.

Nothing here writes outside --out-dir (default /tmp) or the pixel cache.
"""

import os
import sys
import gc
import copy
import json
import time
import argparse
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.config import Config
from src.kaggle_data import RSNADataset, build_lateral_swap
from src import infer as I


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def describe_device(device):
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.1f} GiB"
    return str(device)


def macro_auc(y_true, y_score):
    """Macro ROC-AUC over the columns that HAVE both classes.

    Silently averaging over columns where every study is negative is how a
    benchmark reports a number that the leaderboard will not reproduce, so the
    usable-column count is returned alongside the score.
    """
    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if 0 < col.sum() < len(col):
            aucs.append(roc_auc_score(col, y_score[:, j]))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def inversion_rate(a, b):
    """Fraction of within-column study PAIRS ordered differently by a and b.

    ROC-AUC is a sum over exactly these pairs, so this -- not the absolute
    difference between two probabilities -- is the number that decides whether
    an optimisation changed the score.
    """
    bad = tot = 0
    for j in range(a.shape[1]):
        x, y = a[:, j], b[:, j]
        dx = np.sign(x[:, None] - x[None, :])
        dy = np.sign(y[:, None] - y[None, :])
        iu = np.triu_indices(len(x), k=1)
        bad += int(np.sum(dx[iu] != dy[iu]))
        tot += len(iu[0])
    return bad / max(1, tot), bad, tot


# ── the OLD inference path, kept verbatim so "before" is really before ─────

def legacy_tta_predict(model, images, use_amp, amp_dtype, tta, perm=None):
    """src/infer.py's tta_predict as it stood before this work: one forward per
    TTA view, inside torch.no_grad()."""
    out = None
    ctx = (torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True)
           if use_amp else _null())
    with ctx:
        for aug_fn, needs_swap, weight in tta:
            pred = torch.sigmoid(model(aug_fn(images))).float()
            if needs_swap and perm is not None:
                pred = pred[:, perm]
            out = pred * weight if out is None else out + pred * weight
    return out


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def legacy_predict(model, loader, device, use_amp, amp_dtype, tta, perm):
    chunks = []
    with torch.no_grad():
        for images in loader:
            if not isinstance(images, torch.Tensor):
                images = images[0]
            images = images.to(device)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            chunks.append(legacy_tta_predict(model, images, use_amp, amp_dtype,
                                             tta, perm).cpu().numpy())
    return np.concatenate(chunks, 0) if chunks else np.zeros((0, 1), np.float32)


# ── model / data plumbing shared by the modes ─────────────────────────────

def build(ckpt, backbone, size, cfg, device, compile_model=False):
    m = I.load_model(ckpt, backbone, cfg, device, image_size=size,
                     compile_model=compile_model)
    return m


def make_loader(ids, image_dir, cfg, size, batch, workers, cache_root):
    c = copy.copy(cfg)
    c.image_size = int(size)
    cache = os.path.join(cache_root, f"bench_sz{int(size)}_ch{cfg.in_channels}")
    os.makedirs(cache, exist_ok=True)
    ds = RSNADataset(pd.DataFrame({"StudyInstanceUID": list(ids)}), image_dir, c,
                     is_train=False, cache_dir=cache)
    kw = dict(batch_size=int(batch), shuffle=False, num_workers=int(workers),
              pin_memory=(torch.cuda.is_available()))
    if kw["num_workers"] > 0:
        kw.update(prefetch_factor=4, persistent_workers=True)
    return DataLoader(ds, **kw), cache


def collect_batches(loader, device):
    """Materialise the whole (small) benchmark set once, on the host."""
    out = []
    for b in loader:
        if not isinstance(b, torch.Tensor):
            b = b[0]
        out.append(b)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Mode: equivalence
# ══════════════════════════════════════════════════════════════════════════

def mode_equivalence(args, cfg, device):
    print("\n" + "=" * 74)
    print("EQUIVALENCE  -- does the fast path predict the same thing?")
    print("=" * 74)

    size = args.image_size
    model = build(args.checkpoints[0], args.backbones[0], size, cfg, device)
    perm, pairs, _ = build_lateral_swap(list(args.target_cols))
    tta = I.build_tta(args.tta, size, size)
    print(f"laterality pairs swapped on flip views: {pairs}")

    # REAL pixels wherever possible. Gaussian noise makes the model's outputs
    # collapse into a tight cluster, which inflates the pairwise-inversion rate
    # of any tiny numeric perturbation far above what real, spread-out
    # predictions would show -- i.e. it makes fp16 look much worse than it is.
    x_all, src = None, "synthetic N(0,1)"
    if args.study_ids and os.path.isdir(args.image_dir):
        try:
            dl, _ = make_loader(args.study_ids[:args.n_studies], args.image_dir,
                                cfg, size, 32, args.workers, args.out_dir)
            x_all = torch.cat(collect_batches(dl, device), 0)
            src = f"{args.image_dir}"
            del dl
        except Exception as e:
            print(f"  could not load real studies ({e}); falling back to noise.")
    if x_all is None:
        g = torch.Generator().manual_seed(0)
        x_all = torch.randn(args.n_studies, cfg.in_channels, size, size, generator=g)
    args.n_studies = int(x_all.shape[0])
    print(f"inputs: {args.n_studies} studies from {src}")

    def legacy(batch):
        outs = []
        with torch.no_grad():
            for i in range(0, len(x_all), batch):
                xb = x_all[i:i + batch].to(device)
                outs.append(legacy_tta_predict(model, xb, False, torch.float32,
                                               tta, perm).cpu().numpy())
        return np.concatenate(outs, 0)

    def fast(batch, pad=False):
        outs = []
        with torch.inference_mode():
            for i in range(0, len(x_all), batch):
                xb = x_all[i:i + batch].to(device)
                n = xb.shape[0]
                if pad and n < batch:
                    xb = torch.cat([xb, xb[-1:].repeat(batch - n, 1, 1, 1)], 0)
                y = I.tta_predict(model, xb, False, torch.float32, tta, perm,
                                  max_images=batch * len(tta))
                outs.append(y[:n].cpu().numpy())
        return np.concatenate(outs, 0)

    ref = legacy(8)                              # the shipped "before" behaviour
    rows = []

    def check(tag, got):
        d = float(np.abs(got - ref).max())
        rate, bad, tot = inversion_rate(ref, got)
        rows.append((tag, d, rate, bad, tot))
        print(f"  {tag:<44} max|delta| {d:.3e}   inversions {bad}/{tot} ({rate:.2%})")

    print(f"\nreference = legacy (no_grad, {len(tta)} sequential forwards, batch 8), "
          f"{args.n_studies} synthetic studies, fp32")
    check("legacy no_grad, batch 8  (self)", legacy(8))
    check("legacy no_grad, batch 64", legacy(64))
    check("inference_mode + batched TTA, batch 8", fast(8))
    check("inference_mode + batched TTA, batch 64", fast(64))
    check("inference_mode + batched TTA + last-batch pad", fast(64, pad=True))

    # fp16. Casting the whole model to half is STRICTLY more aggressive than
    # autocast (which keeps norms and reductions in fp32), so this bounds the
    # T4 autocast delta from above.
    if device.type in ("cuda", "mps"):
        try:
            half = copy.deepcopy(model).half()
            outs = []
            with torch.inference_mode():
                for i in range(0, len(x_all), 64):
                    xb = x_all[i:i + 64].to(device).half()
                    outs.append(I.tta_predict(half, xb, False, torch.float16, tta,
                                              perm, max_images=64 * len(tta))
                                .float().cpu().numpy())
            check("fp16 weights+activations (bounds autocast)", np.concatenate(outs, 0))
            del half
        except Exception as e:
            print(f"  fp16 path unavailable on {device}: {e}")

    if device.type == "cuda":
        outs = []
        with torch.inference_mode():
            for i in range(0, len(x_all), 64):
                xb = x_all[i:i + 64].to(device)
                outs.append(I.tta_predict(model, xb, True, torch.float16, tta, perm,
                                          max_images=64 * len(tta)).cpu().numpy())
        check("autocast fp16 (what a T4 actually runs)", np.concatenate(outs, 0))

    I.free(model)
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Mode: throughput
# ══════════════════════════════════════════════════════════════════════════

def mode_throughput(args, cfg, device):
    print("\n" + "=" * 74)
    print("THROUGHPUT  -- per-study latency and forward calls, before vs after")
    print("=" * 74)

    size = args.image_size
    model = build(args.checkpoints[0], args.backbones[0], size, cfg, device)
    perm, _, _ = build_lateral_swap(list(args.target_cols))
    tta = I.build_tta(args.tta, size, size)
    M = len(tta)

    def timed(fn, batch, reps=args.reps):
        x = torch.randn(batch, cfg.in_channels, size, size, device=device)
        for _ in range(2):
            fn(x)
        sync(device)
        I.reset_forward_stats()
        t = time.time()
        for _ in range(reps):
            fn(x)
        sync(device)
        dt = (time.time() - t) / reps
        return dt / batch * 1000, I.FORWARD_STATS["calls"] / reps / batch

    def old(x):
        with torch.no_grad():
            return legacy_tta_predict(model, x, False, torch.float32, tta, perm)

    def new(x):
        with torch.inference_mode():
            return I.tta_predict(model, x, False, torch.float32, tta, perm,
                                 max_images=x.shape[0] * M)

    print(f"\n{M} TTA view(s), fp32, {describe_device(device)}")
    print(f"{'studies':>8} | {'OLD ms/study':>13} {'fwd/study':>10} | "
          f"{'NEW ms/study':>13} {'fwd/study':>10} | {'speedup':>8}")
    print("-" * 74)
    rows = []
    for b in args.batches:
        try:
            ot, oc = timed(old, b)
            # forward-call accounting for the legacy path is analytic: it calls
            # model() once per view, and reset_forward_stats does not see it.
            oc = float(M)
            nt, nc = timed(new, b)
        except Exception as e:
            print(f"{b:>8} | failed: {e}")
            continue
        rows.append(dict(batch=b, old_ms=ot, new_ms=nt, old_fwd=oc, new_fwd=nc))
        print(f"{b:>8} | {ot:>13.2f} {oc:>10.3f} | {nt:>13.2f} {nc:>10.3f} | "
              f"{ot / nt:>7.2f}x")

    if device.type == "cuda":
        for b in args.batches:
            try:
                x = torch.randn(b, cfg.in_channels, size, size, device=device)
                x = x.contiguous(memory_format=torch.channels_last)
                for _ in range(2):
                    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                        I.tta_predict(model, x, True, torch.float16, tta, perm,
                                      max_images=b * M)
                sync(device)
                t = time.time()
                for _ in range(args.reps):
                    with torch.inference_mode():
                        I.tta_predict(model, x, True, torch.float16, tta, perm,
                                      max_images=b * M)
                sync(device)
                ms = (time.time() - t) / args.reps / b * 1000
                print(f"{b:>8} | autocast fp16, batched TTA: {ms:.2f} ms/study")
            except Exception as e:
                print(f"{b:>8} | fp16 failed: {e}")
    I.free(model)
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Mode: memory
# ══════════════════════════════════════════════════════════════════════════

def mode_memory(args, cfg, device):
    print("\n" + "=" * 74)
    print("MEMORY  -- what fraction of the card does inference actually use?")
    print("=" * 74)

    size = args.image_size
    model = build(args.checkpoints[0], args.backbones[0], size, cfg, device)
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    bufs = sum(b.numel() * b.element_size() for b in model.buffers())
    print(f"\nweights resident: {params / 2**20:.2f} MiB params + "
          f"{bufs / 2**20:.2f} MiB buffers")

    # Device-independent: total bytes of every leaf activation, i.e. exactly the
    # set training must keep for backprop. Inference keeps almost none of it.
    cpu_model = build(args.checkpoints[0], args.backbones[0], size, cfg,
                      torch.device("cpu"))
    seen = []
    hooks = [m.register_forward_hook(
        lambda mod, i, o: seen.append(o.numel() * o.element_size())
        if isinstance(o, torch.Tensor) else None)
        for m in cpu_model.modules() if not list(m.children())]
    with torch.inference_mode():
        cpu_model(torch.randn(2, cfg.in_channels, size, size))
    for h in hooks:
        h.remove()
    tot, mx = sum(seen) / 2, max(seen) / 2
    print(f"activation bytes per image @ {size}px, fp32:")
    print(f"  sum of all {len(seen)} leaf outputs (the TRAINING set): {tot / 2**20:7.2f} MiB")
    print(f"  largest single leaf output:                            {mx / 2**20:7.2f} MiB")
    print(f"  -> inference retains no graph, so its live set is a small multiple of")
    print(f"     the largest output, not the sum.")
    del cpu_model
    gc.collect()

    if device.type != "cuda":
        print(f"\n{describe_device(device)} exposes no peak-allocation counter; "
              f"re-run this mode on the T4 for the measured numbers below.")
        I.free(model)
        return []

    T4 = 15.0 * 2**30
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"\n{'images/fwd':>11} {'peak alloc':>12} {'MiB/image':>10} "
          f"{'% of card':>10} {'% of 15.0GiB T4':>17}")
    rows = []
    for n in args.mem_batches:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            x = torch.randn(n, cfg.in_channels, size, size, device=device)
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                model(x)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            del x
            rows.append(dict(images=n, peak=peak))
            print(f"{n:>11} {peak / 2**30:>11.3f}G {(peak - base) / n / 2**20:>10.3f} "
                  f"{peak / total:>9.1%} {peak / T4:>16.1%}")
        except Exception as e:
            print(f"{n:>11} OOM/err: {str(e)[:60]}")
            torch.cuda.empty_cache()
            break
    I.free(model)
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Mode: io  (decode once per preprocessing group)
# ══════════════════════════════════════════════════════════════════════════

def mode_io(args, cfg, device):
    print("\n" + "=" * 74)
    print("I/O  -- is the DICOM decode shared across models, or paid per model?")
    print("=" * 74)

    ids = args.study_ids[:args.n_studies]
    cache_root = os.path.join(args.out_dir, f"iocache_{int(time.time())}")
    os.makedirs(cache_root, exist_ok=True)

    def run(tag, workers, batch):
        dl, cache = make_loader(ids, args.image_dir, cfg, args.image_size, batch,
                                workers, cache_root)
        t0 = time.time()
        n = 0
        for b in dl:
            if not isinstance(b, torch.Tensor):
                b = b[0]
            n += b.shape[0]
        dt = time.time() - t0
        entries = len([f for f in os.listdir(cache) if f.endswith(".npy")])
        print(f"  {tag:<34} nw={workers} bs={batch:>3}  {dt:6.2f}s  "
              f"{dt / max(1, n) * 1000:7.2f} ms/study   cache entries: {entries}")
        del dl
        return dt / max(1, n)

    print(f"\n{len(ids)} studies @ {args.image_size}px, fresh cache dir")
    cold = run("pass 1 (cold: DICOM decode)", args.workers, args.batches[-1])
    warm1 = run("pass 2 (warm: .npy + transform)", 0, args.batches[-1])
    warm2 = run("pass 3 (warm: .npy + transform)", 0, args.batches[-1])
    warm = min(warm1, warm2)
    print(f"\n  decode is paid ONCE per preprocessing group: "
          f"{cold * 1000:.2f} ms/study cold vs {warm * 1000:.2f} ms/study warm "
          f"({cold / max(1e-9, warm):.0f}x).")
    print(f"  With K models at one resolution the old cost was ~{cold * 1000:.0f} + "
          f"{(len(ids) and 1)}*(K-1)*{warm * 1000:.2f} ms/study of data work; the "
          f"in-RAM BatchSource removes the (K-1) term as well.")
    return [dict(cold=cold, warm=warm)]


# ══════════════════════════════════════════════════════════════════════════
# Mode: sweep  (AUC vs N models, AUC vs M TTA views)
# ══════════════════════════════════════════════════════════════════════════

def per_view_predictions(ckpt, backbone, size, cfg, device, batches, perm, n_views):
    """(n_views, S, C) sigmoid predictions, lateral-swap already undone.

    Runs the model ONCE at the maximum view count. build_tta(k) is the first k
    entries of build_tta(n_views) with different weights, so every smaller k is
    recoverable from this array exactly -- no re-running.
    """
    tta = I.build_tta(n_views, size, size)
    model = build(ckpt, backbone, size, cfg, device)
    outs = []
    with torch.inference_mode():
        for xb in batches:
            xb = xb.to(device)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            _, views = I.tta_predict(model, xb, False, torch.float32, tta, perm,
                                     max_images=xb.shape[0] * len(tta),
                                     return_views=True)
            outs.append(views.cpu().numpy())
    I.free(model)
    return np.concatenate(outs, axis=1) if outs else np.zeros((n_views, 0, 1), np.float32)


def combine_views(views, m):
    """Weighted average of the first m views -- exactly what build_tta(m) does."""
    w = np.asarray(I.tta_prefix_weights(m), dtype=np.float64)[:, None, None]
    return (views[:m] * w).sum(axis=0)


def mode_sweep(args, cfg, device):
    print("\n" + "=" * 74)
    print("SWEEP  -- how much AUC does the Nth model and the Mth TTA view buy?")
    print("=" * 74)

    if args.labels is None:
        print("  no labelled studies available (--labels-csv / --fold); skipping.")
        return []

    df, target_cols = args.labels
    y = df[target_cols].values.astype(np.float64)
    ids = df["StudyInstanceUID"].astype(str).tolist()
    perm, _, _ = build_lateral_swap(list(target_cols))
    print(f"\n{len(ids)} labelled study(ies), {len(target_cols)} columns, "
          f"{len(args.checkpoints)} checkpoint(s), {args.tta} TTA view(s)")
    pos = y.sum(0)
    usable = [c for c, p in zip(target_cols, pos) if 0 < p < len(y)]
    print(f"  columns with both classes present: {len(usable)}/{len(target_cols)} "
          f"-> macro AUC is over those only")
    if len(ids) < 20:
        print(f"  WARNING: {len(ids)} studies is far too few for a stable AUC. "
              f"Treat every number below as a point estimate with a very wide "
              f"interval (a bootstrap CI is printed for exactly this reason).")

    dl, cache = make_loader(ids, args.image_dir, cfg, args.image_size,
                            args.batches[-1], args.workers, args.out_dir)
    batches = collect_batches(dl, device)
    del dl

    views = []
    for ck, bb in zip(args.checkpoints, args.backbones):
        t = time.time()
        v = per_view_predictions(ck, bb, args.image_size, cfg, device, batches,
                                 perm, args.tta)
        print(f"  {os.path.basename(ck):<28} {v.shape} in {time.time() - t:.1f}s")
        views.append(v)
    views = np.stack(views, 0)                       # (N, M, S, C)
    N, M = views.shape[0], views.shape[1]

    np.savez_compressed(os.path.join(args.out_dir, "sweep_per_view.npz"),
                        views=views, y=y, ids=np.array(ids, dtype=object),
                        target_cols=np.array(list(target_cols), dtype=object))
    print(f"  per-view predictions cached -> {args.out_dir}/sweep_per_view.npz "
          f"(re-sweep offline without touching the GPU)")

    def auc_for(model_idx, m):
        parts = [combine_views(views[i], m) for i in model_idx]
        return macro_auc(y, I.rank_average(parts))[0]

    def boot_ci(model_idx, m, reps=args.boot):
        rng = np.random.default_rng(0)
        parts = [combine_views(views[i], m) for i in model_idx]
        score = I.rank_average(parts)
        vals = []
        for _ in range(reps):
            k = rng.integers(0, len(y), len(y))
            a, _n = macro_auc(y[k], score[k])
            if np.isfinite(a):
                vals.append(a)
        if not vals:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))

    print(f"\n--- AUC vs TTA views M (all {N} model(s), cost = N*M forwards/study) ---")
    print(f"{'M':>3} {'macro AUC':>10} {'delta vs M=1':>13} {'90% CI':>20} "
          f"{'fwd/study':>10}")
    all_idx = list(range(N))
    base = auc_for(all_idx, 1)
    tta_rows = []
    for m in range(1, M + 1):
        a = auc_for(all_idx, m)
        lo, hi = boot_ci(all_idx, m)
        tta_rows.append(dict(m=m, auc=a))
        print(f"{m:>3} {a:>10.4f} {a - base:>+13.4f} "
              f"[{lo:>7.4f},{hi:>7.4f}] {N * m:>10}")

    print(f"\n--- AUC vs ensemble size N (at M={M}, averaged over every subset) ---")
    print(f"{'N':>3} {'mean AUC':>10} {'best':>9} {'worst':>9} "
          f"{'delta vs N=1':>13} {'fwd/study':>10}")
    ens_rows = []
    for n in range(1, N + 1):
        subs = list(itertools.combinations(all_idx, n))
        vals = [auc_for(list(s), M) for s in subs]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        ens_rows.append(dict(n=n, mean=float(np.mean(vals))))
        d = np.mean(vals) - ens_rows[0]["mean"]
        print(f"{n:>3} {np.mean(vals):>10.4f} {max(vals):>9.4f} {min(vals):>9.4f} "
              f"{d:>+13.4f} {n * M:>10}")

    if N > 1:
        print(f"\n--- pairwise agreement between models (Spearman, no labels) ---")
        for i, j in itertools.combinations(all_idx, 2):
            rho = I.agreement(combine_views(views[i], M), combine_views(views[j], M))
            print(f"  {os.path.basename(args.checkpoints[i]):<24} vs "
                  f"{os.path.basename(args.checkpoints[j]):<24} rho={rho:.4f}")
        print("  rho near 1 means the second model re-ranks almost nothing and its "
              "forward passes buy almost no AUC.")

    print(f"\n--- cascade: AUC when only the top-q most TTA-uncertain studies "
          f"get the full ensemble ---")
    if N > 1 and M > 1:
        # Exactly the routing signal src/infer.py uses: the spread of model 1's
        # predictions across its own TTA views. Free -- the views already exist.
        spread = views[0][:M].std(axis=0).mean(axis=1)       # (S,)
        order = np.argsort(-spread)
        base_pred = combine_views(views[0], M)
        full = macro_auc(y, I.rank_average(
            [combine_views(views[i], M) for i in all_idx]))[0]
        one = macro_auc(y, I.rank_normalise(base_pred))[0]
        print(f"{'q':>6} {'studies':>8} {'macro AUC':>10} {'fwd/study':>10}")
        print(f"{'0.00':>6} {0:>8} {one:>10.4f} {M:>10}   (model 1 only)")
        for q in (0.1, 0.25, 0.5, 1.0):
            k = max(1, int(round(q * len(y))))
            idx = np.sort(order[:k])
            merged = I.cascade_merge(base_pred, idx,
                                     [base_pred[idx]] +
                                     [combine_views(views[i], M)[idx] for i in all_idx[1:]])
            a = macro_auc(y, merged)[0]
            cost = M + (N - 1) * M * k / len(y)
            print(f"{q:>6.2f} {k:>8} {a:>10.4f} {cost:>10.2f}")
        print(f"{'1.00*':>6} {len(y):>8} {full:>10.4f} {N * M:>10}   "
              f"(plain rank-average of all {N})")
        print("  q=1.00 is cascade_merge over every study. It reproduces the "
              "rank-average ORDER, refining ties with model 1's own ranking, so "
              "the two rows agree up to tie-breaking (they can differ slightly "
              "on a small set, where ties are common).")
    else:
        print("  needs >=2 models and >=2 TTA views; skipped.")

    return dict(tta=tta_rows, ensemble=ens_rows)


# ══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", nargs="+",
                   default=["equivalence", "throughput", "memory"],
                   choices=["equivalence", "throughput", "memory", "io", "sweep", "all"])
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--backbones", nargs="+", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--tta", type=int, default=5)
    p.add_argument("--batches", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--mem-batches", nargs="+", type=int,
                   default=[8, 40, 80, 160, 320, 640, 1280])
    p.add_argument("--n-studies", type=int, default=128)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default="/tmp/rsna_bench")
    p.add_argument("--image-dir", default=None,
                   help="Directory of study folders for the io/sweep modes.")
    p.add_argument("--labels-csv", default=None,
                   help="CSV with StudyInstanceUID + target columns (for --mode sweep).")
    p.add_argument("--fold", type=int, default=None,
                   help="Keep only rows with this fold value (the OOF split).")
    p.add_argument("--set", nargs="+", help="Overrides for Config")
    args = p.parse_args()

    if "all" in args.mode:
        args.mode = ["equivalence", "throughput", "memory", "io", "sweep"]

    cfg = Config.from_args(args)
    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device(args.device)
    args.image_size = args.image_size or cfg.image_size
    args.backbones = (list(args.backbones) if args.backbones
                      else [cfg.backbone] * len(args.checkpoints))
    if len(args.backbones) != len(args.checkpoints):
        args.backbones = [args.backbones[0]] * len(args.checkpoints)
    args.image_dir = args.image_dir or os.path.join(cfg.data_dir, "train_series")

    # target columns: the submission template is the authority.
    sub = os.path.join(cfg.data_dir, "sample_submission.csv")
    if os.path.exists(sub):
        args.target_cols = list(pd.read_csv(sub, nrows=1).columns[1:])
    else:
        from src.kaggle_data import KNEE_TARGETS
        args.target_cols = list(KNEE_TARGETS)
    cfg.num_classes = len(args.target_cols)

    # study ids for the io mode
    args.study_ids = []
    if os.path.isdir(args.image_dir):
        args.study_ids = sorted(d for d in os.listdir(args.image_dir)
                                if not d.startswith("."))

    # labelled OOF set for the sweep mode
    args.labels = None
    if args.labels_csv:
        path = args.labels_csv if os.path.exists(args.labels_csv) else \
            os.path.join(cfg.data_dir, args.labels_csv)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if args.fold is not None and "fold" in df.columns:
                df = df[df["fold"] == args.fold]
            if "has_images" in df.columns:
                df = df[df["has_images"].astype(bool)]
            have = set(args.study_ids)
            if have:
                df = df[df["StudyInstanceUID"].astype(str).isin(have)]
            cols = [c for c in args.target_cols if c in df.columns]
            if len(df) and len(cols) == len(args.target_cols):
                args.labels = (df.reset_index(drop=True), args.target_cols)
            else:
                print(f"WARNING: {path} has {len(df)} usable rows and "
                      f"{len(cols)}/{len(args.target_cols)} target columns; "
                      f"sweep will be skipped.")
        else:
            print(f"WARNING: labels csv not found: {args.labels_csv}")

    print("=" * 74)
    print(f"device            {describe_device(device)}")
    print(f"checkpoints       {[os.path.basename(c) for c in args.checkpoints]}")
    print(f"backbones         {args.backbones}")
    print(f"image size        {args.image_size}px, in_channels={cfg.in_channels}")
    print(f"cfg.batch_size    {cfg.batch_size}  <- TRAINING batch size")
    print(f"out dir           {args.out_dir}")
    print("=" * 74)

    results = {}
    for m in args.mode:
        fn = dict(equivalence=mode_equivalence, throughput=mode_throughput,
                  memory=mode_memory, io=mode_io, sweep=mode_sweep)[m]
        try:
            results[m] = fn(args, cfg, device)
        except Exception as e:
            import traceback
            print(f"\nmode {m} failed: {e}")
            traceback.print_exc()

    out = os.path.join(args.out_dir, "benchmark_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
