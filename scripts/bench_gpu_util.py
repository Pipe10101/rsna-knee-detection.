#!/usr/bin/env python3
"""Measure the training-step cost as a function of batch size.

WHY THIS EXISTS
---------------
``src/model.py`` already answers "how much VRAM does batch B cost?" exactly
(``estimate_training_memory_gb`` + the measured activation table).  It does NOT
answer "how much *time* does batch B cost?", and that is the question that
decides whether a bigger batch is worth anything.

The training step in ``train_one_epoch`` is not just fwd+bwd.  Every optimiser
update also pays:

    clip_grad_norm_(model.parameters())   # a full parameter sweep
    centralize_gradient(model)            # a second full parameter sweep
    optimizer.step()                      # AdamW (+ Lookahead every k=6)
    ema_model.update(model)               # a third full parameter sweep
    scheduler.step()

and every *micro*-batch pays ``loss.item()`` (a device sync) plus a tqdm
postfix update.  All of that is per-STEP, not per-SAMPLE, so it is exactly the
term a larger batch amortises.  The model here fits

    t_step(B) = a + b * B

and reports ``a`` (fixed per-step cost) separately from ``b`` (marginal cost
per sample).  ``a / t_step`` at batch 8 is the fraction of wall-clock the
current config spends on overhead.

DEVICE NOTE
-----------
This machine has no CUDA, so the numbers below are measured on MPS (Apple
silicon) or CPU.  They are an INDICATOR of the *shape* of the curve (where the
fixed cost sits, whether channels_last helps, what grad-checkpointing costs),
not a prediction of T4 seconds.  Any T4 number in the final report is computed
from these ratios and is labelled as computed, never as measured.

USAGE
    python3 scripts/bench_gpu_util.py                       # default sweep
    python3 scripts/bench_gpu_util.py --batches 8 16 32 64
    python3 scripts/bench_gpu_util.py --variants base channels_last ckpt
    python3 scripts/bench_gpu_util.py --json out.json
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.config import Config
from src.model import (build_model, count_parameters, activation_mib_per_sample,
                       estimate_training_memory_gb)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def pick_device(name=""):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_cfg(backbone, image_size, grad_ckpt):
    cfg = Config()
    cfg.backbone = backbone
    cfg.image_size = image_size
    cfg.pretrained = False            # offline + we only care about shapes
    cfg.grad_checkpointing = bool(grad_ckpt)
    return cfg


def time_steps(device, backbone, image_size, batch, *, channels_last, grad_ckpt,
               accum, ema, lookahead_k, clip, gc_grads,
               warmup=3, iters=8, in_channels=3, num_classes=12):
    """Median seconds per OPTIMISER UPDATE at this batch size.

    ``accum`` micro-batches are run per update, so the returned time covers
    ``accum * batch`` samples.  All of the per-update parameter sweeps that
    ``train_one_epoch`` performs are reproduced faithfully; that is the whole
    point of the measurement.
    """
    from timm.utils import ModelEmaV2
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.train import Lookahead, centralize_gradient

    cfg = make_cfg(backbone, image_size, grad_ckpt)
    model = build_model(cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.train()

    ema_model = ModelEmaV2(model, decay=0.999) if ema else None

    inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    opt = Lookahead(inner, k=lookahead_k, alpha=0.5) if lookahead_k else inner

    crit = nn.BCEWithLogitsLoss()
    x = torch.randn(batch, in_channels, image_size, image_size, device=device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    y = (torch.rand(batch, num_classes, device=device) > 0.7).float()

    def one_update():
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            logits = model(x)
            loss = crit(logits.float(), y) / accum
            loss.backward()
            # train_one_epoch calls loss.item() on EVERY micro-batch: a hard
            # host<->device sync that serialises the pipeline.
            float(loss.detach())
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if gc_grads:
            centralize_gradient(model)
        opt.step()
        if ema_model is not None:
            ema_model.update(model)

    for _ in range(warmup):
        one_update()
    _sync(device)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        one_update()
        _sync(device)
        times.append(time.perf_counter() - t0)

    del model, ema_model, opt, inner, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return statistics.median(times), times


def linear_fit(xs, ys):
    """Least-squares fit y = a + b*x. Returns (a, b, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return a, b, r2


VARIANTS = {
    # name              channels_last  grad_ckpt
    "base":             (False, False),
    "channels_last":    (True,  False),
    "ckpt":             (False, True),
    "ckpt_chlast":      (True,  True),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="tf_efficientnet_b0_ns")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--variants", nargs="+", default=["base", "channels_last", "ckpt"],
                   choices=sorted(VARIANTS))
    p.add_argument("--accum", type=int, default=1,
                   help="micro-batches per optimiser update (mimics grad_accum_steps)")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--no-lookahead", action="store_true")
    p.add_argument("--no-gc", action="store_true", help="skip gradient centralization")
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="")
    p.add_argument("--json", default="")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    n_params = count_parameters(args.backbone, args.image_size)
    print(f"device={device}  backbone={args.backbone}@{args.image_size}  "
          f"params={n_params/1e6:.2f}M  accum={args.accum}")
    print("NOTE: non-CUDA timings are an INDICATOR of curve shape, not T4 seconds.\n")

    out = {"device": str(device), "backbone": args.backbone,
           "image_size": args.image_size, "params": n_params,
           "accum": args.accum, "variants": {}}

    for vname in args.variants:
        chlast, ckpt = VARIANTS[vname]
        act = activation_mib_per_sample(args.backbone, args.image_size, ckpt)
        print(f"--- variant={vname}  channels_last={chlast}  grad_ckpt={ckpt}  "
              f"act={act:.2f} MiB/sample ---")
        print(f"{'batch':>6} {'s/update':>10} {'samples/s':>10} {'T4 GiB':>8} {'%15G':>6}")
        rows = []
        for b in args.batches:
            try:
                med, _ = time_steps(
                    device, args.backbone, args.image_size, b,
                    channels_last=chlast, grad_ckpt=ckpt, accum=args.accum,
                    ema=not args.no_ema,
                    lookahead_k=0 if args.no_lookahead else 6,
                    clip=True, gc_grads=not args.no_gc,
                    warmup=args.warmup, iters=args.iters)
            except RuntimeError as e:
                print(f"{b:>6}  FAILED: {str(e)[:60]}")
                continue
            samples = b * args.accum
            gib = estimate_training_memory_gb(n_params, act, b)
            print(f"{b:>6} {med:>10.4f} {samples/med:>10.1f} {gib:>8.2f} "
                  f"{100*gib/15.0:>5.1f}%")
            rows.append({"batch": b, "s_per_update": med,
                         "samples_per_s": samples / med, "t4_gib": gib})

        if len(rows) >= 2:
            a, bslope, r2 = linear_fit([r["batch"] for r in rows],
                                       [r["s_per_update"] for r in rows])
            small = rows[0]
            frac = a / small["s_per_update"] if small["s_per_update"] else 0.0
            print(f"  fit: t_update = {a*1000:.1f} ms + {bslope*1000:.3f} ms/sample "
                  f"(r2={r2:.4f})")
            print(f"  fixed per-update overhead is {100*frac:.1f}% of the step at "
                  f"batch {small['batch']}")
            out["variants"][vname] = {"rows": rows, "fixed_s": a,
                                      "per_sample_s": bslope, "r2": r2}
        else:
            out["variants"][vname] = {"rows": rows}
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {args.json}")
    return out


if __name__ == "__main__":
    main()
