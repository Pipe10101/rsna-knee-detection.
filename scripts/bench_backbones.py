#!/usr/bin/env python3
"""Measure the real cost of every candidate backbone, and plan batch sizes.

Everything printed here is either MEASURED or COMPUTED-FROM-MEASURED; nothing is
asserted from a datasheet.

  params            MEASURED  -- exact ``sum(p.numel())`` of the built model.
  act/sample        MEASURED  -- MiB of saved-for-backward tensors per sample,
                                 via ``torch.autograd.graph.saved_tensors_hooks``
                                 under 16-bit autocast, taken as the slope
                                 ``bytes(B=2) - bytes(B=1)`` over unique
                                 storages so parameter-sized saves cancel.
                                 Tensor shapes/dtypes are device-independent,
                                 so this number transfers to a T4.
  latency           MEASURED  -- on whatever accelerator this machine has
                                 (Apple MPS here).  INDICATIVE ONLY for a T4:
                                 use the *ratios* between backbones, not the
                                 absolute seconds.
  peak VRAM         COMPUTED  -- optimiser state + autocast weight cache +
                                 measured activations + a workspace allowance.
                                 See ``model.estimate_training_memory_gb``.

Usage
-----
    python3 scripts/bench_backbones.py                     # default sweep
    python3 scripts/bench_backbones.py --plan-only         # no measuring, use
                                                           # the baked-in table
    python3 scripts/bench_backbones.py --emit-table        # regenerate the
                                                           # MEASURED_* dicts
    python3 scripts/bench_backbones.py --backbones dinov2_base convnext_tiny \
        --sizes 224 384 --device t4
"""

import argparse
import json
import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (  # noqa: E402
    DEVICE_VRAM_GIB,
    MEASURED_ACT_MIB_PER_SAMPLE,
    RSNA25DModel,
    activation_mib_per_sample,
    estimate_training_memory_gb,
    plan_batch_size,
    resolve_backbone,
    valid_image_size,
)

MIB = 1024 ** 2

DEFAULT_BACKBONES = [
    "effnet_b0", "effnetv2_s", "effnetv2_m",
    "convnext_tiny", "convnext_small", "convnext_base",
    "dinov2_small", "dinov2_base", "dinov2_large",
]
DEFAULT_SIZES = [224, 384]


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #

def saved_activation_bytes(model, x):
    """Unique saved-for-backward storage bytes for one fwd+bwd at this batch."""
    seen, total = set(), 0

    def pack(t):
        nonlocal total
        try:
            st = t.untyped_storage()
            key = (st.data_ptr(), st.nbytes())
        except Exception:
            return t
        if key[0] and key not in seen:
            seen.add(key)
            total += key[1]
        return t

    model.train()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(x)
        out.float().pow(2).mean().backward()
    model.zero_grad(set_to_none=True)
    return total


def accel_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def sync(kind):
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def measure_latency(model, x, iters=5, warmup=2):
    """Seconds per training step (fwd + bwd + AdamW step) on the local device."""
    dev, kind = accel_device()
    if kind == "cpu":
        return None, kind
    model = model.to(dev)
    x = x.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    def step():
        opt.zero_grad(set_to_none=True)
        model(x).float().pow(2).mean().backward()
        opt.step()

    for _ in range(warmup):
        step()
    sync(kind)
    t0 = time.time()
    for _ in range(iters):
        step()
    sync(kind)
    dt = (time.time() - t0) / iters
    model.to("cpu")
    del opt
    if kind == "mps":
        torch.mps.empty_cache()
    elif kind == "cuda":
        torch.cuda.empty_cache()
    return dt, kind


def measure_one(alias, size, in_channels=3, latency_batch=4, grad_ckpt=False,
                do_latency=True):
    torch.manual_seed(0)
    timm_name, is_vit, patch, _ = resolve_backbone(alias)
    eff = valid_image_size(alias, size)
    model = RSNA25DModel(backbone_name=alias, pretrained=False,
                         in_channels=in_channels, num_classes=12,
                         image_size=size, grad_checkpointing=grad_ckpt)
    rec = dict(alias=alias, timm_name=timm_name, is_vit=is_vit, patch=patch,
               requested_size=int(size), effective_size=int(eff),
               params_m=sum(p.numel() for p in model.parameters()) / 1e6,
               feat_dim=model.num_features, layout=model.layout,
               grad_ckpt=bool(grad_ckpt))

    b1 = saved_activation_bytes(model, torch.randn(1, in_channels, eff, eff))
    b2 = saved_activation_bytes(model, torch.randn(2, in_channels, eff, eff))
    rec["act_mib_per_sample"] = (b2 - b1) / MIB

    if do_latency:
        try:
            dt, kind = measure_latency(model, torch.randn(latency_batch, in_channels, eff, eff))
            rec["latency_s_per_batch"] = dt
            rec["latency_batch"] = latency_batch
            rec["latency_device"] = kind
            if dt:
                rec["latency_ms_per_sample"] = 1000.0 * dt / latency_batch
        except Exception as e:  # OOM on a small local GPU is expected for the big ones
            rec["latency_error"] = f"{type(e).__name__}: {e}"[:160]
    del model
    return rec


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_measurements(rows):
    print("\n=== MEASURED ===")
    print(f"{'backbone':<15}{'size':>10}{'params':>10}{'act/sample':>13}"
          f"{'latency':>16}{'per-sample':>12}")
    print("-" * 76)
    for r in rows:
        size = (f"{r['requested_size']}" if r["requested_size"] == r["effective_size"]
                else f"{r['requested_size']}->{r['effective_size']}")
        if r.get("grad_ckpt"):
            size += "*"
        lat = (f"{r['latency_s_per_batch']:.3f}s/b{r['latency_batch']}"
               if r.get("latency_s_per_batch") else r.get("latency_error", "n/a")[:15])
        ps = (f"{r['latency_ms_per_sample']:.0f}ms" if r.get("latency_ms_per_sample") else "-")
        print(f"{r['alias']:<15}{size:>10}{r['params_m']:>9.1f}M"
              f"{r['act_mib_per_sample']:>11.1f}MiB{lat:>16}{ps:>12}")
    if any(r.get("grad_ckpt") for r in rows):
        print("* = gradient checkpointing on")


def print_plans(backbones, sizes, device, effective_batch, max_batch):
    vram = DEVICE_VRAM_GIB.get(str(device).lower(), 15.0)
    print(f"\n=== COMPUTED: what fits on a {device.upper()} "
          f"({vram:.1f} GiB total device memory) ===")
    print(f"{'backbone':<15}{'size':>7}{'ckpt':>6}{'params':>9}{'fixed':>8}"
          f"{'bs':>4}{'accum':>7}{'peak':>8}  verdict")
    print("-" * 82)
    for bb in backbones:
        for sz in sizes:
            for ck in (False, True):
                p = plan_batch_size(bb, sz, device=device,
                                    effective_batch=effective_batch,
                                    max_batch=max_batch, grad_checkpointing=ck)
                verdict = ("OK" if p["fits"] else "OOM")
                if p["fits"] and p["batch_size"] == 1:
                    verdict = "OK (bs=1 only)"
                print(f"{bb:<15}{p['image_size']:>7}{'yes' if ck else 'no':>6}"
                      f"{p['params_m']:>8.0f}M{p['fixed_gib']:>7.1f}G"
                      f"{p['batch_size']:>4}{p['grad_accum_steps']:>7}"
                      f"{p['est_peak_gib']:>7.1f}G  {verdict}")
                if p["fits"]:
                    break   # no need to show the checkpointed variant if it fits


def emit_table(rows):
    """Print the literal dicts to paste into src/model.py."""
    plain = {}
    ckpt = {}
    for r in rows:
        key = (r["alias"], r["effective_size"])
        if r["grad_ckpt"]:
            ckpt[key] = r["act_mib_per_sample"]
        else:
            plain[key] = r["act_mib_per_sample"]
    print("\nMEASURED_ACT_MIB_PER_SAMPLE = {")
    for (a, s), v in sorted(plain.items()):
        print(f'    ("{a}", {s}): {v:.1f},')
    print("}")
    if ckpt:
        print("\n# grad-checkpointing ratios (measured / plain)")
        print("MEASURED_GRAD_CKPT_RATIO = {")
        ratios = {}
        for (a, s), v in ckpt.items():
            base = plain.get((a, s))
            if base:
                ratios.setdefault(a, []).append(v / base)
        for a, rs in sorted(ratios.items()):
            print(f'    "{a}": {sum(rs) / len(rs):.3f},')
        print("}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    ap.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    ap.add_argument("--device", default="t4", help="t4 | p100 | v100 | a100 | l4")
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--effective-batch", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--latency-batch", type=int, default=4)
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="also measure each config with gradient checkpointing")
    ap.add_argument("--plan-only", action="store_true",
                    help="skip measurement; plan from the baked-in measured table")
    ap.add_argument("--emit-table", action="store_true")
    ap.add_argument("--json", default="", help="append one JSON record per line here")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    if not args.plan_only:
        rows = []
        for bb in args.backbones:
            for sz in args.sizes:
                variants = [False, True] if args.grad_ckpt else [False]
                for ck in variants:
                    try:
                        r = measure_one(bb, sz, args.in_channels, args.latency_batch,
                                        ck, not args.no_latency)
                    except Exception as e:
                        print(f"  SKIP {bb}@{sz} ckpt={ck}: {type(e).__name__}: {e}")
                        continue
                    rows.append(r)
                    if args.json:
                        with open(args.json, "a") as f:
                            f.write(json.dumps(r) + "\n")
                    print(f"  measured {bb}@{r['effective_size']} ckpt={int(ck)}")
        print_measurements(rows)
        if args.emit_table:
            emit_table(rows)
        missing = [(r["alias"], r["effective_size"]) for r in rows
                   if not r["grad_ckpt"]
                   and (r["alias"], r["effective_size"]) not in MEASURED_ACT_MIB_PER_SAMPLE]
        if missing:
            print(f"\nNOTE: {len(missing)} config(s) are not in the baked-in table "
                  f"({missing[:4]}...). Re-run with --emit-table and paste into "
                  "src/model.py so the planner stops extrapolating.")

    print_plans(args.backbones, args.sizes, args.device,
                args.effective_batch, args.max_batch)

    print("\nMemory model (COMPUTED):")
    print("  fixed  = n_params x 18 bytes  (fp32 master 4 + fp32 grad 4 + AdamW m,v 8")
    print("           + 2-byte autocast weight cache)  +  0.9 GiB workspace/context")
    print("  peak   = fixed + batch_size x act/sample")
    print("  A T4 reports 15360 MiB = 15.0 GiB TOTAL. Any plan needing ~18 GB on a T4")
    print("  is arithmetically impossible, not merely tight.")


if __name__ == "__main__":
    main()
