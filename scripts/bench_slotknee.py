#!/usr/bin/env python3
"""Benchmark the SlotKnee-S inference path on THIS machine (MPS, with a CPU reference).

Standalone on purpose: it does NOT import src/slots.py or src/slotknee.py (they may not
exist yet).  It carries its own minimal header pass / slot assignment / slice selection /
crop+resize / encoder harness that follows docs/slotknee_spec.md §3–§4 closely enough to
time them.  Everything printed is MEASURED here unless a column is marked "est." — the
T4 projection is an extrapolation whose assumptions are printed next to the table.

Stages
  (a) header pass      pydicom stop_before_pixels + specific_tags over EVERY file of the study
  (b) decode           the 6·G·T selected slices -> 140 mm centre crop -> cv2.INTER_AREA P×P ->
                       per-slot 1–99 percentile uint8.  Also "full decode" (every file, no
                       crop) so header-first vs full decode can be compared.
  (c) encoder forward  timm vit_small_patch14_dinov2.lvd142m, 6·G images per study,
                       P ∈ {224, 252}, fp32 / fp16-autocast / fp16-half, encoder_chunk ∈ {18, 64},
                       B ∈ {1, 4} studies, G ∈ {1, 2, 3}
  (d) memory           peak RSS (resource.getrusage) + torch.mps current/driver allocated
  (e) alternatives     vit_tiny_patch16_224 / efficientnet_b0 / convnext_nano /
                       mobilenetv3_small_100 at the same image counts (record only)
  (f) --train-probe    opt-in: fwd+bwd step time and memory for bs ∈ {4, 8} with 2 / 4
                       trainable blocks (fp32 on MPS, as scripts/train_slotknee.py would run)

Usage
  python3 scripts/bench_slotknee.py --quick                 # < 3 min, 4 studies, fewer iters
  python3 scripts/bench_slotknee.py                         # full grid (~6–8 min)
  python3 scripts/bench_slotknee.py --train-probe           # adds the training-memory probe
  python3 scripts/bench_slotknee.py --json bench.json       # also dump every record
"""

import argparse
import copy
import glob
import io
import json
import os
import platform
import resource
import statistics
import sys
import time
from collections import defaultdict

# Never touch the network: the DINOv2 weights live in the local HF cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

try:
    import timm
except ImportError:  # pragma: no cover - timm is a hard dependency of the repo
    timm = None

# ---------------------------------------------------------------------------
# Constants (spec §0, §3) and projection assumptions (printed with the tables)
# ---------------------------------------------------------------------------
SLOT_NAMES = ["SAG_FS", "COR_FS", "AX_FS", "SAG_T1", "COR_T1", "AX_T1"]
PLANE_IDX = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
N_TEST_STUDIES = 1300
J_SECONDS_PER_AUC = 71_700          # 0.01 AUC == 717 s
CROP_MM = 140.0
TRIM_FRAC = 0.15
DINO = "vit_small_patch14_dinov2.lvd142m"
ALT_ENCODERS = [
    "vit_tiny_patch16_224.augreg_in21k",
    "efficientnet_b0",
    "convnext_nano",
    "mobilenetv3_small_100",
]

# --- T4 projection assumptions (estimates, NOT measurements) -----------------
T4_MS_PER_IMG_224_FP16 = 1.0   # ViT-S/14 @224, fp16, batched: ≈1 ms/img on a T4 (brief §4)
T4_FP32_OVER_FP16 = 4.0        # T4 has 8.1 TFLOPS fp32 vs 65 fp16 tensor-core; ViT-S sees ~4×
T4_CPU_SLOWDOWN = 2.5          # Kaggle 2-vCPU Xeon vs one M5 Pro P-core on pydicom/cv2 work
T4_FIXED_OVERHEAD_S = 120.0    # container start + imports + checkpoint load (not per study)
JPEG_MS_PER_SLICE = 7.0        # brief §4: JPEG-lossless / J2K decode budget 5–10 ms/slice

# Tags read in the header pass (spec §3.1–3.4 need all of these; nothing else).
HDR_TAGS = [
    "SeriesInstanceUID", "InstanceNumber", "ImagePositionPatient",
    "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns", "Laterality",
    "ScanOptions", "SeriesDescription", "RepetitionTime", "EchoTime",
]

MIB = float(2 ** 20)


def rss_mb():
    """Peak RSS of this process in MiB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / MIB if sys.platform == "darwin" else v / 1024.0


def mps_mem_mb():
    if torch.backends.mps.is_available():
        try:
            return (torch.mps.current_allocated_memory() / MIB,
                    torch.mps.driver_allocated_memory() / MIB)
        except Exception:  # noqa: BLE001
            return float("nan"), float("nan")
    return float("nan"), float("nan")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def timeit(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
        sync(device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3, min(ts) * 1e3


# ---------------------------------------------------------------------------
# (a) header pass, slot assignment, slice selection  (spec §3.1–3.2)
# ---------------------------------------------------------------------------
def _derive_plane(iop):
    if iop is None or len(iop) != 6:
        return None
    v = np.asarray(iop, dtype=float)
    n = np.cross(v[:3], v[3:])
    if not np.isfinite(n).all() or not n.any():
        return None
    return ("Sagittal", "Coronal", "Axial")[int(np.argmax(np.abs(n)))]


def _fluid_sensitive_from_header(rec):
    so = rec.get("scan_options") or ""
    sd = (rec.get("series_desc") or "").lower()
    if isinstance(so, (list, tuple)):
        so = " ".join(str(s) for s in so)
    so = str(so).upper()
    if "FS" in so.split() or "FS" in so.split("_") or any(k in sd for k in ("fs", "fat", "stir", "t2", "pd")):
        return 1
    tr, te = rec.get("tr"), rec.get("te")
    if tr is not None and te is not None:
        return 1 if (tr > 800) else 0
    return 0


HEAD_BYTES = 8192   # partial-read variant: parse only the first 8 KB (falls back if a tag is missing)


def _study_files(study_dir):
    files = sorted(glob.glob(os.path.join(study_dir, "*", "*.dcm")))
    if not files:
        files = sorted(glob.glob(os.path.join(study_dir, "*.dcm")))
    return files


def header_pass_head(study_dir, n_bytes=HEAD_BYTES):
    """Partial-read variant: dcmread on the first n_bytes only. Returns (ms, n_missing_geometry)."""
    files = _study_files(study_dir)
    t0 = time.perf_counter()
    missing = 0
    for f in files:
        with open(f, "rb") as fh:
            head = fh.read(n_bytes)
        try:
            ds = pydicom.dcmread(io.BytesIO(head), stop_before_pixels=True, specific_tags=HDR_TAGS, force=True)
            ok = all(hasattr(ds, t) for t in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows"))
        except Exception:  # noqa: BLE001 - truncated inside an element
            ok = False
        missing += int(not ok)
    return (time.perf_counter() - t0) * 1e3, missing


def header_pass(study_dir, specific=True):
    """Read every DICOM header in a study. Returns (records, ms, n_files)."""
    files = _study_files(study_dir)
    t0 = time.perf_counter()
    recs = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True,
                             specific_tags=HDR_TAGS if specific else None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        ipp = getattr(ds, "ImagePositionPatient", None)
        pos = None
        if iop is not None and ipp is not None and len(iop) == 6 and len(ipp) == 3:
            v = np.asarray(iop, dtype=float)
            n = np.cross(v[:3], v[3:])
            pos = float(np.dot(np.asarray(ipp, dtype=float), n))
        inst = getattr(ds, "InstanceNumber", None)
        try:
            inst = int(inst)
        except (TypeError, ValueError):
            inst = None
        sp = getattr(ds, "PixelSpacing", None)
        sp = (float(sp[0]), float(sp[1])) if sp is not None and len(sp) == 2 else None
        recs.append(dict(
            path=f, series=str(getattr(ds, "SeriesInstanceUID", os.path.basename(os.path.dirname(f)))),
            pos=pos, inst=inst, spacing=sp,
            rows=int(getattr(ds, "Rows", 0) or 0), cols=int(getattr(ds, "Columns", 0) or 0),
            plane=_derive_plane(iop), laterality=getattr(ds, "Laterality", None),
            scan_options=getattr(ds, "ScanOptions", None), series_desc=getattr(ds, "SeriesDescription", None),
            tr=_float_or_none(getattr(ds, "RepetitionTime", None)),
            te=_float_or_none(getattr(ds, "EchoTime", None)),
        ))
    return recs, (time.perf_counter() - t0) * 1e3, len(files)


def _float_or_none(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def assign_slots(recs, series_rows):
    """{slot_idx: ordered records of the chosen series}. series_rows: {uid: (plane, fs)}."""
    by_series = defaultdict(list)
    for r in recs:
        by_series[r["series"]].append(r)
    chosen = {}
    for suid, rs in by_series.items():
        if suid in series_rows:
            plane, fs = series_rows[suid]
        else:
            plane, fs = rs[0]["plane"], _fluid_sensitive_from_header(rs[0])
        if plane not in PLANE_IDX:
            continue
        s = PLANE_IDX[plane] + (0 if int(fs) == 1 else 3)
        cur = chosen.get(s)
        if cur is None or len(rs) > len(cur) or (len(rs) == len(cur) and suid < cur[0]["series"]):
            chosen[s] = rs
    out = {}
    for s, rs in chosen.items():
        if all(r["pos"] is not None for r in rs):
            rs = sorted(rs, key=lambda r: (r["pos"], r["path"]))
        elif all(r["inst"] is not None for r in rs):
            rs = sorted(rs, key=lambda r: (r["inst"], r["path"]))
        else:
            rs = sorted(rs, key=lambda r: r["path"])
        out[s] = rs
    return out


def select_slices(slots, G, T, trim_frac=TRIM_FRAC):
    """{slot_idx: [G groups of T records]} — spec §3.2 (anchors by linspace, T adjacent)."""
    sel = {}
    for s, rs in slots.items():
        n = len(rs)
        lo = int(round(trim_frac * (n - 1)))
        hi = (n - 1) - lo
        if hi < lo:
            lo = hi = (n - 1) // 2
        anchors = np.linspace(lo, hi, G) if G > 1 else np.array([(lo + hi) / 2.0])
        groups = []
        for a in np.rint(anchors).astype(int):
            idx = [min(max(a + k - T // 2, 0), n - 1) for k in range(T)]
            groups.append([rs[i] for i in idx])
        sel[s] = groups
    return sel


# ---------------------------------------------------------------------------
# (b) decode: read pixels -> fixed-mm centre crop -> INTER_AREA resize -> uint8
# ---------------------------------------------------------------------------
def resize_area(a, P):
    """Spec §3.3: one cv2.INTER_AREA call. Slow at non-integer ratios (478->224: ~1.0 ms;
    exact 2x: 0.015 ms — measured here)."""
    return cv2.resize(a, (P, P), interpolation=cv2.INTER_AREA)


def resize_two_step(a, P):
    """Integer-factor INTER_AREA (fast path) then INTER_LINEAR for the remainder (< 2x, so
    aliasing is mild). Same physical crop, same output size."""
    h, w = a.shape[:2]
    f = min(h // P, w // P)
    if f >= 2:
        a = cv2.resize(a, (w // f, h // f), interpolation=cv2.INTER_AREA)
    return cv2.resize(a, (P, P), interpolation=cv2.INTER_LINEAR)


RESIZERS = {"area": resize_area, "two-step": resize_two_step}


def decode_slice(rec, P, crop_mm=CROP_MM, resize="area"):
    t0 = time.perf_counter()
    ds = pydicom.dcmread(rec["path"])
    a = ds.pixel_array
    t1 = time.perf_counter()
    rows, cols = a.shape[:2]
    sp = rec["spacing"]
    fallback = True
    if sp is not None and sp[0] > 0 and sp[1] > 0:
        ph, pw = int(round(crop_mm / sp[0])), int(round(crop_mm / sp[1]))
        if 0 < ph <= rows and 0 < pw <= cols:
            r0, c0 = (rows - ph) // 2, (cols - pw) // 2
            a = a[r0:r0 + ph, c0:c0 + pw]
            fallback = False
    a = RESIZERS[resize](a.astype(np.float32, copy=False), P)
    t2 = time.perf_counter()
    return a, (t1 - t0) * 1e3, (t2 - t1) * 1e3, fallback, (rows, cols)


def build_study_tensor(sel, P, G, T, resize="area"):
    """x uint8 [6,G,T,P,P], mask [6], timing breakdown (ms)."""
    x = np.zeros((6, G, T, P, P), np.uint8)
    mask = np.zeros(6, np.uint8)
    t_read = t_resize = t_norm = 0.0
    n_dec = n_fb = 0
    shapes = []
    for s, groups in sel.items():
        buf = np.empty((G, T, P, P), np.float32)
        for g, grp in enumerate(groups):
            for t, rec in enumerate(grp):
                img, tr, trs, fb, shp = decode_slice(rec, P, resize=resize)
                buf[g, t] = img
                t_read += tr
                t_resize += trs
                n_dec += 1
                n_fb += int(fb)
                shapes.append(shp)
        t0 = time.perf_counter()
        lo, hi = np.percentile(buf, [1.0, 99.0])
        if hi > lo:
            buf = (buf - lo) * (255.0 / (hi - lo))
        else:
            buf = np.zeros_like(buf)
        x[s] = np.clip(buf, 0, 255).astype(np.uint8)
        mask[s] = 1
        t_norm += (time.perf_counter() - t0) * 1e3
    return x, mask, dict(read=t_read, resize=t_resize, norm=t_norm,
                         total=t_read + t_resize + t_norm, n_decoded=n_dec,
                         n_crop_fallback=n_fb, shapes=shapes)


def full_decode(recs):
    """Decode EVERY file's pixel array (what a non-header-first pipeline pays)."""
    t0 = time.perf_counter()
    for r in recs:
        _ = pydicom.dcmread(r["path"]).pixel_array
    return (time.perf_counter() - t0) * 1e3


# ---------------------------------------------------------------------------
# (c) encoders
# ---------------------------------------------------------------------------
def build_encoder(name, P, pretrained):
    if timm is None:
        raise RuntimeError("timm is required for the encoder benchmark")
    kwargs = dict(num_classes=0, global_pool="")
    if "vit" in name:
        kwargs["img_size"] = P
    try:
        m = timm.create_model(name, pretrained=pretrained, **kwargs)
    except Exception as exc:  # noqa: BLE001 - offline cache miss => untrained copy, timing is identical
        print(f"  [warn] pretrained load failed for {name} ({exc}); using random init (timing is identical)")
        m = timm.create_model(name, pretrained=False, **kwargs)
    cfg = getattr(m, "pretrained_cfg", {}) or {}
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), dtype=torch.float32).view(1, 3, 1, 1)
    return m.eval(), mean, std


def make_forward(model, mean, std, x_u8, chunk, mode, device):
    """Closure that runs all images of x_u8 [N,3,P,P] through the encoder in chunks."""
    N = x_u8.shape[0]
    mean_d, std_d = mean.to(device), std.to(device)
    use_autocast = mode == "fp16-autocast"
    half_in = mode == "fp16-half"
    out_holder = {}

    def run():
        outs = []
        # inference_mode is load-bearing: without it every forward retains ~128 MiB/img of
        # autograd state on MPS (measured: 72 imgs -> 7.6 GiB live, 15.7 GiB driver, 40x slower).
        with torch.inference_mode():
            for i in range(0, N, chunk):
                xb = x_u8[i:i + chunk].float().div_(255.0).sub_(mean_d).div_(std_d)
                if half_in:
                    xb = xb.half()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                    o = model(xb)
                if o.ndim == 3:          # ViT tokens [n, L, D] -> CLS
                    outs.append(o[:, 0].float())
                else:                    # CNN map [n, C, h, w] -> GAP
                    outs.append(o.float().mean((2, 3)))
            out_holder["feat"] = torch.cat(outs)

    return run, out_holder


def bench_encoder_rows(name, rows_cfg, device, warmup, iters, pretrained, ref_feats=None):
    """rows_cfg: list of dicts(P, mode, B, G, chunk). Returns list of result records."""
    results = []
    built = {}
    for cfg in rows_cfg:
        P, mode, B, G, chunk = cfg["P"], cfg["mode"], cfg["B"], cfg["G"], cfg["chunk"]
        key = (P, mode)
        if key not in built:
            if (P, "fp32") not in built:
                m, mean, std = build_encoder(name, P, pretrained)
                built[(P, "fp32")] = (m.to(device), mean, std)
            m32, mean, std = built[(P, "fp32")]
            if mode == "fp16-half":
                built[key] = (copy.deepcopy(m32).half(), mean, std)
            else:
                built[key] = (m32, mean, std)
        model, mean, std = built[key]
        n_img = B * 6 * G
        g = torch.Generator().manual_seed(0)
        x_u8 = torch.randint(0, 256, (n_img, 3, P, P), generator=g, dtype=torch.uint8).to(device)
        empty_cache(device)
        run, holder = make_forward(model, mean, std, x_u8, chunk, mode, device)
        try:
            med, best = timeit(run, warmup, iters, device)
            feat = holder["feat"]
            finite = bool(torch.isfinite(feat).all().item())
            dev_vs_fp32 = float("nan")
            if ref_feats is not None:
                rk = (name, P, B, G)
                if mode == "fp32":
                    ref_feats[rk] = feat.detach().cpu()
                elif rk in ref_feats:
                    dev_vs_fp32 = float((feat.detach().cpu() - ref_feats[rk]).abs().max())
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {name} P={P} {mode} B={B} G={G} chunk={chunk} failed: {exc}")
            med = best = float("nan")
            finite, dev_vs_fp32 = False, float("nan")
        cur, drv = mps_mem_mb()
        results.append(dict(
            encoder=name, device=device.type, P=P, mode=mode, B=B, G=G, chunk=chunk,
            imgs_per_study=6 * G, n_img=n_img, ms_batch=med, ms_batch_min=best,
            ms_study=med / B, ms_img=med / n_img, finite=finite, max_dev_vs_fp32=dev_vs_fp32,
            mps_alloc_mb=cur, mps_driver_mb=drv, rss_mb=rss_mb(),
            params_m=sum(p.numel() for p in model.parameters()) / 1e6,
        ))
        del x_u8
    for k in list(built):
        built[k][0].cpu()
    built.clear()
    empty_cache(device)
    return results


# ---------------------------------------------------------------------------
# (f) training probe: fwd+bwd with the last `trainable` blocks unfrozen (fp32, MPS)
# ---------------------------------------------------------------------------
def train_probe(name, P, bs, G, trainable, device, warmup, iters):
    model, mean, std = build_encoder(name, P, pretrained=True)
    model = model.to(device).train()
    n_blocks = len(model.blocks)
    cut = n_blocks - trainable
    for p in model.parameters():
        p.requires_grad_(False)
    for blk in model.blocks[cut:]:
        for p in blk.parameters():
            p.requires_grad_(True)
    for p in model.norm.parameters():
        p.requires_grad_(True)
    head = nn.Linear(model.num_features * 2, 12).to(device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-5)
    n_img = bs * 6 * G
    g = torch.Generator().manual_seed(0)
    x_u8 = torch.randint(0, 256, (n_img, 3, P, P), generator=g, dtype=torch.uint8).to(device)
    y = torch.rand(bs, 12, generator=g).to(device)
    mean_d, std_d = mean.to(device), std.to(device)

    def step():
        xb = x_u8.float().div_(255.0).sub_(mean_d).div_(std_d)
        with torch.no_grad():
            h = model.patch_embed(xb)
            h = model._pos_embed(h)
            h = model.patch_drop(h)
            h = model.norm_pre(h)
            for blk in model.blocks[:cut]:
                h = blk(h)
        for blk in model.blocks[cut:]:
            h = blk(h)
        h = model.norm(h)
        feat = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)           # CLS + mean patch
        feat = feat.view(bs, 6 * G, -1).mean(1)
        loss = nn.functional.binary_cross_entropy_with_logits(head(feat), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    empty_cache(device)
    med, best = timeit(step, warmup, iters, device)
    cur, drv = mps_mem_mb()
    rec = dict(encoder=name, P=P, bs=bs, G=G, trainable_blocks=trainable, n_img=n_img,
               ms_step=med, ms_step_min=best, ms_img=med / n_img,
               mps_alloc_mb=cur, mps_driver_mb=drv, rss_mb=rss_mb(),
               trainable_params_m=sum(p.numel() for p in params) / 1e6)
    model.cpu()
    del model, head, opt, x_u8
    empty_cache(device)
    return rec


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def f1(v):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.1f}"


def f2(v):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2f}"


def f3(v):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"


def t4_model_ms_per_img(row, k_fp16):
    """Estimated T4 ms/img for an MPS row: scale by the fp16 calibration; fp32 rows ×T4_FP32_OVER_FP16."""
    if row["device"] != "mps" or not np.isfinite(row["ms_img"]):
        return float("nan")
    est = row["ms_img"] * k_fp16
    if row["mode"] == "fp32":
        est *= T4_FP32_OVER_FP16
    return est


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data_subset")
    ap.add_argument("--image-dir", default="train_images")
    ap.add_argument("--series-csv", default="train_series.csv")
    ap.add_argument("--n-studies", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="4 studies, fewer timing iterations (< 3 min)")
    ap.add_argument("--P", nargs="+", type=int, default=[224, 252])
    ap.add_argument("--token-proxy-P", nargs="*", type=int, default=[196, 168],
                    help="extra ViT sizes run only as token-count proxies (fp16-half, B=4, G=3)")
    ap.add_argument("--device", default="auto", help="auto | mps | cpu")
    ap.add_argument("--no-cpu-ref", action="store_true")
    ap.add_argument("--no-alt", action="store_true", help="skip the alternative-encoder table")
    ap.add_argument("--train-probe", action="store_true", help="fwd+bwd memory probe (bs 4/8, 2/4 trainable blocks)")
    ap.add_argument("--train-bs", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--json", default="", help="dump all records here")
    args = ap.parse_args()

    if args.quick:
        n_studies = min(args.n_studies, 4)
        warmup, iters = 1, 3
    else:
        n_studies = args.n_studies
        warmup, iters = 2, 7

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    cpu = torch.device("cpu")
    torch.manual_seed(0)

    print("# SlotKnee-S benchmark\n")
    print(f"- machine: {platform.machine()} / {platform.platform()}; python {platform.python_version()}; "
          f"torch {torch.__version__}; timm {getattr(timm, '__version__', 'n/a')}; pydicom {pydicom.__version__}; "
          f"cv2 {cv2.__version__}; device {device.type}; cv2 threads {cv2.getNumThreads()}")
    print(f"- settings: n_studies={n_studies} warmup={warmup} iters={iters} quick={args.quick} P={args.P}")
    print(f"- RSS at start: {rss_mb():.0f} MiB\n")

    records = dict(env=dict(device=device.type, torch=torch.__version__, quick=args.quick))

    # ------------------------------------------------------------------ (a)+(b)
    img_root = os.path.join(args.data_dir, args.image_dir)
    studies = sorted(d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d)))[:n_studies]
    series_rows = {}
    scsv = os.path.join(args.data_dir, args.series_csv)
    if os.path.exists(scsv):
        sdf = pd.read_csv(scsv)
        for r in sdf.itertuples(index=False):
            series_rows[str(r.SeriesInstanceUID)] = (str(r.Anatomical_Plane), int(r.Fluid_Sensitive))

    stage_rows = []
    per_study = []
    print("## (a)+(b) header pass and decode, per study (P=224, G=3, T=3, crop 140 mm)\n")
    lat_present = 0
    missing_hist = np.zeros(7, int)
    head_missing = 0
    for uid in studies:
        sdir = os.path.join(img_root, uid)
        # header pass: warm the page cache once, then time each mode twice and keep the min
        _ = header_pass(sdir, specific=True)
        recs, ms_hdr, n_files = header_pass(sdir, specific=True)
        ms_hdr = min(ms_hdr, header_pass(sdir, specific=True)[1])
        ms_hdr_all = min(header_pass(sdir, specific=False)[1], header_pass(sdir, specific=False)[1])
        ms_hdr_head, miss = min(header_pass_head(sdir), header_pass_head(sdir))
        head_missing += miss
        slots = assign_slots(recs, series_rows)
        n_missing = 6 - len(slots)
        missing_hist[n_missing] += 1
        lat_present += int(any(r["laterality"] for r in recs))
        sel = select_slices(slots, G=3, T=3)
        x, mask, tb = build_study_tensor(sel, P=224, G=3, T=3)          # warm
        x, mask, tb = build_study_tensor(sel, P=224, G=3, T=3)
        _, _, tb2 = build_study_tensor(sel, P=224, G=3, T=3, resize="two-step")
        decode_g = {}
        for G in (1, 2):
            sg = select_slices(slots, G=G, T=3)
            _, _, tbg = build_study_tensor(sg, P=224, G=G, T=3)
            decode_g[G] = tbg["total"]
        _, _, tb_t1 = build_study_tensor(select_slices(slots, G=3, T=1), P=224, G=3, T=1)
        _, _, tb_t5 = build_study_tensor(select_slices(slots, G=3, T=5), P=224, G=3, T=5)
        _, _, tb_252 = build_study_tensor(sel, P=252, G=3, T=3)
        ms_full = full_decode(recs)
        shp = tb["shapes"][0] if tb["shapes"] else (0, 0)
        per_study.append(dict(
            uid=uid, n_files=n_files, ms_hdr=ms_hdr, ms_hdr_alltags=ms_hdr_all, ms_hdr_head=ms_hdr_head,
            head_missing=miss, n_slots=len(slots), n_decoded=tb["n_decoded"], ms_decode=tb["total"],
            ms_read=tb["read"], ms_resize=tb["resize"], ms_norm=tb["norm"],
            ms_decode_twostep=tb2["total"], ms_resize_twostep=tb2["resize"],
            ms_decode_G1=decode_g[1], ms_decode_G2=decode_g[2], ms_decode_T1=tb_t1["total"],
            ms_decode_T5=tb_t5["total"], ms_decode_252=tb_252["total"], ms_full_decode=ms_full,
            crop_fallback=tb["n_crop_fallback"], shape=f"{shp[0]}x{shp[1]}", rss_mb=rss_mb(),
        ))
        stage_rows.append([uid[-8:], n_files, f1(ms_hdr), f1(ms_hdr_all), f"{f1(ms_hdr_head)} ({miss})",
                           len(slots), tb["n_decoded"],
                           f1(tb["total"]), f1(tb["read"]), f1(tb["resize"]), f1(tb["norm"]),
                           f1(tb2["total"]), f1(decode_g[1]), f1(decode_g[2]), f1(tb_t1["total"]),
                           f1(tb_t5["total"]), f1(tb_252["total"]),
                           f1(ms_full), tb["n_crop_fallback"], f"{shp[0]}x{shp[1]}", f"{rss_mb():.0f}"])
    print(md_table(["study", "files", "hdr ms (12 tags)", "hdr ms (all)", "hdr ms (8KB head) (miss)", "slots",
                    "decoded", "decode ms", "·read", "·resize", "·norm", "decode ms 2-step", "G=1 ms",
                    "G=2 ms", "T=1 ms", "T=5 ms", "P=252 ms", "FULL decode ms", "crop fb", "shape",
                    "RSS MiB"], stage_rows))
    ps = pd.DataFrame(per_study)
    mean = ps.mean(numeric_only=True)
    print(f"\nmean over {len(ps)} studies: {mean.n_files:.0f} files/study; header {mean.ms_hdr:.1f} ms "
          f"({mean.ms_hdr / mean.n_files:.3f} ms/file; all-tags {mean.ms_hdr_alltags:.1f} ms; "
          f"8 KB head {mean.ms_hdr_head:.1f} ms with {head_missing} files missing geometry tags); "
          f"decode of {mean.n_decoded:.0f} slices {mean.ms_decode:.1f} ms "
          f"({mean.ms_decode / max(mean.n_decoded, 1):.2f} ms/slice: read {mean.ms_read:.1f} + resize {mean.ms_resize:.1f} + norm {mean.ms_norm:.1f}); "
          f"two-step resize {mean.ms_decode_twostep:.1f} ms (resize part {mean.ms_resize_twostep:.1f}); "
          f"G=1 {mean.ms_decode_G1:.1f} ms, G=2 {mean.ms_decode_G2:.1f} ms, T=1 {mean.ms_decode_T1:.1f} ms, "
          f"T=5 {mean.ms_decode_T5:.1f} ms, P=252 {mean.ms_decode_252:.1f} ms; "
          f"FULL decode {mean.ms_full_decode:.0f} ms ({mean.ms_full_decode / mean.n_files:.2f} ms/file); "
          f"Laterality tag present in {lat_present}/{len(ps)} studies; missing-slot histogram {missing_hist.tolist()} "
          f"(index = #missing slots)")
    records["studies"] = per_study
    records["decode_mean"] = {k: float(v) for k, v in mean.items()}

    # ------------------------------------------------------------------ (c)
    print("\n## (c) DINOv2-S encoder forward (images per study = 6·G; ms/study = ms per batch / B)\n")
    rows_cfg = []
    for P in args.P:
        for mode in ("fp32", "fp16-autocast", "fp16-half"):
            rows_cfg.append(dict(P=P, mode=mode, B=1, G=3, chunk=64))
    for P in args.P:
        rows_cfg.append(dict(P=P, mode="fp32", B=4, G=3, chunk=64))
        rows_cfg.append(dict(P=P, mode="fp16-half", B=4, G=3, chunk=64))
        rows_cfg.append(dict(P=P, mode="fp16-half", B=4, G=3, chunk=18))
    rows_cfg.append(dict(P=args.P[0], mode="fp32", B=4, G=3, chunk=18))
    rows_cfg.append(dict(P=args.P[0], mode="fp16-autocast", B=4, G=3, chunk=64))
    for B in (1, 4):
        for G in (1, 2):
            rows_cfg.append(dict(P=args.P[0], mode="fp16-half", B=B, G=G, chunk=64))
            rows_cfg.append(dict(P=args.P[0], mode="fp32", B=B, G=G, chunk=64))
    # token-count proxies for token pruning / a lower-resolution slot: 196 -> 14x14 = 196 patches
    # (-24% tokens vs 256), 168 -> 12x12 = 144 (-44%). Cost per image, not a quality claim.
    for Pp in args.token_proxy_P:
        rows_cfg.append(dict(P=Pp, mode="fp16-half", B=4, G=3, chunk=64))
    ref = {}
    enc_rows = bench_encoder_rows(DINO, rows_cfg, device, warmup, iters, pretrained=True, ref_feats=ref)
    if device.type != "cpu" and not args.no_cpu_ref:
        cpu_cfg = [dict(P=args.P[0], mode="fp32", B=1, G=3, chunk=64)]
        if not args.quick:
            cpu_cfg.append(dict(P=args.P[0], mode="fp32", B=1, G=1, chunk=64))
        enc_rows += bench_encoder_rows(DINO, cpu_cfg, cpu, 1, 2 if args.quick else 3, pretrained=True)

    # calibration: the batched fp16-half @224 row is the throughput reference
    cal = [r for r in enc_rows if r["device"] == "mps" and r["P"] == 224 and r["mode"] == "fp16-half"
           and r["B"] == 4 and r["G"] == 3 and r["chunk"] == 64 and np.isfinite(r["ms_img"])]
    k_fp16 = (T4_MS_PER_IMG_224_FP16 / cal[0]["ms_img"]) if cal else float("nan")

    tbl = []
    for r in enc_rows:
        t4_img = t4_model_ms_per_img(r, k_fp16)
        t4_study = t4_img * r["imgs_per_study"]
        r["t4_ms_study_est"] = t4_study
        r["t4_s_1300_est"] = t4_study * N_TEST_STUDIES / 1e3
        r["mps_s_1300"] = r["ms_study"] * N_TEST_STUDIES / 1e3
        tbl.append([r["device"], r["P"], r["mode"], r["B"], r["G"], r["chunk"], r["n_img"],
                    f1(r["ms_batch"]), f1(r["ms_study"]), f2(r["ms_img"]),
                    "yes" if r["finite"] else "NO", f3(r["max_dev_vs_fp32"]),
                    f"{r['mps_alloc_mb']:.0f}" if np.isfinite(r["mps_alloc_mb"]) else "n/a",
                    f"{r['mps_driver_mb']:.0f}" if np.isfinite(r["mps_driver_mb"]) else "n/a",
                    f"{r['rss_mb']:.0f}", f1(r["mps_s_1300"]), f2(t4_study), f1(r["t4_s_1300_est"])])
    print(md_table(["dev", "P", "dtype", "B", "G", "chunk", "imgs", "ms/batch", "ms/study", "ms/img",
                    "finite", "max|Δ| vs fp32", "mps alloc MiB", "mps driver MiB", "RSS MiB",
                    "s/1300 (this dev)", "T4 ms/study est.", "T4 s/1300 est."], tbl))
    print(f"\nT4 projection (ESTIMATE): calibration k = {T4_MS_PER_IMG_224_FP16} ms/img (assumed T4 fp16 ViT-S/14@224) "
          f"÷ measured MPS fp16-half @224 B=4 G=3 = {cal[0]['ms_img'] if cal else float('nan'):.2f} ms/img → k={k_fp16:.3f}; "
          f"every MPS row is scaled by k; fp32 rows additionally ×{T4_FP32_OVER_FP16} (T4 fp32 vs tensor-core fp16). "
          f"'max|Δ| vs fp32' is the largest abs. difference of the CLS feature against the fp32 row with the same P/B/G.")
    records["encoder"] = enc_rows
    records["k_fp16"] = k_fp16

    # ------------------------------------------------------------------ (e)
    alt_rows = []
    if not args.no_alt:
        print("\n## (e) alternative encoders, for the record (P=224, random init, same image counts)\n")
        alt_cfg = [dict(P=224, mode="fp16-half", B=4, G=3, chunk=64), dict(P=224, mode="fp32", B=4, G=3, chunk=64)]
        if not args.quick:
            alt_cfg.append(dict(P=224, mode="fp16-half", B=1, G=3, chunk=64))
        for name in ALT_ENCODERS:
            alt_rows += bench_encoder_rows(name, alt_cfg, device, warmup, iters, pretrained=False)
        dino224 = [r for r in enc_rows if r["device"] == "mps" and r["P"] == 224 and r["B"] == 4
                   and r["G"] == 3 and r["chunk"] == 64 and r["mode"] in ("fp16-half", "fp32")]
        tbl = []
        for r in dino224 + alt_rows:
            t4_img = t4_model_ms_per_img(r, k_fp16)
            t4_s = t4_img * r["imgs_per_study"] * N_TEST_STUDIES / 1e3
            r["t4_s_1300_est"] = t4_s
            tbl.append([r["encoder"], f"{r['params_m']:.1f}", r["mode"], r["B"], r["G"], r["n_img"],
                        f1(r["ms_study"]), f2(r["ms_img"]), f"{r['mps_driver_mb']:.0f}" if np.isfinite(r["mps_driver_mb"]) else "n/a",
                        f1(r["ms_study"] * N_TEST_STUDIES / 1e3), f1(t4_s),
                        f"{(t4_s / J_SECONDS_PER_AUC):.4f}" if np.isfinite(t4_s) else "n/a"])
        print(md_table(["encoder", "params M", "dtype", "B", "G", "imgs", "ms/study", "ms/img", "mps driver MiB",
                        "s/1300 (MPS)", "T4 s/1300 est.", "J cost (AUC-equiv)"], tbl))
        print("\nNote: DINOv2-S with 2 vs 4 trainable blocks has IDENTICAL inference cost (the rows above apply to both); "
              "trainable depth only changes training memory/time — see --train-probe. Alt encoders are random-init "
              "(timing is weight-independent); their T4 estimate reuses the ViT calibration, which flatters/penalises "
              "CNNs by an unknown factor — use the MPS ratio, not the absolute seconds.")
    records["alt"] = alt_rows

    # ------------------------------------------------------------------ (f)
    tp_rows = []
    if args.train_probe and device.type != "cpu":
        print("\n## (f) training probe: fwd+bwd step, fp32 on MPS, 6·G·bs images, last-k blocks trainable (+norm, head)\n")
        combos = [(bs, tb) for bs in args.train_bs for tb in ((2, 4) if bs == args.train_bs[0] else (4,))]
        for bs, tb in combos:
            try:
                tp_rows.append(train_probe(DINO, 224, bs, 3, tb, device, 1, 3 if args.quick else 5))
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] train probe bs={bs} trainable={tb} failed: {exc}")
        print(md_table(["bs", "trainable blocks", "imgs/step", "trainable params M", "ms/step", "ms/img",
                        "mps alloc MiB", "mps driver MiB", "RSS MiB", "s/epoch 4,407 est."],
                       [[r["bs"], r["trainable_blocks"], r["n_img"], f"{r['trainable_params_m']:.1f}", f1(r["ms_step"]),
                         f2(r["ms_img"]), f"{r['mps_alloc_mb']:.0f}", f"{r['mps_driver_mb']:.0f}", f"{r['rss_mb']:.0f}",
                         f1(r["ms_step"] / r["bs"] * 4407 / 1e3)] for r in tp_rows]))
    records["train_probe"] = tp_rows

    # ------------------------------------------------------------------ summary
    best = [r for r in enc_rows if r["device"] == "mps" and r["P"] == 224 and r["mode"] == "fp16-half"
            and r["B"] == 4 and r["G"] == 3 and r["chunk"] == 64]
    best1 = [r for r in enc_rows if r["device"] == "mps" and r["P"] == 224 and r["mode"] == "fp16-half"
             and r["B"] == 1 and r["G"] == 3]
    print("\n## per-study budget and 1,300-study projection\n")
    if best:
        b = best[0]
        hdr, dec, full = float(mean.ms_hdr), float(mean.ms_decode), float(mean.ms_full_decode)
        mdl_mps = b["ms_study"]
        mdl_t4 = b["t4_ms_study_est"]
        rows = []

        def proj(label, h, d, m, overhead, note):
            serial = (h + d + m) * N_TEST_STUDIES / 1e3 + overhead
            overlap = max(h + d, m) * N_TEST_STUDIES / 1e3 + overhead
            rows.append([label, f1(h), f1(d), f1(m), f1(h + d + m), f1(serial), f1(overlap),
                         f"{serial / J_SECONDS_PER_AUC:.4f}", note])

        hdr_fast, dec_fast = float(mean.ms_hdr_head), float(mean.ms_decode_twostep)
        proj("this machine (MPS, measured)", hdr, dec, mdl_mps, 0.0, "no fixed overhead added")
        proj("this machine, 8KB header + 2-step resize", hdr_fast, dec_fast, mdl_mps, 0.0, "measured variants")
        proj("T4 est., uncompressed DICOM", hdr * T4_CPU_SLOWDOWN, dec * T4_CPU_SLOWDOWN, mdl_t4, T4_FIXED_OVERHEAD_S,
             f"CPU ×{T4_CPU_SLOWDOWN}, +{T4_FIXED_OVERHEAD_S:.0f} s fixed")
        proj("T4 est., 8KB header + 2-step resize", hdr_fast * T4_CPU_SLOWDOWN, dec_fast * T4_CPU_SLOWDOWN, mdl_t4,
             T4_FIXED_OVERHEAD_S, "optimised CPU path")
        proj(f"T4 est., all {mean.n_decoded:.0f} slices JPEG-lossless", hdr * T4_CPU_SLOWDOWN,
             dec * T4_CPU_SLOWDOWN + JPEG_MS_PER_SLICE * float(mean.n_decoded), mdl_t4, T4_FIXED_OVERHEAD_S,
             f"+{JPEG_MS_PER_SLICE} ms/slice decode")
        proj("T4 est., FULL decode (no header-first)", hdr * T4_CPU_SLOWDOWN, full * T4_CPU_SLOWDOWN, mdl_t4,
             T4_FIXED_OVERHEAD_S, "every file decoded")
        print(md_table(["scenario", "hdr ms", "decode ms", "model ms", "total ms/study", "s/1300 serial",
                        "s/1300 overlapped", "J cost serial", "note"], rows))
        print(f"\nHeadline: decode {dec:.1f} ms/study ({mean.n_decoded:.0f} slices = 6·G·T over present slots) + header {hdr:.1f} ms; model {mdl_mps:.1f} ms/study "
              f"(MPS fp16 @224, B=4) / {best1[0]['ms_study']:.1f} ms (B=1); T4 est. {mdl_t4:.1f} ms/study; "
              f"peak RSS {rss_mb():.0f} MiB.")
    print(f"\nPeak RSS for the whole run: {rss_mb():.0f} MiB")
    records["peak_rss_mb"] = rss_mb()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(records, fh, indent=1, default=float)
        print(f"records written to {args.json}")


if __name__ == "__main__":
    main()
