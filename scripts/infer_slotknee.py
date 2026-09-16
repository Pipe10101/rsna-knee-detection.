#!/usr/bin/env python3
"""SlotKnee-S inference pipeline.

Implements the inference loop specified in docs/slotknee_spec.md §6.
"""

import argparse
import copy
import json
import os
import resource
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from src.infer import rank_average
from src.llm_labels import LABELS
from src.slotknee import SlotKneeS
from src.slots import SlotCache, build_study_tensor, index_study, series_lookup, set_decoder, is_medial_slot


LAT_PAIRS = (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"))


def _lat_perm(labels):
    perm = list(range(len(labels)))
    for a, b in LAT_PAIRS:
        if a in labels and b in labels:
            ia, ib = labels.index(a), labels.index(b)
            perm[ia], perm[ib] = ib, ia
    return perm


def _mirror_study(x, slot_names):
    """[B,S,G,T,P,P] -> laterality-mirrored copy: COR/AX flip columns, SAG reverse anchors (src.slots.apply_laterality).
    A medial-centred zoom slot (``src.slots.is_medial_slot``) is ZEROED, not flipped: its mirror is not
    lateral anatomy, so it must be absent on the pass whose labels are swapped (see ``_mirror_mask``)."""
    out = x.clone()
    for s, name in enumerate(slot_names):
        plane = name.split("_")[0]
        if is_medial_slot(name):
            out[:, s] = 0
        elif plane in ("COR", "AX"):
            out[:, s] = x[:, s].flip(-1)
        elif plane == "SAG":
            out[:, s] = x[:, s].flip(1)
        else:
            raise ValueError(f"cannot infer plane from slot name {name!r}")
    return out


def _mirror_mask(mask, slot_names):
    """[B,S] slot mask for the mirrored pass: medial-centred zoom slots become absent."""
    out = mask.clone()
    for s, name in enumerate(slot_names):
        if is_medial_slot(name):
            out[:, s] = 0
    return out


_LAYOUT_KEYS = ("G", "T", "slot_names", "zoom_mm", "zoom_slots", "zoom_center", "zoom_spec", "trim_frac", "crop_mm")


def _layout_of(ckpt_path):
    """The study-rebuild layout a checkpoint was trained on (slot_layout + T from hparams)."""
    ck = torch.load(ckpt_path, map_location="cpu")
    layout = dict(ck.get("slot_layout") or {})
    layout.setdefault("G", 3)                                   # checkpoints before slot_layout: 6 slots, G=3
    layout["T"] = int(ck.get("hparams", {}).get("T", layout.get("T", 3)))
    layout.setdefault("trim_frac", 0.15)                        # pre-2026-09-09 checkpoints: default band
    layout.setdefault("crop_mm", 140.0)
    del ck
    return layout


def _check_layouts(ckpt_paths):
    """Every --ckpt member must share ONE input layout: studies are decoded once per run with the
    first checkpoint's layout, so a member trained on other anchors (e.g. the t35 central band vs
    the 0.15 default), other zoom slots or another crop would be scored on the wrong pixels
    silently.  Returns the common layout or raises SystemExit naming the offenders."""
    layouts = [(os.path.basename(c), _layout_of(c)) for c in ckpt_paths]
    ref_name, ref = layouts[0]
    def sig(l):
        return tuple(json.dumps(l.get(k), sort_keys=True, default=str) for k in _LAYOUT_KEYS)
    bad = [(n, {k: l.get(k) for k in _LAYOUT_KEYS if json.dumps(l.get(k), sort_keys=True, default=str) != json.dumps(ref.get(k), sort_keys=True, default=str)})
           for n, l in layouts[1:] if sig(l) != sig(ref)]
    if bad:
        raise SystemExit("checkpoints do not share one input layout (group them per layout, as the submit "
                         "kernel does, and run infer once per group): reference %s = %s; differing: %s"
                         % (ref_name, {k: ref.get(k) for k in _LAYOUT_KEYS}, bad))
    return ref


def _main_logits(out):
    """aux_slot_logits=True checkpoints return (logits, aux); keep the main head."""
    return out[0] if isinstance(out, tuple) else out


def get_amp_device(amp_arg):
    amp_arg = (amp_arg or "auto").lower()
    if amp_arg == "cpu":
        return torch.device("cpu"), "fp32"
    if torch.cuda.is_available():
        return torch.device("cuda"), ("fp32" if amp_arg == "fp32" else "fp16")
    if torch.backends.mps.is_available():
        # fp16 on MPS is opt-in (--amp fp16), applied as model.half() after a
        # finite-check on the first batch; autocast engages only on cuda.
        return torch.device("mps"), ("fp16" if amp_arg == "fp16" else "fp32")
    return torch.device("cpu"), "fp32"


def peak_rss_gb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024 ** 3 if sys.platform == "darwin" else ru / 1024 ** 2  # bytes vs KB


def device_mem_gb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    if device.type == "mps":
        # RSS is blind to Metal memory; driver-allocated is the usable proxy.
        return torch.mps.driver_allocated_memory() / 1024 ** 3
    return 0.0


def _decode_study(task):
    """(study_dir, lookup, P, layout, shifts) -> (xs, mask, decode_ms, err).

    xs is a list with one [S, G, T, P, P] array per anchor shift (TTA along the
    slice axis -- laterality-safe, unlike a horizontal flip).  The slot layout
    (zoom slots, G, T) comes from the checkpoint so train and inference match.
    """
    study_dir, lookup, P, layout, shifts = task
    try:
        xs, mask, ms = [], None, 0.0
        # The header walk is anchor-shift-independent and it is the expensive half of a
        # cold study (measured on Kaggle: index 2113 ms vs decode 307 ms on first touch),
        # so index ONCE and reuse it for every TTA shift instead of re-walking per shift.
        index = index_study(study_dir, lookup)
        for sh in shifts:
            x, mask, info = build_study_tensor(
                study_dir, series_df=lookup, P=P, G=layout.get("G", 3), T=layout.get("T", 3),
                laterality=True, zoom_mm=layout.get("zoom_mm"),
                zoom_slots=tuple(layout.get("zoom_slots") or ()), anchor_shift=int(sh),
                zoom_center=layout.get("zoom_center", "image"),
                zoom_spec=layout.get("zoom_spec"), index=index,
                # checkpoints since 2026-09-09 carry the training cache's anchor band and crop
                # size; older ones trained on the default 0.15 / 140 mm layout.
                trim_frac=float(layout.get("trim_frac", 0.15)), crop_mm=float(layout.get("crop_mm", 140.0)))
            xs.append(x); ms += float(info.get("ms", 0.0))
        if mask is None or not mask.any():
            return None, None, ms, "no readable series"
        return xs, mask, ms, None
    except Exception as exc:  # a failed study keeps its seeded 0.5 row
        return None, None, 0.0, f"{type(exc).__name__}: {exc}"


def _bounded_map(fn, tasks, workers, decoder="pydicom"):
    """Ordered map over a small process pool with a bounded in-flight window,
    so decode overlaps the model while decoded RAM stays ~2*workers studies.

    The decoder choice is a module global, which a 'spawn' start method does not inherit —
    hence the initializer, so workers decode the same way on macOS and on Kaggle."""
    if workers <= 0:
        for t in tasks:
            yield fn(t)
        return
    window = 2 * workers
    with ProcessPoolExecutor(max_workers=workers, initializer=set_decoder,
                             initargs=(decoder,)) as pool:
        pending = deque()
        for t in tasks:
            pending.append(pool.submit(fn, t))
            if len(pending) >= window:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--P", type=int, default=224)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--amp", default="auto", choices=["auto", "cpu", "fp32", "fp16"])
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--flip-tta", action="store_true",
                        help="mirror TTA: also score the laterality-mirrored study (COR/AX columns flipped, SAG anchor "
                             "order reversed -- the inverse of the cache normalisation) with the medial/lateral labels "
                             "swapped back, and average. Exact under laterality normalisation for the base slots; a "
                             "medial-centred zoom slot (*_Z<mm>M) is dropped on the mirrored pass, not flipped.")
    parser.add_argument("--seed", type=int, default=42,
                        help="seeds the --bags anchor sampling so a submission is reproducible")
    parser.add_argument("--bag", type=int, default=0,
                        help="multi-bag inference: use K of the study's anchors per forward "
                             "(0 = all, the old behaviour). Match the --anchor-bag used in training.")
    parser.add_argument("--bags", type=int, default=1,
                        help="how many independent random bags to average (1 = off). Costs one "
                             "forward per bag per member, so accuracy entry only.")
    parser.add_argument("--decoder", default="pydicom", choices=["pydicom", "dicomsdl"],
                        help="pixel decoder. Decoding is ~70%% of submission wall-clock, and "
                             "dicomsdl is several times faster; it falls back to pydicom per file "
                             "on any surprise, and silently stays on pydicom if not installed.")
    parser.add_argument("--write-every", type=int, default=200)
    parser.add_argument("--anchor-shift", default="0",
                        help="comma list of slice-anchor shifts to average, e.g. 0,1,-1 (TTA)")
    parser.add_argument("--tta", action="store_true",
                        help="deprecated alias for --anchor-shift 0,1,-1 (the old horizontal-flip TTA mirrored medial/lateral and is gone)")
    parser.add_argument("--cache", default=None,
                        help="score a prebuilt slot cache instead of decoding on the fly")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    # Multi-bag inference draws random anchors, so without a fixed generator the same
    # checkpoint scores differently on every run and an LB result cannot be reproduced.
    bag_gen = torch.Generator().manual_seed(args.seed)
    if args.bag and args.bags <= 1:
        print(f"note: --bag {args.bag} does nothing without --bags > 1; using all anchors",
              flush=True)
    print(f"decoder: {set_decoder(args.decoder)}", flush=True)

    t_start = time.time()

    # 1. Seed submission.csv FIRST; keep the template for aligned rewrites.
    # A corrupt-but-present sample_submission must degrade to the test.csv template,
    # not crash before anything scoreable is written.
    sample_sub = os.path.join(args.data_dir, "sample_submission.csv")
    template = None
    if os.path.exists(sample_sub):
        try:
            template = pd.read_csv(sample_sub)
            if template.shape[1] < 2:
                raise ValueError(f"only {template.shape[1]} column(s)")
            shutil.copy2(sample_sub, args.out)
        except Exception as e:
            print(f"Warning: {sample_sub} unreadable ({e}); template built from test.csv.")
            template = None
    if template is None:
        test_csv = os.path.join(args.data_dir, "test.csv")
        ids = (pd.read_csv(test_csv)["StudyInstanceUID"].astype(str).tolist()
               if os.path.exists(test_csv) else [])
        template = pd.DataFrame({"StudyInstanceUID": ids})
        for lab in LABELS:
            template[lab] = 0.5
        template.to_csv(args.out, index=False)
        if not os.path.exists(sample_sub):
            print(f"Warning: {sample_sub} not found; template built from test.csv.")
    id_col = template.columns[0]
    label_cols = list(template.columns[1:])
    template[id_col] = template[id_col].astype(str)

    # 2. Study stream: a prebuilt cache, else test.csv uids decoded on the fly.
    if args.cache:
        cache = SlotCache(args.cache, split=args.split)
        uids = [str(u) for u in cache.uids]

        def _cache_stream():
            for i in range(cache.N):
                x, mask = cache[i]
                yield np.asarray(x), np.asarray(mask), 0.0, None
        results = _cache_stream()
    else:
        test_images_dir = os.path.join(args.data_dir, "test_images")
        if not os.path.isdir(test_images_dir) and os.path.isdir(os.path.join(args.data_dir, f"{args.split}_series")):
            test_images_dir = os.path.join(args.data_dir, f"{args.split}_series")   # Kaggle layout
        test_csv = os.path.join(args.data_dir, "test.csv")
        if os.path.exists(test_csv):
            uids = pd.read_csv(test_csv)["StudyInstanceUID"].astype(str).tolist()
        elif os.path.isdir(test_images_dir):
            uids = sorted(d for d in os.listdir(test_images_dir)
                          if os.path.isdir(os.path.join(test_images_dir, d)))
        else:
            print(f"Warning: neither {test_csv} nor {test_images_dir} found; "
                  "submission stays seeded at 0.5.")
            return
        series_csv = os.path.join(args.data_dir, f"{args.split}_series.csv")
        series_df = pd.read_csv(series_csv) if os.path.exists(series_csv) else None
        layout = _check_layouts(args.ckpt)                           # one layout for every member, or exit
        shifts = [int(v) for v in str(args.anchor_shift).split(",") if v.strip() != ""] or [0]
        if getattr(args, "tta", False) and shifts == [0]:
            shifts = [0, 1, -1]
        print(f"slot layout from checkpoint: {layout} | anchor shifts: {shifts}")
        tasks = ((os.path.join(test_images_dir, uid), series_lookup(series_df, uid), args.P, layout, shifts)
                 for uid in uids)
        results = _bounded_map(_decode_study, tasks, args.decode_workers, args.decoder)

    device, amp_dtype = get_amp_device(args.amp)
    use_amp_autocast = (device.type == "cuda" and amp_dtype == "fp16")
    print(f"Device: {device}, AMP: {amp_dtype}, ckpts: {len(args.ckpt)}, "
          f"studies: {len(uids)}")

    # All checkpoints stay resident (<= 3 ViT-S) so each decoded batch is shared.
    models = []
    for ckpt_path in args.ckpt:
        print(f"  Loading {os.path.basename(ckpt_path)}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Pass the FULL saved hparams through so future constructor args survive.
        hparams = {k: v for k, v in ckpt["hparams"].items()
                   if k not in ("pretrained", "pretrained_path")}
        model = SlotKneeS(pretrained=False, **hparams)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
    # mirror TTA needs the slot layout (plane per slot) and the label order the checkpoints were trained with
    _lay = (torch.load(args.ckpt[0], map_location="cpu").get("slot_layout") or {})
    slot_names_tta = list(_lay.get("slot_names") or [])
    lat_perm = _lat_perm(LABELS)
    if args.flip_tta and any(is_medial_slot(n) for n in slot_names_tta):
        print("flip-tta: medial-centred zoom slot(s) %s are DROPPED on the mirrored pass (their mirror is not "
              "lateral anatomy); the plain pass keeps them." % [n for n in slot_names_tta if is_medial_slot(n)], flush=True)
    if args.flip_tta and not slot_names_tta:
        raise SystemExit("--flip-tta needs slot_names in the checkpoint's slot_layout (older checkpoints lack it)")

    n_total = len(uids)
    state = {"half_pending": device.type == "mps" and amp_dtype == "fp16",
             "models": models, "model_ms": 0.0}
    pred_uids = []
    per_model = [[] for _ in models]
    failures = []
    decode_ms_total = 0.0

    def flush(buf_uids, buf_x, buf_mask):
        n_shift = len(buf_x[0])
        xs_t = [torch.from_numpy(np.stack([bx[k] for bx in buf_x])).to(device) for k in range(n_shift)]
        x_t = xs_t[0]
        mask_t = torch.from_numpy(np.stack(buf_mask)).to(device)
        if state["half_pending"]:
            probe = copy.deepcopy(state["models"][0]).half()
            with torch.no_grad():
                out = _main_logits(probe(x_t, mask_t))
            if torch.isfinite(out).all():
                state["models"] = [m.half() for m in state["models"]]
                print("MPS fp16: probe batch finite; running model.half().")
            else:
                print("MPS fp16 probe produced non-finite logits; staying fp32.")
            del probe
            state["half_pending"] = False
        t0 = time.time()
        with torch.no_grad(), torch.autocast(device_type=device.type,
                                             enabled=use_amp_autocast, dtype=torch.float16):
            outs = []
            for m in state["models"]:
                acc, n_acc = None, 0
                for xk in xs_t:                      # anchor-shift TTA: mean of logits
                    # MULTI-BAG INFERENCE (--bag K --bags N).  The recipe now trains with
                    # random slice bagging (--anchor-bag), so the model is used to seeing a
                    # SUBSET of anchors.  Averaging several independent bags at inference is
                    # the matching read-out: free ensembling from a single set of weights,
                    # and it matches the train-time input distribution instead of showing the
                    # model more anchors than it ever trained on.  bags<=1 keeps the old path.
                    if args.bag and args.bag < xk.shape[2] and args.bags > 1:
                        for _ in range(args.bags):
                            keep = torch.randperm(xk.shape[2], generator=bag_gen)[: args.bag].sort().values
                            o = _main_logits(m(xk[:, :, keep], mask_t)).float()
                            acc = o if acc is None else acc + o
                            n_acc += 1
                    else:
                        o = _main_logits(m(xk, mask_t)).float()
                        if args.flip_tta:
                            om = _main_logits(m(_mirror_study(xk, slot_names_tta), _mirror_mask(mask_t, slot_names_tta))).float()
                            o = 0.5 * (o + om[:, lat_perm])       # swap medial/lateral back, then average
                        acc = o if acc is None else acc + o
                        n_acc += 1
                outs.append(acc / float(max(1, n_acc)))

            outs = [o.float().cpu().numpy() for o in outs]
        state["model_ms"] += (time.time() - t0) * 1000.0
        pred_uids.extend(buf_uids)
        for j, o in enumerate(outs):
            per_model[j].append(o)

    def current_preds():
        mats = [np.concatenate(chunks, axis=0) for chunks in per_model]
        if len(mats) > 1:
            preds = rank_average(mats)                    # already in (0, 1]
        else:
            preds = 1.0 / (1.0 + np.exp(-mats[0]))        # sigmoid
        return np.clip(np.nan_to_num(preds, nan=0.5), 0.0, 1.0)

    def write_aligned():
        # Same row set/order as sample_submission; unpredicted studies keep 0.5.
        sub = template.copy()
        if pred_uids:
            preds = current_preds()
            pos = {u: i for i, u in enumerate(sub[id_col])}
            vals = sub[label_cols].to_numpy(dtype=float)
            for u, row in zip(pred_uids, preds):
                i = pos.get(u)
                if i is not None:
                    vals[i] = row
            sub[label_cols] = vals
        tmp = args.out + ".tmp"
        sub.to_csv(tmp, index=False)
        os.replace(tmp, args.out)

    done = 0
    buf_uids, buf_x, buf_mask = [], [], []
    for uid, (x, mask, dec_ms, err) in zip(uids, results):
        decode_ms_total += dec_ms
        if err is not None:
            failures.append((uid, err))
        else:
            buf_uids.append(uid)
            buf_x.append(x)
            buf_mask.append(mask)
            if len(buf_x) == args.bs:
                flush(buf_uids, buf_x, buf_mask)
                buf_uids, buf_x, buf_mask = [], [], []
        done += 1
        if args.write_every and done % args.write_every == 0:
            if buf_x:
                flush(buf_uids, buf_x, buf_mask)
                buf_uids, buf_x, buf_mask = [], [], []
            write_aligned()
            print(f"  {done}/{n_total} studies; submission rewritten.")
    if buf_x:
        flush(buf_uids, buf_x, buf_mask)
    write_aligned()

    n_pred = len(pred_uids)
    total_sec = time.time() - t_start
    proj_sec = (total_sec / max(1, done)) * 1300

    print(f"\nInference complete. {n_pred}/{n_total} studies predicted, "
          f"{len(failures)} failed (kept at 0.5).")
    for uid, err in failures[:10]:
        print(f"  FAILED {uid}: {err}")
    print(f"Decode ms/study: {decode_ms_total / max(1, done):.1f}")
    print(f"Model ms/study:  {state['model_ms'] / max(1, n_pred):.1f}")
    print(f"Total seconds:   {total_sec:.1f}")
    print(f"Projected 1300:  {proj_sec:.1f}s")
    print(f"Peak RSS:        {peak_rss_gb():.2f} GB, device mem: {device_mem_gb(device):.2f} GB")


if __name__ == "__main__":
    main()
