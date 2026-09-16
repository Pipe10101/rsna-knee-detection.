#!/usr/bin/env python3
"""Build the SlotKnee-S slot cache (docs/slotknee_spec.md §3.6).

    python3 scripts/build_slot_cache.py --data-dir data_subset --split train --out cache/slots_P224 \
        [--limit N] [--P 224] [--crop-mm 140] [--G 3] [--T 3] [--workers 4] \
        [--shard-size 512] [--limit-minutes M] [--image-dir DIR] [--series-csv CSV]

Output layout of <out>/ (all flat, memory-mapped by ``src.slots.SlotCache``):

    <split>_x.u8, <split>_x.001.u8, ...   uint8 shards; shard k holds rows
                                          [k*S, (k+1)*S) as [n_k, 6, G, T, P, P]
    <split>_mask.u8                       [N, 6] uint8 slot-presence mask
    <split>_index.json                    studies, done flags, shards, params, stats

Every study is written straight into its shard row by the worker that built
it (np.memmap opened 'r+' for that one write), so no process ever holds more
than one study: the 6*G*T selected slices (54 at defaults; ~45 after missing
slots) decoded one at a time plus one slot's float32
crops.  Resumable: rows marked done in the index are skipped; a run that stops
on --limit-minutes / Ctrl-C leaves a consistent cache to resume from.

Image directory: Kaggle's competition data uses ``train_series/`` and
``test_series/``; the local subset uses ``train_images/`` / ``test_images/``.
Both are tried (``*_series`` first) unless --image-dir is given.

Zoom slots: ``--zoom-mm 100 --zoom-slots SAG_FS,COR_FS`` appends ``SAG_FS_Z`` and
``COR_FS_Z`` (same files/anchors, 100 mm crop from the same decoded array, mask
copied from the base slot); rows become ``[S, G, T, P, P]`` with S = 6 + zoom
slots and the index records ``slot_names`` (read via ``SlotCache.S``).  Size at
P=224, G=T=3 for 4,407 studies: 6 slots 11.9 GB, 8 slots 4,407×8×9×224² =
15.9 GB — inside Kaggle's 20 GB output cap.  Default (no zoom) layout is
unchanged; a cache refuses to resume under a different zoom layout.
``--zoom-center joint`` centres the SAG/COR zoom crops on the tibiofemoral
joint found by ``slots.locate_joint`` (image centre on low confidence; the
index records ``zoom_center`` and the stats count ``n_joint_fallback``).
``--zoom-spec "BASE:MM[:CENTER]"`` (comma-separated and/or repeated) appends
zoom slots with per-entry size and centre, named ``<BASE>_Z<mm>[J]`` — the
Phase B ACL/notch layout is ``--zoom-spec "SAG_FS:100:joint,SAG_FS:80:joint"``
-> 8 slots (exactly what is listed; no COR entry is implied).  The legacy trio
keeps its ``<BASE>_Z`` names and stays bit-identical.
"""

import argparse
import bisect
import json
import multiprocessing
import os
import resource
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Dict, List, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src import slots                                  # noqa: E402
from src.kaggle_data import available_cpus             # noqa: E402

N_SLOTS = slots.N_SLOTS


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def rss_bytes(children: bool = False) -> int:
    """Peak RSS in bytes (ru_maxrss is bytes on macOS, kilobytes on Linux)."""
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    v = int(resource.getrusage(who).ru_maxrss)
    return v if sys.platform == "darwin" else v * 1024


def gb(n: float) -> str:
    return "%.2f GB" % (n / 1e9)


def log(msg: str) -> None:
    print("[build] " + msg, flush=True)


def resolve_image_dir(data_dir: str, split: str, explicit: Optional[str] = None) -> str:
    if explicit:
        if not os.path.isdir(explicit):
            raise SystemExit("--image-dir %s is not a directory" % explicit)
        return explicit
    tried = []
    for name in ("%s_series" % split, "%s_images" % split):
        p = os.path.join(data_dir, name)
        tried.append(p)
        if os.path.isdir(p):
            return p
    raise SystemExit("no image directory found; tried: %s" % ", ".join(tried))


def resolve_series_csv(data_dir: str, split: str, explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    p = os.path.join(data_dir, "%s_series.csv" % split)
    return p if os.path.isfile(p) else None


def load_lookup(csv_path: Optional[str]) -> Dict[str, Dict[str, tuple]]:
    """study uid -> {series uid: (plane, fluid)}; empty when there is no CSV."""
    if csv_path is None:
        return {}
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str)
    out: Dict[str, Dict[str, tuple]] = {}
    if "StudyInstanceUID" not in df.columns:
        return out
    for study, g in df.groupby("StudyInstanceUID", sort=False):
        out[str(study)] = slots.series_lookup(g)
    return out


def list_studies(image_dir: str, limit: Optional[int] = None, skip: int = 0) -> List[str]:
    uids = sorted(e.name for e in os.scandir(image_dir) if e.is_dir())
    if skip:
        uids = uids[skip:]
    if limit is not None and limit >= 0:
        uids = uids[:limit]
    return uids


def ensure_size(path: str, nbytes: int) -> None:
    """Create (sparse) or extend a flat file; never shrink, never read it."""
    if not os.path.exists(path):
        with open(path, "wb"):
            pass
    if os.path.getsize(path) < nbytes:
        with open(path, "r+b") as f:
            f.truncate(nbytes)


def write_index_atomic(path: str, index: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Index (study list, done flags, shard layout)
# --------------------------------------------------------------------------

def load_or_init_index(out: str, split: str, requested: List[str], args) -> dict:
    paths = slots.cache_paths(out, split)
    params = {"P": int(args.P), "G": int(args.G), "T": int(args.T),
              "crop_mm": float(args.crop_mm), "trim_frac": float(args.trim_frac)}
    zoom = {"zoom_mm": args.zoom_mm_eff, "zoom_slots": list(args.zoom_slots_list),
            "zoom_center": str(args.zoom_center),
            "zoom_spec": [list(v) for v in args.zoom_spec_list]}
    if os.path.isfile(paths["index"]):
        with open(paths["index"]) as f:
            index = json.load(f)
        for k, v in params.items():
            if index.get(k) != v:
                raise SystemExit("existing cache at %s has %s=%r, requested %r; use a new --out"
                                 % (out, k, index.get(k), v))
        if index.get("version") != slots.CACHE_VERSION:
            raise SystemExit("existing cache version %r != %r" % (index.get("version"), slots.CACHE_VERSION))
        have = {"zoom_mm": index.get("zoom_mm"), "zoom_slots": list(index.get("zoom_slots") or []),
                "zoom_center": str(index.get("zoom_center") or "image"),
                "zoom_spec": [list(v) for v in (index.get("zoom_spec") or [])]}
        if have != zoom:
            raise SystemExit("existing cache at %s has zoom layout %r, requested %r; use a new --out"
                             % (out, have, zoom))
        if int(index.get("shard_size", 0)) != int(args.shard_size):
            log("resuming with the existing shard_size=%s (ignoring --shard-size %s)"
                % (index.get("shard_size"), args.shard_size))
        known = set(index["studies"])
        new = [u for u in requested if u not in known]
        index["studies"] = list(index["studies"]) + new
        index["done"] = list(index.get("done", [])) + [0] * len(new)
        index["side"] = list(index.get("side", [])) + [None] * len(new)
        index.setdefault("failed", {})
    else:
        index = dict(params)
        index.update(zoom)
        index.update({
            "version": slots.CACHE_VERSION,
            "split": split,
            "slot_names": list(args.slot_names),
            "shard_size": int(args.shard_size),
            "studies": list(requested),
            "done": [0] * len(requested),
            "side": [None] * len(requested),
            "failed": {},
            "stats": {},
            "cumulative": {},
        })
    n = len(index["studies"])
    index["shards"] = slots.shard_layout(n, int(index["shard_size"]), split)
    index["mask"] = os.path.basename(paths["mask"])
    return index


# --------------------------------------------------------------------------
# Worker (one study in flight per process)
# --------------------------------------------------------------------------

def _worker(task: dict) -> dict:
    """Build one study and write it into its shard row.  Never raises."""
    t0 = time.perf_counter()
    try:
        x, mask, info = slots.build_study_tensor(
            task["study_dir"], task["lookup"], P=task["P"], crop_mm=task["crop_mm"],
            G=task["G"], T=task["T"], laterality=task["laterality"], trim_frac=task["trim_frac"],
            zoom_mm=task.get("zoom_mm"), zoom_slots=task.get("zoom_slots", ()),
            zoom_center=task.get("zoom_center", "image"), zoom_spec=task.get("zoom_spec") or None)
        row_shape = x.shape
        mm = np.memmap(task["x_path"], dtype=np.uint8, mode="r+",
                       shape=(task["shard_rows"],) + row_shape)
        mm[task["local_row"]] = x
        mm.flush()
        del mm
        del x
    except Exception as e:                      # pragma: no cover - defensive
        return {"row": task["row"], "uid": task["uid"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e), "rss": rss_bytes(),
                "wall_ms": (time.perf_counter() - t0) * 1000.0}
    return {"row": task["row"], "uid": task["uid"], "ok": True, "mask": mask.tolist(),
            "info": info, "rss": rss_bytes(), "wall_ms": (time.perf_counter() - t0) * 1000.0}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-dir", default=None, help="explicit image root (default: <data-dir>/<split>_series or <split>_images)")
    ap.add_argument("--series-csv", default=None, help="explicit series CSV (default: <data-dir>/<split>_series.csv)")
    ap.add_argument("--limit", type=int, default=None, help="first N studies (sorted UIDs)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N studies (sorted UIDs) before --limit; two kernels can build one big cache in halves")
    ap.add_argument("--P", type=int, default=224)
    ap.add_argument("--crop-mm", type=float, default=140.0)
    ap.add_argument("--G", type=int, default=3)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--trim-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=None, help="default: kaggle_data.available_cpus()")
    ap.add_argument("--shard-size", type=int, default=512, help="studies per memmap shard (0 = one file)")
    ap.add_argument("--limit-minutes", type=float, default=None, help="stop submitting new studies after M minutes")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--no-laterality", action="store_true", help="do not mirror right knees")
    ap.add_argument("--zoom-mm", type=float, default=None, help="physical size (mm) of the zoom crop used by --zoom-slots")
    ap.add_argument("--zoom-slots", default="", help="comma-separated base slots that get a <BASE>_Z zoom slot, e.g. SAG_FS,COR_FS")
    ap.add_argument("--zoom-center", default="image", choices=list(slots.ZOOM_CENTERS),
                    help="centre of the zoom crops: the image centre (default) or the located tibiofemoral joint")
    ap.add_argument("--zoom-spec", action="append", default=None, metavar="BASE:MM[:CENTER]",
                    help='per-entry zoom slots (comma-separated, flag repeatable), e.g. "SAG_FS:100:joint,SAG_FS:80:joint"')
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.P % 14 != 0:
        raise SystemExit("--P must be a multiple of 14")
    try:
        zoom_slots = list(slots.parse_zoom_slots(args.zoom_slots))
        if zoom_slots and (args.zoom_mm is None or args.zoom_mm <= 0):
            raise ValueError("--zoom-slots requires a positive --zoom-mm")
        zoom_mm = float(args.zoom_mm) if zoom_slots else None
        zoom_spec = [list(t) for t in slots.parse_zoom_spec(
            ",".join(args.zoom_spec) if args.zoom_spec else None)]
        zooms = slots.normalize_zoom(zoom_mm, zoom_slots, args.zoom_center, zoom_spec)
        slot_names = list(slots.SLOT_NAMES) + [z.name for z in zooms]
    except ValueError as e:
        raise SystemExit(str(e))
    args.zoom_mm_eff, args.zoom_slots_list, args.slot_names = zoom_mm, zoom_slots, slot_names
    args.zoom_spec_list = zoom_spec
    S = len(slot_names)
    workers = int(args.workers) if args.workers else max(1, available_cpus())
    image_dir = resolve_image_dir(args.data_dir, args.split, args.image_dir)
    csv_path = resolve_series_csv(args.data_dir, args.split, args.series_csv)
    os.makedirs(args.out, exist_ok=True)

    requested = list_studies(image_dir, args.limit, args.skip)
    if not requested:
        raise SystemExit("no study directories under %s" % image_dir)
    lookup = load_lookup(csv_path)
    if csv_path is None:
        log("WARNING: no series CSV found for split %r; plane/contrast come from DICOM headers" % args.split)

    index = load_or_init_index(args.out, args.split, requested, args)
    paths = slots.cache_paths(args.out, args.split)
    n_total = len(index["studies"])
    row_shape = (S, args.G, args.T, args.P, args.P)
    row_bytes = int(np.prod(row_shape))
    for s in index["shards"]:
        ensure_size(os.path.join(args.out, s["x"]), int(s["n"]) * row_bytes)
    ensure_size(paths["mask"], n_total * S)
    write_index_atomic(paths["index"], index)

    row_of = {u: i for i, u in enumerate(index["studies"])}
    todo_rows = [row_of[u] for u in requested if not index["done"][row_of[u]]]
    n_already = len(requested) - len(todo_rows)
    shard_starts = [int(s["start"]) for s in index["shards"]]

    log("split=%s image_dir=%s csv=%s" % (args.split, image_dir, csv_path or "-"))
    log("P=%d G=%d T=%d crop_mm=%g trim=%g laterality=%s | workers=%d shard_size=%d (%d shards) | row=%.2f MB"
        % (args.P, args.G, args.T, args.crop_mm, args.trim_frac, not args.no_laterality, workers,
           int(index["shard_size"]), len(index["shards"]), row_bytes / 1e6))
    log("slots S=%d [%s] | zoom_mm=%s | zoom_center=%s | zoom_spec=%s"
        % (S, ",".join(slot_names), zoom_mm, args.zoom_center,
           ",".join(":".join(str(q) for q in v) for v in zoom_spec) or "-"))
    log("cache=%s | %d studies in index | requested %d | todo: %d new studies (%d already done)"
        % (args.out, n_total, len(requested), len(todo_rows), n_already))

    mask_mm = np.memmap(paths["mask"], dtype=np.uint8, mode="r+", shape=(n_total, S)) if n_total else None

    def make_task(row: int) -> dict:
        uid = index["studies"][row]
        k = bisect.bisect_right(shard_starts, row) - 1
        shard = index["shards"][k]
        return {
            "row": row, "uid": uid, "study_dir": os.path.join(image_dir, uid),
            "lookup": lookup.get(uid, {}), "P": args.P, "crop_mm": args.crop_mm,
            "G": args.G, "T": args.T, "trim_frac": args.trim_frac,
            "laterality": not args.no_laterality,
            "zoom_mm": zoom_mm, "zoom_slots": list(zoom_slots), "zoom_center": str(args.zoom_center),
            "zoom_spec": [list(v) for v in zoom_spec],
            "x_path": os.path.join(args.out, shard["x"]), "shard_rows": int(shard["n"]),
            "local_row": row - int(shard["start"]),
        }

    # ---- accumulators ---------------------------------------------------
    st = {
        "n_new": 0, "n_failed_studies": 0, "n_files": 0, "n_decoded": 0, "n_decode_fail": 0,
        "n_header_fallback": 0,
        "n_crop_fallback": 0, "n_joint_fallback": 0, "n_joint_slots": 0, "n_medial_side_fallback": 0,
        "decode_fail_by_syntax": {}, "syntax_hist": {},
        "missing_slot_hist": {name: 0 for name in slot_names},
        "side_hist": {"L": 0, "R": 0, "None": 0},
        "side_source_hist": {"tag": 0, "geometry": 0, "unresolved": 0},
        "worker_peak_rss_bytes": 0, "ms_per_study": [], "wall_ms_per_study": [],
    }
    t_start = time.time()
    deadline = t_start + args.limit_minutes * 60.0 if args.limit_minutes else None
    last_print = [t_start]
    stopped_early = [False]

    def handle(res: dict) -> None:
        row = res["row"]
        st["worker_peak_rss_bytes"] = max(st["worker_peak_rss_bytes"], int(res.get("rss", 0)))
        if not res["ok"]:
            st["n_failed_studies"] += 1
            index["failed"][res["uid"]] = res["error"]
            log("FAILED %s: %s" % (res["uid"], res["error"]))
            return
        info = res["info"]
        mask_mm[row] = np.asarray(res["mask"], dtype=np.uint8)
        index["done"][row] = 1
        index["side"][row] = info["side"]
        index.setdefault("joint", {})[res["uid"]] = info.get("joint", {})   # per-study locator estimates (~150 B/study)
        index["failed"].pop(res["uid"], None)
        st["n_new"] += 1
        st["n_files"] += info["n_files"]
        st["n_decoded"] += info["n_decoded"]
        st["n_decode_fail"] += info["n_decode_fail"]
        st["n_header_fallback"] += info.get("n_header_fallback", 0)
        st["n_crop_fallback"] += info["n_crop_fallback"]
        st["n_joint_fallback"] += int(info.get("n_joint_fallback", 0))
        st["n_joint_slots"] += len(info.get("joint", {}) or {})
        st["n_medial_side_fallback"] += int(info.get("n_medial_side_fallback", 0))
        for k, v in info.get("decode_fail_by_syntax", {}).items():
            st["decode_fail_by_syntax"][k] = st["decode_fail_by_syntax"].get(k, 0) + v
        for k, v in info.get("syntax_hist", {}).items():
            st["syntax_hist"][k] = st["syntax_hist"].get(k, 0) + v
        for s, present in enumerate(res["mask"]):
            if not present:
                st["missing_slot_hist"][slot_names[s]] += 1
        st["side_hist"][str(info["side"])] += 1
        st["side_source_hist"][info["side_source"]] = st["side_source_hist"].get(info["side_source"], 0) + 1
        st["ms_per_study"].append(float(info["ms"]))
        st["wall_ms_per_study"].append(float(res["wall_ms"]))

    def progress(force: bool = False) -> None:
        now = time.time()
        n_done = st["n_new"] + st["n_failed_studies"]
        if not force and n_done % args.progress_every != 0 and now - last_print[0] < 60.0:
            return
        last_print[0] = now
        elapsed = max(now - t_start, 1e-6)
        rate = n_done / elapsed
        remaining = len(todo_rows) - n_done
        eta = remaining / rate if rate > 0 else float("inf")
        log("%d/%d this run (%d/%d in cache) | %.2f studies/s | %.0f ms/study | elapsed %.1f min | ETA %.1f min | peak RSS parent %s worker %s"
            % (n_done, len(todo_rows), int(sum(index["done"])), n_total, rate,
               float(np.mean(st["ms_per_study"])) if st["ms_per_study"] else 0.0,
               elapsed / 60.0, eta / 60.0, gb(rss_bytes()), gb(st["worker_peak_rss_bytes"])))
        if n_done % 25 == 0 or force:
            if mask_mm is not None:
                mask_mm.flush()
            write_index_atomic(paths["index"], index)

    def past_deadline() -> bool:
        if deadline is not None and time.time() >= deadline:
            if not stopped_early[0]:
                log("--limit-minutes %.1f reached: no new studies submitted; finishing in-flight ones" % args.limit_minutes)
            stopped_early[0] = True
            return True
        return False

    # ---- run --------------------------------------------------------------
    try:
        if todo_rows:
            if workers <= 1:
                for row in todo_rows:
                    if past_deadline():
                        break
                    handle(_worker(make_task(row)))
                    progress()
            else:
                it = iter(todo_rows)
                pending = set()
                # 'spawn' everywhere: identical on Kaggle (Linux) and macOS, and no
                # fork-after-torch-import surprises; the import cost is paid once.
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                    for _ in range(workers):
                        row = next(it, None)
                        if row is None:
                            break
                        pending.add(ex.submit(_worker, make_task(row)))
                    while pending:
                        done_set, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done_set:
                            handle(fut.result())
                            progress()
                            if not past_deadline():
                                row = next(it, None)
                                if row is not None:
                                    pending.add(ex.submit(_worker, make_task(row)))
    except KeyboardInterrupt:
        log("interrupted; flushing index so the run can resume")
        stopped_early[0] = True
    finally:
        elapsed = time.time() - t_start
        if mask_mm is not None:
            mask_mm.flush()
        ms = st["ms_per_study"]
        wall = st["wall_ms_per_study"]
        n_done_cache = int(sum(index["done"]))
        peak_children = rss_bytes(children=True)
        stats = {
            "elapsed_s": elapsed,
            "workers": workers,
            "n_new": st["n_new"],
            "n_failed_studies": st["n_failed_studies"],
            "studies_per_s": (st["n_new"] / elapsed) if elapsed > 0 else 0.0,
            "ms_per_study_mean": float(np.mean(ms)) if ms else 0.0,
            "ms_per_study_median": float(np.median(ms)) if ms else 0.0,
            "wall_ms_per_study_mean": float(np.mean(wall)) if wall else 0.0,
            "n_files": st["n_files"], "n_decoded": st["n_decoded"],
            "n_header_fallback": st["n_header_fallback"],
            "n_decode_fail": st["n_decode_fail"], "decode_fail_by_syntax": st["decode_fail_by_syntax"],
            "syntax_hist": st["syntax_hist"],
            "n_crop_fallback": st["n_crop_fallback"],
            "n_joint_fallback": st["n_joint_fallback"], "n_joint_slots": st["n_joint_slots"],
            "n_medial_side_fallback": st["n_medial_side_fallback"],
            "missing_slot_hist": st["missing_slot_hist"],
            "side_hist": st["side_hist"], "side_source_hist": st["side_source_hist"],
            "peak_rss_parent_bytes": rss_bytes(),
            "peak_rss_worker_bytes": max(st["worker_peak_rss_bytes"], peak_children),
            "stopped_early": stopped_early[0],
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        index["stats"] = stats                      # this run (may be a no-op resume)
        if st["n_new"]:
            index["build_stats"] = stats            # last run that actually built rows
        cum = index.setdefault("cumulative", {})
        for k in ("n_new", "n_files", "n_decoded", "n_decode_fail", "n_crop_fallback", "elapsed_s"):
            cum[k] = cum.get(k, 0) + stats[k]
        for hk in ("missing_slot_hist", "side_hist", "syntax_hist", "decode_fail_by_syntax"):
            d = cum.setdefault(hk, {})
            for k, v in stats[hk].items():
                d[k] = d.get(k, 0) + v
        index["n_done"] = n_done_cache
        write_index_atomic(paths["index"], index)

        on_disk = sum(os.path.getsize(os.path.join(args.out, s["x"])) for s in index["shards"]
                      if os.path.exists(os.path.join(args.out, s["x"])))
        log("finished: %d new studies (%d failed) in %.1f s | %.2f studies/s | %.0f ms/study mean, %.0f median (per-worker build time)"
            % (st["n_new"], st["n_failed_studies"], elapsed, stats["studies_per_s"],
               stats["ms_per_study_mean"], stats["ms_per_study_median"]))
        log("files seen %d | slices decoded %d | decode failures %d %s | transfer syntaxes %s"
            % (st["n_files"], st["n_decoded"], st["n_decode_fail"],
               json.dumps(st["decode_fail_by_syntax"]), json.dumps(st["syntax_hist"])))
        log("crop fallbacks (window > image): %d slices | header fallbacks (8 KB partial insufficient): %d files"
            % (st["n_crop_fallback"], st["n_header_fallback"]))
        if st["n_joint_slots"]:
            log("joint-centred zoom: %d / %d slots fell back to the image centre"
                % (st["n_joint_fallback"], st["n_joint_slots"]))
        if st["n_medial_side_fallback"]:
            log("medial zoom: %d studies had no resolved side (distal shift only)" % st["n_medial_side_fallback"])
        log("missing-slot histogram: " + "  ".join("%s %d" % (k, v) for k, v in st["missing_slot_hist"].items()))
        log("side: L %d  R %d  unresolved %d | source: tag %d  geometry %d  unresolved %d"
            % (st["side_hist"]["L"], st["side_hist"]["R"], st["side_hist"]["None"],
               st["side_source_hist"].get("tag", 0), st["side_source_hist"].get("geometry", 0),
               st["side_source_hist"].get("unresolved", 0)))
        log("peak RSS: parent %s | worker max %s (RUSAGE_CHILDREN %s)"
            % (gb(stats["peak_rss_parent_bytes"]), gb(stats["peak_rss_worker_bytes"]), gb(peak_children)))
        log("cache: %s | %d/%d rows done | %d shards | %s on disk"
            % (args.out, n_done_cache, n_total, len(index["shards"]), gb(on_disk)))
        if st["n_new"] and stats["studies_per_s"] > 0:
            for n_proj in (1300, 4407):
                log("projection @ %d studies at this rate (%d workers): %.1f min"
                    % (n_proj, workers, n_proj / stats["studies_per_s"] / 60.0))
        if index["failed"]:
            log("failed studies (not marked done, retried on the next run): %d" % len(index["failed"]))
        if stopped_early[0] and n_done_cache < n_total:
            log("STOPPED EARLY: %d rows remain; rerun the same command to resume" % (n_total - n_done_cache))
    return 3 if (stopped_early[0] and n_done_cache < n_total) else 0


if __name__ == "__main__":
    sys.exit(main())
