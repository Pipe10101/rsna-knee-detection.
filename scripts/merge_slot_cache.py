"""Merge two (or more) partial slot caches built with --skip/--limit into one cache dir.

    python3 scripts/merge_slot_cache.py --parts cache/g20A/slots_P224_g20t1_partA cache/g20B/slots_P224_g20t1_partB \
        --out cache/slots_P224_g20t1_full/slots_P224_g20t1 [--split train] [--link] [--test-from FIRST_PART]

Parts must share every layout parameter (P, G, T, crop, trim, zoom, slot_names, version) and have
disjoint study lists; studies are concatenated in the given part order, shard files are renumbered
(hardlinked with --link, else copied), masks are concatenated, and the merged index carries explicit
per-shard start/n entries.  The result is READ-ONLY for training (`src.slots.SlotCache` reads the
explicit shard table); it is not resumable by build_slot_cache.py, whose loader would recompute a
regular shard layout and refuse the irregular boundaries.  test_* files are taken verbatim from the
first part that has them (or --test-from).
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import slots


PARAMS = ("P", "G", "T", "crop_mm", "trim_frac", "version", "slot_names", "zoom_mm", "zoom_center")


def load_index(part, split):
    p = slots.cache_paths(part, split)["index"]
    with open(p) as f:
        return json.load(f)


def place(src, dst, link):
    if os.path.exists(dst):
        os.remove(dst)
    if link:
        try:
            os.link(src, dst)
            return "linked"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copied"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--link", action="store_true", help="hardlink shard files instead of copying (same filesystem)")
    ap.add_argument("--test-from", default=None, help="part dir whose test_* files are taken verbatim")
    args = ap.parse_args(argv)
    split = args.split

    idxs = [load_index(p, split) for p in args.parts]
    ref = idxs[0]
    for p, ix in zip(args.parts[1:], idxs[1:]):
        for k in PARAMS:
            a, b = ref.get(k), ix.get(k)
            if (list(a) if isinstance(a, (list, tuple)) else a) != (list(b) if isinstance(b, (list, tuple)) else b):
                raise SystemExit(f"{p}: {k}={b!r} != {ref.get(k)!r} of {args.parts[0]}")
        zs_a, zs_b = list(ref.get("zoom_slots") or []), list(ix.get("zoom_slots") or [])
        if zs_a != zs_b:
            raise SystemExit(f"{p}: zoom_slots {zs_b} != {zs_a}")
    seen = set()
    for p, ix in zip(args.parts, idxs):
        dup = seen & set(ix["studies"])
        if dup:
            raise SystemExit(f"{p}: {len(dup)} studies overlap an earlier part (e.g. {sorted(dup)[0]})")
        seen |= set(ix["studies"])
        if not all(ix.get("done", [])):
            n_missing = sum(1 for d in ix.get("done", []) if not d)
            print(f"WARNING: {p} has {n_missing} studies not done; they merge as not-done rows")

    os.makedirs(args.out, exist_ok=True)
    S = len(ref["slot_names"])
    row_bytes = S * int(ref["G"]) * int(ref["T"]) * int(ref["P"]) * int(ref["P"])

    merged = {k: ref.get(k) for k in ("P", "G", "T", "crop_mm", "trim_frac", "version", "split",
                                      "slot_names", "zoom_mm", "zoom_slots", "zoom_center")}
    merged.update({"split": split, "shard_size": int(ref.get("shard_size", 0)),
                   "studies": [], "done": [], "side": [], "failed": {}, "stats": {}, "cumulative": {},
                   "shards": [], "mask": os.path.basename(slots.cache_paths(args.out, split)["mask"]),
                   "merged_from": [os.path.abspath(p) for p in args.parts]})
    k_out, verb = 0, {}
    for p, ix in zip(args.parts, idxs):
        offset = len(merged["studies"])
        merged["studies"] += list(ix["studies"])
        merged["done"] += list(ix.get("done", [1] * len(ix["studies"])))
        merged["side"] += list(ix.get("side", [None] * len(ix["studies"])))
        merged["failed"].update(ix.get("failed", {}))
        merged["stats"][os.path.basename(p)] = ix.get("stats", {})
        for sh in ix["shards"]:
            src = os.path.join(p, sh["x"])
            want = int(sh["n"]) * row_bytes
            have = os.path.getsize(src)
            if have != want:
                raise SystemExit(f"{src}: {have} bytes != expected {want}")
            name = slots.shard_filename(split, k_out)
            v = place(src, os.path.join(args.out, name), args.link)
            verb[v] = verb.get(v, 0) + 1
            merged["shards"].append({"x": name, "start": offset + int(sh["start"]), "n": int(sh["n"])})
            k_out += 1
    # mask: straight byte concatenation (N x S uint8 per part)
    with open(slots.cache_paths(args.out, split)["mask"], "wb") as out_m:
        for p, ix in zip(args.parts, idxs):
            src = slots.cache_paths(p, split)["mask"]
            want = len(ix["studies"]) * S
            if os.path.getsize(src) != want:
                raise SystemExit(f"{src}: {os.path.getsize(src)} bytes != expected {want}")
            with open(src, "rb") as f:
                shutil.copyfileobj(f, out_m)
    with open(slots.cache_paths(args.out, split)["index"], "w") as f:
        json.dump(merged, f)
    # test split: verbatim from --test-from or the first part that has one
    src_test = None
    for p in ([args.test_from] if args.test_from else []) + args.parts:
        if p and os.path.isfile(slots.cache_paths(p, "test")["index"]):
            src_test = p
            break
    if src_test:
        for f in os.listdir(src_test):
            if f.startswith("test_"):
                place(os.path.join(src_test, f), os.path.join(args.out, f), args.link)
        print(f"test_* taken from {src_test}")
    # verify: SlotCache opens and the boundaries look sane
    c = slots.SlotCache(args.out, split=split)
    n_done = int(c.done.sum())
    print(f"merged {len(args.parts)} parts -> {args.out}: {c.N} studies ({n_done} done), "
          f"{len(merged['shards'])} shards ({verb}), S={c.S} G={c.G} T={c.T} P={c.P}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
