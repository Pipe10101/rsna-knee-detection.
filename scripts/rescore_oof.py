#!/usr/bin/env python3
"""Re-score a fold checkpoint's held-out studies from the slot cache, with and without
MULTI-BAG inference -- the read-out that matches how the model was trained.

    python3 scripts/rescore_oof.py --cache CACHE_DIR --ckpt fold_0_best.pt --oof oof_fold_0.npz \
        --bag 6 --bags 4 [--out oof_fold_0_bags.npz] [--bs 8] [--device auto] [--seed 42]

Why this exists (2026-09-03).  The adopted recipe trains with `--anchor-bag 6`: every epoch the
model sees a random 6 of the cache's 10 anchors per slot, and `group_embed` is indexed by
POSITION (0..5), so positions 6..9 are never trained.  Validation and submission then show it
all 10 anchors.  Averaging N independent 6-anchor bags at inference (a) matches the training
distribution, (b) never touches an untrained position, and (c) is free ensembling from one set
of weights.  It costs N*0.6 forward-equivalents per study instead of 1.0.  This script measures
whether that buys AUC, on exactly the studies the fold held out (their uids and labels come from
the OOF npz, so no fold logic is re-derived).  Runs on CPU; a T4 does 1,760 studies in minutes.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score
from src.slots import SlotCache, is_medial_slot
from src.slotknee import SlotKneeS

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


LAT_PAIRS = (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"))


def lat_perm(labels):
    """Index permutation that swaps each medial/lateral label pair (an involution)."""
    perm = list(range(len(labels)))
    for a, b in LAT_PAIRS:
        if a in labels and b in labels:
            ia, ib = labels.index(a), labels.index(b)
            perm[ia], perm[ib] = ib, ia
    return perm


def mirror_study(x, slot_names):
    """Inverse of the cache's laterality normalisation on a [B,S,G,T,P,P] batch: COR/AX slots flip
    columns, SAG slots reverse anchor order (mirrors src.slots.apply_laterality)."""
    out = x.clone()
    for s, name in enumerate(slot_names):
        plane = name.split("_")[0]
        if is_medial_slot(name):          # medial-centred zoom: its mirror is not lateral anatomy -> drop
            out[:, s] = 0
        elif plane in ("COR", "AX"):
            out[:, s] = x[:, s].flip(-1)
        elif plane == "SAG":
            out[:, s] = x[:, s].flip(1)
        else:
            raise ValueError(f"cannot infer plane from slot name {name!r}")
    return out


def mirror_mask(mask, slot_names):
    """[B,S] slot mask for the mirrored pass: medial-centred zoom slots become absent."""
    out = mask.clone()
    for s, name in enumerate(slot_names):
        if is_medial_slot(name):
            out[:, s] = 0
    return out


def macro_auc(logits, y, w):
    per = {}
    for i, name in enumerate(LABELS[: logits.shape[1]]):
        active = w[:, i] > 0
        if active.sum() == 0:
            continue
        yt = (y[active, i] > 0.5).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        per[name] = float(roc_auc_score(yt, logits[active, i]))
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--oof", required=True, help="oof_fold_k.npz written by train_slotknee.py (uids, y, w)")
    ap.add_argument("--bag", type=int, default=6, help="anchors per bag (< the cache's G)")
    ap.add_argument("--bags", type=int, default=4, help="independent bags averaged per study")
    ap.add_argument("--out", default=None, help="write the multi-bag logits as an OOF-style npz")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N held-out studies (smoke runs)")
    ap.add_argument("--flip-tta", action="store_true",
                    help="also score the laterality-MIRRORED study (COR/AX columns flipped, SAG anchor order "
                         "reversed) with the medial/lateral labels swapped back, averaged with the plain read-out")
    a = ap.parse_args(argv)

    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    z = np.load(a.oof, allow_pickle=True)
    uids, y, w = [str(u) for u in z["uids"]], np.asarray(z["y"], dtype=np.float32), np.asarray(z["w"], dtype=np.float32)
    if a.limit:
        uids, y, w = uids[: a.limit], y[: a.limit], w[: a.limit]
    cache = SlotCache(a.cache, split="train")
    rows = [cache.row_of(u) for u in uids]
    G = int(cache.G)
    slot_names = list(getattr(cache, "slot_names", []))
    perm = lat_perm(LABELS)
    if not (0 < a.bag < G):
        raise SystemExit(f"--bag {a.bag} must be in 1..{G-1} for this cache (G={G})")

    ck = torch.load(a.ckpt, map_location="cpu")
    hp = {k: v for k, v in ck["hparams"].items() if k not in ("pretrained", "pretrained_path")}
    model = SlotKneeS(pretrained=False, **hp)
    model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
    gen = torch.Generator().manual_seed(a.seed)

    full, bagged, flipped, t0 = [], [], [], time.time()
    with torch.no_grad():
        for s in range(0, len(rows), a.bs):
            xs, ms = zip(*(cache[r] for r in rows[s: s + a.bs]))
            x = torch.from_numpy(np.stack([np.array(v, copy=True) for v in xs])).to(dev)
            m = torch.from_numpy(np.stack([np.array(v, copy=True) for v in ms])).to(dev)
            out = model(x, m); lg = (out[0] if isinstance(out, tuple) else out).float().cpu()
            full.append(lg)                                   # plain read-out (the baseline)
            if a.flip_tta:
                om = model(mirror_study(x, slot_names), mirror_mask(m, slot_names)); lm = (om[0] if isinstance(om, tuple) else om).float().cpu()
                flipped.append(0.5 * (lg + lm[:, perm]))      # mirrored view, medial/lateral swapped back, averaged
            acc = None
            for _ in range(a.bags):
                keep = torch.randperm(G, generator=gen)[: a.bag].sort().values.to(dev)
                o = model(x[:, :, keep], m); o = (o[0] if isinstance(o, tuple) else o).float().cpu()
                acc = o if acc is None else acc + o
            bagged.append(acc / a.bags)
    full, bagged = torch.cat(full).numpy(), torch.cat(bagged).numpy()
    flipped = torch.cat(flipped).numpy() if flipped else None

    auc_full, per_full = macro_auc(full, y, w)
    auc_bag, per_bag = macro_auc(bagged, y, w)
    rep = {"n": len(uids), "G": G, "bag": a.bag, "bags": a.bags, "flip_tta": bool(a.flip_tta), "seconds": round(time.time() - t0, 1),
           "macro_full": auc_full, "macro_bags": auc_bag, "delta": (auc_bag - auc_full) if np.isfinite(auc_bag) and np.isfinite(auc_full) else None,
           "per_label_delta": {k: round(per_bag[k] - per_full[k], 4) for k in per_full if k in per_bag}}
    if flipped is not None:
        auc_flip, per_flip = macro_auc(flipped, y, w)
        rep.update({"macro_flip_tta": auc_flip,
                    "delta_flip": (auc_flip - auc_full) if np.isfinite(auc_flip) and np.isfinite(auc_full) else None,
                    "per_label_delta_flip": {k: round(per_flip[k] - per_full[k], 4) for k in per_full if k in per_flip}})
    print(json.dumps(rep, indent=1), flush=True)
    if a.out:
        np.savez_compressed(a.out, uids=np.array(uids), logits=bagged, logits_full=full, y=y, w=w, report=json.dumps(rep),
                            **({"logits_flip": flipped} if flipped is not None else {}))
        print("wrote", a.out, flush=True)
    return rep


if __name__ == "__main__":
    main()
