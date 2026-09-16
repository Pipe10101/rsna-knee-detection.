#!/usr/bin/env python3
"""compare_arms.py -- the campaign's decision tool: pooled-OOF tables with per-label deltas.

Compares one or more ablation arms against a baseline on the out-of-fold predictions written
by scripts/train_slotknee.py (oof_fold_<k>.npz with keys uids, logits, y, w, mask), scored
exactly like its validation metric:
  * soft targets binarised at 0.5, only cells with w > 0 count, one-class labels skipped;
  * POOLED = concatenate the folds first, then one ROC-AUC per label, macro over labels
    (NOT the mean of per-fold AUCs).

Fold consistency.  An arm is scored only on folds present in BOTH the baseline and the arm
(further restricted by --folds), and the baseline is re-scored on those same folds for the
delta.  The table also reports the uid overlap of the common folds: if the two runs used a
different fold split (e.g. the 5-fold T4 run vs a 2-fold laptop run) the overlap is low and
--common-uids switches to matching on studies instead -- every available fold of each run
(subject to --folds) is pooled and both runs are scored on the intersection of study uids.

Verdict.  ADOPT if delta > 2*noise, REGRESS if delta < -2*noise, else NULL, where noise is
--noise <float>, or --noise-from <seedA_dir> <seedB_dir> (|pooled-AUC difference| of two seeds
of the same recipe over their common folds), or 0.006 assumed (docs/slotknee_runbook.md).

--combine A B adds a "combined" row: per fold, rank-average the two arms' logits (ranks
normalised to [0, 1] within the fold) over their matched uids, then pool -- the cheap
estimate of what ensembling the two arms would give.

Run directories are searched recursively for oof_fold_*.npz, so Kaggle-fetched layouts
(models/kaggle_<kernel>_v<N>/...) work as-is.

Examples:
  python3 scripts/compare_arms.py models/ablate_full/*_s42 --baseline models/ablate_full/baseline_s42
  python3 scripts/compare_arms.py --baseline models/ablate_full/baseline_s42 \
      --arms models/ablate_full/*_s* --folds 0 1 --noise models/ablate_full/baseline_s1337 --json out.json
  python3 scripts/compare_arms.py models/slotknee_t4/models/slotknee_full5 \
      --baseline models/ablate_full/baseline_s42 --folds 0 1 --common-uids
  python3 scripts/compare_arms.py models/ablate_full/tb6_s42 models/ablate_full/g10_s42 \
      --baseline models/ablate_full/baseline_s42 \
      --noise-from models/ablate_full/baseline_s42 models/ablate_full/baseline_s1337 \
      --combine models/ablate_full/tb6_s42 models/ablate_full/g10_s42 --json out.json
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

# Spelling matches the competition CSV headers (src/llm_labels.py); copied so this tool has
# no import-side effects (pandas, torch) and stays fast in tests.
LABELS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
SHORT: List[str] = ["ACL", "MCL", "MedMen", "LatMen", "MedOA", "LatOA", "PFOA",
                    "Effus", "Synov", "Baker", "Contus", "Fract"]

DEFAULT_NOISE = 0.006          # docs/slotknee_runbook.md: assume until measured
# "Weak-label focus": the findings where the labels are least reliable / the model weakest;
# the table reports the mean per-label delta over them next to the macro delta.
DEFAULT_FOCUS: List[str] = ["ACL", "MCL", "Lateral Meniscus"]
_OOF_RE = re.compile(r"oof_fold_(\d+)\.npz$")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class FoldData:
    uids: np.ndarray     # (N,) str
    logits: np.ndarray   # (N, L) float
    y: np.ndarray        # (N, L) float, soft targets
    w: np.ndarray        # (N, L) float, 0 = silent cell

    def subset(self, idx: np.ndarray) -> "FoldData":
        return FoldData(self.uids[idx], self.logits[idx], self.y[idx], self.w[idx])


@dataclass
class Run:
    name: str
    path: str
    folds: Dict[int, FoldData]


def discover_oof(run_dir: str) -> Dict[int, str]:
    """Map fold index -> oof_fold_<k>.npz under run_dir (recursive). If a fold appears more
    than once (nested Kaggle layouts), the shallowest path wins and a warning is printed."""
    found: Dict[int, List[str]] = {}
    pattern = os.path.join(glob.escape(run_dir), "**", "oof_fold_*.npz")
    for p in sorted(glob.glob(pattern, recursive=True)):
        m = _OOF_RE.search(os.path.basename(p))
        if m:
            found.setdefault(int(m.group(1)), []).append(p)
    out: Dict[int, str] = {}
    for k, paths in found.items():
        paths = sorted(paths, key=lambda p: (p.count(os.sep), p))
        if len(paths) > 1:
            print(f"[compare_arms] warning: fold {k} appears {len(paths)}x under {run_dir}; "
                  f"using {paths[0]}", file=sys.stderr)
        out[k] = paths[0]
    return out


def load_fold(path: str) -> FoldData:
    z = np.load(path, allow_pickle=True)
    uids = np.asarray(z["uids"]).astype(str)
    logits = np.asarray(z["logits"], dtype=np.float64)
    y = np.asarray(z["y"], dtype=np.float64)
    w = np.asarray(z["w"], dtype=np.float64)
    if not (len(uids) == len(logits) == len(y) == len(w)):
        raise ValueError(f"{path}: inconsistent row counts")
    if logits.shape[1] != len(LABELS):
        raise ValueError(f"{path}: expected {len(LABELS)} label columns, got {logits.shape[1]}")
    return FoldData(uids, logits, y, w)


def load_run(run_dir: str, name: Optional[str] = None) -> Run:
    run_dir = os.path.normpath(run_dir)
    files = discover_oof(run_dir)
    if not files:
        raise FileNotFoundError(f"no oof_fold_*.npz under {run_dir}")
    return Run(name or os.path.basename(run_dir), run_dir,
               {k: load_fold(p) for k, p in sorted(files.items())})


# ---------------------------------------------------------------------------
# Scoring (mirrors validate() in scripts/train_slotknee.py, pooled over folds)
# ---------------------------------------------------------------------------
def pool(run: Run, folds: Iterable[int], keep_uids: Optional[Set[str]] = None) -> FoldData:
    """Concatenate the given folds (sorted), optionally keeping only uids in keep_uids."""
    parts = [run.folds[f] for f in sorted(folds)]
    if not parts:
        empty = np.zeros((0, len(LABELS)))
        return FoldData(np.zeros((0,), dtype=str), empty, empty.copy(), empty.copy())
    fd = FoldData(np.concatenate([p.uids for p in parts]),
                  np.concatenate([p.logits for p in parts]),
                  np.concatenate([p.y for p in parts]),
                  np.concatenate([p.w for p in parts]))
    if keep_uids is not None:
        idx = np.array([u in keep_uids for u in fd.uids], dtype=bool)
        fd = fd.subset(idx)
    return fd


def per_label_auc(fd: FoldData) -> np.ndarray:
    """One ROC-AUC per label on cells with w > 0, targets binarised at 0.5; nan where the
    label has < 2 classes (or no active cells)."""
    out = np.full(len(LABELS), np.nan)
    for i in range(len(LABELS)):
        m = fd.w[:, i] > 0
        if not m.any():
            continue
        yt = (fd.y[m, i] > 0.5).astype(int)
        if len(np.unique(yt)) > 1:
            out[i] = roc_auc_score(yt, fd.logits[m, i])
    return out


def macro(aucs: np.ndarray) -> float:
    ok = ~np.isnan(aucs)
    return float(np.mean(aucs[ok])) if ok.any() else float("nan")


def pooled_macro(run: Run, folds: Iterable[int], keep_uids: Optional[Set[str]] = None) -> float:
    return macro(per_label_auc(pool(run, folds, keep_uids)))


def verdict(delta: float, noise: float) -> str:
    if delta is None or math.isnan(delta):
        return "n/a"
    if delta > 2 * noise:
        return "ADOPT"
    if delta < -2 * noise:
        return "REGRESS"
    return "NULL"


# ---------------------------------------------------------------------------
# Combining two arms (ensembling estimate)
# ---------------------------------------------------------------------------
def rank_average_fold(a: FoldData, b: FoldData, tag: str = "") -> FoldData:
    """Rank-average the two arms' logits over their matched uids (ranks normalised to
    [0, 1] within the fold). Targets/weights are taken from `a`."""
    pos_b = {u: i for i, u in enumerate(b.uids)}
    ia = np.array([i for i, u in enumerate(a.uids) if u in pos_b], dtype=int)
    if len(ia) == 0:
        raise ValueError(f"combine{tag}: no shared uids between the two arms")
    ib = np.array([pos_b[a.uids[i]] for i in ia], dtype=int)
    if len(ia) < len(a.uids) or len(ia) < len(b.uids):
        print(f"[compare_arms] warning: combine{tag}: arms share {len(ia)} uids "
              f"(of {len(a.uids)} / {len(b.uids)}); using the intersection", file=sys.stderr)
    sa, sb = a.subset(ia), b.subset(ib)
    if not (np.allclose(sa.y, sb.y) and np.allclose(sa.w, sb.w)):
        print(f"[compare_arms] warning: combine{tag}: targets/weights differ between arms; "
              "using the first arm's", file=sys.stderr)
    n = len(ia)
    ra = np.column_stack([rankdata(sa.logits[:, j]) / n for j in range(sa.logits.shape[1])])
    rb = np.column_stack([rankdata(sb.logits[:, j]) / n for j in range(sb.logits.shape[1])])
    return FoldData(sa.uids, 0.5 * (ra + rb), sa.y, sa.w)


def combine_runs(a: Run, b: Run, folds: Optional[Iterable[int]] = None) -> Run:
    common = sorted(set(a.folds) & set(b.folds) & (set(folds) if folds is not None else set(a.folds)))
    if not common:
        raise ValueError(f"combine: no common folds between {a.name} and {b.name}")
    return Run(f"combine({a.name}+{b.name})", f"{a.path} + {b.path}",
               {f: rank_average_fold(a.folds[f], b.folds[f], f" fold {f}") for f in common})


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _clean(x: float) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def common_folds(base: Run, arm: Run, folds_req: Optional[Sequence[int]]) -> List[int]:
    c = set(base.folds) & set(arm.folds)
    if folds_req is not None:
        c &= set(folds_req)
    return sorted(c)


def uid_overlap_per_fold(base: Run, arm: Run, folds: Iterable[int]) -> Dict[int, float]:
    """Per fold: fraction of the arm's studies that the baseline holds in the SAME fold --
    1.0 everywhere means the two runs share the fold split."""
    out = {}
    for f in sorted(folds):
        au = set(arm.folds[f].uids.tolist())
        out[f] = (len(au & set(base.folds[f].uids.tolist())) / len(au)) if au else float("nan")
    return out


def uid_overlap(base: Run, arm: Run, folds: Iterable[int]) -> float:
    """Pooled version of uid_overlap_per_fold (weighted by studies per fold)."""
    folds = list(folds)
    tot = sum(len(arm.folds[f].uids) for f in folds)
    if tot == 0:
        return float("nan")
    hit = sum(len(set(arm.folds[f].uids.tolist()) & set(base.folds[f].uids.tolist())) for f in folds)
    return hit / tot


def per_fold_macro(run: Run, folds: Iterable[int]) -> Dict[int, float]:
    """Per-fold macro AUC (for reference only; decisions use the pooled number)."""
    return {f: pooled_macro(run, [f]) for f in sorted(folds)}


def focus_delta(per_label_delta: Dict[str, Optional[float]], focus: Sequence[str]) -> Optional[float]:
    """Mean per-label delta over the focus labels (labels with no AUC are skipped)."""
    vals = [per_label_delta[l] for l in focus if per_label_delta.get(l) is not None]
    return float(np.mean(vals)) if vals else None


def compare(base: Run, arm: Run, noise: float, folds_req: Optional[Sequence[int]] = None,
            common_uids: bool = False, focus: Sequence[str] = DEFAULT_FOCUS) -> dict:
    """Score `arm` against `base` on their common folds (or common uids) and return a record."""
    rec = {"arm": arm.name, "dir": arm.path, "mode": "uids" if common_uids else "folds"}
    if common_uids:
        fb = sorted(set(base.folds) & set(folds_req)) if folds_req is not None else sorted(base.folds)
        fa = sorted(set(arm.folds) & set(folds_req)) if folds_req is not None else sorted(arm.folds)
        pb_all, pa_all = pool(base, fb), pool(arm, fa)
        keep = set(pb_all.uids.tolist()) & set(pa_all.uids.tolist())
        pb, pa = pool(base, fb, keep), pool(arm, fa, keep)
        rec.update(folds=fa, base_folds=fb,
                   uid_overlap=(len(keep) / len(pa_all.uids)) if len(pa_all.uids) else float("nan"),
                   uid_overlap_per_fold={})
    else:
        folds = common_folds(base, arm, folds_req)
        pb, pa = pool(base, folds), pool(arm, folds)
        rec.update(folds=folds, base_folds=folds,
                   uid_overlap=uid_overlap(base, arm, folds) if folds else float("nan"),
                   uid_overlap_per_fold={str(f): _clean(v)
                                         for f, v in uid_overlap_per_fold(base, arm, folds).items()})
    rec["n_studies"], rec["n_baseline"] = int(len(pa.uids)), int(len(pb.uids))
    rec["per_fold_auc"] = {str(f): _clean(v) for f, v in per_fold_macro(arm, rec["folds"]).items()}
    rec["base_per_fold_auc"] = {str(f): _clean(v) for f, v in per_fold_macro(base, rec["base_folds"]).items()}
    if rec["n_studies"] == 0 or rec["n_baseline"] == 0:
        rec.update(auc=float("nan"), base_auc=float("nan"), delta=float("nan"),
                   per_label_auc={l: None for l in LABELS},
                   base_per_label_auc={l: None for l in LABELS},
                   per_label_delta={l: None for l in LABELS}, focus_delta=None,
                   verdict="n/a", win=False, note="no common folds/studies with the baseline")
        return rec
    la, lb = per_label_auc(pa), per_label_auc(pb)
    rec["auc"], rec["base_auc"] = macro(la), macro(lb)
    rec["delta"] = rec["auc"] - rec["base_auc"]
    rec["per_label_auc"] = {l: _clean(v) for l, v in zip(LABELS, la)}
    rec["base_per_label_auc"] = {l: _clean(v) for l, v in zip(LABELS, lb)}
    rec["per_label_delta"] = {l: _clean(v) for l, v in zip(LABELS, la - lb)}
    rec["focus_delta"] = focus_delta(rec["per_label_delta"], focus)
    rec["verdict"] = verdict(rec["delta"], noise)
    rec["win"] = rec["verdict"] == "ADOPT"
    if not common_uids and rec["uid_overlap"] < 0.999:
        rec["note"] = (f"fold split differs from baseline (same-fold uid overlap "
                       f"{rec['uid_overlap']:.0%}); consider --common-uids")
    return rec


def noise_from_seeds(a: Run, b: Run, folds_req: Optional[Sequence[int]] = None) -> dict:
    """Seed-to-seed noise floor from two runs of the same recipe, two ways:
      naive  = |pooled(a) - pooled(b)| on their common fold ids (fold-matched);
      paired = the same on the intersection of study uids, pooling EVERY fold of each run
               (--folds is ignored here: noise is a property of the recipe, and more shared
               studies make a better estimate) -- the right estimate when the seeds use
               different fold splits (train_slotknee.py derives the split from --seed).
    The gate uses the LARGER of the two ("used")."""
    folds = common_folds(a, b, folds_req)
    naive = abs(pooled_macro(a, folds) - pooled_macro(b, folds)) if folds else None
    fa, fb = sorted(a.folds), sorted(b.folds)
    keep = set(pool(a, fa).uids.tolist()) & set(pool(b, fb).uids.tolist())
    paired = abs(pooled_macro(a, fa, keep) - pooled_macro(b, fb, keep)) if keep else None
    same_split = bool(folds) and abs(uid_overlap(a, b, folds) - 1.0) < 1e-9
    if naive is None and paired is None:
        raise ValueError(f"noise: {a.name} and {b.name} share neither folds nor studies")
    cands = {k: v for k, v in (("naive", naive), ("paired", paired)) if v is not None}
    used = max(cands, key=cands.get)
    return {"value": cands[used], "used": used, "naive": naive, "naive_folds": folds,
            "paired": paired, "paired_n": len(keep), "same_split": same_split,
            "runs": [a.name, b.name]}


def _noise_src(nz: dict) -> str:
    parts = [f"seed-to-seed |{nz['runs'][0]} - {nz['runs'][1]}|"]
    if nz["naive"] is not None:
        parts.append(f"naive fold-matched {nz['naive']:.4f} on folds {_folds_str(nz['naive_folds'])}")
    if nz["paired"] is not None:
        parts.append(f"paired common-uids {nz['paired']:.4f} on {nz['paired_n']} studies")
    parts.append(f"using {nz['used']}" + ("" if nz["same_split"] else "; splits differ"))
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _f(v: Optional[float], signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.4f}" if signed else f"{v:.4f}"


def _folds_str(folds: Sequence[int]) -> str:
    return ",".join(str(f) for f in folds) if folds else "-"


def _per_fold_str(d: Dict[str, Optional[float]]) -> str:
    return "/".join(_f(d[k]) for k in sorted(d, key=int)) if d else "-"


def render_markdown(base_rec: dict, arm_recs: List[dict], noise: float, noise_src: str,
                    absolute: bool = False, focus: Sequence[str] = DEFAULT_FOCUS) -> str:
    hdr = ["arm", "folds", "n", "uid-ovl", "AUC", "per-fold", "delta", "d focus", "verdict"] + [f"d {s}" for s in SHORT]
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    base_cells = [f"**{base_rec['arm']}** (abs)", _folds_str(base_rec["folds"]),
                  str(base_rec["n_studies"]), "-", _f(base_rec["auc"]),
                  _per_fold_str(base_rec["per_fold_auc"]), "-", _f(base_rec["focus_auc"]), "base"]
    base_cells += [_f(base_rec["per_label_auc"][l]) for l in LABELS]
    lines.append("| " + " | ".join(base_cells) + " |")
    for r in arm_recs:
        cells = [r["arm"], _folds_str(r["folds"]), str(r["n_studies"]),
                 _per_fold_str(r["uid_overlap_per_fold"]) if r["uid_overlap_per_fold"]
                 else (_f(r["uid_overlap"]) if not math.isnan(r["uid_overlap"]) else "n/a"),
                 _f(r["auc"]), _per_fold_str(r["per_fold_auc"]),
                 _f(r["delta"], signed=True), _f(r["focus_delta"], signed=True), r["verdict"]]
        cells += [_f(r["per_label_delta"][l], signed=True) for l in LABELS]
        lines.append("| " + " | ".join(cells) + " |")
    out = "\n".join(lines)
    out += (f"\n\nnoise = {noise:.4f} ({noise_src}); ADOPT if delta > {2 * noise:+.4f}, "
            f"REGRESS if delta < {-2 * noise:+.4f}, else NULL. Pooled = folds concatenated "
            "before per-label AUC; baseline row shows absolute per-label AUCs, arm rows show "
            f"deltas vs the baseline re-scored on that arm's folds/studies. uid-ovl = per-fold "
            "fraction of the arm's studies that the baseline has in the SAME fold (1.0 = same "
            "split; pooled fraction in --common-uids mode). d focus = mean delta over "
            f"{', '.join(focus)} (baseline row: their mean AUC).")
    notes = [f"- {r['arm']}: {r['note']}" for r in arm_recs if r.get("note")]
    if notes:
        out += "\n\nNotes:\n" + "\n".join(notes)
    if absolute:
        hdr2 = ["arm", "folds", "base AUC", "AUC"] + SHORT
        lines2 = ["| " + " | ".join(hdr2) + " |", "|" + "|".join(["---"] * len(hdr2)) + "|"]
        for r in arm_recs:
            cells = [r["arm"], _folds_str(r["folds"]), _f(r["base_auc"]), _f(r["auc"])]
            cells += [_f(r["per_label_auc"][l]) for l in LABELS]
            lines2.append("| " + " | ".join(cells) + " |")
        out += "\n\nAbsolute per-label AUCs (arm on its common folds/studies):\n\n" + "\n".join(lines2)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("arms", nargs="*", help="run directories holding oof_fold_*.npz (searched recursively)")
    p.add_argument("--arms", dest="arms_opt", nargs="+", default=[], metavar="DIR",
                   help="same as the positional arms (flag form)")
    p.add_argument("--baseline", required=True, help="baseline run directory")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="restrict to these fold indices (only folds present in BOTH runs are used)")
    p.add_argument("--common-uids", action="store_true",
                   help="match on studies instead of folds: pool every available fold of each run "
                        "and score both on the intersection of uids (use when fold splits differ)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--noise", default=None, metavar="FLOAT|DIR",
                   help="seed-to-seed noise floor (macro AUC), or a directory holding a second "
                        "seed of the baseline recipe: noise = |pooled(baseline) - pooled(DIR)|")
    g.add_argument("--noise-from", nargs=2, metavar=("DIR_SEED_A", "DIR_SEED_B"),
                   help="measure the noise floor as |pooled AUC(A) - pooled AUC(B)| of two seeds")
    p.add_argument("--combine", nargs=2, metavar=("DIR_A", "DIR_B"),
                   help="add a row with the per-fold rank-average of two arms' logits")
    p.add_argument("--json", default=None, help="also write the records to this JSON file")
    p.add_argument("--absolute", action="store_true", help="also print absolute per-label AUCs per arm")
    p.add_argument("--focus", nargs="+", default=list(DEFAULT_FOCUS), metavar="LABEL",
                   help="labels for the weak-label focus column (mean per-label delta); "
                        f"default: {', '.join(DEFAULT_FOCUS)}")
    return p


def _unique_names(runs: List[Run]) -> None:
    seen: Dict[str, int] = {}
    for r in runs:
        seen[r.name] = seen.get(r.name, 0) + 1
    for r in runs:
        if seen[r.name] > 1:
            r.name = os.path.join(os.path.basename(os.path.dirname(r.path)), os.path.basename(r.path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.arms = list(args.arms) + list(args.arms_opt)
    if not args.arms and not args.combine:
        parser.error("give at least one arm directory or --combine")
    bad = [l for l in args.focus if l not in LABELS]
    if bad:
        parser.error(f"--focus: unknown label(s) {bad}; choose from {LABELS}")

    base = load_run(args.baseline)
    base_path = os.path.normpath(os.path.abspath(args.baseline))
    cache: Dict[str, Run] = {base_path: base}

    def get(d: str) -> Run:
        key = os.path.normpath(os.path.abspath(d))
        if key not in cache:
            cache[key] = load_run(d)
        return cache[key]

    arms: List[Run] = []
    for d in args.arms:
        key = os.path.normpath(os.path.abspath(d))
        if key == base_path:
            print(f"[compare_arms] skipping {d}: it is the baseline", file=sys.stderr)
            continue
        if any(os.path.normpath(os.path.abspath(a.path)) == key for a in arms):
            continue
        arms.append(get(d))
    _unique_names(arms)

    noise_detail: Optional[dict] = None
    if args.noise is not None and os.path.isdir(args.noise):
        noise_detail = noise_from_seeds(base, get(args.noise), args.folds)
        noise, noise_src = noise_detail["value"], _noise_src(noise_detail)
    elif args.noise is not None:
        try:
            noise = float(args.noise)
        except ValueError:
            parser.error(f"--noise: {args.noise!r} is neither a number nor a directory")
        noise_src = "given"
    elif args.noise_from:
        noise_detail = noise_from_seeds(get(args.noise_from[0]), get(args.noise_from[1]), args.folds)
        noise, noise_src = noise_detail["value"], _noise_src(noise_detail)
    else:
        noise, noise_src = DEFAULT_NOISE, "ASSUMED, docs/slotknee_runbook.md; measure with --noise-from"

    base_folds = sorted(set(base.folds) & set(args.folds)) if args.folds is not None else sorted(base.folds)
    base_pooled = pool(base, base_folds)
    base_aucs = per_label_auc(base_pooled)
    base_rec = {"arm": base.name, "dir": base.path, "folds": base_folds,
                "n_studies": int(len(base_pooled.uids)), "auc": macro(base_aucs),
                "per_fold_auc": {str(f): _clean(v) for f, v in per_fold_macro(base, base_folds).items()},
                "per_label_auc": {l: _clean(v) for l, v in zip(LABELS, base_aucs)}}
    base_rec["focus_auc"] = focus_delta(base_rec["per_label_auc"], args.focus)

    recs = [compare(base, arm, noise, args.folds, args.common_uids, args.focus) for arm in arms]
    combined_rec = None
    if args.combine:
        comb = combine_runs(get(args.combine[0]), get(args.combine[1]), args.folds)
        combined_rec = compare(base, comb, noise, args.folds, args.common_uids, args.focus)
        recs.append(combined_rec)

    print(render_markdown(base_rec, recs, noise, noise_src, absolute=args.absolute, focus=args.focus))

    if args.json:
        payload = {"labels": LABELS, "focus_labels": list(args.focus),
                   "folds_requested": args.folds, "common_uids": args.common_uids,
                   "noise": {"value": noise, "source": noise_src, "detail": noise_detail},
                   "baseline": base_rec,
                   "arms": [r for r in recs if r is not combined_rec],
                   "combined": combined_rec}

        def default(o):
            if isinstance(o, (np.floating, float)):
                return None if math.isnan(float(o)) else float(o)
            if isinstance(o, np.integer):
                return int(o)
            raise TypeError(f"not serialisable: {type(o)}")

        # nan -> null (json.dumps would emit NaN otherwise)
        def scrub(x):
            if isinstance(x, dict):
                return {k: scrub(v) for k, v in x.items()}
            if isinstance(x, list):
                return [scrub(v) for v in x]
            if isinstance(x, float) and math.isnan(x):
                return None
            return x

        with open(args.json, "w") as fh:
            json.dump(scrub(payload), fh, indent=2, default=default)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
