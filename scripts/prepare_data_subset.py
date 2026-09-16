#!/usr/bin/env python3
"""Populate ``data_subset/`` for the RSNA Knee Abnormality Detection pipeline.

Layout produced (verified against src/config.py, src/kaggle_data.py,
src/train.py and src/infer.py):

    data_subset/
        train.csv                 # verbatim copy, 4407 rows, 12 label cols + Report
        train_series.csv          # verbatim copy
        test.csv                  # verbatim copy
        test_series.csv           # verbatim copy
        sample_submission.csv     # verbatim copy
        train_gold.csv            # derived: only the 58 fully-labelled studies
        subset_manifest.csv       # derived: which studies actually have pixels on disk
        train_images/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
        test_images/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
        train_series -> train_images   (compat symlink)
        test_series  -> test_images    (compat symlink)

``src/train.py``  reads ``<data_dir>/train.csv`` and walks ``<data_dir>/train_images``.
``src/infer.py``  reads ``<data_dir>/test.csv``  and walks ``<data_dir>/test_images``.
``RSNADataset.__getitem__`` uses ``os.walk``, so the series sub-directory nesting is fine.

SAFETY CONTRACT (this project was once wiped by a careless script):
  * This script NEVER deletes, moves, truncates a directory, or renames anything.
  * The kagglehub cache is opened strictly read-only.
  * Every write target is asserted to live under data_subset/.
  * Extraction stops the moment the next study would push free space below the
    60 GiB floor (plus a safety margin).
  * Re-running is safe and resumable: complete studies are skipped, incomplete
    files are re-written in place.

Usage:
    python3 scripts/prepare_data_subset.py                 # full run
    python3 scripts/prepare_data_subset.py --verify-only   # no writes, just audit
    python3 scripts/prepare_data_subset.py --max-studies 200
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import shutil
import sys
import time
import zipfile

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(PROJECT_ROOT, "data_subset")

CACHE = os.path.expanduser(
    "~/.cache/kagglehub/competitions/rsna-knee-abnormality-detection"
)
ARCHIVE = os.path.expanduser(
    "~/.cache/kagglehub/competitions/rsna-knee-abnormality-detection.archive"
)

CSVS = [
    "train.csv",
    "train_series.csv",
    "test.csv",
    "test_series.csv",
    "sample_submission.csv",
]

# The 12 binary targets scored by macro ROC-AUC (must match src/kaggle_data.KNEE_TARGETS).
KNEE_TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

GIB = 1024 ** 3
FREE_FLOOR = 60 * GIB      # hard floor: never let the volume drop below this
SAFETY_MARGIN = 2 * GIB    # stop this far above the floor so we never touch it
MAX_STUDIES = 800          # cap on total train studies (gold + extra)
SAMPLE_SEED = 42           # deterministic choice of the extra (unlabelled) studies
COPY_BUF = 4 * 1024 * 1024


# --------------------------------------------------------------------------
# Safety helpers
# --------------------------------------------------------------------------

def assert_inside_dest(path: str) -> str:
    """Refuse to write anywhere outside data_subset/."""
    real_dest = os.path.realpath(DEST)
    real_path = os.path.realpath(os.path.abspath(path))
    if not (real_path == real_dest or real_path.startswith(real_dest + os.sep)):
        raise RuntimeError("REFUSING to write outside data_subset/: %r" % path)
    return path


def free_bytes() -> int:
    return shutil.disk_usage(DEST if os.path.isdir(DEST) else PROJECT_ROOT).free


def fmt(n: float) -> str:
    return "%.2f GiB" % (n / GIB)


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def dir_bytes(path: str):
    """Return (total_bytes, file_count) for a directory tree."""
    total = 0
    count = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                count += 1
            except OSError:
                pass
    return total, count


# --------------------------------------------------------------------------
# Step 1 - CSVs
# --------------------------------------------------------------------------

def copy_csvs() -> None:
    log("--- Step 1: CSVs ---")
    os.makedirs(DEST, exist_ok=True)
    for name in CSVS:
        src = os.path.join(CACHE, name)
        dst = assert_inside_dest(os.path.join(DEST, name))
        src_size = os.path.getsize(src)
        if os.path.exists(dst) and os.path.getsize(dst) == src_size:
            log("  ok (already present, %d B): %s" % (src_size, name))
            continue
        shutil.copy2(src, dst)  # copy2, never move
        got = os.path.getsize(dst)
        assert got == src_size, "size mismatch for %s: %d != %d" % (name, got, src_size)
        log("  copied %s (%d B)" % (name, got))


def assign_gold_folds(gold, n_splits: int = 5):
    """Deterministic 5-fold assignment over the 58 gold studies.

    src/train.py currently uses ``df.index % 5`` over all 4407 rows, which puts
    only 4-18 gold studies in each fold - far too few and far too unbalanced for
    a stable macro ROC-AUC. This gives a multilabel-stratified alternative that a
    fold-aware trainer can consume directly.
    """
    import numpy as np

    y = gold[KNEE_TARGETS].values.astype(int)
    x = np.zeros((len(gold), 1))
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        splitter = MultilabelStratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=SAMPLE_SEED)
        folds = np.zeros(len(gold), dtype=int)
        for f, (_tr, va) in enumerate(splitter.split(x, y)):
            folds[va] = f
        return folds
    except Exception as exc:                                  # noqa: BLE001
        log("  (iterstrat unavailable: %r - falling back to positive-count strata)" % exc)
        from sklearn.model_selection import StratifiedKFold
        strata = y.sum(axis=1)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=SAMPLE_SEED)
        folds = np.zeros(len(gold), dtype=int)
        for f, (_tr, va) in enumerate(splitter.split(x, strata)):
            folds[va] = f
        return folds


def write_derived_csvs(gold_ids, present_train, present_test) -> None:
    """Small helper CSVs. The verbatim competition CSVs are left untouched."""
    import pandas as pd

    log("--- Step 5: derived helper CSVs ---")
    train = pd.read_csv(os.path.join(DEST, "train.csv"))

    gold = train[train["StudyInstanceUID"].isin(gold_ids)].copy()
    gold["has_images"] = gold["StudyInstanceUID"].isin(present_train)
    gold["fold"] = assign_gold_folds(gold)
    dst = assert_inside_dest(os.path.join(DEST, "train_gold.csv"))
    gold.to_csv(dst, index=False)
    log("  wrote train_gold.csv: %d rows, %d with images"
        % (len(gold), int(gold["has_images"].sum())))
    log("  gold fold sizes: %s" % gold["fold"].value_counts().sort_index().to_dict())

    rows = []
    for sid in sorted(present_train):
        rows.append({"StudyInstanceUID": sid, "split": "train",
                     "is_gold": sid in gold_ids})
    for sid in sorted(present_test):
        rows.append({"StudyInstanceUID": sid, "split": "test", "is_gold": False})
    man = pd.DataFrame(rows)
    dst = assert_inside_dest(os.path.join(DEST, "subset_manifest.csv"))
    man.to_csv(dst, index=False)
    log("  wrote subset_manifest.csv: %d rows" % len(man))


# --------------------------------------------------------------------------
# Step 2 - copy already-extracted studies out of the kagglehub cache
# --------------------------------------------------------------------------

def copy_study_from_cache(src_study_dir: str, dst_study_dir: str) -> int:
    """Copy one study tree. Skips files that already match by size. Never deletes."""
    copied = 0
    for root, _dirs, files in os.walk(src_study_dir):
        rel = os.path.relpath(root, src_study_dir)
        out_dir = dst_study_dir if rel == "." else os.path.join(dst_study_dir, rel)
        assert_inside_dest(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            if f.startswith("."):
                continue
            s = os.path.join(root, f)
            d = assert_inside_dest(os.path.join(out_dir, f))
            s_size = os.path.getsize(s)
            if os.path.exists(d) and os.path.getsize(d) == s_size:
                continue
            shutil.copy2(s, d)
            copied += 1
    return copied


def missing_bytes_for_copy(src_study_dir: str, dst_study_dir: str) -> int:
    """Bytes still needed for this study - 0 when it is already fully copied.

    Using *remaining* bytes (not total) means a re-run sitting exactly at the
    free-space floor is not blocked from re-verifying work it already did.
    """
    need = 0
    for root, _dirs, files in os.walk(src_study_dir):
        rel = os.path.relpath(root, src_study_dir)
        out_dir = dst_study_dir if rel == "." else os.path.join(dst_study_dir, rel)
        for f in files:
            if f.startswith("."):
                continue
            s_size = os.path.getsize(os.path.join(root, f))
            d = os.path.join(out_dir, f)
            if os.path.exists(d) and os.path.getsize(d) == s_size:
                continue
            need += s_size
    return need


def copy_cached_split(cache_sub: str, dest_sub: str) -> list:
    """Copy every study present in <cache>/<cache_sub> into data_subset/<dest_sub>."""
    src_root = os.path.join(CACHE, cache_sub)
    dst_root = assert_inside_dest(os.path.join(DEST, dest_sub))
    os.makedirs(dst_root, exist_ok=True)
    studies = sorted(d for d in os.listdir(src_root)
                     if not d.startswith(".") and os.path.isdir(os.path.join(src_root, d)))
    done = []
    for i, sid in enumerate(studies, 1):
        need = missing_bytes_for_copy(os.path.join(src_root, sid),
                                      os.path.join(dst_root, sid))
        fb = free_bytes()
        if fb - need < FREE_FLOOR + SAFETY_MARGIN:
            log("  STOP: free %s, study needs %s, floor %s"
                % (fmt(fb), fmt(need), fmt(FREE_FLOOR)))
            break
        n = copy_study_from_cache(os.path.join(src_root, sid),
                                  os.path.join(dst_root, sid))
        done.append(sid)
        if i % 10 == 0 or i == len(studies):
            log("  %s: %d/%d studies (free %s)" % (dest_sub, i, len(studies), fmt(free_bytes())))
        del n
    return done


# --------------------------------------------------------------------------
# Step 3 - extract extra studies from the 247 GB archive
# --------------------------------------------------------------------------

def build_archive_index(zf: zipfile.ZipFile):
    """study_uid -> (total_uncompressed_bytes, [ZipInfo, ...]) for train_series/."""
    per_study = collections.defaultdict(list)
    sizes = collections.Counter()
    for info in zf.infolist():
        name = info.filename
        if not name.startswith("train_series/") or not name.endswith(".dcm"):
            continue
        parts = name.split("/")
        if len(parts) < 3:
            continue
        per_study[parts[1]].append(info)
        sizes[parts[1]] += info.file_size
    return per_study, sizes


def extract_study(zf: zipfile.ZipFile, infos, dst_study_dir: str) -> int:
    """Extract one study's members. Resumable: skips files that already match size."""
    written = 0
    for info in infos:
        parts = info.filename.split("/")          # train_series/<study>/<series>/<sop>.dcm
        rel = os.path.join(*parts[2:])
        out = assert_inside_dest(os.path.join(dst_study_dir, rel))
        if os.path.exists(out) and os.path.getsize(out) == info.file_size:
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with zf.open(info) as fsrc, open(out, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, COPY_BUF)
        got = os.path.getsize(out)
        if got != info.file_size:
            raise RuntimeError("short write %s: %d != %d" % (out, got, info.file_size))
        written += 1
    return written


def extract_extra_studies(already: set, max_studies: int) -> list:
    log("--- Step 4: extra studies from archive ---")
    dst_root = assert_inside_dest(os.path.join(DEST, "train_images"))
    os.makedirs(dst_root, exist_ok=True)

    budget = max_studies - len(already)
    if budget <= 0:
        log("  study cap already met (%d); nothing to extract" % len(already))
        return []

    zf = zipfile.ZipFile(ARCHIVE)                  # read-only handle
    per_study, sizes = build_archive_index(zf)
    log("  archive holds %d train studies, %s uncompressed total"
        % (len(per_study), fmt(sum(sizes.values()))))

    candidates = sorted(s for s in per_study if s not in already)
    random.Random(SAMPLE_SEED).shuffle(candidates)   # deterministic representative sample

    added = []
    t0 = time.time()
    for sid in candidates:
        if len(added) >= budget:
            log("  reached study cap: %d extra studies" % len(added))
            break
        dst_study = os.path.join(dst_root, sid)
        if os.path.isdir(dst_study):
            # resumable: only count the members not yet written
            need = 0
            for info in per_study[sid]:
                out = os.path.join(dst_study, *info.filename.split("/")[2:])
                if not (os.path.exists(out) and os.path.getsize(out) == info.file_size):
                    need += info.file_size
        else:
            need = sizes[sid]
        fb = free_bytes()
        if fb - need < FREE_FLOOR + SAFETY_MARGIN:
            log("  STOP at free-space floor: free %s, next study needs %s, floor %s"
                % (fmt(fb), fmt(need), fmt(FREE_FLOOR)))
            break
        extract_study(zf, per_study[sid], os.path.join(dst_root, sid))
        added.append(sid)
        if len(added) % 25 == 0:
            log("  +%d studies (free %s, %.1f min elapsed)"
                % (len(added), fmt(free_bytes()), (time.time() - t0) / 60))
    zf.close()
    log("  extracted %d extra studies in %.1f min" % (len(added), (time.time() - t0) / 60))
    return added


# --------------------------------------------------------------------------
# Compat symlinks
# --------------------------------------------------------------------------

def make_compat_symlinks() -> None:
    """train_series -> train_images, test_series -> test_images.

    Only created when the name is free; an existing entry is left completely alone.
    """
    for link_name, target in (("train_series", "train_images"),
                              ("test_series", "test_images")):
        link = assert_inside_dest(os.path.join(DEST, link_name))
        if os.path.lexists(link):
            log("  symlink %s: already exists, left untouched" % link_name)
            continue
        if not os.path.isdir(os.path.join(DEST, target)):
            continue
        os.symlink(target, link)
        log("  symlink %s -> %s" % (link_name, target))


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def verify() -> int:
    import pandas as pd

    log("--- Verification ---")
    problems = 0

    for name in CSVS:
        dst = os.path.join(DEST, name)
        if not os.path.exists(dst):
            log("  MISSING csv: %s" % name)
            problems += 1
        else:
            src_size = os.path.getsize(os.path.join(CACHE, name))
            dst_size = os.path.getsize(dst)
            status = "ok" if src_size == dst_size else "SIZE MISMATCH"
            if src_size != dst_size:
                problems += 1
            log("  csv %-24s %10d B  %s" % (name, dst_size, status))

    train = pd.read_csv(os.path.join(DEST, "train.csv"))
    gold_ids = set(train.loc[train[KNEE_TARGETS].notna().all(axis=1), "StudyInstanceUID"])
    log("  gold (fully-labelled) studies in train.csv: %d" % len(gold_ids))

    tr_root = os.path.join(DEST, "train_images")
    te_root = os.path.join(DEST, "test_images")
    present_train = set(os.listdir(tr_root)) if os.path.isdir(tr_root) else set()
    present_train = {d for d in present_train if not d.startswith(".")}
    present_test = set(os.listdir(te_root)) if os.path.isdir(te_root) else set()
    present_test = {d for d in present_test if not d.startswith(".")}

    missing_gold = gold_ids - present_train
    log("  train studies on disk: %d  (gold covered: %d/%d)"
        % (len(present_train), len(gold_ids & present_train), len(gold_ids)))
    if missing_gold:
        log("  MISSING GOLD IMAGE DIRS: %d" % len(missing_gold))
        problems += 1

    test_ids = set(pd.read_csv(os.path.join(DEST, "test.csv"))["StudyInstanceUID"])
    log("  test studies on disk: %d/%d" % (len(test_ids & present_test), len(test_ids)))
    if test_ids - present_test:
        problems += 1

    tr_bytes, tr_files = dir_bytes(tr_root) if os.path.isdir(tr_root) else (0, 0)
    te_bytes, te_files = dir_bytes(te_root) if os.path.isdir(te_root) else (0, 0)
    log("  train_images: %s in %d files" % (fmt(tr_bytes), tr_files))
    log("  test_images : %s in %d files" % (fmt(te_bytes), te_files))

    # empty-study check
    empties = [s for s in sorted(present_train)
               if dir_bytes(os.path.join(tr_root, s))[1] == 0]
    if empties:
        log("  EMPTY STUDY DIRS: %d (e.g. %s)" % (len(empties), empties[0]))
        problems += 1

    # pydicom smoke test on one gold slice
    try:
        import pydicom
        sample = None
        for sid in sorted(gold_ids & present_train):
            for root, _d, files in os.walk(os.path.join(tr_root, sid)):
                dcms = [f for f in files if f.endswith(".dcm")]
                if dcms:
                    sample = os.path.join(root, dcms[0])
                    break
            if sample:
                break
        if sample is None:
            log("  pydicom test: NO .dcm FOUND")
            problems += 1
        else:
            ds = pydicom.dcmread(sample)
            arr = ds.pixel_array
            log("  pydicom ok: %s" % os.path.basename(sample))
            log("    shape=%s dtype=%s min=%s max=%s Modality=%s"
                % (arr.shape, arr.dtype, arr.min(), arr.max(),
                   getattr(ds, "Modality", "?")))
    except Exception as exc:                                  # noqa: BLE001
        log("  pydicom test FAILED: %r" % exc)
        problems += 1

    log("  free space now: %s (floor %s)" % (fmt(free_bytes()), fmt(FREE_FLOOR)))
    log("  VERIFY: %s (%d problem(s))" % ("PASS" if problems == 0 else "FAIL", problems))
    return problems


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-studies", type=int, default=MAX_STUDIES,
                    help="cap on total train studies (gold + extra)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-archive", action="store_true",
                    help="only copy CSVs + already-extracted cache studies")
    args = ap.parse_args()

    log("project : %s" % PROJECT_ROOT)
    log("dest    : %s" % DEST)
    log("cache   : %s (READ-ONLY)" % CACHE)
    log("free    : %s   floor: %s   margin: %s"
        % (fmt(free_bytes()), fmt(FREE_FLOOR), fmt(SAFETY_MARGIN)))

    if args.verify_only:
        return 1 if verify() else 0

    if free_bytes() < FREE_FLOOR + SAFETY_MARGIN:
        log("ABORT: free space is already below the floor.")
        return 2

    copy_csvs()

    log("--- Step 2: test studies from cache -> test_images ---")
    copy_cached_split("test_series", "test_images")

    log("--- Step 3: gold studies from cache -> train_images ---")
    gold_copied = copy_cached_split("train_series", "train_images")
    log("  copied %d gold studies" % len(gold_copied))

    if not args.skip_archive:
        # "already" = everything already on disk, so re-runs resume rather than
        # re-count studies extracted by an earlier invocation.
        tr_root = os.path.join(DEST, "train_images")
        on_disk = {d for d in os.listdir(tr_root) if not d.startswith(".")}
        extract_extra_studies(on_disk | set(gold_copied), args.max_studies)

    log("--- Compat symlinks ---")
    make_compat_symlinks()

    import pandas as pd
    train = pd.read_csv(os.path.join(DEST, "train.csv"))
    gold_ids = set(train.loc[train[KNEE_TARGETS].notna().all(axis=1), "StudyInstanceUID"])
    tr_root = os.path.join(DEST, "train_images")
    te_root = os.path.join(DEST, "test_images")
    present_train = {d for d in os.listdir(tr_root) if not d.startswith(".")}
    present_test = {d for d in os.listdir(te_root) if not d.startswith(".")}
    write_derived_csvs(gold_ids, present_train, present_test)

    return 1 if verify() else 0


if __name__ == "__main__":
    sys.exit(main())
