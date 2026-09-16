"""Prepare a Kaggle notebook to run this pipeline against the mounted competition data.

Why this exists
---------------
/kaggle/input is READ-ONLY and its layout does not match what the pipeline
expects:

    Kaggle mounts        /kaggle/input/rsna-knee-abnormality-detection/
                           train.csv  train_series.csv  test.csv ...
                           train_series/<study>/<series>/*.dcm
                           test_series/<study>/<series>/*.dcm

    the pipeline wants   <data_dir>/train.csv ...
                         <data_dir>/train_images/<study>/<series>/*.dcm
                         <data_dir>/test_images/<study>/<series>/*.dcm

So this builds a writable directory under /kaggle/working whose image
directories are SYMLINKS back into the read-only mount (no copying: the
competition data is ~531 GB) and whose small CSVs are real files, so
derived label files can sit alongside them.

Usage inside a notebook cell:

    !python3 scripts/kaggle_bootstrap.py
    # -> prints the data_dir to pass as --set data_dir=...
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

COMP = "rsna-knee-abnormality-detection"

# (source name in the mount, name the pipeline expects)
IMAGE_DIRS = [("train_series", "train_images"), ("test_series", "test_images")]
CSVS = ["train.csv", "train_series.csv", "test.csv", "test_series.csv",
        "sample_submission.csv"]


def find_competition_root(explicit: str | None = None) -> str:
    """Locate the mounted competition directory."""
    if explicit:
        if not os.path.isdir(explicit):
            raise SystemExit(f"ERROR: --competition-dir {explicit} does not exist")
        return explicit
    direct = f"/kaggle/input/{COMP}"
    if os.path.isdir(direct):
        return direct
    # Fall back to scanning: a Kaggle competition is sometimes attached under a
    # different directory name than its slug.
    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            cand = os.path.join(base, name)
            if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "train.csv")) \
               and os.path.isdir(os.path.join(cand, "train_series")):
                return cand
    raise SystemExit(
        "ERROR: could not find the competition data under /kaggle/input.\n"
        "  Attach the competition to this notebook (Add data -> Competitions),\n"
        "  or pass --competition-dir explicitly."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition-dir", default=None)
    ap.add_argument("--out", default="/kaggle/working/data",
                    help="writable data_dir to build")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra CSVs to copy in (e.g. train_gold.csv train_derived.csv)")
    args = ap.parse_args()

    root = find_competition_root(args.competition_dir)
    out = args.out
    os.makedirs(out, exist_ok=True)
    print(f"competition data : {root}")
    print(f"data_dir         : {out}")

    # Images: symlink, never copy.
    for src_name, dst_name in IMAGE_DIRS:
        src = os.path.join(root, src_name)
        dst = os.path.join(out, dst_name)
        if not os.path.isdir(src):
            print(f"  WARN  {src_name} missing in mount; skipped")
            continue
        if os.path.islink(dst) or os.path.exists(dst):
            # Never remove a real directory -- only refresh our own symlink.
            if os.path.islink(dst):
                os.unlink(dst)
            else:
                print(f"  SKIP  {dst} exists and is not a symlink; leaving it alone")
                continue
        os.symlink(src, dst)
        n = len(os.listdir(dst))
        print(f"  link  {dst_name} -> {src_name}  ({n:,} studies)")
        # Both names are used across the codebase; provide the alias too.
        alias = os.path.join(out, src_name)
        if not os.path.exists(alias) and not os.path.islink(alias):
            os.symlink(src, alias)

    # CSVs: real copies, so derived files can live beside them.
    for name in CSVS:
        src = os.path.join(root, name)
        dst = os.path.join(out, name)
        if os.path.isfile(src):
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
            print(f"  copy  {name}  ({os.path.getsize(dst):,} B)")
        else:
            print(f"  WARN  {name} missing in mount")

    # Extra files shipped inside the code dataset (gold folds, derived labels).
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in args.extra:
        for cand in (name, os.path.join(here, name), os.path.join(here, "data", name)):
            if os.path.isfile(cand):
                dst = os.path.join(out, os.path.basename(name))
                if os.path.abspath(cand) != os.path.abspath(dst):
                    shutil.copy2(cand, dst)
                print(f"  copy  {os.path.basename(name)}  ({os.path.getsize(dst):,} B)")
                break
        else:
            print(f"  WARN  extra file not found: {name}")

    print()
    print("Run training with:")
    print(f"  python3 -m src.train --set data_dir={out}")


if __name__ == "__main__":
    main()
