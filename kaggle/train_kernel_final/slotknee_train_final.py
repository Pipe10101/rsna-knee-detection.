"""Kaggle GPU kernel: FINAL SlotKnee-S training — 5 folds x 10 epochs of the best recipe from
`slotknee-ablate-g10`, one T4 session (--limit-minutes 470 keeps it inside the 9 h cap).

Attached sources (kernel-metadata.json):
  competition  rsna-knee-abnormality-detection          (labels CSV only; pixels come from the caches)
  dataset      felipedeleon11/slotknee-code             (src/, scripts/, labels_external/, weights/, wheels/)
  kernel       felipedeleon11/slotknee-cache-g10t1      (slots_P224_g10t1)
  kernel       felipedeleon11/slotknee-cache-p336g4     (slots_P336_g4t1)
  kernel       felipedeleon11/slotknee-cache-zoomj      (slots_P224_g10t1_z100j)

RECIPE below selects the cache by name and the extra train flags; SEED lets a second version
train a second seed for the ensemble.  The data_dir exposes the 649 laptop-resident studies so
src/folds.build_groups reproduces the laptop folds (OOFs comparable with every other run).
Outputs: /kaggle/working/models/slotknee_final/<recipe>_s<seed>/fold_*_best.pt, oof_fold_*.npz, logs.
"""
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
FOLDS = os.environ.get("SK_FOLDS", "0 1 2 3 4").split()
LIMIT_MIN = os.environ.get("SK_LIMIT_MINUTES", "470")   # GPU sessions are capped at 9 h
BS = os.environ.get("SK_BS", "8")
OUT_ROOT = "/kaggle/working/models/slotknee_final"
TMP = "/kaggle/temp"


def walk_find(pred, prune=("train_series", "test_series", "train_images", "test_images")):
    for base, dirs, files in os.walk("/kaggle/input"):
        hit = pred(base, list(dirs), files)      # test BEFORE pruning: the predicate may look for train_series
        if hit:
            return hit
        dirs[:] = [d for d in dirs if d not in prune]   # never descend into the 800k-file image trees
    return None


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def check_gpu():
    """Kaggle's PyTorch build dropped sm_60: a P100 cannot run any CUDA kernel.

    The API cannot choose the accelerator; it is a kernel setting.  Fail fast with
    the fix instead of dying 10 minutes later inside the first forward pass.
    """
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Kernel settings -> Accelerator -> 'GPU T4 x2', then Save & Run.")
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} sm_{major}{minor}", flush=True)
    if major < 7:
        raise SystemExit(f"{name} (sm_{major}{minor}) is not supported by this PyTorch build. "
                         "Open the kernel on Kaggle -> Settings -> Accelerator -> 'GPU T4 x2' "
                         "(the choice persists across API pushes), then Save & Run.")



# ---- THE RECIPE (fill from slotknee-ablate-g10's verdict, then `kaggle_ops.py push-train-final`) ----
# name, extra train flags, cache dir name.  Coverage layout + the reg4 backbone.
# reg4 ADOPTED 2026-08-24: +0.0187 pooled vs the same-split baseline (folds 0+1, noise 0.0038),
# ACL +0.0533 — registers absorb the high-norm artifact tokens that pollute the patch tokens our
# per-finding head attends over.  Weights ship in the code dataset; no warm-start here, so the
# architecture change is safe (a reg4 model cannot load non-reg4 checkpoints).
# Self-distillation ADOPTED 2026-08-24 (+0.0185 pooled on the base layout, every label up,
# 0 s inference cost). Expect LESS here: it correlates 0.829 with the coverage layout this
# cache already provides (both are variance reduction), so ~+0.006-0.009 is the honest
# expectation. The teacher path is resolved at runtime by the kernel (labels_external/).
# NOTE: a distilled run's OOF is INFLATED (cross-fold teacher information) and is not a gate
# metric — but the hidden test set is in no fold, so distillation is sound for a SUBMISSION.
# Random slice bagging ADOPTED 2026-08-25: +0.0090 vs the champion on the SAME layout/folds,
# and it lands on the weak findings (ACL +0.032, MCL +0.019, LatMen +0.012).
# CENTRAL-BLOCK ANCHORS ADOPTED 2026-09-15: the same 140 mm crop with the 10 anchors over the middle
# 30 % of the stack (cache slots_P224_t35_g10t1, trim 0.35) measured +0.0038 / +0.0027 at 8 epochs on
# both controls and +0.0025 at 16 epochs vs recipe_e16 (ACL and lateral meniscus up every time).  The
# checkpoints record trim_frac 0.35 in slot_layout (code dataset 2026-09-09+) so inference rebuilds the
# same slices; the previous g10t1 folds must not be mixed into the same submit group.
RECIPE = ("t35_reg4_distil_bag6",
          ["--backbone", "vit_small_patch14_reg4_dinov2.lvd142m",
           "--distill-targets", "labels_external/teacher_g10_oof.csv",
           "--distill-weight", "0.5",
           "--anchor-bag", "6"],
          "slots_P224_t35_g10t1")
# NOTE (2026-08-24 11:2x): --cutout/--rot-aug were inserted here at 10:56 without a
# measurement — REVERTED. cutrot is an
# UNGATED ablation arm; it enters the final recipe only after passing the gate
# (docs/slotknee_runbook.md, Δ>0.012 vs the coverage baseline). Do not add ungated flags here.
# Only MEASURED, gate-passing changes belong in RECIPE — reg4 above is one; cutrot is not (yet).
SEED = os.environ.get("SK_SEED", "42")
EPOCHS = os.environ.get("SK_EPOCHS", "16")   # ADOPTED 2026-09-06: +0.0160 pooled vs 8 epochs (gate 0.0011)
ARMS = [(f"{RECIPE[0]}_e{EPOCHS}", RECIPE[1], RECIPE[2])]
FOLDS = os.environ.get("SK_FOLDS", "0 1 2 3 4").split()
TOTAL_MIN = int(os.environ.get("SK_TOTAL_MINUTES", "500"))


def main():
    print("input mounts:", os.listdir("/kaggle/input"), flush=True)
    code = walk_find(lambda b, d, f: b if ("build_slot_cache.py" in f and os.path.basename(b) == "scripts") else None)
    code = os.path.dirname(code) if code else None
    comp = walk_find(lambda b, d, f: b if ("train.csv" in f and "train_series" in d) else None)
    caches = {}
    for base, dirs, files in os.walk("/kaggle/input"):
        if "train_index.json" in files and os.path.basename(base).startswith("slots_"):
            caches[os.path.basename(base)] = base
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series", "train_images", "test_images")]
    print("code:", code, "\ncompetition:", comp, "\ncaches:", caches, flush=True)
    if not (code and comp and caches):
        raise SystemExit("missing input: code/competition/caches not all found under /kaggle/input")

    # GPU probe deliberately runs AFTER input discovery.  A CPU push (the only kind the
    # Kaggle API can make -- it always lands on a banned P100) then still validates every
    # mount, cache name and label path before aborting, so the cheap run is a real
    # pre-flight instead of a wasted "No GPU" line that proves nothing.
    check_gpu()

    work = "/kaggle/working/code"
    if os.path.isdir(work):
        shutil.rmtree(work)
    shutil.copytree(code, work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.chdir(work)

    data_dir = os.path.join(TMP, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name in ("train.csv", "train_series.csv", "sample_submission.csv", "test.csv"):
        src = os.path.join(comp, name)
        if os.path.isfile(src):
            shutil.copy(src, data_dir)
    shutil.copy(os.path.join(work, "labels_external", "train_gold.csv"), data_dir)
    # Expose the competition DICOMs as <data_dir>/train_images (symlink, no copy) so
    # src/folds.build_groups can read one header per study and reproduce the SAME
    # scanner-grouped folds as local runs (without it Kaggle fell back to report-hash groups).
    # Only the studies the laptop has on disk get a fingerprint there (649), so link exactly
    # those uids (list shipped in the code dataset) — a whole-tree link would fingerprint all
    # 4,407 and produce a third, different split.
    link = os.path.join(data_dir, "train_images")
    uid_list = os.path.join(work, "labels_external", "fold_image_uids.txt")
    if not os.path.isdir(link) and os.path.isdir(os.path.join(comp, "train_series")):
        os.makedirs(link, exist_ok=True)
        if os.path.isfile(uid_list):
            n = 0
            for uid in open(uid_list).read().split():
                srcd = os.path.join(comp, "train_series", uid)
                if os.path.isdir(srcd):
                    os.symlink(srcd, os.path.join(link, uid)); n += 1
            print(f"fold groups: linked {n} laptop-resident studies for scanner fingerprints", flush=True)
        else:
            print("WARN: fold_image_uids.txt missing -> whole-tree link (folds will NOT match the laptop split)", flush=True)
            os.rmdir(link); os.symlink(os.path.join(comp, "train_series"), link)
    # The encoder weights MUST match the recipe's --backbone.  Until 2026-09-07 this line
    # hardcoded the plain-DINOv2 file while RECIPE passes the reg4 backbone; the register
    # variant has a different pos_embed shape, so timm's resample_abs_pos_embed crashed with
    # "shape [1, 37, 37, -1] is invalid" the first time this kernel ever ran on a GPU.
    _bb = RECIPE[1][RECIPE[1].index("--backbone") + 1] if "--backbone" in RECIPE[1] else "vit_small_patch14_dinov2.lvd142m"
    weights = os.path.join(work, "weights", f"{_bb}.safetensors")
    labels = os.path.join(work, "labels_external", "stevenleehans", "llm_labels_v4_blend.csv")
    wfrom = os.path.join(work, "labels_external", "stevenleehans", "llm_labels_v2.csv")
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")

    report = {}
    for name, extra, cache_name in ARMS:
        left = TOTAL_MIN - (time.time() - T0) / 60
        if left < 60:
            print(f"[ablate] {left:.0f} min left: stopping before {name}", flush=True)
            break
        if cache_name not in caches:
            print(f"[ablate] skip {name}: cache {cache_name} not attached", flush=True)
            continue
        local_cache = os.path.join(TMP, cache_name)
        for other in os.listdir(TMP) if os.path.isdir(TMP) else []:   # local disk is finite: keep one cache copy
            if other.startswith("slots_") and other != cache_name:
                shutil.rmtree(os.path.join(TMP, other), ignore_errors=True)
                print(f"evicted local copy of {other}", flush=True)
        if not os.path.isfile(os.path.join(local_cache, "train_index.json")):
            t = time.time()
            shutil.copytree(caches[cache_name], local_cache, dirs_exist_ok=True)
            print(f"copied {cache_name} in {time.time()-t:.0f}s", flush=True)
        out = os.path.join(OUT_ROOT, f"{name}_s{SEED}")
        cmd = [sys.executable, "scripts/train_slotknee.py", "--cache", local_cache, "--data-dir", data_dir,
               "--labels", labels, "--weights-from", wfrom, "--pretrained-path", weights,
               "--folds", *FOLDS, "--epochs", EPOCHS, "--bs", "8", "--workers", "2", "--out", out,
               "--seed", SEED, "--limit-minutes", str(min(470, int(left - 15)))] + extra
        print(f"[ablate] == {name} ({left:.0f} min left)", flush=True)
        try:
            run(cmd, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[ablate] {name} FAILED: {e}", flush=True)
        for k in FOLDS:
            lf = os.path.join(out, f"fold_{k}_log.json")
            if os.path.isfile(lf):
                with open(lf) as fh:
                    report[f"{name}/fold{k}"] = json.load(fh)
    with open("/kaggle/working/train_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print({k: (v[-2] if len(v) > 1 else v) for k, v in report.items()}, flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
