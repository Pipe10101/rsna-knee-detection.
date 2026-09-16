"""Kaggle GPU kernel: run several SlotKnee-S ablation ARMS (2 folds each) in one T4 session.

Attached sources (kernel-metadata.json):
  competition  rsna-knee-abnormality-detection          (labels CSV only; pixels come from the cache)
  dataset      felipedeleon11/slotknee-code             (src/, scripts/, labels_external/, weights/, wheels/)
  kernel       felipedeleon11/slotknee-cache-builder    (slots_P224/ = the verified v2 cache, 11.9 GB)

Robust to Kaggle's nested mount layout: every input is LOCATED by walking /kaggle/input,
never by a hard-coded path.  Runs offline: DINOv2 weights come from the code dataset.
Outputs: /kaggle/working/models/slotknee_full5/fold_*_best.pt, oof_fold_*.npz, logs.
"""
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
EPOCHS = int(os.environ.get("SK_EPOCHS", "8"))
FOLDS = os.environ.get("SK_FOLDS", "0 1 2 3 4").split()
LIMIT_MIN = os.environ.get("SK_LIMIT_MINUTES", "470")   # GPU sessions are capped at 9 h
BS = os.environ.get("SK_BS", "8")
OUT_ROOT = "/kaggle/working/models/ablate"
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



# One T4 run ≈ 9 h.  A 224-px, 2-fold, 8-epoch arm takes ≈ 2 h on a T4, so four arms fit.
# Edit this list and `kaggle_ops.py push-code && push-ablate` to queue a different set.
ARMS = [
    ("baseline", [], "slots_P224"),                 # T4 baseline on the laptop-matching folds
    ("tau3",    ["--attn-tau-init", "3.0"], "slots_P224"),
    ("mil",     ["--mil-pool", "lse"], "slots_P224"),
    ("nomixer", ["--mixer-layers", "0"], "slots_P224"),
]
FOLDS = os.environ.get("SK_FOLDS", "0 1").split()
TOTAL_MIN = int(os.environ.get("SK_TOTAL_MINUTES", "500"))


def main():
    print("input mounts:", os.listdir("/kaggle/input"), flush=True)
    check_gpu()
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
    weights = os.path.join(work, "weights", "vit_small_patch14_dinov2.lvd142m.safetensors")
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
        if not os.path.isfile(os.path.join(local_cache, "train_index.json")):
            t = time.time()
            shutil.copytree(caches[cache_name], local_cache, dirs_exist_ok=True)
            print(f"copied {cache_name} in {time.time()-t:.0f}s", flush=True)
        out = os.path.join(OUT_ROOT, f"{name}_s42")
        cmd = [sys.executable, "scripts/train_slotknee.py", "--cache", local_cache, "--data-dir", data_dir,
               "--labels", labels, "--weights-from", wfrom, "--pretrained-path", weights,
               "--folds", *FOLDS, "--epochs", "8", "--bs", "8", "--workers", "2", "--out", out,
               "--seed", "42", "--limit-minutes", str(int(left - 15))] + extra
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
        # free the per-arm cache copy when the next arm uses a different one
    with open("/kaggle/working/ablate_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print({k: (v[-2] if len(v) > 1 else v) for k, v in report.items()}, flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
