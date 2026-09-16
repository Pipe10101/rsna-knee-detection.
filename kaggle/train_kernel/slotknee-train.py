# %% [code]
"""Kaggle GPU kernel: train SlotKnee-S (5 folds) on the full 4,407-study slot cache.

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
OUT = "/kaggle/working/models/slotknee_full5"
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


def main():
    print("input mounts:", os.listdir("/kaggle/input"), flush=True)
    check_gpu()
    code = walk_find(lambda b, d, f: b if ("build_slot_cache.py" in f and os.path.basename(b) == "scripts") else None)
    code = os.path.dirname(code) if code else None
    comp = walk_find(lambda b, d, f: b if ("train.csv" in f and "train_series" in d) else None)
    cache = walk_find(lambda b, d, f: b if "train_index.json" in f else None)
    print("code:", code, "\ncompetition:", comp, "\ncache:", cache, flush=True)
    if not (code and comp and cache):
        raise SystemExit("missing input: code/competition/cache not all found under /kaggle/input")

    work = "/kaggle/working/code"
    if os.path.isdir(work):
        shutil.rmtree(work)
    shutil.copytree(code, work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.chdir(work)

    # The cache is a memmap read thousands of times per epoch: local disk beats the
    # network mount.  /kaggle/temp is not preserved, so it does not count against the
    # 20 GB output cap.
    local_cache = os.path.join(TMP, "slots_P224")
    if not os.path.isfile(os.path.join(local_cache, "train_index.json")):
        t = time.time()
        shutil.copytree(cache, local_cache, dirs_exist_ok=True)
        print(f"copied cache to {local_cache} in {time.time()-t:.0f}s", flush=True)

    # data_dir must hold train_gold.csv + train.csv (labels/folds) — build a tiny one.
    data_dir = os.path.join(TMP, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name in ("train.csv", "train_series.csv", "sample_submission.csv", "test.csv"):
        src = os.path.join(comp, name)
        if os.path.isfile(src):
            shutil.copy(src, data_dir)
    gold_src = os.path.join(work, "labels_external", "train_gold.csv")
    if os.path.isfile(gold_src):
        shutil.copy(gold_src, data_dir)
    else:
        # derive the 58-row gold file from train.csv (rows with any label present)
        import pandas as pd
        df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        lab = [c for c in df.columns if c not in ("StudyInstanceUID", "Report")]
        g = df[df[lab].notna().all(axis=1)].copy()
        g["fold"] = (list(range(5)) * (len(g) // 5 + 1))[:len(g)]
        g.to_csv(os.path.join(data_dir, "train_gold.csv"), index=False)
        print(f"derived train_gold.csv with {len(g)} rows", flush=True)

    weights = os.path.join(work, "weights", "vit_small_patch14_dinov2.lvd142m.safetensors")
    labels = os.path.join(work, "labels_external", "stevenleehans", "llm_labels_v4_blend.csv")
    wfrom = os.path.join(work, "labels_external", "stevenleehans", "llm_labels_v2.csv")
    for p in (weights, labels, wfrom):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}")

    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "scripts/train_slotknee.py", "--cache", local_cache,
           "--data-dir", data_dir, "--labels", labels, "--weights-from", wfrom,
           "--pretrained-path", weights, "--folds", *FOLDS, "--epochs", str(EPOCHS),
           "--bs", BS, "--workers", "2", "--out", OUT, "--seed", "42",
           "--limit-minutes", LIMIT_MIN]
    run(cmd, env=env)

    report = {"elapsed_min": round((time.time() - T0) / 60, 1), "epochs": EPOCHS, "folds": FOLDS}
    for k in FOLDS:
        lf = os.path.join(OUT, f"fold_{k}_log.json")
        if os.path.isfile(lf):
            with open(lf) as fh:
                report[f"fold_{k}"] = json.load(fh)
    with open("/kaggle/working/train_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("fold_")}), flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
