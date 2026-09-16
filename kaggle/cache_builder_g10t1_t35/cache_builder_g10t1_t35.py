"""Kaggle CPU kernel: the g10t1 coverage cache with the 10 anchors CONCENTRATED in the central 30 % of each
stack (SLOT_TRIM 0.35) instead of spread over the central 70 % (0.15).

Evidence (this competition's forum, 2026-09): 9 ADJACENT central slices beat 9 spread over 24 by +0.018;
crop geometry moved 10/12 labels while encoder scaling moved nothing.  Same crop (140 mm), size (224), G=10,
T=1, laterality normalisation and percentile scaling as g10t1, so the ablate arm `central` differs from
the control in anchor placement only.  Output: /kaggle/working/slots_P224_t35_g10t1 (~12 GB, ~45 min).
"""
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
P = int(os.environ.get("SLOT_P", "224"))
G = int(os.environ.get("SLOT_G", "10"))
T = int(os.environ.get("SLOT_T", "1"))
CROP_MM = os.environ.get("SLOT_CROP_MM", "140")
TRIM = os.environ.get("SLOT_TRIM", "0.35")   # CENTRAL BLOCK: 10 anchors over the middle 30% of the stack (near-adjacent), not spread over 70%
LIMIT = os.environ.get("SLOT_LIMIT", "")            # "" = all studies
LIMIT_MIN = os.environ.get("SLOT_LIMIT_MINUTES", "500")   # leave margin under the 9 h cap
OUT = f"/kaggle/working/slots_P{P}" + ("" if TRIM == "0.15" else f"_t{int(float(TRIM)*100)}") + ("" if (G, T) == (3, 3) else f"_g{G}t{T}")

CODE_CANDIDATES = [
    "/kaggle/input/nidhogg/pytorch/default",
    "/kaggle/input/slotknee-code",
    "/kaggle/input/datasets/felipedeleon11/slotknee-code",
]


def find_code_dir():
    for c in CODE_CANDIDATES:
        if os.path.isdir(os.path.join(c, "src")) and os.path.isdir(os.path.join(c, "scripts")):
            return c
    base = "/kaggle/input"
    for root, dirs, files in os.walk(base):
        # never descend into the competition image tree
        for d in ("train_series", "test_series", "train_images", "test_images"):
            if d in dirs:
                dirs.remove(d)
        if "build_slot_cache.py" in files and os.path.basename(root) == "scripts":
            return os.path.dirname(root)
    raise SystemExit("code dataset not found under /kaggle/input")


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main():
    code = find_code_dir()
    work_code = "/kaggle/working/code"
    if os.path.isdir(work_code):
        shutil.rmtree(work_code)
    shutil.copytree(code, work_code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.chdir(work_code)
    sys.path.insert(0, work_code)

    print("input mounts:", os.listdir("/kaggle/input"), flush=True)

    # Newer Kaggle images nest sources (/kaggle/input/competitions/<slug>,
    # /kaggle/input/datasets/<user>/<slug>); the bootstrap scans one level, so
    # locate the competition root here and hand it over explicitly.
    comp_root = None
    for base, dirs, files in os.walk("/kaggle/input"):
        for d in ("train_series", "test_series", "train_images", "test_images"):
            if d in dirs:
                dirs.remove(d)
        if "train.csv" in files and os.path.isdir(os.path.join(base, "train_series")):
            comp_root = base
            break
    print("competition root:", comp_root, flush=True)
    boot_cmd = [sys.executable, "scripts/kaggle_bootstrap.py"]
    if comp_root:
        boot_cmd += ["--competition-dir", comp_root]

    # Writable data_dir whose image dirs are symlinks into the read-only mount.
    boot = subprocess.run(boot_cmd, capture_output=True, text=True)
    print(boot.stdout, flush=True)
    if boot.returncode != 0:
        print("kaggle_bootstrap.py FAILED, stderr:\n" + boot.stderr, flush=True)
        raise SystemExit(1)
    data_dir = None
    for line in boot.stdout.splitlines():
        if "data_dir=" in line:
            data_dir = line.split("data_dir=", 1)[1].strip().split()[0]
    if not data_dir or not os.path.isdir(data_dir):
        raise SystemExit("kaggle_bootstrap.py did not report a data_dir")

    common = ["--data-dir", data_dir, "--out", OUT, "--P", str(P), "--G", str(G), "--T", str(T),
              "--crop-mm", str(CROP_MM), "--trim-frac", str(TRIM), "--shard-size", "512"]
    if LIMIT:
        common += ["--limit", LIMIT]

    # Test path first: 3 example studies, seconds, proves the inference-time builder works
    # on the real test layout (test_series/ + test_series.csv).
    run([sys.executable, "scripts/build_slot_cache.py", "--split", "test"] + common)

    # Train: the long one.  --limit-minutes makes it stop cleanly and resumably.
    run([sys.executable, "scripts/build_slot_cache.py", "--split", "train",
         "--limit-minutes", str(LIMIT_MIN)] + common)

    report = {"elapsed_min": round((time.time() - T0) / 60, 1), "P": P, "G": G, "T": T,
              "crop_mm": CROP_MM, "out": OUT}
    for split in ("train", "test"):
        idx = os.path.join(OUT, f"{split}_index.json")
        if os.path.isfile(idx):
            with open(idx) as fh:
                j = json.load(fh)
            report[split] = {"n_studies": len(j.get("studies", [])), "stats": j.get("stats", {})}
    total = 0
    for root, _, files in os.walk(OUT):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    report["output_gb"] = round(total / 2**30, 2)
    with open("/kaggle/working/cache_builder_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    shutil.rmtree(work_code, ignore_errors=True)   # keep the output small


if __name__ == "__main__":
    main()
