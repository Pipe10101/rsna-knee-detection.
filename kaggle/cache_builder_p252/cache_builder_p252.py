"""Kaggle kernel: build the SlotKnee uint8 slot cache on the FULL competition data.

Runs as a CPU script kernel with the competition attached read-only under
/kaggle/input/rsna-knee-abnormality-detection and the code dataset
(felipedeleon11/slotknee-code) attached under /kaggle/input/slotknee-code.

Output (under /kaggle/working, downloadable with `kaggle kernels output`):
    slots_P224/train_x.u8 (+ shards), train_mask.u8, train_index.json
    slots_P224/test_*  (the 3 example test studies; proves the test path)
    cache_builder_report.json

Nothing here needs the network.  Total output must stay under Kaggle's 20 GB
working-directory cap: 4,407 x 6 x 3 x 3 x 224 x 224 bytes = 11.9 GB.
"""
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
P = int(os.environ.get("SLOT_P", "252"))
G = int(os.environ.get("SLOT_G", "3"))
T = int(os.environ.get("SLOT_T", "3"))
CROP_MM = os.environ.get("SLOT_CROP_MM", "140")
LIMIT = os.environ.get("SLOT_LIMIT", "")            # "" = all studies
LIMIT_MIN = os.environ.get("SLOT_LIMIT_MINUTES", "500")   # leave margin under the 9 h cap
OUT = f"/kaggle/working/slots_P{P}"

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
              "--crop-mm", str(CROP_MM), "--shard-size", "512"]
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
