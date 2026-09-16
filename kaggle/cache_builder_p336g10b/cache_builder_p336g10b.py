"""Kaggle CPU kernel: HALF B of the P=336 COVERAGE cache (336 px, 10 single-slice anchors/slot).

Resolution IN ADDITION to coverage, the one combination never tested: the champion layout is
g10t1 @224 (60 imgs/study, +0.0325 pooled OOF over the 18-img layout — the biggest win of the
campaign) and the only 336 px cache so far is `slots_P336_g4t1`, which bought resolution by
GIVING UP coverage (24 imgs/study, equal compute) and measured -0.0070.  This cache keeps
G=10 and raises P to 336: 60 imgs/study at 24x24 patches = 2.25x the tokens of the champion.

    full cache = 4,407 x 6 x 10 x 1 x 336 x 336 = 29,851,960,320 B = 29.85 GB (27.80 GiB)

which is OVER Kaggle's 20 GB working-directory cap, so it is built as two halves over the SAME
sorted uid list (part A: first 2204 studies = 14.93 GB; part B: --skip 2204 = 14.92 GB), exactly
like cache_builder_g20a/g20b.  The consumer (kaggle/ablate_kernel_g10) attaches BOTH halves as
kernel_sources and merges them with scripts/merge_slot_cache.py; nothing has to be downloaded to
the laptop, whose 18 GiB of free disk could never hold the merged 27.80 GiB anyway.

--shard-size 256 (not the template's 512): a study row is 6.77 MB at this layout, so 512 rows
would be a 3.47 GB shard file — larger than anything this pipeline has moved.  256 keeps each
shard at 1.73 GB, in the proven range of the existing caches (1.4-1.5 GB), and makes a resume
after a dropped session cheaper.  Layout parameters are otherwise identical to g10t1.

Output: /kaggle/working/slots_P336_g10t1_partB/ (~14.92 GB).  No test split here: part A builds it.
"""
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
P = int(os.environ.get("SLOT_P", "336"))
G = int(os.environ.get("SLOT_G", "10"))
T = int(os.environ.get("SLOT_T", "1"))
CROP_MM = os.environ.get("SLOT_CROP_MM", "140")
TRIM = os.environ.get("SLOT_TRIM", "0.15")
ZOOM_MM = os.environ.get("SLOT_ZOOM_MM", "")          # no zoom slots in this cache
ZOOM_SLOTS = os.environ.get("SLOT_ZOOM_SLOTS", "SAG_FS,COR_FS")
ZOOM_CENTER = os.environ.get("SLOT_ZOOM_CENTER", "image")
SKIP = os.environ.get("SLOT_SKIP", "2204")
LIMIT_N = os.environ.get("SLOT_LIMIT_N", "")   # "" = to the end
PART = os.environ.get("SLOT_PART", "B")
SHARD = os.environ.get("SLOT_SHARD_SIZE", "256")   # 6.77 MB/study -> 1.73 GB per shard file
LIMIT = os.environ.get("SLOT_LIMIT", "")            # "" = all studies
LIMIT_MIN = os.environ.get("SLOT_LIMIT_MINUTES", "500")   # leave margin under the session cap
OUT = f"/kaggle/working/slots_P{P}" + ("" if TRIM == "0.15" else f"_t{int(float(TRIM)*100)}") + ("" if (G, T) == (3, 3) else f"_g{G}t{T}") + ("" if not ZOOM_MM else f"_z{ZOOM_MM}") + ("j" if ZOOM_CENTER == "joint" else "") + (f"_part{PART}" if PART else "")

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
              "--crop-mm", str(CROP_MM), "--trim-frac", str(TRIM), "--shard-size", str(SHARD)]
    if ZOOM_MM and ZOOM_SLOTS:
        common += ["--zoom-mm", ZOOM_MM, "--zoom-slots", ZOOM_SLOTS, "--zoom-center", ZOOM_CENTER]
    if LIMIT:
        common += ["--limit", LIMIT]
    # --skip/--limit select this kernel's HALF of the TRAIN studies only; the 3-study test
    # split is built whole, and only by part A (part B would skip past it entirely).
    train_only = []
    if LIMIT_N:
        train_only += ["--limit", LIMIT_N]
    if SKIP and SKIP != "0":
        train_only += ["--skip", SKIP]

    # Test path first: 3 example studies, seconds, proves the inference-time builder works
    # on the real test layout (test_series/ + test_series.csv).
    if not (SKIP and SKIP != "0"):
        run([sys.executable, "scripts/build_slot_cache.py", "--split", "test"] + common)

    # Train: the long one.  --limit-minutes makes it stop cleanly and resumably.
    run([sys.executable, "scripts/build_slot_cache.py", "--split", "train",
         "--limit-minutes", str(LIMIT_MIN)] + common + train_only)

    report = {"elapsed_min": round((time.time() - T0) / 60, 1), "P": P, "G": G, "T": T,
              "crop_mm": CROP_MM, "skip": SKIP, "limit": LIMIT_N, "part": PART,
              "shard_size": SHARD, "out": OUT}
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
