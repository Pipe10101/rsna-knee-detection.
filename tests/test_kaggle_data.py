"""Tests for src/kaggle_data.py — slice geometry, selection, cache, augmentation.

Run:  python3 -m tests.test_kaggle_data        (from the repo root)
No pytest required; plain asserts so it works on the Kaggle image too.

Tests that need real DICOMs are skipped (loudly) when data_subset/ is absent,
so this file still runs inside a Kaggle notebook that only mounts the test set.
"""

import os
import sys
import glob
import json
import random
import shutil
import tempfile
import types

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
import src.kaggle_data as kd

PASSED, FAILED, SKIPPED = [], [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except _Skip as e:
        SKIPPED.append((name, str(e)))
        print(f"  SKIP  {name}: {e}")
    except AssertionError as e:
        FAILED.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAILED.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


class _Skip(Exception):
    pass


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(REPO, "data_subset", "train_series")


def real_studies(n=8):
    if not os.path.isdir(IMAGES):
        raise _Skip("data_subset/train_series not present")
    ids = sorted(os.listdir(IMAGES))[:n]
    if not ids:
        raise _Skip("no studies under data_subset/train_series")
    return ids


def cfg(**kw):
    c = Config()
    c.image_size, c.in_channels = 224, 3
    for k, v in kw.items():
        setattr(c, k, v)
    return c


LEGACY = dict(slice_order="filename", slice_selection="study_pooled",
              slice_trim_frac=0.15, slice_sample="endpoints")


# ══════════════════════════════════════════════════════════════════════════
# 1. Slices are geometrically ordered BEFORE anything is trimmed or sampled
# ══════════════════════════════════════════════════════════════════════════

def test_series_are_sorted_by_image_position_not_filename():
    """The original defect: `all_dcm_paths.sort()` sorted SOP-Instance-UID
    filenames, which carry no anatomical meaning. Assert the shipped ordering
    is monotone in the true slice position for every real series."""
    import pydicom
    ids = real_studies(6)
    checked = 0
    for sid in ids:
        for s in kd.build_series_index(os.path.join(IMAGES, sid)):
            paths = s["paths"]
            if len(paths) < 4:
                continue
            pos = []
            for p in paths:
                _, z, _ = kd._read_geometry(p)
                if z is None:
                    break
                pos.append(z)
            if len(pos) != len(paths):
                continue
            d = np.diff(pos)
            assert np.all(d > 0) or np.all(d < 0), (
                f"series {s['uid'][-8:]} is not monotone along the slice normal: "
                f"{np.round(pos, 2)}")
            checked += 1
    assert checked >= 5, f"only {checked} series had usable geometry"


def test_filename_order_is_not_geometric_order():
    """The premise of the bug, asserted rather than assumed: if filename order
    ever DID match geometry this whole fix would be unnecessary."""
    import pydicom
    ids = real_studies(6)
    disagree = total = 0
    for sid in ids:
        for uid, files in kd._list_series_dirs(os.path.join(IMAGES, sid)):
            if len(files) < 6:
                continue
            geo, _ = kd.order_series(files)
            total += 1
            if geo != sorted(files):
                disagree += 1
    assert total >= 5, "not enough series to test"
    assert disagree == total, (
        f"{total - disagree}/{total} series had filename order == geometric "
        "order; the geometric sort would be a no-op there")


def test_trim_is_applied_after_ordering_and_defaults_to_zero():
    """The 15% trim was justified as 'mostly air/skin'. Measurement says the
    dropped slices carry as much tissue as the kept ones (89.7% vs 90.6% of
    slices at >=50% of peak tissue area on Sagittal), and on Sagittal they are
    the medial and lateral compartments. Default must therefore be 0.0."""
    c = cfg()
    assert float(kd._cfg(c, "slice_trim_frac", 0.0)) == 0.0, \
        "slice_trim_frac must default to 0.0"
    # and when a trim IS requested it must cut the geometric ends, not a
    # hash-random subset: index 0 and n-1 of an ordered run.
    idx = kd._sample_positions(20, 3, 0.15, "endpoints")
    assert min(idx) >= 3 and max(idx) <= 16, idx
    idx0 = kd._sample_positions(20, 3, 0.0, "endpoints")
    assert min(idx0) == 0 and max(idx0) == 19, idx0


def test_bin_center_sampling_never_picks_the_outermost_slice():
    """The one slice measurement DOES show is air/skin is the very outermost
    (10% tissue fraction, 54 mm A-P extent vs 119-131 mm one slice in).
    bin_center spans the full range without ever landing on it."""
    for n in (9, 20, 29, 40, 64):
        for k in (1, 2, 3, 5, 8):
            idx = kd._sample_positions(n, k, 0.0, "bin_center")
            assert len(idx) == k, (n, k, idx)
            assert all(0 <= i < n for i in idx), (n, k, idx)
            assert idx == sorted(idx), (n, k, idx)
            if n > 2 * k:
                assert 0 not in idx and (n - 1) not in idx, (n, k, idx)


def test_selection_spans_the_full_medial_to_lateral_range():
    """With in_channels=3 on one sagittal series the three channels must land
    in the medial third, the middle and the lateral third — that is the
    evidence for Medial/Lateral Meniscus and Medial/Lateral OA."""
    n = 29
    idx = kd._sample_positions(n, 3, 0.0, "bin_center")
    assert idx[0] < n / 3 and idx[1] > n / 3 and idx[1] < 2 * n / 3 \
        and idx[2] > 2 * n / 3, idx


def test_plane_is_derived_from_image_orientation():
    """447/447 agreement against train_series.csv, so the fixed pipeline needs
    no CSV on the offline Kaggle test set."""
    assert kd.derive_plane([1, 0, 0, 0, 1, 0]) == "Axial"        # normal = z
    assert kd.derive_plane([1, 0, 0, 0, 0, -1]) == "Coronal"     # normal = y
    assert kd.derive_plane([0, 1, 0, 0, 0, -1]) == "Sagittal"    # normal = x
    assert kd.derive_plane(None) is None
    assert kd.derive_plane([0, 0, 0, 0, 0, 0]) is None
    csv = os.path.join(REPO, "data_subset", "train_series.csv")
    if not os.path.isfile(csv):
        return
    truth = pd.read_csv(csv).set_index("SeriesInstanceUID")["Anatomical_Plane"].to_dict()
    ids = real_studies(10)
    agree = total = 0
    for sid in ids:
        for s in kd.build_series_index(os.path.join(IMAGES, sid), order_uids=set()):
            if s["uid"] in truth:
                total += 1
                agree += (s["plane"] == truth[s["uid"]])
    assert total >= 10 and agree == total, f"plane agreement {agree}/{total}"


def test_channels_have_a_stable_anatomical_meaning():
    """Legacy pooling drew channels from >1 plane in 95.8% of studies with the
    plane->channel mapping decided by UID hash. Assert plane_balanced gives
    every study the same channel->plane mapping."""
    ids = real_studies(10)
    c = cfg()
    seen = []
    for sid in ids:
        idx = kd.build_series_index(os.path.join(IMAGES, sid), order_uids=set())
        planes = {s["uid"]: s["plane"] for s in idx}
        idx2 = kd.build_series_index(os.path.join(IMAGES, sid),
                                     order_uids=kd.plan_series_uids(idx, 3, c))
        picked = kd.select_slice_paths(idx2, 3, c)
        seen.append(tuple(planes.get(os.path.basename(os.path.dirname(p)))
                          for p in picked))
    assert len(set(seen)) == 1, f"channel->plane mapping varies per study: {set(seen)}"


def test_lazy_ordering_selects_exactly_what_eager_ordering_selects():
    """The cold-path optimisation: only the series that will be sampled are
    header-sorted. It must be a pure speedup, never a different selection."""
    ids = real_studies(8)
    c = cfg()
    for sid in ids:
        d = os.path.join(IMAGES, sid)
        eager = kd.build_series_index(d, geometric=True)
        for k in (1, 3, 6, 12):
            c.in_channels = k
            survey = kd.build_series_index(d, geometric=True, order_uids=set())
            lazy = kd.build_series_index(
                d, geometric=True, order_uids=kd.plan_series_uids(survey, k, c))
            assert kd.select_slice_paths(eager, k, c) == \
                   kd.select_slice_paths(lazy, k, c), (sid, k)


# ══════════════════════════════════════════════════════════════════════════
# 2. Cache: exact round-trip, no all-zero entries, no stale reuse
# ══════════════════════════════════════════════════════════════════════════

def test_cache_round_trips_exactly():
    d = tempfile.mkdtemp(prefix="kdtest_cache_")
    try:
        c = cfg()
        rng = np.random.RandomState(0)
        stack = rng.randint(0, 256, (3, 224, 224)).astype(np.uint8)
        p = kd.cache_path_for(d, "STUDY", 3, c)
        assert kd.write_slice_cache(p, stack, c) is True
        back = np.load(p)
        assert back.dtype == np.uint8, back.dtype
        assert back.shape == stack.shape, back.shape
        assert np.array_equal(back, stack), "cache round-trip is not bit-exact"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cache_refuses_an_all_zero_study():
    """A previous version of this project accumulated 13,230 all-zero cache
    entries: one transient read failure per study, frozen forever."""
    d = tempfile.mkdtemp(prefix="kdtest_zero_")
    try:
        c = cfg()
        z = np.zeros((3, 224, 224), np.uint8)
        p = kd.cache_path_for(d, "BLACK", 3, c)
        assert kd.write_slice_cache(p, z, c) is False, "all-zero stack was cached"
        assert not os.path.exists(p), "all-zero cache file was created"
        assert os.listdir(d) == [], f"stray files left behind: {os.listdir(d)}"
        # a stack with a single non-zero pixel is legitimate and must be kept
        nz = z.copy()
        nz[1, 100, 100] = 1
        p2 = kd.cache_path_for(d, "NEARLY", 3, c)
        assert kd.write_slice_cache(p2, nz, c) is True
        assert np.array_equal(np.load(p2), nz)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cache_write_is_atomic_and_leaves_no_temp_files():
    """np.save appends '.npy' to a name that lacks it — a path-based temp file
    would be written to '<tmp>.npy' and the os.replace would then fail, so the
    cache would silently never be written. Guard that regression."""
    d = tempfile.mkdtemp(prefix="kdtest_atomic_")
    try:
        c = cfg()
        stack = np.full((3, 8, 8), 7, np.uint8)
        p = kd.cache_path_for(d, "ATOMIC", 3, c)
        assert kd.write_slice_cache(p, stack, c) is True
        assert os.listdir(d) == [os.path.basename(p)], os.listdir(d)
        assert np.array_equal(np.load(p), stack)
        # a failing writer must clean up after itself
        def boom(fh):
            raise RuntimeError("disk full")
        assert kd._atomic_write(os.path.join(d, "x.npy"), boom) is False
        assert os.listdir(d) == [os.path.basename(p)], os.listdir(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cache_key_changes_when_the_slice_policy_changes():
    """A cache built under the old filename-ordered policy must never be served
    to the fixed one — the pixels are different studies' worth of different."""
    base = cfg()
    legacy = cfg(**LEGACY)
    assert kd.selection_signature(base, 3) != kd.selection_signature(legacy, 3)
    assert kd.selection_signature(base, 3) != kd.selection_signature(base, 6)
    assert kd.selection_signature(base, 3) != \
        kd.selection_signature(cfg(image_size=288), 3)
    assert kd.selection_signature(base, 3) == kd.selection_signature(cfg(), 3)
    for a, b in (("slice_order", "filename"), ("slice_selection", "single_series"),
                 ("slice_trim_frac", 0.15), ("slice_sample", "endpoints"),
                 ("slice_plane_priority", "Axial,Coronal,Sagittal")):
        assert kd.selection_signature(base, 3) != \
            kd.selection_signature(cfg(**{a: b}), 3), a


def test_cache_hits_on_the_second_pass():
    ids = real_studies(4)
    d = tempfile.mkdtemp(prefix="kdtest_hit_")
    try:
        c = cfg()
        df = pd.DataFrame({"StudyInstanceUID": ids})
        for t in kd.KNEE_TARGETS:
            df[t] = 0.0
        ds = kd.RSNADataset(df, IMAGES, c, is_train=False, cache_dir=d)
        for i in range(len(ds)):
            ds[i]
        assert ds.cache_stats.get("miss", 0) == len(ids), ds.cache_stats
        ds.cache_stats.clear()
        for i in range(len(ds)):
            ds[i]
        assert ds.cache_stats.get("hit", 0) == len(ids), \
            f"second pass did not hit the cache: {ds.cache_stats}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_missing_study_yields_blanks_and_is_never_cached():
    d = tempfile.mkdtemp(prefix="kdtest_missing_")
    try:
        c = cfg()
        sl, status = kd.load_study_slices("NO_SUCH_STUDY", IMAGES, c, cache_dir=d)
        assert status == "nodir", status
        assert len(sl) == 3 and all(s.shape == (224, 224) for s in sl)
        assert all(not s.any() for s in sl)
        assert glob.glob(os.path.join(d, "*.npy")) == [], "blank study was cached"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
# 3. Augmentation invariants
# ══════════════════════════════════════════════════════════════════════════

def test_augmentation_output_shape_and_dtype_are_invariant():
    """Whatever the random draw, the tensor handed to the model is always
    (in_channels, image_size, image_size) float32 — a shape/dtype wobble would
    surface as a cryptic conv error a hundred batches into a fold."""
    rng = np.random.RandomState(3)
    for size in (224, 288):
        for ch in (1, 3, 6):
            c = cfg(image_size=size, in_channels=ch)
            for is_train in (True, False):
                tf = kd.build_transforms(c, is_train=is_train, in_channels=ch)
                for _ in range(12):
                    src = rng.randint(0, 256, (rng.randint(64, 700),
                                               rng.randint(64, 700), ch)).astype(np.uint8)
                    out = tf(image=src)["image"]
                    assert isinstance(out, torch.Tensor), type(out)
                    assert out.dtype == torch.float32, out.dtype
                    assert tuple(out.shape) == (ch, size, size), tuple(out.shape)
                    assert torch.isfinite(out).all(), "non-finite pixel in augmented tensor"


def test_dataset_item_shape_and_dtype_are_invariant():
    ids = real_studies(4)
    d = tempfile.mkdtemp(prefix="kdtest_item_")
    try:
        df = pd.DataFrame({"StudyInstanceUID": ids})
        for t in kd.KNEE_TARGETS:
            df[t] = 0.0
        for is_train in (True, False):
            c = cfg()
            ds = kd.RSNADataset(df, IMAGES, c, is_train=is_train, cache_dir=d)
            for i in range(len(ds)):
                img, lab = ds[i]
                assert tuple(img.shape) == (3, 224, 224), tuple(img.shape)
                assert img.dtype == torch.float32
                assert torch.isfinite(img).all()
                assert tuple(lab.shape) == (len(kd.KNEE_TARGETS),)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_slice_stack_is_transformed_as_one_registered_unit():
    """aug_slice_consistent: ONE geometric draw for the whole stack, otherwise
    the 2.5D channels are mutually shifted views of the same knee and a conv
    filter sees three unregistered images inside its own receptive field.

    Only the GEOMETRIC ops are asserted to be shared. The per-channel intensity
    ops (GaussNoise(per_channel=True), and CoarseDropout/brightness draws) are
    allowed to differ per channel — they do not move anatomy.
    """
    c = cfg(aug_hflip_p=0.0, aug_noise_p=0.0, aug_dropout_p=0.0,
            aug_brightness_contrast_p=0.0, aug_clahe_p=0.0,
            aug_sharpen_p=0.0, aug_gamma_p=0.0)
    ds_df = pd.DataFrame({"StudyInstanceUID": ["x"]})
    for t in kd.KNEE_TARGETS:
        ds_df[t] = 0.0
    d = tempfile.mkdtemp(prefix="kdtest_reg_")
    try:
        ds = kd.RSNADataset(ds_df, IMAGES, c, is_train=True, cache_dir=d)
        assert ds.slice_consistent is True
        marker = np.zeros((224, 224), np.uint8)
        marker[40:60, 40:60] = 255
        for _ in range(8):
            out = ds._apply_transform([marker.copy() for _ in range(3)])
            assert tuple(out.shape) == (3, 224, 224)
            a = out[0].numpy()
            for ch in range(1, 3):
                assert np.array_equal(a, out[ch].numpy()), (
                    "identical input slices landed in different places — the "
                    "geometric draw was re-rolled per channel")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
# 4. The legacy policy still reproduces the pre-fix tensors bit-exactly
# ══════════════════════════════════════════════════════════════════════════

def test_legacy_knobs_reproduce_the_old_pooled_filename_selection():
    """A/B-ability: the old behaviour must remain reachable, exactly."""
    ids = real_studies(6)
    c = cfg(**LEGACY)
    for sid in ids:
        sd = os.path.join(IMAGES, sid)
        old = []
        for root, _, files in os.walk(sd):
            for f in files:
                if f.endswith(".dcm"):
                    old.append(os.path.join(root, f))
        old.sort()
        n = len(old)
        m = int(n * 0.15)
        s, e = m, n - 1 - m
        if e - s + 1 < 3:
            s, e = 0, n - 1
        want = [old[i] for i in np.linspace(s, e, 3, dtype=int)]
        idx = kd.build_series_index(sd, geometric=False)
        assert kd.select_slice_paths(idx, 3, c) == want, sid


def test_decode_is_bit_identical_to_the_reference_implementation():
    """The 1.53x decode speedup must not move a single grey level."""
    import pydicom
    import cv2
    ids = real_studies(3)
    paths = []
    for sid in ids:
        paths += sorted(glob.glob(os.path.join(IMAGES, sid, "*", "*.dcm")))[:4]
    assert paths, "no DICOMs found"

    def reference(p, size):
        dcm = pydicom.dcmread(p)
        img = dcm.pixel_array.astype(np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(img) > np.min(img):
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        else:
            img = np.zeros_like(img)
        img = cv2.resize(img, (size, size))
        return (img * 255).astype(np.uint8)

    for p in paths:
        a, b = reference(p, 224), kd.decode_slice(p, 224)
        assert a.dtype == b.dtype == np.uint8
        assert np.array_equal(a, b), \
            f"decode diverged on {os.path.basename(p)}: max|d|={np.abs(a.astype(int)-b.astype(int)).max()}"


# ══════════════════════════════════════════════════════════════════════════
# 5. Robustness
# ══════════════════════════════════════════════════════════════════════════

def test_selection_handles_degenerate_studies():
    c = cfg()
    assert kd.select_slice_paths([], 3, c) == []
    one = [{"uid": "a", "plane": "Sagittal", "paths": ["/p0.dcm"]}]
    assert kd.select_slice_paths(one, 3, c) == ["/p0.dcm"] * 3
    two = [{"uid": "a", "plane": "Sagittal", "paths": ["/p0.dcm", "/p1.dcm"]}]
    got = kd.select_slice_paths(two, 3, c)
    assert len(got) == 3 and set(got) <= {"/p0.dcm", "/p1.dcm"}, got
    for k in (1, 2, 3, 4, 5, 8, 12):
        idx = [{"uid": f"s{i}", "plane": p,
                "paths": [f"/{p}/{j}.dcm" for j in range(7 + i)]}
               for i, p in enumerate(("Sagittal", "Coronal", "Axial"))]
        got = kd.select_slice_paths(idx, k, c)
        assert len(got) == k, (k, len(got))


def test_quotas_are_exhaustive_and_front_loaded():
    for k in range(0, 13):
        for g in range(1, 5):
            q = kd._quotas(k, g)
            assert len(q) == g and sum(q) == k, (k, g, q)
            assert q == sorted(q, reverse=True), (k, g, q)


def test_plane_priority_is_total_and_deduplicated():
    c = cfg(slice_plane_priority="Axial,Axial,Sagittal")
    assert kd._plane_priority(c) == ["Axial", "Sagittal", "Coronal"]
    assert kd._plane_priority(cfg(slice_plane_priority="")) == \
        ["Sagittal", "Coronal", "Axial"]
    assert kd._plane_priority(cfg(slice_plane_priority="nonsense")) == \
        ["Sagittal", "Coronal", "Axial"]


def test_dataset_runs_against_a_config_that_predates_every_new_knob():
    """Every knob is read through _cfg/getattr, so an old Config must still
    work — train.py, infer.py and pseudo_label.py all construct their own."""
    class OldConfig:
        image_size = 224
        in_channels = 3
    ids = real_studies(2)
    d = tempfile.mkdtemp(prefix="kdtest_oldcfg_")
    try:
        df = pd.DataFrame({"StudyInstanceUID": ids})
        for t in kd.KNEE_TARGETS:
            df[t] = 0.0
        ds = kd.RSNADataset(df, IMAGES, OldConfig(), is_train=False, cache_dir=d)
        img, lab = ds[0]
        assert tuple(img.shape) == (3, 224, 224)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    print("\n=== src/kaggle_data.py tests ===")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            check(name[5:], fn)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed, {len(SKIPPED)} skipped")
    for n, e in FAILED:
        print(f"  - {n}: {e}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
