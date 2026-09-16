"""Host-memory / dataloader invariants for src/kaggle_data.py.

These guard the changes made when the training set grew from 58 studies to
4,407, where each DataLoader worker is a forked copy of the dataset object:

  * __getitem__ must never touch the pandas DataFrame — a forked worker that
    walks an object-dtype column dirties one page per Python object, so the
    frame's *page* cost dwarfs its byte count. Measured on the real 4,407-row
    frame: 4.70 MB of private memory per worker (train_gold.csv columns) and
    12.34 MB (train.csv, report text present), against 0.61 MB after.
  * The compact id/label arrays must reproduce the old per-row pandas result
    BIT-FOR-BIT, including NaN, +-inf, int64 and float32 label columns.
  * The cache budget arithmetic must be exact, because it is what decides
    whether a 9-hour Kaggle session survives to write a submission.

Plain asserts, pytest-compatible, and skipped (not failed) without data_subset.
"""

import os
import sys
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
import src.kaggle_data as kd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(REPO, "data_subset", "train_series")


def cfg(**kw):
    c = Config()
    c.image_size, c.in_channels = 224, 3
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def real_studies(n=6):
    if not os.path.isdir(IMAGES):
        pytest.skip("data_subset/train_series not present")
    ids = [d for d in sorted(os.listdir(IMAGES)) if not d.startswith(".")][:n]
    if not ids:
        pytest.skip("no studies under data_subset/train_series")
    return ids


def frame(ids, kind="float", targets=None):
    """A training frame shaped like the real one, with awkward label dtypes."""
    targets = list(kd.KNEE_TARGETS if targets is None else targets)
    n = len(ids)
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"StudyInstanceUID": list(ids)})
    for j, t in enumerate(targets):
        if kind == "int":
            df[t] = (rng.random(n) < 0.5).astype(np.int64)
        elif kind == "float32":
            df[t] = rng.random(n).astype(np.float32)
        else:
            v = rng.random(n)
            if kind == "nan":
                v[rng.random(n) < 0.4] = np.nan
            elif kind == "inf":
                v[0] = np.inf
                v[min(1, n - 1)] = -np.inf
                v[min(2, n - 1)] = np.nan
            df[t] = v
    # the columns the real frame carries alongside the targets
    df["Report"] = ["finding text " * 90] * n
    df["fold"] = np.arange(n) % 5
    df["is_gold"] = rng.random(n) < 0.5
    df["is_pseudo"] = False
    return df, targets


class _Landmine:
    """Any attribute access raises. Stands in for the DataFrame."""

    def __getattr__(self, name):
        raise AssertionError(
            f"__getitem__ touched the DataFrame (self.df.{name}). Every "
            "DataLoader worker is a fork of this object; reaching into pandas "
            "here copies object-dtype pages into every worker.")


# ══════════════════════════════════════════════════════════════════════════
# 1. The frame must not be on the hot path
# ══════════════════════════════════════════════════════════════════════════

def test_getitem_never_touches_the_dataframe():
    ids = real_studies(4)
    df, targets = frame(ids, "nan")
    d = tempfile.mkdtemp(prefix="kdmem_land_")
    try:
        ds = kd.RSNADataset(df, IMAGES, cfg(), is_train=True, cache_dir=d)
        ds.df = _Landmine()               # from here on, any pandas touch fails
        assert len(ds) == len(ids)
        for i in range(len(ids)):
            img, lab = ds[i]
            assert tuple(img.shape) == (3, 224, 224)
            assert tuple(lab.shape) == (len(targets),)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_hot_path_state_is_contiguous_and_object_free():
    """The two arrays a worker touches must be single flat buffers.

    An object-dtype array would put the Python objects back, which is the whole
    cost being avoided; a non-contiguous label matrix would make the per-item
    row read a gather instead of a memcpy.
    """
    ids = real_studies(4)
    df, targets = frame(ids)
    d = tempfile.mkdtemp(prefix="kdmem_dtype_")
    try:
        ds = kd.RSNADataset(df, IMAGES, cfg(), is_train=True, cache_dir=d)
        assert ds._ids.dtype.kind in ("S", "U"), ds._ids.dtype
        assert ds._ids.flags["C_CONTIGUOUS"]
        assert ds._labels.dtype == np.float32, ds._labels.dtype
        assert ds._labels.flags["C_CONTIGUOUS"]
        assert ds._labels.shape == (len(ids), len(targets))
        # and the whole hot-path state is small enough to be irrelevant
        assert ds._ids.nbytes + ds._labels.nbytes < 200 * len(ids)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
# 2. Equivalence with the pandas path it replaced
# ══════════════════════════════════════════════════════════════════════════

def _old_style_labels(df, idx, targets):
    """Exactly what __getitem__ used to compute, per row, through pandas."""
    row = df.iloc[idx]
    return np.nan_to_num(row[targets].values.astype(np.float32), nan=0.0)


@pytest.mark.parametrize("kind", ["float", "nan", "inf", "int", "float32"])
def test_label_matrix_is_bit_identical_to_the_old_per_row_pandas_path(kind):
    ids = real_studies(6)
    df, targets = frame(ids, kind)
    d = tempfile.mkdtemp(prefix="kdmem_lab_")
    try:
        ds = kd.RSNADataset(df, IMAGES, cfg(), is_train=True, cache_dir=d)
        for i in range(len(ids)):
            want = _old_style_labels(df, i, targets)
            got = ds._labels[i]
            assert want.dtype == got.dtype == np.float32
            # bytes, not allclose: a 1-ULP drift would be a silent label change
            assert want.tobytes() == got.tobytes(), (kind, i, want, got)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_study_ids_round_trip_exactly():
    ids = real_studies(6)
    df, _ = frame(ids)
    d = tempfile.mkdtemp(prefix="kdmem_ids_")
    try:
        ds = kd.RSNADataset(df, IMAGES, cfg(), is_train=True, cache_dir=d)
        for i in range(len(ids)):
            assert ds._study_id(i) == str(df.iloc[i]["StudyInstanceUID"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_non_ascii_and_numeric_study_ids_still_resolve():
    """'S' (bytes) is the compact form; anything it cannot hold falls back to
    'U' rather than mangling the id into a path that does not exist."""
    d = tempfile.mkdtemp(prefix="kdmem_uni_")
    try:
        for raw in (["studyA", "studyB"], ["étude", "studyB"], [101, 202]):
            df = pd.DataFrame({"StudyInstanceUID": raw})
            for t in kd.KNEE_TARGETS:
                df[t] = 0.0
            ds = kd.RSNADataset(df, d, cfg(), is_train=True, cache_dir=d)
            for i in range(len(raw)):
                assert ds._study_id(i) == str(raw[i]), (raw, i, ds._study_id(i))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_len_matches_the_frame():
    ids = real_studies(5)
    df, _ = frame(ids)
    d = tempfile.mkdtemp(prefix="kdmem_len_")
    try:
        ds = kd.RSNADataset(df, IMAGES, cfg(), is_train=True, cache_dir=d)
        assert len(ds) == len(df) == len(ids)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_returned_labels_never_alias_the_shared_matrix():
    """The label matrix is shared by every worker through fork. A returned
    tensor that aliased it would let one in-place op downstream (mixup,
    label smoothing) rewrite the dataset for the rest of the run."""
    ids = real_studies(3)
    df, targets = frame(ids)
    d = tempfile.mkdtemp(prefix="kdmem_alias_")
    try:
        ds = kd.RSNADataset(df, IMAGES, cfg(), is_train=True, cache_dir=d)
        before = ds._labels.copy()
        for i in range(len(ids)):
            _, lab = ds[i]
            lab.add_(7.0)
            lab.zero_()
        assert np.array_equal(before, ds._labels), \
            "mutating a returned label tensor changed the dataset's matrix"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_target_list_monkeypatch_rebinds_the_labels():
    """train.py sets kaggle_data.KNEE_TARGETS after CSV column auto-detection,
    sometimes after a dataset exists. The cached matrix has to follow."""
    ids = real_studies(3)
    df, targets = frame(ids)
    d = tempfile.mkdtemp(prefix="kdmem_patch_")
    original = list(kd.KNEE_TARGETS)
    try:
        ds = kd.RSNADataset(df, IMAGES, cfg(), is_train=True, cache_dir=d)
        reordered = list(reversed(targets))
        kd.KNEE_TARGETS = reordered
        _, lab = ds[0]
        want = _old_style_labels(df, 0, reordered)
        assert lab.numpy().tobytes() == want.tobytes()
        assert ds._label_targets == reordered
    finally:
        kd.KNEE_TARGETS = original
        shutil.rmtree(d, ignore_errors=True)


def test_frame_without_target_columns_keeps_the_old_contract():
    """is_train=False -> image only (inference/pseudo-labelling path).
    is_train=True  -> raise, because a silently zero label is the exact
    failure filter_labelled() exists to prevent."""
    ids = real_studies(2)
    df = pd.DataFrame({"StudyInstanceUID": ids})
    d = tempfile.mkdtemp(prefix="kdmem_nolab_")
    try:
        ds_eval = kd.RSNADataset(df, IMAGES, cfg(), is_train=False, cache_dir=d)
        out = ds_eval[0]
        assert isinstance(out, torch.Tensor), type(out)
        ds_train = kd.RSNADataset(df, IMAGES, cfg(), is_train=True, cache_dir=d)
        with pytest.raises(KeyError):
            ds_train[0]
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
# 3. Cache budget arithmetic
# ══════════════════════════════════════════════════════════════════════════

def test_estimate_cache_bytes_matches_a_real_npy_on_disk():
    """The pixel term is a claim about np.save's output, so check np.save."""
    d = tempfile.mkdtemp(prefix="kdmem_est_")
    try:
        for C in (3, 6, 9):
            for size in (224, 384):
                p = os.path.join(d, f"probe_{C}_{size}.npy")
                np.save(p, np.zeros((C, size, size), dtype=np.uint8),
                        allow_pickle=False)
                actual = os.path.getsize(p)
                est = kd.estimate_cache_bytes(1, C, size, index_bytes_per_study=0,
                                              block_size=0)
                assert est == actual, (C, size, est, actual)
                # with block rounding the estimate may only ever be >= actual
                blocked = kd.estimate_cache_bytes(1, C, size,
                                                  index_bytes_per_study=0)
                assert blocked >= actual and blocked - actual < 4096
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_estimate_cache_bytes_scales_linearly_and_handles_zero():
    assert kd.estimate_cache_bytes(0, 3, 224) == 0
    one = kd.estimate_cache_bytes(1, 3, 224)
    assert kd.estimate_cache_bytes(4407, 3, 224) == 4407 * one
    # more channels and more pixels both cost strictly more
    assert kd.estimate_cache_bytes(100, 6, 224) > kd.estimate_cache_bytes(100, 3, 224)
    assert kd.estimate_cache_bytes(100, 3, 384) > kd.estimate_cache_bytes(100, 3, 224)


def test_available_cpus_is_sane():
    n = kd.available_cpus()
    assert isinstance(n, int) and n >= 1
    assert n <= (os.cpu_count() or 1), \
        "available_cpus must never exceed the host core count"


def test_cache_location_report_measures_rather_than_guesses():
    d = tempfile.mkdtemp(prefix="kdmem_loc_")
    unwritable = "/proc/definitely/not/writable/rsna"
    try:
        rep = kd.cache_location_report([unwritable, d], need_bytes=0)
        assert [r["path"] for r in rep] == [unwritable, d]
        good = rep[1]
        assert good["writable"] is True
        assert good["total"] > 0 and good["free"] > 0
        assert good["fits"] is True
        assert rep[0]["fits"] is False
        # nothing on earth has an exabyte free
        huge = kd.cache_location_report([d], need_bytes=2 ** 60)[0]
        assert huge["fits"] is False
        # the probe file must not be left behind
        assert not [f for f in os.listdir(d) if f.startswith(".rsna_probe")]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_resolve_cache_dir_picks_a_root_that_fits_and_honours_opt_out():
    d = tempfile.mkdtemp(prefix="kdmem_res_")
    try:
        c = cfg(cache_dir="/proc/definitely/not/writable/rsna")
        got = kd.resolve_cache_dir(c, need_bytes=1024, candidates=[d],
                                   verbose=False)
        assert got == d, got

        c2 = cfg(cache_dir="/some/explicit/path", cache_auto_locate=False)
        assert kd.resolve_cache_dir(c2, need_bytes=1024, candidates=[d],
                                    verbose=False) == "/some/explicit/path"

        # nothing fits -> keep the configured directory rather than invent one
        c3 = cfg(cache_dir=d)
        assert kd.resolve_cache_dir(c3, need_bytes=2 ** 60, candidates=[d],
                                    verbose=False) == d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_kaggle_working_is_the_last_cache_candidate():
    """Anything under /kaggle/working is committed as notebook OUTPUT against
    the ~19.5 GiB limit the checkpoints also draw on, so it must never be
    preferred over a scratch filesystem."""
    cands = kd.default_cache_candidates()
    working = [i for i, p in enumerate(cands) if p.startswith("/kaggle/working")]
    if working:
        assert working[0] == len(cands) - 1, cands


def test_precache_budget_reports_the_arithmetic():
    d = tempfile.mkdtemp(prefix="kdmem_bud_")
    # NOTE the nesting: _index_dir() puts the slice-index sidecars one level
    # ABOVE cache_dir (they are shared by every resolution phase), so handing a
    # bare directory to the cache scatters _slice_index into its PARENT. Always
    # pass the resolution-scoped subdirectory, exactly as train.py does.
    cache = os.path.join(d, "sz224_ch6")
    try:
        c = cfg(in_channels=6, image_size=224, cache_reserve_gb=0.5)
        b = kd.precache_budget(1000, c, cache)
        assert b["n_todo"] == 1000 and b["in_channels"] == 6
        assert b["need"] >= 1000 * 6 * 224 * 224
        assert b["reserve"] == int(0.5 * 2 ** 30)
        assert b["total"] > 0
        assert b["index_measured"] is False      # nothing written yet
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
# 4. Pre-cache behaviour
# ══════════════════════════════════════════════════════════════════════════

def test_precache_is_resumable_and_says_so():
    """The second run must rebuild nothing: a 4,407-study warm-up that silently
    repeated itself would cost the session twice."""
    ids = real_studies(4)
    d = tempfile.mkdtemp(prefix="kdmem_resume_")
    cache = os.path.join(d, "sz224_ch3")
    try:
        c = cfg()
        df = pd.DataFrame({"StudyInstanceUID": ids})
        first = kd.precache_dataset(df, IMAGES, c, cache_dir=cache)
        assert first.get("miss", 0) == len(ids), first
        stamps = {f: os.path.getmtime(os.path.join(cache, f))
                  for f in os.listdir(cache) if f.endswith(".npy")}
        assert stamps, "nothing was cached"

        second = kd.precache_dataset(df, IMAGES, c, cache_dir=cache)
        assert second.get("hit", 0) == len(ids), second
        assert second.get("miss", 0) == 0, second
        for f, t in stamps.items():
            assert os.path.getmtime(os.path.join(cache, f)) == t, \
                f"{f} was rewritten by a resumed pre-cache"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_precache_reports_studies_that_never_reached_the_disk():
    """_atomic_write swallows ENOSPC and load_study_slices ignores the result,
    so a full filesystem used to look exactly like a normal 'miss'. Every such
    study is re-decoded from DICOM on every epoch, so it has to be visible."""
    ids = real_studies(3)
    d = tempfile.mkdtemp(prefix="kdmem_nospace_")
    cache = os.path.join(d, "sz224_ch3")
    real_write = kd._atomic_write
    try:
        kd._atomic_write = lambda path, fn: (
            real_write(path, fn) if path.endswith(".json") else False)
        counts = kd.precache_dataset(
            pd.DataFrame({"StudyInstanceUID": ids}), IMAGES, cfg(),
            cache_dir=cache)
        assert counts.get("unwritten", 0) == len(ids), counts
        assert counts.get("miss", 0) == 0, counts
    finally:
        kd._atomic_write = real_write
        shutil.rmtree(d, ignore_errors=True)


def test_precache_stops_before_exhausting_the_reserve():
    """A reserve larger than the whole filesystem means the very first check
    must stop the run rather than write until ENOSPC."""
    ids = real_studies(6)
    d = tempfile.mkdtemp(prefix="kdmem_reserve_")
    cache = os.path.join(d, "sz224_ch3")
    try:
        free_gb = shutil.disk_usage(d).free / 2 ** 30
        c = cfg(cache_reserve_gb=free_gb + 1024.0)
        b = kd.precache_budget(len(ids), c, d)
        assert b["shortfall"] > 0, b
        # cache_require_space turns the shortfall into a refusal to start
        c.cache_require_space = True
        with pytest.raises(SystemExit):
            kd.precache_dataset(pd.DataFrame({"StudyInstanceUID": ids}),
                                IMAGES, c, cache_dir=cache)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_worker_processes_see_the_same_ids_and_labels_as_the_parent():
    """End-to-end: the compact arrays must survive whatever the DataLoader does
    to get the dataset into a worker (fork on Linux, spawn on macOS). Run with
    augmentation off so the only thing that can differ is the data itself."""
    from torch.utils.data import DataLoader
    ids = real_studies(6)
    df, targets = frame(ids, "nan")
    d = tempfile.mkdtemp(prefix="kdmem_dl_")
    cache = os.path.join(d, "sz224_ch3")
    try:
        c = cfg(aug_enabled=False)
        ds = kd.RSNADataset(df, IMAGES, c, is_train=False, cache_dir=cache)
        serial = [ds[i] for i in range(len(ds))]

        dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)
        got_img, got_lab = [], []
        for img, lab in dl:
            got_img.append(img)
            got_lab.append(lab)
        img = torch.cat(got_img)
        lab = torch.cat(got_lab)
        want_img = torch.stack([s[0] for s in serial])
        want_lab = torch.stack([s[1] for s in serial])
        assert torch.equal(img, want_img), "worker images differ from serial"
        assert torch.equal(lab, want_lab), "worker labels differ from serial"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_index_dir_is_shared_across_resolutions():
    """The slice index does not depend on image_size or in_channels, so the
    224px and 384px phases must not each pay for their own copy."""
    root = "/tmp/rsna_cache"
    a = kd._index_dir(os.path.join(root, "sz224_ch3"))
    b = kd._index_dir(os.path.join(root, "sz384_ch9"))
    assert a == b == os.path.join(root, "_slice_index")


def test_measured_index_bytes_prefers_measurement_to_the_constant():
    ids = real_studies(3)
    d = tempfile.mkdtemp(prefix="kdmem_idx_")
    cache = os.path.join(d, "sz224_ch3")
    try:
        assert kd.measured_index_bytes(cache) is None
        kd.precache_dataset(pd.DataFrame({"StudyInstanceUID": ids}), IMAGES,
                            cfg(), cache_dir=cache)
        got = kd.measured_index_bytes(cache)
        assert got is not None and got > 0, got
        b = kd.precache_budget(10, cfg(), cache)
        assert b["index_measured"] is True
        assert b["index_bytes"] == int(got)
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
