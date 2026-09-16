"""Tests for src/slots.py and scripts/build_slot_cache.py (spec §3).

    python3 -m pytest tests/test_slots.py -q

Real-data tests use <= 8 studies from data_subset/train_images and skip
when the subset is absent.  Synthetic-study tests (laterality, missing
slots, JPEG2000 decode) always run.
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, JPEG2000Lossless, generate_uid

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src import slots                      # noqa: E402
from src import kaggle_data as kd          # noqa: E402

DATA = os.path.join(REPO, "data_subset")
IMAGES = os.path.join(DATA, "train_images")
CSV = os.path.join(DATA, "train_series.csv")
HAVE_DATA = os.path.isdir(IMAGES) and os.path.isfile(CSV)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="data_subset/train_images not present")
BUILDER = os.path.join(REPO, "scripts", "build_slot_cache.py")
N_REAL = 8


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def studies():
    if not HAVE_DATA:
        pytest.skip("no data")
    uids = sorted(e.name for e in os.scandir(IMAGES) if e.is_dir())[:N_REAL]
    if not uids:
        pytest.skip("no studies")
    return uids


@pytest.fixture(scope="module")
def series_df():
    return pd.read_csv(CSV) if HAVE_DATA else None


@pytest.fixture(scope="module")
def indexes(studies, series_df):
    return {u: slots.index_study(os.path.join(IMAGES, u), series_df) for u in studies}


@pytest.fixture(scope="module")
def built(studies, series_df, indexes):
    out = {}
    for u in studies[:3]:
        x, m, info = slots.build_study_tensor(os.path.join(IMAGES, u), series_df, index=indexes[u])
        out[u] = (x, m, info)
    return out


# --------------------------------------------------------------------------
# Synthetic studies (no data needed)
# --------------------------------------------------------------------------

_SAG_IOP = [0, 1, 0, 0, 0, -1]       # normal along x
_COR_IOP = [1, 0, 0, 0, 0, -1]       # normal along y
_AX_IOP = [1, 0, 0, 0, 1, 0]         # normal along z
_ROWS = _COLS = 64
_SPACING = 2.5                        # 140 mm -> 56 px window inside a 64 px image
_P = 28


def _write_dcm(path, arr, ipp, iop, inst, series_uid, study_uid, desc, scan_options,
               laterality, compress=False, thickness=3.0, pad_before_geometry=0):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = JPEG2000Lossless if compress else ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.InstanceNumber = inst
    ds.ImagePositionPatient = [float(v) for v in ipp]
    ds.ImageOrientationPatient = [float(v) for v in iop]
    ds.PixelSpacing = [_SPACING, _SPACING]
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.SeriesDescription = desc
    ds.ScanOptions = scan_options
    if thickness is not None:
        ds.SliceThickness = thickness
    if pad_before_geometry:
        # A large private element with tag < (0020,0032) pushes the geometry
        # tags past the 8 KB partial-read window.
        ds.add_new(0x00191001, "OB", b"\0" * int(pad_before_geometry))
    ds.RepetitionTime, ds.EchoTime = 3000.0, 30.0
    if laterality:
        ds.Laterality = laterality
    if compress:
        import openjpeg
        from pydicom.encaps import encapsulate
        ds.PixelData = encapsulate([openjpeg.encode(arr, bits_stored=16, photometric_interpretation=2, use_mct=False)])
        ds["PixelData"].is_undefined_length = True
    else:
        ds.PixelData = arr.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path)


def _synthetic_image(k, bright_left=True):
    """Uniform 100 + 40k with a brighter left half (so column flips are visible)."""
    a = np.full((_ROWS, _COLS), 100 + 40 * k, dtype=np.uint16)
    half = slice(0, _COLS // 2) if bright_left else slice(_COLS // 2, _COLS)
    a[:, half] += 400
    return a


def make_synthetic_study(root, laterality_tag=None, centre_x=-110.0, compress_sag=False):
    """Three series (SAG-FS 10, COR-T1 8, AX-FS 6 slices) -> slots 0, 4, 2 present."""
    study_uid = "1.2.3.4.%d" % (abs(hash(root)) % 10 ** 8)
    sdir = os.path.join(root, study_uid)
    os.makedirs(sdir)
    half_fov = _SPACING * _COLS / 2.0
    specs = [
        ("1.2.3.4.1", _SAG_IOP, 10, "pd_tse_fs_sag", "FS", compress_sag,
         lambda k: [centre_x - 10 + 2 * k, -half_fov, half_fov]),           # x varies
        ("1.2.3.4.2", _COR_IOP, 8, "t1_tse_cor", "", False,
         lambda k: [centre_x - half_fov, -10 + 2 * k, half_fov]),            # y varies
        ("1.2.3.4.3", _AX_IOP, 6, "t2_tse_fs_tra", "FS", False,
         lambda k: [centre_x - half_fov, -half_fov, 20 - 2 * k]),            # z varies
    ]
    for suid, iop, n, desc, so, comp, ipp_fn in specs:
        d = os.path.join(sdir, suid)
        os.makedirs(d)
        for k in range(n):
            # Random-looking filenames: ordering must come from geometry, not names.
            name = "%s.dcm" % generate_uid()
            _write_dcm(os.path.join(d, name), _synthetic_image(k), ipp_fn(k), iop, n - k,
                       suid, study_uid, desc, so, laterality_tag, compress=comp)
    return sdir


# --------------------------------------------------------------------------
# Pure-logic tests
# --------------------------------------------------------------------------

def test_slot_names_and_slot_of():
    assert slots.SLOT_NAMES == ["SAG_FS", "COR_FS", "AX_FS", "SAG_T1", "COR_T1", "AX_T1"]
    assert slots.slot_of("Sagittal", 1) == 0
    assert slots.slot_of("Coronal", 1) == 1
    assert slots.slot_of("Axial", 1) == 2
    assert slots.slot_of("Sagittal", 0) == 3
    assert slots.slot_of("Coronal", 0) == 4
    assert slots.slot_of("Axial", 0) == 5
    assert slots.slot_of(None, 1) is None and slots.slot_of("Axial", None) is None


def test_anchor_indices_adjacent_ordered_clamped():
    idx = slots.anchor_indices(30, G=3, T=3, trim_frac=0.15)
    assert idx.shape == (3, 3)
    assert idx.tolist() == [[3, 4, 5], [13, 14, 15], [24, 25, 26]]
    for row in idx:
        assert np.all(np.diff(row) == 1)            # physically adjacent
    assert np.all(np.diff(idx[:, 1]) > 0)           # anchors ascend along the stack
    assert idx.min() >= 0 and idx.max() <= 29
    assert slots.anchor_indices(30, G=1, T=3).tolist() == [[13, 14, 15]]   # centre
    assert slots.anchor_indices(1, G=3, T=3).tolist() == [[0, 0, 0]] * 3  # clamped
    assert slots.anchor_indices(2, G=2, T=3, trim_frac=0.0).tolist() == [[0, 0, 1], [0, 1, 1]]
    assert slots.anchor_indices(30, G=2, T=1).shape == (2, 1)
    with pytest.raises(ValueError):
        slots.anchor_indices(0)


def test_crop_window_synthetic():
    # 256 px @ 0.625 mm: 140 mm = 224 px, centred.
    assert slots.crop_window(256, 256, (0.625, 0.625), 140.0) == (16, 240, 16, 240, False)
    # 512 px @ 0.1953 mm: window 717 px > image -> whole image + fallback.
    assert slots.crop_window(512, 512, (0.1953, 0.1953), 140.0) == (0, 512, 0, 512, True)
    # Missing / invalid spacing -> whole image + fallback.
    assert slots.crop_window(640, 640, None, 140.0) == (0, 640, 0, 640, True)
    assert slots.crop_window(640, 640, (0.0, 0.5), 140.0)[-1] is True
    # Anisotropic spacing gives a rectangular window (per-axis px).
    r0, r1, c0, c1, fb = slots.crop_window(640, 640, (0.25, 0.5), 140.0)
    assert (r1 - r0, c1 - c0, fb) == (560, 280, False)


def test_resolve_side():
    assert slots.resolve_side(["R", "R", None], [-90.0, -80.0])[:2] == ("R", "tag")
    assert slots.resolve_side(["L"], [-90.0])[:2] == ("L", "tag")         # tag beats geometry
    assert slots.resolve_side([None, None], [-90.0, -80.0, -100.0])[:2] == ("R", "geometry")
    assert slots.resolve_side([], [80.0])[:2] == ("L", "geometry")
    assert slots.resolve_side([], [5.0, -3.0, 10.0])[:2] == (None, "unresolved")   # |x| < 20 mm
    assert slots.resolve_side(["L", "R"], [])[:2] == (None, "unresolved")          # tie, no geometry
    assert slots.resolve_side([], [])[:2] == (None, "unresolved")


def test_fluid_from_header_means_fat_suppressed():
    assert slots.fluid_from_header("FS", "LT_pd_tse_fs_cor", 4370, 36) == (1, "pd")
    assert slots.fluid_from_header("", "Sag T2 FSE", 6483, 73) == (0, "t2")       # FSE is not FS
    assert slots.fluid_from_header("", "t1_tse_cor", 675, 11) == (0, "t1")
    assert slots.fluid_from_header("", "cor STIR", 4000, 40)[0] == 1
    assert slots.fluid_from_header(["FAST_GEMS", "FS"], "Ax PD FSE FS", 4704, 39.5) == (1, "pd")
    assert slots.fluid_from_header("PFP SAT2 WE", "t1_vibe_we_tra_cartilage", 17.1, 9.4) == (0, "t1")
    assert slots.fluid_from_header(None, None, 3000, 80) == (0, "t2")             # TR/TE weighting only
    assert slots.fluid_from_header(None, None, 2000, 30) == (0, "pd")


def test_apply_laterality_synthetic():
    rng = np.random.RandomState(0)
    x = rng.randint(0, 256, size=(6, 2, 3, 4, 4)).astype(np.uint8)
    y = slots.apply_laterality(x, "R")
    for s in (1, 2, 4, 5):                                  # COR / AX: columns mirrored
        assert np.array_equal(y[s], x[s][..., ::-1])
    for s in (0, 3):                                        # SAG: anchor order reversed, no mirror
        assert np.array_equal(y[s][0], x[s][1]) and np.array_equal(y[s][1], x[s][0])
        assert np.array_equal(y[s], x[s][::-1])
    assert np.array_equal(slots.apply_laterality(x, "L"), x)
    assert np.array_equal(slots.apply_laterality(x, None), x)
    y[0, 0, 0, 0, 0] = 7                                    # returned array is a copy
    assert x[0, 1, 0, 0, 0] != 7 or x[0, 1, 0, 0, 0] == 7  # no aliasing with x
    assert y.flags["C_CONTIGUOUS"]


def test_shard_layout():
    assert slots.shard_layout(5, 2, "train") == [
        {"x": "train_x.u8", "start": 0, "n": 2},
        {"x": "train_x.001.u8", "start": 2, "n": 2},
        {"x": "train_x.002.u8", "start": 4, "n": 1}]
    assert slots.shard_layout(5, 0, "test") == [{"x": "test_x.u8", "start": 0, "n": 5}]
    assert slots.shard_layout(0, 512) == []


# --------------------------------------------------------------------------
# Synthetic right-knee study: slots, masks, laterality, compressed decode
# --------------------------------------------------------------------------

def test_synthetic_right_knee_tag_flip(tmp_path):
    sdir = make_synthetic_study(str(tmp_path), laterality_tag="R", centre_x=-110.0)
    idx = slots.index_study(sdir)                       # no CSV -> header fallback
    assert sorted(idx.slots) == [0, 2, 4]
    assert [len(idx.slots[s]) for s in (0, 2, 4)] == [10, 6, 8]
    assert idx.side == "R" and idx.side_source == "tag"
    assert idx.n_files == 24
    # Geometry ordering: sagittal stack sorted by IPP.normal regardless of filename order.
    pos = [kd._read_geometry(p)[1] for p in idx.slots[0]]
    assert pos == sorted(pos)

    x_raw, m_raw, info_raw = slots.build_study_tensor(sdir, P=_P, laterality=False)
    x, m, info = slots.build_study_tensor(sdir, P=_P, laterality=True)
    assert x.shape == (6, 3, 3, _P, _P) and x.dtype == np.uint8
    assert m.tolist() == [1, 0, 1, 0, 1, 0]
    for s in (1, 3, 5):
        assert not x[s].any()
    assert info["flipped"] and not info_raw["flipped"]
    # Only the selected files are decoded, each once: 9 + 8 + 6 = 23 unique
    # paths here (adjacent anchor groups overlap on the 8- and 6-slice stacks).
    sel = slots.select_slices(idx)
    n_unique = sum(len(set(p for g in groups for p in g)) for groups in sel.values())
    assert n_unique == 23 and info["n_decoded"] == n_unique
    assert info["n_decode_fail"] == 0 and info["n_crop_fallback"] == 0
    assert np.array_equal(x, slots.apply_laterality(x_raw, "R"))
    # Unflipped COR/AX: bright half on the LEFT columns; flipped: on the RIGHT.
    for s in (2, 4):
        left, right = x_raw[s][..., :_P // 2].mean(), x_raw[s][..., _P // 2:].mean()
        assert left > right
        left, right = x[s][..., :_P // 2].mean(), x[s][..., _P // 2:].mean()
        assert right > left
    # SAG: slice brightness is monotonic in k, so the unflipped group order must follow
    # the geometry order (first ordered slice -> group 0) and the flip must reverse it.
    first = pydicom.dcmread(idx.slots[0][0]).pixel_array.mean()
    last = pydicom.dcmread(idx.slots[0][-1]).pixel_array.mean()
    g_raw = [x_raw[0][g].mean() for g in range(3)]
    g_flip = [x[0][g].mean() for g in range(3)]
    assert first != last
    assert (g_raw[0] > g_raw[2]) == (first > last)
    assert (g_flip[0] > g_flip[2]) == (first < last)
    assert np.array_equal(x[0][..., :_P // 2], x_raw[0][::-1][..., :_P // 2])   # no column mirror in SAG


def test_synthetic_side_from_geometry_and_unresolved(tmp_path):
    right = make_synthetic_study(str(tmp_path / "r"), laterality_tag=None, centre_x=-110.0)
    idx = slots.index_study(right)
    assert (idx.side, idx.side_source) == ("R", "geometry")
    assert idx.centre_x is not None and idx.centre_x < -20
    _, _, info = slots.build_study_tensor(right, P=_P)
    assert info["flipped"]

    mid = make_synthetic_study(str(tmp_path / "m"), laterality_tag=None, centre_x=5.0)
    idx = slots.index_study(mid)
    assert (idx.side, idx.side_source) == (None, "unresolved")
    _, _, info = slots.build_study_tensor(mid, P=_P)
    assert not info["flipped"]

    # A series CSV (dict form) overrides the header-derived plane/contrast.
    left = make_synthetic_study(str(tmp_path / "l"), laterality_tag="L", centre_x=110.0)
    override = {"1.2.3.4.2": ("Coronal", 1)}            # call the T1 coronal fluid-sensitive
    idx = slots.index_study(left, override)
    assert 1 in idx.slots and 4 not in idx.slots and idx.side == "L"


def test_synthetic_jpeg2000_decodes_like_uncompressed(tmp_path):
    pytest.importorskip("openjpeg")
    pytest.importorskip("pylibjpeg")
    plain = make_synthetic_study(str(tmp_path / "plain"), laterality_tag="L", centre_x=110.0)
    comp = make_synthetic_study(str(tmp_path / "comp"), laterality_tag="L", centre_x=110.0, compress_sag=True)
    xp, mp, ip = slots.build_study_tensor(plain, P=_P)
    xc, mc, ic = slots.build_study_tensor(comp, P=_P)
    assert ic["n_decode_fail"] == 0 and ic["decode_fail_by_syntax"] == {}
    assert any("JPEG 2000" in k for k in ic["syntax_hist"]), ic["syntax_hist"]
    assert np.array_equal(xp, xc) and np.array_equal(mp, mc)
    # A corrupt file is counted, zero-filled and does not kill the study.
    bad_dir = os.path.join(comp, "1.2.3.4.3")
    for name in sorted(os.listdir(bad_dir))[:2]:
        with open(os.path.join(bad_dir, name), "wb") as f:
            f.write(b"\0" * 300)
    xb, mb, ib = slots.build_study_tensor(comp, P=_P)
    assert ib["n_decode_fail"] >= 1 and "unreadable" in ib["decode_fail_by_syntax"]
    assert mb.tolist() == [1, 0, 1, 0, 1, 0]


def test_series_preference_tiers():
    def si(uid, n, thickness=None):
        info = slots.SeriesInfo(uid=uid, files=["f%d" % i for i in range(n)])
        info.thickness = thickness
        return info
    # 3-D thin stack loses to the routine 2-D TSE despite 5x the slices
    assert slots.preferred_series([si("1.9", 160, 0.6), si("1.5", 30, 3.0)]).uid == "1.5"
    # slice count in [12, 60] beats out-of-range
    assert slots.preferred_series([si("1.1", 8, 3.0), si("1.2", 20, 3.0)]).uid == "1.2"
    assert slots.preferred_series([si("1.1", 100, 3.0), si("1.2", 20, 3.0)]).uid == "1.2"
    # inside the range, thickness >= 2 mm beats thinner even with fewer slices
    assert slots.preferred_series([si("1.1", 30, 1.0), si("1.2", 20, 3.0)]).uid == "1.2"
    # unknown thickness is not penalised
    assert slots.preferred_series([si("1.1", 30, None), si("1.2", 20, 3.0)]).uid == "1.1"
    # same tier -> most slices, then smallest UID (numeric-aware)
    assert slots.preferred_series([si("1.1", 20, 3.0), si("1.2", 30, 3.0)]).uid == "1.2"
    assert slots.preferred_series([si("1.10", 30, 3.0), si("1.2", 30, 3.0)]).uid == "1.2"
    # fallback: no candidate in range -> the comparison still resolves
    assert slots.preferred_series([si("1.9", 160, 0.6)]).uid == "1.9"
    assert slots.preferred_series([si("1.9", 160, 0.6), si("1.8", 200, 0.6)]).uid == "1.8"
    assert slots.preferred_series([]) is None


def test_synthetic_3d_vs_2d_series_selection(tmp_path):
    study_uid = "9.8.7"
    sdir = os.path.join(str(tmp_path), study_uid)
    arr = np.full((16, 16), 500, dtype=np.uint16)
    for suid, n, thick, desc in (("9.1.160", 160, 0.6, "t1_vibe_we_tra"),
                                 ("9.1.30", 30, 3.0, "t1_tse_tra")):
        d = os.path.join(sdir, suid)
        os.makedirs(d)
        for k in range(n):
            _write_dcm(os.path.join(d, "%s.dcm" % generate_uid()), arr,
                       [-110, -40, 20 - 0.7 * k], _AX_IOP, k + 1, suid, study_uid,
                       desc, "", "L", thickness=thick)
    idx = slots.index_study(sdir)
    assert idx.slot_series[5] == "9.1.30"          # AX_T1: 2-D TSE beats the 160-slice 3-D
    assert [x.thickness for x in idx.series if x.uid == "9.1.160"] == [0.6]
    # with the 2-D series gone the 3-D one still fills the slot (fallback tier)
    import shutil
    shutil.rmtree(os.path.join(sdir, "9.1.30"))
    idx2 = slots.index_study(sdir)
    assert idx2.slot_series[5] == "9.1.160"


def test_resize_to_square_two_step():
    rng = np.random.RandomState(1)
    for shape in ((560, 560), (561, 563), (640, 640), (448, 448), (896, 896),
                  (336, 336), (300, 200), (100, 100), (224, 224)):
        base = cv2.GaussianBlur((rng.rand(*shape) * 400).astype(np.float32), (0, 0), 3)
        out = slots.resize_to_square(base, 224)
        assert out.shape == (224, 224) and out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]
        ref = cv2.resize(base, (224, 224), interpolation=cv2.INTER_AREA)
        assert float(np.abs(out - ref).mean()) < 2.0, shape      # 'negligible pixel difference'
        assert float(np.abs(out - ref).max()) < 40.0, shape
    # uint16 input comes back float32 and matches the same input as float32
    u16 = (rng.rand(640, 640) * 4000).astype(np.uint16)
    out16 = slots.resize_to_square(u16, 224)
    assert out16.dtype == np.float32
    assert float(np.abs(out16 - slots.resize_to_square(u16.astype(np.float32), 224)).max()) <= 1.0
    # identity path
    same = np.zeros((224, 224), np.float32)
    assert slots.resize_to_square(same, 224).shape == (224, 224)


def test_resize_helper_shared_by_tensor_and_cache_builder(tmp_path, monkeypatch):
    calls = {"n": 0}
    real = slots.resize_to_square
    def counting(a, P):
        calls["n"] += 1
        return real(a, P)
    monkeypatch.setattr(slots, "resize_to_square", counting)
    sdir = make_synthetic_study(str(tmp_path / "s"), laterality_tag="L", centre_x=110.0)
    x, m, info = slots.build_study_tensor(sdir, P=_P)
    assert calls["n"] == info["n_decoded"] > 0     # one shared resize per decoded slice
    # the cache builder's worker goes through the very same module-level helper
    import importlib.util
    spec = importlib.util.spec_from_file_location("bsc_under_test", BUILDER)
    bsc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsc)
    out = tmp_path / "cache"
    out.mkdir()
    row_shape = (6, 3, 3, _P, _P)
    x_path = str(out / "train_x.u8")
    bsc.ensure_size(x_path, int(np.prod(row_shape)))
    before = calls["n"]
    res = bsc._worker({"row": 0, "uid": os.path.basename(sdir), "study_dir": sdir,
                       "lookup": {}, "P": _P, "crop_mm": 140.0, "G": 3, "T": 3,
                       "trim_frac": 0.15, "laterality": True, "x_path": x_path,
                       "shard_rows": 1, "local_row": 0})
    assert res["ok"] and calls["n"] > before
    mm = np.memmap(x_path, dtype=np.uint8, mode="r", shape=(1,) + row_shape)
    assert np.array_equal(np.asarray(mm[0]), x)    # builder row == direct build, bit-identical


def test_partial_header_read_falls_back_cleanly(tmp_path):
    arr = np.full((16, 16), 300, dtype=np.uint16)
    plain = str(tmp_path / "plain.dcm")
    _write_dcm(plain, arr, [-110, -40, 3], _AX_IOP, 7, "8.1", "8.0", "t1_tse_tra", "", "L")
    padded = str(tmp_path / "padded.dcm")
    _write_dcm(padded, arr, [-110, -40, 3], _AX_IOP, 7, "8.1", "8.0", "t1_tse_tra", "", "L",
               pad_before_geometry=10240)
    inst_a, pos_a, plane_a, fb_a = slots._read_geometry_fast(plain)
    assert fb_a is False and plane_a == "Axial" and inst_a == 7
    inst_b, pos_b, plane_b, fb_b = slots._read_geometry_fast(padded)
    assert fb_b is True                            # geometry sat past 8 KB -> full read used
    assert (inst_b, plane_b) == (inst_a, plane_a) and pos_b == pos_a
    assert kd._read_geometry(padded) == (inst_b, pos_b, plane_b)
    # unreadable file: both paths fail -> (None, ...) and counted as fallback
    junk = str(tmp_path / "junk.dcm")
    with open(junk, "wb") as f:
        f.write(b"\0" * 300)
    assert slots._read_geometry_fast(junk) == (None, None, None, True)
    # a padded-header study still indexes correctly end to end
    sdir = os.path.join(str(tmp_path), "8.9")
    d = os.path.join(sdir, "8.9.1")
    os.makedirs(d)
    for k in range(12):
        _write_dcm(os.path.join(d, "%s.dcm" % generate_uid()), arr,
                   [-110, -40, 20 - 2 * k], _AX_IOP, k + 1, "8.9.1", "8.9",
                   "t1_tse_tra", "", "R", pad_before_geometry=10240)
    idx = slots.index_study(sdir)
    assert idx.slot_series[5] == "8.9.1" and idx.n_header_fallback >= 12
    pos = [kd._read_geometry(p)[1] for p in idx.slots[5]]
    assert pos == sorted(pos)


@needs_data
def test_order_series_fast_matches_kaggle_data(studies, indexes):
    checked = 0
    for u in studies[:3]:
        for suid in indexes[u].slot_series.values():
            files = sorted(os.path.join(IMAGES, u, suid, f)
                           for f in os.listdir(os.path.join(IMAGES, u, suid)))
            paths_fast, plane_fast, n_fb = slots._order_series_fast(files)
            paths_ref, plane_ref = kd.order_series(files)
            assert paths_fast == paths_ref and plane_fast == plane_ref
            assert n_fb == 0                       # local corpus: 8 KB always suffices
            checked += 1
    assert checked >= 6


def test_anchor_shift_moves_every_anchor(tmp_path):
    base = slots.anchor_indices(30, G=3, T=3)
    plus = slots.anchor_indices(30, G=3, T=3, anchor_shift=1)
    minus = slots.anchor_indices(30, G=3, T=3, anchor_shift=-1)
    assert np.array_equal(plus, base + 1) and np.array_equal(minus, base - 1)
    assert np.array_equal(plus[:, 1], base[:, 1] + 1)          # every anchor, exactly one slice
    assert slots.anchor_indices(30, G=3, T=3, anchor_shift=0).tolist() == base.tolist()
    assert slots.anchor_indices(1, G=3, T=3, anchor_shift=5).tolist() == [[0, 0, 0]] * 3   # no room
    big = slots.anchor_indices(30, G=3, T=3, anchor_shift=100)
    assert big.max() == 29 and big.min() == 28                  # clamped, still adjacent
    # threaded through select_slices (no I/O needed)
    idx = slots.StudyIndex(study_dir="x", uid="x", slots={0: ["p%02d" % i for i in range(30)]})
    g0 = slots.select_slices(idx)[0]
    g1 = slots.select_slices(idx, anchor_shift=1)[0]
    gm = slots.select_slices(idx, anchor_shift=-1)[0]
    for a, b, c in zip(g0, g1, gm):
        assert int(b[1][1:]) == int(a[1][1:]) + 1 and int(c[1][1:]) == int(a[1][1:]) - 1
    # and through build_study_tensor (default 0 = unchanged)
    sdir = make_synthetic_study(str(tmp_path), laterality_tag="L", centre_x=110.0)
    x0, m0, i0 = slots.build_study_tensor(sdir, P=_P)
    x1, m1, i1 = slots.build_study_tensor(sdir, P=_P, anchor_shift=1)
    assert x1.shape == x0.shape and np.array_equal(m0, m1) and i1["anchor_shift"] == 1
    assert not np.array_equal(x0, x1) and i0["anchor_shift"] == 0
    assert np.array_equal(x0, slots.build_study_tensor(sdir, P=_P, anchor_shift=0)[0])


def test_zoom_slots_synthetic(tmp_path):
    sdir = make_synthetic_study(str(tmp_path), laterality_tag="R", centre_x=-110.0)
    zs = ("SAG_FS", "COR_FS", "AX_FS")
    names = slots.slot_names_ext(zs)
    assert names == slots.SLOT_NAMES + ["SAG_FS_Z", "COR_FS_Z", "AX_FS_Z"]
    assert slots.SLOT_NAMES_EXT("SAG_FS,COR_FS") == slots.SLOT_NAMES + ["SAG_FS_Z", "COR_FS_Z"]
    assert slots.SLOT_NAMES == ["SAG_FS", "COR_FS", "AX_FS", "SAG_T1", "COR_T1", "AX_T1"]   # BC
    x_raw, m_raw, i_raw = slots.build_study_tensor(sdir, P=_P, laterality=False, zoom_mm=100.0, zoom_slots=zs)
    x, m, info = slots.build_study_tensor(sdir, P=_P, zoom_mm=100.0, zoom_slots="SAG_FS,COR_FS,AX_FS")
    assert x.shape == (9, 3, 3, _P, _P) and x.dtype == np.uint8 and m.shape == (9,)
    assert info["slot_names"] == names and info["zoom_mm"] == 100.0
    assert m.tolist() == [1, 0, 1, 0, 1, 0, 1, 0, 1]            # zoom masks copy their base
    assert not x[7].any()                                        # COR_FS absent -> COR_FS_Z zero
    assert info["n_decoded"] == i_raw["n_decoded"] == 23         # zoom crops add NO decodes
    # zoom crop == centre sub-window of the base crop, up-sampled (corr > 0.9)
    r = int(round(_P * 100.0 / 140.0))
    o = (_P - r) // 2
    for base_s, zoom_s in ((2, 8), (0, 6)):
        b = x_raw[base_s][1, 1].astype(np.float32)[o:o + r, o:o + r]
        up = cv2.resize(b, (_P, _P), interpolation=cv2.INTER_LINEAR)
        z = x_raw[zoom_s][1, 1].astype(np.float32)
        assert np.corrcoef(up.ravel(), z.ravel())[0, 1] > 0.9
    # laterality covers zoom slots: AX_FS_Z columns mirrored, SAG_FS_Z anchors reversed
    assert np.array_equal(x, slots.apply_laterality(x_raw, "R", names))
    assert x[8][..., _P // 2:].mean() > x[8][..., :_P // 2].mean()
    assert np.array_equal(x[6], x_raw[6][::-1])
    with pytest.raises(ValueError):
        slots.apply_laterality(x_raw, "R")                       # 9 slots need their names
    # the base 6 are bit-identical with or without zoom slots
    x6, m6, _ = slots.build_study_tensor(sdir, P=_P)
    assert x6.shape[0] == 6 and np.array_equal(x6, x[:6]) and np.array_equal(m6, m[:6])
    with pytest.raises(ValueError):
        slots.build_study_tensor(sdir, P=_P, zoom_slots=("SAG_FS",))     # zoom_mm missing
    with pytest.raises(ValueError):
        slots.slot_names_ext(("NOPE",))
    with pytest.raises(ValueError):
        slots.slot_names_ext(("SAG_FS", "SAG_FS"))


@needs_data
def test_zoom_slots_real_and_cache_roundtrip(tmp_path, studies, series_df, indexes):
    u = studies[0]
    zs = ("SAG_FS", "COR_FS")
    x, m, info = slots.build_study_tensor(os.path.join(IMAGES, u), series_df, index=indexes[u],
                                          zoom_mm=100.0, zoom_slots=zs)
    assert x.shape == (8, 3, 3, 224, 224) and m[6] == m[0] and m[7] == m[1]
    r = int(round(224 * 100.0 / 140.0))
    o = (224 - r) // 2
    for b_s, z_s in ((0, 6), (1, 7)):
        if not m[b_s]:
            continue
        b = x[b_s][1, 1].astype(np.float32)[o:o + r, o:o + r]
        up = cv2.resize(b, (224, 224), interpolation=cv2.INTER_LINEAR)
        assert np.corrcoef(up.ravel(), x[z_s][1, 1].astype(np.float32).ravel())[0, 1] > 0.9
    out = tmp_path / "zoom_cache"
    _run_builder(out, 2, extra=["--zoom-mm", "100", "--zoom-slots", "SAG_FS,COR_FS"])
    cache = slots.SlotCache(str(out), "train")
    assert cache.S == 8 and cache.slot_names == slots.slot_names_ext(zs)
    assert cache.zoom_mm == 100.0 and cache.zoom_slots == list(zs)
    xc, mc = cache[0]
    assert xc.shape == (8, 3, 3, 224, 224) and mc.shape == (8,)
    assert np.array_equal(np.asarray(xc), x) and np.array_equal(np.asarray(mc), m)   # bit-identical
    allmask = np.asarray(cache.mask)
    assert np.array_equal(allmask[:, 6], allmask[:, 0]) and np.array_equal(allmask[:, 7], allmask[:, 1])
    assert (out / "train_x.u8").stat().st_size == 2 * 8 * 9 * 224 * 224
    index = json.loads((out / "train_index.json").read_text())
    assert index["slot_names"] == slots.slot_names_ext(zs) and set(index["stats"]["missing_slot_hist"]) == set(index["slot_names"])
    # resuming with the 6-slot layout on this 8-slot cache is refused
    r2 = subprocess.run([sys.executable, BUILDER, "--data-dir", DATA, "--out", str(out), "--limit", "1"],
                        capture_output=True, text=True, timeout=120)
    assert r2.returncode != 0 and "use a new --out" in (r2.stdout + r2.stderr)


# --------------------------------------------------------------------------
# Real-data tests (<= 8 studies)
# --------------------------------------------------------------------------

@needs_data
def test_index_study_real(studies, series_df, indexes):
    csv = series_df.set_index("SeriesInstanceUID")
    for u in studies:
        idx = indexes[u]
        assert set(idx.slots) <= set(range(6)) and idx.slots
        n_files = sum(len(f) for _, _, f in os.walk(os.path.join(IMAGES, u)))
        assert idx.n_files == n_files
        assert idx.side in ("L", "R", None)
        by_slot = {}
        for si in idx.series:
            if si.slot is not None:
                by_slot.setdefault(si.slot, []).append(si)
        for s, paths in idx.slots.items():
            suid = idx.slot_series[s]
            assert all(os.path.isfile(p) for p in paths)
            assert len(paths) == len(os.listdir(os.path.join(IMAGES, u, suid)))
            # chosen series = the tiered preference over the slot's candidates
            assert suid == slots.preferred_series(by_slot[s]).uid
            # and the slot agrees with the CSV's plane / Fluid_Sensitive
            row = csv.loc[suid]
            assert slots.slot_of(row["Anatomical_Plane"], int(row["Fluid_Sensitive"])) == s
            sp = idx.spacing[suid]
            assert sp is not None and len(sp) == 2 and sp[0] > 0 and sp[1] > 0


@needs_data
def test_selected_slices_are_geometry_ordered_and_adjacent(studies, indexes):
    checked = 0
    for u in studies[:3]:
        idx = indexes[u]
        sel = slots.select_slices(idx, G=3, T=3, trim_frac=0.15)
        assert set(sel) == set(idx.slots)
        for s, groups in sel.items():
            paths = idx.slots[s]
            pos_of = {p: i for i, p in enumerate(paths)}
            assert len(groups) == 3 and all(len(g) == 3 for g in groups)
            flat = [pos_of[p] for g in groups for p in g]
            assert flat == sorted(flat)                                   # ordered along the stack
            for g in groups:
                i = [pos_of[p] for p in g]
                assert all(0 <= d <= 1 for d in np.diff(i))               # adjacent (dup only at clamp)
            anchors = [pos_of[g[1]] for g in groups]
            assert anchors[0] >= round(0.15 * (len(paths) - 1)) - 1
            assert anchors[-1] <= round(0.85 * (len(paths) - 1)) + 1
            # physical positions along the slice normal are non-decreasing
            geo = [kd._read_geometry(p)[1] for g in groups for p in g]
            if all(v is not None for v in geo):
                assert all(b >= a for a, b in zip(geo, geo[1:]))
            checked += 1
    assert checked > 0


@needs_data
def test_crop_window_matches_header(studies, indexes):
    idx = indexes[studies[0]]
    s = sorted(idx.slots)[0]
    path = slots.select_slices(idx)[s][1][1]
    ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["Rows", "Columns", "PixelSpacing"])
    sp = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
    r0, r1, c0, c1, fb = slots.crop_window(int(ds.Rows), int(ds.Columns), sp, 140.0)
    exp_h, exp_w = int(round(140.0 / sp[0])), int(round(140.0 / sp[1]))
    if exp_h <= int(ds.Rows) and exp_w <= int(ds.Columns):
        assert (r1 - r0, c1 - c0, fb) == (exp_h, exp_w, False)
        assert abs((r0 + r1) / 2 - int(ds.Rows) / 2) <= 1 and abs((c0 + c1) / 2 - int(ds.Columns) / 2) <= 1
    else:
        assert (r0, r1, c0, c1, fb) == (0, int(ds.Rows), 0, int(ds.Columns), True)


@needs_data
def test_build_study_tensor_real(built, indexes):
    for u, (x, m, info) in built.items():
        idx = indexes[u]
        assert x.shape == (6, 3, 3, 224, 224) and x.dtype == np.uint8
        assert m.shape == (6,) and m.dtype == np.uint8 and set(m.tolist()) <= {0, 1}
        for s in range(6):
            if s in idx.slots:
                assert m[s] == 1
                assert x[s].max() == 255 and x[s].min() == 0          # per-slot 1-99 pct stretch
            else:
                assert m[s] == 0 and not x[s].any()
        for k in ("side", "n_decoded", "n_decode_fail", "n_crop_fallback", "ms", "ms_index", "ms_decode"):
            assert k in info
        assert info["n_decode_fail"] == 0
        assert 0 < info["n_decoded"] <= 6 * 3 * 3
        assert info["flipped"] == (idx.side == "R")


@needs_data
def test_real_right_knee_is_mirrored(studies, series_df, indexes):
    right = [u for u in studies if indexes[u].side == "R"]
    if not right:
        pytest.skip("no right knee among the first %d studies" % N_REAL)
    u = right[0]
    x_raw, _, _ = slots.build_study_tensor(os.path.join(IMAGES, u), series_df, laterality=False, index=indexes[u])
    x, _, info = slots.build_study_tensor(os.path.join(IMAGES, u), series_df, laterality=True, index=indexes[u])
    assert info["flipped"]
    assert np.array_equal(x, slots.apply_laterality(x_raw, "R"))
    assert not np.array_equal(x, x_raw)


@needs_data
def test_p252_build(studies, series_df, indexes):
    u = studies[0]
    x, m, _ = slots.build_study_tensor(os.path.join(IMAGES, u), series_df, P=252, G=2, T=3, index=indexes[u])
    assert x.shape == (6, 2, 3, 252, 252)
    with pytest.raises(ValueError):
        slots.build_study_tensor(os.path.join(IMAGES, u), series_df, P=100, index=indexes[u])


def _run_builder(out, limit, extra=()):
    cmd = [sys.executable, BUILDER, "--data-dir", DATA, "--split", "train", "--out", str(out),
           "--limit", str(limit), "--workers", "2", "--shard-size", "2", "--progress-every", "1"] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@needs_data
def test_cache_roundtrip_resume_and_shards(tmp_path, studies, series_df, built):
    out = tmp_path / "cache"
    log = _run_builder(out, 3)
    assert "todo: 3 new studies" in log
    for name in ("train_x.u8", "train_x.001.u8", "train_mask.u8", "train_index.json"):
        assert (out / name).is_file(), name
    index = json.loads((out / "train_index.json").read_text())
    assert index["studies"] == studies[:3] and index["done"] == [1, 1, 1]
    assert index["version"] == slots.CACHE_VERSION and index["P"] == 224 and index["G"] == 3 and index["T"] == 3
    assert index["stats"]["n_new"] == 3 and index["stats"]["n_decode_fail"] == 0
    assert index["stats"]["peak_rss_worker_bytes"] < 4e9 and index["stats"]["peak_rss_parent_bytes"] < 4e9
    assert set(index["stats"]["missing_slot_hist"]) == set(slots.SLOT_NAMES)
    assert (out / "train_x.u8").stat().st_size == 2 * 6 * 9 * 224 * 224
    assert (out / "train_x.001.u8").stat().st_size == 1 * 6 * 9 * 224 * 224

    cache = slots.SlotCache(str(out), "train")
    assert (cache.N, cache.P, cache.G, cache.T, len(cache)) == (3, 224, 3, 3, 3)
    assert cache.S == 6 and cache.slot_names == slots.SLOT_NAMES and cache.zoom_slots == []
    assert cache.uids == studies[:3] and cache.done.all()
    for i, u in enumerate(studies[:3]):
        x, m = cache[i]
        assert x.shape == (6, 3, 3, 224, 224) and x.dtype == np.uint8 and m.shape == (6,)
        assert isinstance(x, np.memmap)                                   # zero-copy view
        x_ref, m_ref, _ = built[u]
        assert np.array_equal(np.asarray(x), x_ref) and np.array_equal(np.asarray(m), m_ref)
        assert cache.row_of(u) == i
        assert cache.side(i) in ("L", "R", None)
    assert np.array_equal(cache.get(studies[2])[0], built[studies[2]][0])
    with pytest.raises(AttributeError):
        _ = cache.x                                                       # multi-shard: no flat view

    # Resume: nothing to do.
    log2 = _run_builder(out, 3)
    assert "todo: 0 new studies (3 already done)" in log2
    index2 = json.loads((out / "train_index.json").read_text())
    assert index2["stats"]["n_new"] == 0 and index2["done"] == [1, 1, 1]

    # Extend by one study: old rows untouched, new shard row appended.
    before = np.array(slots.SlotCache(str(out))[0][0])
    log3 = _run_builder(out, 4)
    assert "todo: 1 new studies (3 already done)" in log3
    cache3 = slots.SlotCache(str(out), "train")
    assert cache3.N == 4 and cache3.uids == studies[:4] and cache3.done.all()
    assert np.array_equal(np.asarray(cache3[0][0]), before)
    assert cache3[3][0].shape == (6, 3, 3, 224, 224) and cache3[3][1].sum() > 0
    assert len(cache3.shards) == 2

    # Parameter mismatch is refused rather than silently mixing layouts.
    r = subprocess.run([sys.executable, BUILDER, "--data-dir", DATA, "--out", str(out), "--limit", "1",
                        "--P", "252"], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "use a new --out" in (r.stdout + r.stderr)

    # A pre-zoom index (no slot_names / zoom keys at all) still reads as S = 6.
    idx_path = out / "train_index.json"
    legacy = json.loads(idx_path.read_text())
    for k in ("slot_names", "zoom_mm", "zoom_slots"):
        legacy.pop(k, None)
    idx_path.write_text(json.dumps(legacy))
    old = slots.SlotCache(str(out), "train")
    assert old.S == 6 and old.slot_names == slots.SLOT_NAMES and old.zoom_slots == []
    assert old[0][0].shape == (6, 3, 3, 224, 224) and old[0][1].shape == (6,)


# --------------------------------------------------------------------------
# Joint-centre locator (opt-in zoom_center="joint"; no data needed)
# --------------------------------------------------------------------------

def _joint_image(rows, cols, spacing, joint_rc, polarity="dark", seed=0):
    """Synthetic knee slice: a grey leg, femur / tibia blobs and a joint band.

    ``joint_rc`` is the (row, col) of the joint centre in px; blob and band
    sizes are in mm so the same picture works at any ``spacing``.  FS
    (``polarity="dark"``): marrow dark, band bright.  T1 (``"bright"``):
    marrow bright, band dark.
    """
    rng = np.random.RandomState(seed)
    jr, jc = joint_rc
    yy, xx = np.mgrid[:rows, :cols]
    mm_y, mm_x = (yy - jr) * spacing, (xx - jc) * spacing
    a = np.zeros((rows, cols), dtype=np.float32)
    leg = (xx * spacing >= 0.12 * cols * spacing) & (xx * spacing <= 0.88 * cols * spacing)
    a[leg] = 400.0
    fem = ((mm_y + 30.0) / 26.0) ** 2 + (mm_x / 30.0) ** 2 <= 1.0
    tib = ((mm_y - 30.0) / 26.0) ** 2 + (mm_x / 34.0) ** 2 <= 1.0
    a[fem | tib] = 60.0 if polarity == "dark" else 900.0
    band = (np.abs(mm_y) <= 3.0) & (np.abs(mm_x) <= 32.0)
    a[band] = 1000.0 if polarity == "dark" else 150.0
    a += rng.normal(0.0, 8.0, a.shape).astype(np.float32)
    return np.clip(a, 0, 4095).astype(np.uint16)


def test_crop_window_offset():
    rows, cols, sp = 512, 512, (0.3, 0.3)
    base = slots.crop_window(rows, cols, sp, 100.0)
    assert slots.crop_window(rows, cols, sp, 100.0, None) == base
    assert slots.crop_window(rows, cols, sp, 100.0, (0.0, 0.0)) == base
    r0, r1, c0, c1, fb = slots.crop_window(rows, cols, sp, 100.0, (15.0, -9.0))
    assert (r1 - r0, c1 - c0) == (base[1] - base[0], base[3] - base[2]) and not fb
    assert r0 == base[0] + 50 and c0 == base[2] - 30             # 15 mm / 0.3 = 50 px
    r0, r1, c0, c1, fb = slots.crop_window(rows, cols, sp, 100.0, (500.0, -500.0))   # clamped inside
    assert (r0, r1, c0, c1) == (rows - 333, rows, 0, 333) and not fb
    assert slots.crop_window(rows, cols, sp, 200.0, (10.0, 10.0)) == (0, rows, 0, cols, True)   # window > image
    assert slots.crop_window(rows, cols, sp, 100.0, (float("nan"), 0.0)) == base


def test_locate_joint_synthetic_blobs():
    rows, cols, sp = 200, 200, 1.0
    for pol in ("dark", "bright"):
        for jr, jc in ((100, 100), (118, 76), (84, 121)):
            a = _joint_image(rows, cols, sp, (jr, jc), polarity=pol)
            est = slots.locate_joint(a, (sp, sp), polarity=pol)
            assert not est.fallback and est.conf >= slots.JOINT_MIN_CONF, (pol, jr, jc, est)
            assert abs(est.row - jr) <= 2.0 and abs(est.col - jc) <= 2.0, (pol, jr, jc, est.row, est.col)
            assert abs(est.dy_mm - (jr - (rows - 1) / 2.0) * sp) <= 2.0
            assert abs(est.dx_mm - (jc - (cols - 1) / 2.0) * sp) <= 2.0
            assert est.offset_mm == (est.dy_mm, est.dx_mm) and est.ms < 20.0
            d = est.as_dict()
            assert json.dumps(d) and d["fallback"] is False
    # anisotropic spacing: offsets come back in mm, rows / cols in px
    a = _joint_image(160, 240, 1.0, (96, 130), polarity="dark")
    est = slots.locate_joint(a, (1.0, 0.75), polarity="dark")
    assert not est.fallback and abs(est.row - 96) <= 2.0 and abs(est.col - 130) <= 3.0
    with pytest.raises(ValueError):
        slots.locate_joint(a, (1.0, 1.0), polarity="sideways")


def test_locate_joint_flat_and_degenerate_fall_back():
    flat = np.full((128, 128), 500, dtype=np.uint16)
    est = slots.locate_joint(flat, (1.0, 1.0))
    assert est.fallback and est.conf == 0.0 and est.reason == "flat image"
    assert (est.row, est.col, est.dy_mm, est.dx_mm) == (63.5, 63.5, 0.0, 0.0) and est.offset_mm is None
    zeros = np.zeros((128, 128), dtype=np.int16)
    assert slots.locate_joint(zeros, (1.0, 1.0)).fallback
    noise = np.random.RandomState(1).randint(0, 50, size=(128, 128)).astype(np.uint16)
    est = slots.locate_joint(noise, (1.0, 1.0))
    assert est.fallback or est.conf < 1.0                                    # never raises
    assert slots.locate_joint(flat, None).reason == "no spacing"
    assert slots.locate_joint(flat, (0.0, 1.0)).reason == "no spacing"
    assert slots.locate_joint(np.zeros((3, 3), np.uint16), (1.0, 1.0)).reason == "tiny image"
    assert slots.locate_joint(np.zeros((0, 5), np.uint16), (1.0, 1.0)).fallback
    # the joint must stay inside the middle band / 40 mm: a band at the very top is clamped
    a = _joint_image(200, 200, 1.0, (30, 100), polarity="dark")
    est = slots.locate_joint(a, (1.0, 1.0), polarity="dark")
    assert est.fallback or abs(est.dy_mm) <= slots.JOINT_MAX_OFFSET_MM


def test_combine_joint_estimates_rules():
    def E(dy, conf, fb=False):
        return slots.JointEstimate(50.0 + dy, 50.0, dy_mm=dy, dx_mm=0.0, conf=conf, fallback=fb, reason="ok" if not fb else "x")
    assert slots.combine_joint_estimates([]).fallback
    # centre anchor confident: agreeing slices are averaged, outliers ignored
    c = slots.combine_joint_estimates([E(30, 0.9), E(2, 0.5), E(4, 0.6)], centre=1)
    assert not c.fallback and c.dy_mm == 3.0 and "centre" in c.reason and c.n_slices == 3
    # centre not confident -> majority of ALL slices must agree
    c = slots.combine_joint_estimates([E(2, 0.5), E(0, 0.1), E(4, 0.6)], centre=1)
    assert not c.fallback and c.dy_mm == 3.0 and "majority" in c.reason
    c = slots.combine_joint_estimates([E(2, 0.5), E(0, 0.1), E(40, 0.6)], centre=1)
    assert c.fallback and "1/3" in c.reason
    c = slots.combine_joint_estimates([E(2, 0.5, fb=True), E(0, 0.2, fb=True), E(4, 0.9, fb=True)])
    assert c.fallback and c.row == 52.0                         # image centre of the first slice
    assert not slots.combine_joint_estimates([E(5, 0.4)]).fallback          # 1 of 1
    assert slots.combine_joint_estimates([E(5, 0.2)]).fallback
    # sagittal takes the vote, coronal honours the anchor, axial never locates
    ests = [(np.zeros((64, 64), np.uint16), (1.0, 1.0))]
    assert slots.locate_joint_for_slot(2, ests).reason == "axial"
    assert slots.locate_joint_for_slot(5, ests, centre=0).fallback
    assert slots.joint_polarity(0) == "dark" and slots.joint_polarity(4) == "bright"
    # slice choice: centre anchor's T slices + the nearest other anchors' middle slices
    groups = [["a0", "a1", "a2"], ["b0", "b1", "b2"], ["c0", "c1", "c2"]]
    assert slots.joint_slice_paths(groups) == (["b0", "b1", "b2", "a1", "c1"], 1)
    groups10 = [["g%d" % g] for g in range(10)]
    paths, centre = slots.joint_slice_paths(groups10)
    assert paths == ["g5", "g4", "g6", "g3", "g7"] and centre == 0
    assert slots.joint_slice_paths([]) == ([], 0)


def _make_joint_study(root, joint_mm, laterality_tag="L"):
    """SAG-FS (12 slices, dark marrow) + COR-T1 (10, bright marrow) with the
    joint ``joint_mm`` = (dy, dx) mm from the image centre; AX-FS (6) plain."""
    study_uid = "1.2.3.5.%d" % (abs(hash(root)) % 10 ** 8)
    sdir = os.path.join(root, study_uid)
    os.makedirs(sdir)
    half_fov = _SPACING * _COLS / 2.0
    jr = (_ROWS - 1) / 2.0 + joint_mm[0] / _SPACING
    jc = (_COLS - 1) / 2.0 + joint_mm[1] / _SPACING
    specs = [
        ("1.2.3.5.1", _SAG_IOP, 12, "pd_tse_fs_sag", "FS", "dark",
         lambda k: [-110.0 - 12 + 2 * k, -half_fov, half_fov]),
        ("1.2.3.5.2", _COR_IOP, 10, "t1_tse_cor", "", "bright",
         lambda k: [-110.0 - half_fov, -10 + 2 * k, half_fov]),
        ("1.2.3.5.3", _AX_IOP, 6, "t2_tse_fs_tra", "FS", None,
         lambda k: [-110.0 - half_fov, -half_fov, 20 - 2 * k]),
    ]
    for suid, iop, n, desc, so, pol, ipp_fn in specs:
        d = os.path.join(sdir, suid)
        os.makedirs(d)
        for k in range(n):
            arr = _joint_image(_ROWS, _COLS, _SPACING, (jr, jc), pol, seed=k) if pol else _synthetic_image(k)
            _write_dcm(os.path.join(d, "%s.dcm" % generate_uid()), arr, ipp_fn(k), iop, n - k,
                       suid, study_uid, desc, so, laterality_tag)
    return sdir


def test_zoom_center_joint_synthetic(tmp_path):
    zs = ("SAG_FS", "COR_T1", "AX_FS")
    # plain synthetic study: no joint to find -> fallback everywhere, tensors bit-identical
    sdir = make_synthetic_study(str(tmp_path / "plain"), laterality_tag="R")
    x0, m0, i0 = slots.build_study_tensor(sdir, P=_P, zoom_mm=100.0, zoom_slots=zs)
    x1, m1, i1 = slots.build_study_tensor(sdir, P=_P, zoom_mm=100.0, zoom_slots=zs, zoom_center="image")
    x2, m2, i2 = slots.build_study_tensor(sdir, P=_P, zoom_mm=100.0, zoom_slots=zs, zoom_center="joint")
    assert np.array_equal(x0, x1) and i0["zoom_center"] == "image" and i0["joint"] == {} and i0["n_joint_fallback"] == 0
    assert np.array_equal(x0, x2) and np.array_equal(m0, m2) and i2["zoom_center"] == "joint"
    assert set(i2["joint"]) == {"SAG_FS", "COR_T1", "AX_FS"} and i2["n_joint_fallback"] == 3
    assert i2["joint"]["AX_FS"]["reason"] == "axial" and i2["joint"]["SAG_FS"]["fallback"] is True
    assert i2["joint"]["COR_T1"]["fallback"] is True and i2["joint"]["COR_T1"]["n_slices"] == 5
    assert i2["n_decoded"] == i0["n_decoded"]                                            # no extra decodes
    json.dumps(i2)
    with pytest.raises(ValueError):
        slots.build_study_tensor(sdir, P=_P, zoom_mm=100.0, zoom_slots=zs, zoom_center="knee")
    # a study whose joint sits 25 mm below / 12 mm right of the image centre
    dy, dx = 25.0, 12.0
    jdir = _make_joint_study(str(tmp_path / "joint"), (dy, dx))
    xi, mi, ii = slots.build_study_tensor(jdir, P=_P, zoom_mm=100.0, zoom_slots=zs)
    xj, mj, ij = slots.build_study_tensor(jdir, P=_P, zoom_mm=100.0, zoom_slots=zs, zoom_center="joint")
    assert mj.tolist() == [1, 0, 1, 0, 1, 0, 1, 1, 1] and np.array_equal(mi, mj)
    assert np.array_equal(xi[:6], xj[:6])                                                # base slots untouched
    assert ij["n_joint_fallback"] == 1 and ij["joint"]["AX_FS"]["fallback"]             # axial only
    for name in ("SAG_FS", "COR_T1"):
        j = ij["joint"][name]
        assert j["fallback"] is False and j["conf"] >= slots.JOINT_MIN_CONF, j
        assert abs(j["dy_mm"] - dy) <= 3.0 and abs(j["dx_mm"] - dx) <= 3.0, j
        assert j["n_slices"] == 5 and j["ms"] < 50.0
    assert np.array_equal(xj[8], xi[8])                                                  # AX zoom: image centre
    for s in (6, 7):
        assert not np.array_equal(xj[s], xi[s])
        # the bright (SAG_FS) / dark (COR_T1) joint band now runs through the crop's middle row
        prof = xj[s][1, 1].astype(np.float32).mean(axis=1)
        peak = int(np.argmax(prof) if s == 6 else np.argmin(prof[4:-4]) + 4)
        assert abs(peak - _P // 2) <= 2, (s, peak)
        # ... and was off-centre in the image-centred crop
        prof_i = xi[s][1, 1].astype(np.float32).mean(axis=1)
        peak_i = int(np.argmax(prof_i) if s == 6 else np.argmin(prof_i[4:-4]) + 4)
        assert peak_i - _P // 2 >= 4, (s, peak_i)
    # the cache builder records the choice, SlotCache exposes it, resume refuses a change
    out = tmp_path / "jcache"
    cmd = [sys.executable, BUILDER, "--data-dir", str(tmp_path), "--image-dir", str(tmp_path / "joint"),
           "--out", str(out), "--P", str(_P), "--workers", "1", "--zoom-mm", "100", "--zoom-slots", ",".join(zs),
           "--zoom-center", "joint"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "zoom_center=joint" in r.stdout and "joint-centred zoom: 1 / 3 slots fell back" in r.stdout
    cache = slots.SlotCache(str(out), "train")
    assert cache.zoom_center == "joint" and cache.S == 9 and cache.zoom_slots == list(zs)
    assert np.array_equal(np.asarray(cache[0][0]), xj) and np.array_equal(np.asarray(cache[0][1]), mj)
    index = json.loads((out / "train_index.json").read_text())
    assert index["zoom_center"] == "joint" and index["stats"]["n_joint_fallback"] == 1 and index["stats"]["n_joint_slots"] == 3
    r2 = subprocess.run(cmd[:-2], capture_output=True, text=True, timeout=300)                 # default "image"
    assert r2.returncode != 0 and "use a new --out" in (r2.stdout + r2.stderr)
    index.pop("zoom_center")
    (out / "train_index.json").write_text(json.dumps(index))
    assert slots.SlotCache(str(out), "train").zoom_center == "image"                            # old caches


def test_zoom_center_medial_synthetic(tmp_path):
    """"medial" centre: 80 mm window at MEDIAL_CENTRE_MM = (+20 distal, 32 medial) from the
    located joint.  Harness scale: 80 mm = 28 px (2.857 mm/px); the joint sits 5 mm below
    the image centre so joint+60 mm and the medial edge (-72 mm) stay inside the +-80 mm
    FOV (no clamping).  The synthetic leg is left-right symmetric, so the medial check is
    the joint column's position: the dark band / blobs lie RIGHT of the crop centre."""
    dy, dx = 5.0, 0.0
    for tag in ("L", "R"):
        jdir = _make_joint_study(str(tmp_path / ("m" + tag)), (dy, dx), laterality_tag=tag)
        xi, mi, ii = slots.build_study_tensor(jdir, P=_P, zoom_spec="COR_T1:80:medial")
        assert ii["slot_names"] == slots.SLOT_NAMES + ["COR_T1_Z80M"] and mi.tolist() == [1, 0, 1, 0, 1, 0, 1]
        j = ii["joint"]["COR_T1"]
        assert j["fallback"] is False and j["medial_side_fallback"] is False and ii["n_medial_side_fallback"] == 0
        sgn = -1.0 if tag == "L" else 1.0
        assert abs(j["medial_offset"][0] - (j["dy_mm"] + 20.0)) < 1e-6
        assert abs(j["medial_offset"][1] - (j["dx_mm"] + sgn * 32.0)) < 1e-6
        assert ii["flipped"] is (tag == "R")
        json.dumps(ii)
        img = xi[6][1, 1].astype(np.float32)
        prof = img.mean(axis=1)
        peak = int(np.argmin(prof[2:-2]) + 2)                 # T1: dark joint band
        assert abs(peak - (_P // 2 - 7)) <= 2, (tag, peak)    # 20 mm proximal of centre = 7 px at 28 px / 80 mm
        row = img[peak]
        # the crop centre is 32 mm MEDIAL of the joint column, so the dark T1 band (+-32 mm
        # about that column) starts at the crop's middle column and runs off its right edge;
        # left of it is soft tissue (the leg) and, at the very edge, background.
        bright = row >= 80
        first = int(len(row) - np.argmax(bright[::-1]))       # first band column after the last bright one
        assert bright.any() and abs(first - _P // 2) <= 1 and row[first:].max() < 80, (tag, row)
        leg = row[:first][row[:first] > 10]                   # inside the leg (background ~0)
        assert leg.mean() > row[first:].mean() + 40, (tag, row)
    # SAG bases cannot take a medial centre; the builder persists the per-study estimates
    with pytest.raises(ValueError):
        slots.build_study_tensor(jdir, P=_P, zoom_spec="SAG_FS:80:medial")
    out = tmp_path / "mcache"
    cmd = [sys.executable, BUILDER, "--data-dir", str(tmp_path), "--image-dir", str(tmp_path / "mL"),
           "--out", str(out), "--P", str(_P), "--workers", "1", "--zoom-spec", "COR_T1:80:medial"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    cache = slots.SlotCache(str(out), "train")
    assert cache.S == 7 and slots.parse_zoom_spec(cache.zoom_spec) == (("COR_T1", 80.0, "medial"),)
    index = json.loads((out / "train_index.json").read_text())
    assert index["stats"]["n_medial_side_fallback"] == 0 and len(index["joint"]) == 1
    (uid, jinfo), = index["joint"].items()
    assert uid == cache.uids[0] and jinfo["COR_T1"]["fallback"] is False
    assert abs(jinfo["COR_T1"]["medial_offset"][1] - (jinfo["COR_T1"]["dx_mm"] - 32.0)) < 1e-6


# --------------------------------------------------------------------------
# Zoom spec (per-entry mm + centre; Phase B ACL/notch view)
# --------------------------------------------------------------------------

def test_zoom_spec_parsing():
    assert slots.parse_zoom_spec(None) == ()
    assert slots.parse_zoom_spec("SAG_FS:80:joint") == (("SAG_FS", 80.0, "joint"),)
    assert slots.parse_zoom_spec("SAG_FS:100") == (("SAG_FS", 100.0, "image"),)
    assert slots.parse_zoom_spec("SAG_FS:100:joint, SAG_FS:80:joint") == (
        ("SAG_FS", 100.0, "joint"), ("SAG_FS", 80.0, "joint"))
    assert slots.parse_zoom_spec(["COR_FS:90", ("SAG_FS", 80, "joint")]) == (
        ("COR_FS", 90.0, "image"), ("SAG_FS", 80.0, "joint"))
    assert slots.zoom_slot_name("SAG_FS", 80, "joint") == "SAG_FS_Z80J"
    assert slots.zoom_slot_name("COR_FS", 100.0) == "COR_FS_Z100"
    assert slots.zoom_slot_name("SAG_FS", 87.5, "joint") == "SAG_FS_Z87.5J"
    assert slots.zoom_slot_name("COR_FS", 80, "medial") == "COR_FS_Z80M"
    for bad in ("SAG_FS:80:medial", "COR_T1:80:knee"):
        with pytest.raises(ValueError):
            slots.normalize_zoom(zoom_spec=bad)
    assert slots.slot_names_ext(zoom_spec="SAG_FS:100:joint,SAG_FS:80:joint") == (
        slots.SLOT_NAMES + ["SAG_FS_Z100J", "SAG_FS_Z80J"])
    # legacy trio + spec combine; legacy keeps its historical _Z name and comes first
    zooms = slots.normalize_zoom(100.0, ("COR_FS",), "image", "SAG_FS:80:joint")
    assert [z.name for z in zooms] == ["COR_FS_Z", "SAG_FS_Z80J"]
    assert tuple(zooms[1]) == ("SAG_FS", 80.0, "joint", "SAG_FS_Z80J")
    for bad in ("SAG_FS", "SAG_FS:x", ":80"):
        with pytest.raises(ValueError):
            slots.parse_zoom_spec(bad)
    for bad in ("NOPE:80:joint", "SAG_FS:0", "SAG_FS:80:elbow",
                "SAG_FS:80:joint,SAG_FS:80:joint"):
        with pytest.raises(ValueError):
            slots.normalize_zoom(None, (), "image", bad)


def test_zoom_spec_build_synthetic(tmp_path):
    sdir = make_synthetic_study(str(tmp_path), laterality_tag="L", centre_x=110.0)
    x, m, info = slots.build_study_tensor(sdir, P=_P, zoom_spec="SAG_FS:100,SAG_FS:80")
    assert x.shape == (8, 3, 3, _P, _P) and x.dtype == np.uint8
    assert info["slot_names"] == slots.SLOT_NAMES + ["SAG_FS_Z100", "SAG_FS_Z80"]
    assert info["zoom_spec"] == [["SAG_FS", 100.0, "image"], ["SAG_FS", 80.0, "image"]]
    assert m.tolist() == [1, 0, 1, 0, 1, 0, 1, 1]
    assert info["n_decoded"] == 23                     # still exactly one decode per slice
    # the 80 mm view is a centre sub-window of the 100 mm view
    r = int(round(_P * 80.0 / 100.0))
    o = (_P - r) // 2
    b = x[6][1, 1].astype(np.float32)[o:o + r, o:o + r]
    up = cv2.resize(b, (_P, _P), interpolation=cv2.INTER_LINEAR)
    assert np.corrcoef(up.ravel(), x[7][1, 1].astype(np.float32).ravel())[0, 1] > 0.9
    # a 100/image spec entry carries the same pixels as the legacy trio (name differs)
    x_leg, m_leg, i_leg = slots.build_study_tensor(sdir, P=_P, zoom_mm=100.0, zoom_slots=("SAG_FS",))
    assert i_leg["slot_names"][6] == "SAG_FS_Z" and np.array_equal(x_leg[6], x[6])
    # joint-centred spec runs end to end (locator may fall back on synthetic slices)
    xj, mj, ij = slots.build_study_tensor(sdir, P=_P, zoom_spec="SAG_FS:80:joint")
    assert ij["slot_names"][6] == "SAG_FS_Z80J" and "SAG_FS" in ij["joint"]
    assert mj.tolist() == [1, 0, 1, 0, 1, 0, 1]
    # defaults bit-identical: no zoom kwargs -> the original 6-slot tensor
    x6, m6, _ = slots.build_study_tensor(sdir, P=_P)
    assert x6.shape[0] == 6 and np.array_equal(x6, x[:6]) and np.array_equal(m6, m[:6])


@needs_data
def test_zoom_spec_real_and_builder_smoke(tmp_path, studies, series_df, indexes):
    u = studies[0]
    spec = "SAG_FS:100:joint,SAG_FS:80:joint"
    x, m, info = slots.build_study_tensor(os.path.join(IMAGES, u), series_df,
                                          index=indexes[u], zoom_spec=spec)
    assert x.shape == (8, 3, 3, 224, 224)
    assert info["slot_names"][6:] == ["SAG_FS_Z100J", "SAG_FS_Z80J"]
    assert m[6] == m[0] and m[7] == m[0]               # masks copy the base slot
    base6, _, _ = slots.build_study_tensor(os.path.join(IMAGES, u), series_df, index=indexes[u])
    assert np.array_equal(x[:6], base6)                # base slots untouched by the spec
    # the 80 mm joint view is a centre sub-window of the 100 mm joint view
    r = int(round(224 * 80.0 / 100.0))
    o = (224 - r) // 2
    b = x[6][1, 1].astype(np.float32)[o:o + r, o:o + r]
    up = cv2.resize(b, (224, 224), interpolation=cv2.INTER_LINEAR)
    assert np.corrcoef(up.ravel(), x[7][1, 1].astype(np.float32).ravel())[0, 1] > 0.9
    # builder smoke on 6 studies; spec passed as two repeated flags
    out = tmp_path / "spec_cache"
    _run_builder(out, 6, extra=["--zoom-spec", "SAG_FS:100:joint", "--zoom-spec", "SAG_FS:80:joint"])
    cache = slots.SlotCache(str(out), "train")
    assert cache.S == 8 and cache.slot_names[6:] == ["SAG_FS_Z100J", "SAG_FS_Z80J"]
    assert cache.zoom_spec == [("SAG_FS", 100.0, "joint"), ("SAG_FS", 80.0, "joint")]
    assert cache.N == 6 and cache.done.all()
    xc, mc = cache[cache.row_of(u)]
    assert np.array_equal(np.asarray(xc), x) and np.array_equal(np.asarray(mc), m)  # bit-identical
    index = json.loads((out / "train_index.json").read_text())
    assert index["zoom_spec"] == [["SAG_FS", 100.0, "joint"], ["SAG_FS", 80.0, "joint"]]
    assert index["slot_names"] == cache.slot_names
    # resume with the comma form of the same spec: nothing to do
    log2 = _run_builder(out, 6, extra=["--zoom-spec", "SAG_FS:100:joint,SAG_FS:80:joint"])
    assert "todo: 0 new studies (6 already done)" in log2
    # a different spec on the same cache is refused
    r2 = subprocess.run([sys.executable, BUILDER, "--data-dir", DATA, "--out", str(out),
                         "--limit", "1", "--zoom-spec", "SAG_FS:80:joint"],
                        capture_output=True, text=True, timeout=120)
    assert r2.returncode != 0 and "use a new --out" in (r2.stdout + r2.stderr)
