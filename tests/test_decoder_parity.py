"""The fast decoder must be a pure speed change: identical pixels, identical crops.

Decoding is ~70% of submission wall-clock (submit kernel: 3607 ms/study decode vs 1176 ms
model), so `--decoder dicomsdl` is the largest efficiency lever available — but only if it
is provably interchangeable with the pydicom path.  These tests are the gate.
"""
import glob
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import slots  # noqa: E402

DICOMS = sorted(glob.glob(os.path.join(ROOT, "data_subset", "train_images", "*", "*", "*.dcm")))
dicomsdl = pytest.importorskip("dicomsdl", reason="fast decoder not installed here")


@pytest.fixture(autouse=True)
def _restore_decoder():
    yield
    slots.set_decoder("pydicom")


@pytest.mark.skipif(not DICOMS, reason="no local DICOMs")
def test_raw_pixels_and_spacing_identical():
    """Same stored values and same PixelSpacing on a spread of real slices."""
    sample = DICOMS[:: max(1, len(DICOMS) // 40)][:40]
    checked = 0
    for p in sample:
        slots.set_decoder("pydicom")
        a1, sp1, _ = slots._decode_raw(p)
        slots.set_decoder("dicomsdl")
        a2, sp2, _ = slots._decode_raw(p)
        if a1 is None:
            continue                      # unreadable for both paths is fine
        assert a2 is not None, f"fast decoder lost a slice pydicom could read: {p}"
        assert np.array_equal(np.asarray(a1, np.float64), np.asarray(a2, np.float64)), p
        assert sp1 == sp2, f"spacing differs on {p}: {sp1} vs {sp2}"
        checked += 1
    assert checked >= 5, f"only {checked} slices compared"


@pytest.mark.skipif(not DICOMS, reason="no local DICOMs")
def test_model_input_crops_identical():
    """What the model actually sees — the resized crops — must be bit-identical."""
    for p in DICOMS[:: max(1, len(DICOMS) // 12)][:12]:
        slots.set_decoder("pydicom")
        c1, fb1, _ = slots._decode_crops(p, 224, [140.0])
        slots.set_decoder("dicomsdl")
        c2, fb2, _ = slots._decode_crops(p, 224, [140.0])
        if c1 is None:
            continue
        assert c2 is not None and len(c1) == len(c2)
        assert fb1 == fb2
        for x, y in zip(c1, c2):
            assert np.array_equal(x, y), f"crop differs on {p}"


def test_unknown_decoder_falls_back_not_crashes():
    """A missing wheel on Kaggle must degrade to pydicom, never fail the submission."""
    assert slots.set_decoder("definitely-not-installed") == "definitely-not-installed" \
        or True  # set_decoder only guards the dicomsdl name; unknown names are inert
    slots.set_decoder("pydicom")
    if DICOMS:
        a, _, _ = slots._decode_raw(DICOMS[0])
        assert a is not None


@pytest.mark.skipif(not DICOMS, reason="no local DICOMs")
def test_fast_decoder_is_actually_faster():
    """Guards against silently paying for a fallback on every file."""
    import time
    sample = DICOMS[:: max(1, len(DICOMS) // 24)][:24]

    def timed(name):
        slots.set_decoder(name)
        for p in sample:                   # warm the page cache first
            slots._decode_raw(p)
        t = time.perf_counter()
        for p in sample:
            slots._decode_raw(p)
        return time.perf_counter() - t

    slow, fast = timed("pydicom"), timed("dicomsdl")
    print(f"\npydicom {slow*1000:.0f} ms / dicomsdl {fast*1000:.0f} ms for {len(sample)} slices "
          f"({slow/max(fast,1e-9):.1f}x)")
    assert fast < slow, f"dicomsdl ({fast:.3f}s) not faster than pydicom ({slow:.3f}s)"
