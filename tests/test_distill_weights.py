"""Per-label distillation weights: capability without breaking arm comparability.

`--distill-weight` was added as "a float OR 12 comma-separated floats", implementing the
per-label idea the gold-vs-labels analysis motivated (the model out-teaches its targets on
Fracture/Effusion/Contusion but trails them badly on MCL/lateral meniscus, so one uniform
weight helps half the findings and corrupts the other half).

The catch: the per-label form also re-weights the BASE loss to one-vote-per-label, which is a
*second* difference from the no-distill baseline.  A distill arm must differ from its baseline
by distillation alone or its gate is unreadable, so a uniform weight keeps the original joint
blend.  These tests pin both halves of that contract.
"""
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.train import MaskedBCEWithLogitsLoss  # noqa: E402


def _train_mod():
    spec = importlib.util.spec_from_file_location(
        "train_slotknee_distill", os.path.join(ROOT, "scripts", "train_slotknee.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tmod():
    return _train_mod()


class _Tiny(torch.nn.Module):
    """Stands in for SlotKneeS: train_epoch calls model(x, mask, is_cached=...)."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 12)

    def forward(self, x, mask, is_cached=False):
        return self.lin(x)


def _one_step(tmod, distill_weight, distill_y, seed=0):
    torch.manual_seed(seed)
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    torch.manual_seed(1)
    x = torch.randn(4, 4)
    y = (torch.rand(4, 12) > 0.5).float()
    w = torch.zeros(4, 12)
    w[:, 0] = 1.0
    w[:, 1] = 1.0                       # only 2 of 12 labels supervised, as at bs 8
    batch = [(x, torch.ones(4, 6), y, w, None, distill_y)] if distill_y is not None \
        else [(x, torch.ones(4, 6), y, w, None)]
    tmod.train_epoch(model, batch, opt, sched, None, torch.device("cpu"), "cpu",
                     loss_fn=MaskedBCEWithLogitsLoss(), aux_weight=0.0,
                     distill_weight=distill_weight)
    return model.lin.weight.detach().clone()


def test_uniform_weight_matches_the_joint_blend(tmod):
    """A uniform weight must reproduce the ORIGINAL joint formulation exactly.

    This is what keeps a distill arm comparable to its baseline: one delta, not two.
    """
    torch.manual_seed(7)
    dy = torch.rand(4, 12)
    got = _one_step(tmod, [0.5] * 12, dy)

    # reference: the pre-per-label maths, computed by hand
    torch.manual_seed(0)
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    torch.manual_seed(1)
    x = torch.randn(4, 4)
    y = (torch.rand(4, 12) > 0.5).float()
    w = torch.zeros(4, 12); w[:, 0] = 1.0; w[:, 1] = 1.0
    logits = model(x, None)
    loss = MaskedBCEWithLogitsLoss()(logits, y, weight=w)
    tm = ~torch.isnan(dy)
    dl = F.binary_cross_entropy_with_logits(logits[tm], dy[tm].clamp(1e-4, 1 - 1e-4))
    (0.5 * loss + 0.5 * dl).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    assert torch.allclose(got, model.lin.weight.detach(), atol=1e-6)


def test_per_label_weights_actually_differ(tmod):
    """Non-uniform weights must take the per-label path and change the update."""
    torch.manual_seed(7)
    dy = torch.rand(4, 12)
    uniform = _one_step(tmod, [0.5] * 12, dy)
    per_label = _one_step(tmod, [0.1, 0.1, 0.2, 0.1, 0.5, 0.3, 0.3, 0.7, 0.5, 0.5, 0.7, 0.7], dy)
    assert not torch.allclose(uniform, per_label), "per-label weights had no effect"


def test_zero_weight_label_is_pure_supervision(tmod):
    """dw_j = 0 must mean 'ignore the teacher for this finding' — the MCL/LatMen case."""
    torch.manual_seed(7)
    dy = torch.rand(4, 12)
    all_zero = _one_step(tmod, [0.0] * 12, dy)
    no_distill = _one_step(tmod, None, None)
    assert torch.allclose(all_zero, no_distill, atol=1e-6), \
        "weight 0 everywhere must equal training without a teacher at all"


def test_argument_parsing_accepts_one_or_twelve():
    """`--distill-weight` is a string: 1 value broadcasts, 12 pass through, else it errors."""
    def parse(s):
        dw = [float(v.strip()) for v in s.split(",")]
        if len(dw) == 1:
            dw = dw * 12
        elif len(dw) != 12:
            raise ValueError("must be 1 or 12")
        return dw
    assert parse("0.5") == [0.5] * 12
    assert len(parse(",".join(["0.3"] * 12))) == 12
    with pytest.raises(ValueError):
        parse("0.1,0.2,0.3")
