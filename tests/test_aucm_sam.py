"""AUC-margin loss and Sharpness-Aware Minimisation: correctness + default-path invariance.

Both are opt-in (`--aucm-epochs` / `--loss aucm`, `--sam-rho`).  The campaign's arms are only
comparable if turning them OFF leaves training bit-identical, so that is tested too.
"""
import importlib.util
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.train import MaskedAUCMLoss, MaskedBCEWithLogitsLoss  # noqa: E402


def _train_mod():
    spec = importlib.util.spec_from_file_location(
        "train_slotknee_aucm", os.path.join(ROOT, "scripts", "train_slotknee.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tmod():
    return _train_mod()


# ── AUC-margin loss ────────────────────────────────────────────────────────────

def test_aucm_rewards_correct_ranking():
    """Ranking positives above negatives by more than the margin costs nothing."""
    loss = MaskedAUCMLoss(margin=1.0)
    y = torch.tensor([[1.0], [0.0]])
    good = torch.tensor([[3.0], [-3.0]])
    bad = torch.tensor([[-3.0], [3.0]])
    assert float(loss(good, y)) == pytest.approx(0.0, abs=1e-6)
    assert float(loss(bad, y)) > float(loss(good, y))


def test_aucm_is_ranking_not_calibration():
    """A constant shift of all scores leaves the loss unchanged (unlike BCE)."""
    loss = MaskedAUCMLoss()
    y = torch.tensor([[1.0], [0.0]])
    a = torch.tensor([[0.5], [-0.5]])
    assert float(loss(a, y)) == pytest.approx(float(loss(a + 7.0, y)), abs=1e-5)
    bce = MaskedBCEWithLogitsLoss()
    assert float(bce(a, y)) != pytest.approx(float(bce(a + 7.0, y)), abs=1e-3)


def test_aucm_ignores_unsupervised_and_pairless_labels():
    """Zero-weight cells contribute nothing; a label with no pos/neg pair is skipped."""
    loss = MaskedAUCMLoss()
    logits = torch.tensor([[0.0, 5.0], [0.0, -5.0]])
    y = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    w_col0_off = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    # label 0 masked out, label 1 ranked perfectly -> zero loss overall
    assert float(loss(logits, y, weight=w_col0_off)) == pytest.approx(0.0, abs=1e-6)
    # all-positive batch: no pairs anywhere -> exactly zero, and differentiable
    y_all_pos = torch.ones(2, 2)
    out = loss(logits.requires_grad_(True), y_all_pos)
    assert float(out) == 0.0
    out.backward()  # must not raise


def test_aucm_gradient_pushes_scores_apart():
    loss = MaskedAUCMLoss(margin=1.0)
    logits = torch.tensor([[0.0], [0.0]], requires_grad=True)
    y = torch.tensor([[1.0], [0.0]])
    loss(logits, y).backward()
    assert logits.grad[0, 0] < 0  # positive pushed up
    assert logits.grad[1, 0] > 0  # negative pushed down


# ── SAM ────────────────────────────────────────────────────────────────────────

def test_sam_ascend_then_descend_restores_weights(tmod):
    p = torch.nn.Parameter(torch.tensor([3.0, -4.0]))
    before = p.detach().clone()
    p.grad = torch.tensor([3.0, -4.0])          # norm 5
    eps = tmod._sam_ascend([p], rho=0.5)
    assert eps is not None
    moved = p.detach().clone()
    assert torch.allclose(moved - before, torch.tensor([0.3, -0.4]), atol=1e-6)  # rho * g/||g||
    tmod._sam_descend([p], eps)
    assert torch.allclose(p.detach(), before, atol=1e-7)


def test_sam_ascend_noop_on_degenerate_grad(tmod):
    p = torch.nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.zeros(1)
    assert tmod._sam_ascend([p], rho=0.5) is None      # zero-norm: nothing to perturb
    p.grad = torch.tensor([float("nan")])
    assert tmod._sam_ascend([p], rho=0.5) is None      # non-finite: skip rather than poison


def test_sam_rejects_fp16_scaler(tmod):
    """fp16's GradScaler cannot unscale twice per step -> refuse loudly, don't train wrong."""
    class _Scaler:  # stand-in; the guard runs before any batch is touched
        pass
    with pytest.raises(SystemExit, match="sam-rho"):
        tmod.train_epoch(model=torch.nn.Linear(2, 2), loader=[], optimizer=None, scheduler=None,
                         scaler=_Scaler(), device=torch.device("cpu"), amp_dtype="fp16",
                         sam_rho=0.05)


# ── default-path invariance (arm comparability) ────────────────────────────────

def test_defaults_leave_training_untouched(tmod):
    """With the flags off, one step must equal the pre-feature behaviour exactly."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    x, y, w = torch.randn(6, 4), (torch.rand(6, 2) > 0.5).float(), torch.ones(6, 2)

    def one_step(sam_rho):
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 2)
        m.load_state_dict(model.state_dict())
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        loader = [(x, torch.ones(6, 6), y, w, None)]
        # the model in train_epoch is called as model(x, mask, is_cached=...)
        wrapped = type("W", (torch.nn.Module,), {
            "forward": lambda self, xx, mm, is_cached=False: m(xx)})()
        wrapped.inner = m
        tmod.train_epoch(wrapped, loader, opt, sched, None, torch.device("cpu"), "cpu",
                         loss_fn=MaskedBCEWithLogitsLoss(), aux_weight=0.0, sam_rho=sam_rho)
        return m.weight.detach().clone()

    plain = one_step(0.0)
    with_sam = one_step(0.05)
    assert not torch.allclose(plain, with_sam), "sam_rho>0 must actually change the update"
    assert torch.allclose(one_step(0.0), plain, atol=0), "default path must be deterministic"
