"""Fitting a recipe's batch size onto a smaller GPU (``examples/training__e1.py --MICRO_BATCH_SIZE``).

The recorded ZINC recipe uses 256 graphs per optimizer step, which needs more than the 24 GB of a consumer GPU.
Putting 128 on the GPU and accumulating two micro-batches per step keeps the optimizer, the number of steps per
epoch and the learning-rate schedule as they were. The one thing that does not carry over by itself is the EMA:
its callback updates after every batch it is handed, i.e. every micro-batch, which would silently shorten the
average. These tests are about that arithmetic; they need neither data nor PyComex.
"""

import sys
from pathlib import Path

import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples/ is not part of an installed wheel")


@pytest.fixture(scope="module")
def common():
    sys.path.insert(0, str(EXAMPLES))
    try:
        import _common
    finally:
        sys.path.remove(str(EXAMPLES))
    return _common


def test_without_a_micro_batch_nothing_changes(common):
    assert common.accumulation_plan(256, None, 0.9999) == (256, 1, 0.9999)
    assert common.accumulation_plan(256, 0, 0.9999) == (256, 1, 0.9999)
    assert common.accumulation_plan(256, 256, 0.9999) == (256, 1, 0.9999)


def test_a_smaller_micro_batch_accumulates_to_the_recorded_batch(common):
    for micro, accumulate in ((128, 2), (64, 4), (32, 8)):
        got_micro, got_accumulate, _ = common.accumulation_plan(256, micro, 0.9999)
        assert (got_micro, got_accumulate) == (micro, accumulate)
        assert got_micro * got_accumulate == 256


@pytest.mark.parametrize("micro", [100, 512, -4, 255])
def test_a_micro_batch_that_does_not_divide_the_batch_is_refused(common, micro):
    """A remainder would train on a different number of graphs per step than the recipe says."""
    with pytest.raises(ValueError, match="must divide"):
        common.accumulation_plan(256, micro, 0.9999)


def test_a_disabled_ema_stays_disabled(common):
    assert common.accumulation_plan(256, 64, None)[2] is None
    assert common.accumulation_plan(256, 64, 0)[2] == 0


@pytest.mark.parametrize("accumulate", [2, 4, 8])
def test_the_real_ema_callback_decays_once_per_optimizer_step_not_once_per_micro_batch(common, accumulate):
    """The property that matters, with the library's own callback: parameters only move at optimizer steps, so
    ``accumulate`` micro-batch updates between two steps must add up to exactly one update at the recorded decay.
    Compared with the per-micro-batch decay left uncompensated, which averages ``accumulate`` times too short."""
    from defog_adapt.core import EMACallback

    decay = 0.9
    _, got_accumulate, per_micro = common.accumulation_plan(256, 256 // accumulate, decay)
    assert got_accumulate == accumulate

    def shadow_after(callback, updates, value):
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(0.0)
        callback.on_fit_start(None, module)                  # the shadow starts at 0
        with torch.no_grad():
            module.weight.fill_(value)                       # one optimizer step moved the parameter
        for _ in range(updates):
            callback.on_train_batch_end(None, module, None, None, 0)
        return float(callback.shadow["weight"])

    reference = shadow_after(EMACallback(decay=decay), 1, 1.0)                  # the recorded recipe: one update
    accumulated = shadow_after(EMACallback(decay=per_micro), accumulate, 1.0)   # ``accumulate`` micro-batches
    uncompensated = shadow_after(EMACallback(decay=decay), accumulate, 1.0)
    assert accumulated == pytest.approx(reference, abs=1e-6)             # the shadow is float32
    assert abs(uncompensated - reference) > 0.05, "premise: skipping the compensation must visibly change the EMA"
