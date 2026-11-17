"""`BestValLossCheckpoint`: keep the weights from the best validation epoch.

The reason it exists is the lesson that cost a recorded run its best result: a training run that
only saves the LAST model loses its peak when the metric degrades afterwards. So the callback has
to save on every new best, never on a regression, never on the sanity-check pass (which would
write an untrained model as the "best"), and never on a missing or NaN metric. It also has to
survive a resume without overwriting a better checkpoint with a worse one.

It is driven here with a stub trainer and module, so the logic is tested without a Lightning run.
"""

import math
import os

import pytest
import pytorch_lightning as pl

from defog_adapt.core import BestValLossCheckpoint


class FakeTrainer:
    def __init__(self, metrics, epoch=0, sanity=False):
        self.callback_metrics = metrics
        self.current_epoch = epoch
        self.sanity_checking = sanity


class FakeModule:
    def __init__(self):
        self.saved = []

    def save(self, path):
        self.saved.append(path)
        return path + ".ckpt"


def feed(cb, module, values, key="val/loss", start_epoch=0):
    for k, v in enumerate(values):
        cb.on_validation_epoch_end(FakeTrainer({key: v}, epoch=start_epoch + k), module)


def make(tmp_path, **kw):
    kw.setdefault("verbose", False)
    return BestValLossCheckpoint(str(tmp_path / "ckpt"), **kw)


def test_it_saves_on_a_first_value_and_records_where(tmp_path):
    cb, m = make(tmp_path), FakeModule()
    feed(cb, m, [0.9])
    assert m.saved == [str(tmp_path / "ckpt" / "best_model")]
    assert cb.best == pytest.approx(0.9) and cb.best_epoch == 0
    assert cb.saved_path == m.saved[0] + ".ckpt"
    assert os.path.isdir(tmp_path / "ckpt"), "the checkpoint directory must be created"


def test_it_saves_on_every_new_best_and_only_then(tmp_path):
    cb, m = make(tmp_path), FakeModule()
    feed(cb, m, [0.9, 0.7, 0.8, 0.95, 0.6, 0.6, 0.7])
    # improvements at epochs 0 (0.9), 1 (0.7), 4 (0.6); the tie at epoch 5 is not an improvement
    assert len(m.saved) == 3
    assert cb.best == pytest.approx(0.6) and cb.best_epoch == 4


def test_a_regression_after_the_peak_never_overwrites_it(tmp_path):
    """The failure this callback exists to prevent: the peak is early, the metric degrades, and
    only the last model survives."""
    cb, m = make(tmp_path), FakeModule()
    feed(cb, m, [0.5] + [1.0 + 0.1 * k for k in range(10)])
    assert len(m.saved) == 1 and cb.best == pytest.approx(0.5) and cb.best_epoch == 0


def test_max_mode_keeps_the_highest(tmp_path):
    cb, m = make(tmp_path, monitor="val/validity", mode="max"), FakeModule()
    feed(cb, m, [0.4, 0.9, 0.5, 0.95], key="val/validity")
    assert cb.best == pytest.approx(0.95) and cb.best_epoch == 3 and len(m.saved) == 3


def test_the_sanity_check_pass_never_saves(tmp_path):
    """Lightning runs validation on two batches before epoch 0; saving then would write an
    untrained model as the 'best'."""
    cb, m = make(tmp_path), FakeModule()
    cb.on_validation_epoch_end(FakeTrainer({"val/loss": 0.01}, sanity=True), m)
    assert m.saved == [] and cb.best == math.inf


@pytest.mark.parametrize("metrics", [{}, {"other": 0.1}, {"val/loss": float("nan")}])
def test_a_missing_or_nan_metric_is_ignored(tmp_path, metrics):
    cb, m = make(tmp_path), FakeModule()
    cb.on_validation_epoch_end(FakeTrainer(metrics), m)
    assert m.saved == [] and cb.best == math.inf


def test_a_bad_mode_is_refused(tmp_path):
    with pytest.raises(AssertionError, match="mode"):
        make(tmp_path, mode="sideways")


def test_the_best_survives_a_resume_and_a_worse_value_does_not_overwrite_it(tmp_path):
    first, m1 = make(tmp_path), FakeModule()
    feed(first, m1, [0.9, 0.5])
    state = first.state_dict()

    resumed, m2 = make(tmp_path), FakeModule()
    resumed.load_state_dict(state)
    assert resumed.best == pytest.approx(0.5) and resumed.best_epoch == 1
    feed(resumed, m2, [0.6, 0.55], start_epoch=2)
    assert m2.saved == [], "a worse value after a resume overwrote the better checkpoint"
    feed(resumed, m2, [0.4], start_epoch=4)
    assert len(m2.saved) == 1 and resumed.best == pytest.approx(0.4)


def test_it_hooks_the_epoch_end_not_the_validation_end():
    """Load-bearing: EMACallback swaps its shadow weights in at on_validation_start and back out
    at on_validation_end. on_validation_epoch_end fires strictly between, so the weights saved
    are the EMA ones whatever the callback order; on_validation_end would make the saved weights
    depend on which callback Lightning happened to call first."""
    assert BestValLossCheckpoint.on_validation_epoch_end is not pl.Callback.on_validation_epoch_end
    assert BestValLossCheckpoint.on_validation_end is pl.Callback.on_validation_end
