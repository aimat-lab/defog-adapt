"""The examples are the documented way to assemble the building blocks, so they have to keep running.

``quickstart_cpu.py`` is the one that needs no data, no checkpoints and no PyComex, so it is
exercised here (in its ``--smoke`` mode: a few seconds, checking that every stage executes and
reports finite numbers, NOT that the toy models steer well). The recipes that need the reference
data and PyComex are covered in ``test_recipes.py``.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def run_example(*args, timeout=600):
    return subprocess.run([sys.executable, "-W", "ignore", *map(str, args)], capture_output=True,
                          text=True, timeout=timeout)


@pytest.fixture(scope="module")
def quickstart():
    script = EXAMPLES / "quickstart_cpu.py"
    if not script.exists():
        pytest.skip("examples/ is not part of an installed wheel")
    proc = run_example(script, "--smoke")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc.stdout


def test_the_quickstart_runs_every_stage(quickstart):
    for stage in ("molecules", "base:", "adapter:", "size model gain", "steered towards", "done."):
        assert stage in quickstart, f"stage {stage!r} missing from the quickstart output"


def test_the_quickstart_reports_all_six_conditions_with_finite_numbers(quickstart):
    rows = [line for line in quickstart.splitlines()
            if line.startswith("adapter") and re.search(r"-?\d+\.\d\d\s*$", line)]
    assert len(rows) == 6, rows                      # (adapter, +size, +FK) x (low, high)
    for line in rows:
        numbers = re.findall(r"-?\d+\.\d+", line)
        assert len(numbers) >= 3 and all(abs(float(x)) < 1e3 for x in numbers), line


# ============================================================================================
# steer_zinc.py: the assembly's own logic, on tiny models (no data, no checkpoints, no PyComex)
# ============================================================================================
import importlib.util  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402


@pytest.fixture(scope="module")
def steer():
    script = EXAMPLES / "steer_zinc.py"
    if not script.exists():
        pytest.skip("examples/ is not part of an installed wheel")
    name = "steer_zinc_under_test"
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(name, script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module              # a dataclass in the script looks itself up here
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXAMPLES))
    yield module
    sys.modules.pop(name, None)


def fake_adapter(mean=2.0, std=1.0):
    return SimpleNamespace(cond_mean=torch.tensor([mean]), cond_std=torch.tensor([std]))


def fake_head(mean=2.0, std=1.0):
    return SimpleNamespace(prop_mean=torch.tensor(mean), prop_std=torch.tensor(std))


def spec(steer, *, head=None, size_model=None, adapter=None, target=1.0, name="logp"):
    return steer.Steered(name=name, adapter=adapter or fake_adapter(), head=head,
                         size_model=size_model, target=target, weight=1.5)


def test_a_head_trained_under_another_label_convention_is_refused(steer):
    """The real failure is a third of a standard deviation (source against decoded labels); 1% of one
    is the tolerance, so parts fit on slightly different subsets still pair."""
    steer.pair_check(spec(steer, head=fake_head(mean=2.004)))                 # 0.4% of a std: fine
    with pytest.raises(RuntimeError, match="label conventions"):
        steer.pair_check(spec(steer, head=fake_head(mean=2.317)))            # 0.317 std: refused


def test_the_size_model_pairing_check_is_scale_free(steer):
    seen = {}

    class Size:
        def check_compatible(self, adapter, tol):
            seen["tol"] = tol

    steer.pair_check(spec(steer, size_model=Size(), adapter=fake_adapter(std=4.0)))
    assert seen["tol"] == pytest.approx(0.04), "the tolerance must be 1% of the property's own std"


def test_the_feynman_kac_energy_is_standardised_by_the_heads_own_std(steer, small_model, small_dataloader):
    """The point of StandardisedEnergy: beta is dimensionless only if the energy is divided by the
    property's variance. Wiring the raw energy in would still run, and would make beta mean something
    different for every property."""
    from defog_adapt.core import CrossAttentionAdapter, AdaptedSampler, FeynmanKacSampler, StandardisedEnergy

    adapter = CrossAttentionAdapter.for_base(small_model, cond_dim=1, hidden=16, cond_mean=[2.0], cond_std=[1.0])
    head = fake_head(mean=2.0, std=1.16)
    head_spec = spec(steer, head=head, adapter=adapter)
    kwargs = dict(mode="product", ctmc=dict(sample_steps=4), beta=7.5, domain=None, atom_enc={}, bond_enc={})

    class Energy:                                          # stand-in so no molecule decoding is needed
        invalid = 1.0

        def __init__(self, *a, **k):
            pass

        def __call__(self, X1, E1, node_mask):
            return torch.zeros(X1.shape[0])

    original = steer.LearnedPropertyEnergy
    steer.LearnedPropertyEnergy = Energy
    try:
        fk = steer.build_sampler(small_model, [head_spec], method="fk", **kwargs)
        plain = steer.build_sampler(small_model, [head_spec], method="adapter", **kwargs)
    finally:
        steer.LearnedPropertyEnergy = original
    assert isinstance(fk, FeynmanKacSampler) and isinstance(fk.energy_fn, StandardisedEnergy)
    assert [t[1] for t in fk.energy_fn.terms] == [pytest.approx(1.16)], "the scale must be the head's std"
    assert fk.beta == 7.5, "--beta must reach the sampler (and 7.5 is not the preset's 20)"
    assert isinstance(plain, AdaptedSampler) and not isinstance(plain, FeynmanKacSampler)


def test_without_a_head_the_energy_falls_back_to_the_closed_form_property(steer, small_model):
    from defog_adapt.core import CrossAttentionAdapter, MoleculePropertyEnergy

    adapter = CrossAttentionAdapter.for_base(small_model, cond_dim=1, hidden=16, cond_mean=[2.0], cond_std=[1.5])
    fk = steer.build_sampler(small_model, [spec(steer, adapter=adapter)], method="fk", mode="product",
                             ctmc=dict(sample_steps=4), beta=20.0, domain=None, atom_enc={}, bond_enc={})
    energy, scale, _ = fk.energy_fn.terms[0]
    assert isinstance(energy, MoleculePropertyEnergy) and scale == pytest.approx(1.5)


def test_size_draws_one_model_as_it_is_and_several_as_a_product_of_experts(steer):
    from defog_adapt.core import ComposedSizeDistribution, fit_size_model

    torch.manual_seed(0)
    conditions = torch.randn(300)
    sizes = (10 + (conditions * 2).round().clamp(-3, 3)).long() + 5
    model, _ = fit_size_model(conditions, sizes, min_size=int(sizes.min()), max_size=int(sizes.max()),
                              hidden=8, layers=1, epochs=2, seed=0, property_name="logp")
    one = spec(steer, size_model=model, target=0.5)
    dist, condition = steer.size_draw([one], 7, "product")
    assert dist is model and condition.shape == (7, model.cond_dim) and float(condition[0, 0]) == 0.5

    two = [spec(steer, size_model=model, target=0.5), spec(steer, size_model=model, target=-0.5)]
    dist, condition = steer.size_draw(two, 7, "mean")
    assert isinstance(dist, ComposedSizeDistribution) and condition is None

    assert steer.size_draw([one, spec(steer, size_model=None)], 7, "product") == (None, None), \
        "one property without a size model means the marginal for the whole draw, never a half-conditioned one"
