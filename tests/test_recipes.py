"""The recipes under ``examples/`` run end to end, in their smoke configuration.

A recipe that is not run is a recipe that has already rotted: it calls the library through names and
arguments that can change under it, and nothing else in the suite touches it. These tests run each one
as a subprocess in the mode its author wrote for exactly this (``--__TESTING__ True`` for the PyComex
recipes, tiny settings for the scripts), on a copy of ``examples/`` so that nothing is written into the
repository.

They are OPT-IN, because they need what a clean checkout does not have:

* ``DEFOG_ADAPT_HOME``   a directory with the reference data (``zinc250k/`` and ``moses/``);
* ``DEFOG_ADAPT_CKPTS``  a directory with ``zinc_e1_seed42_kek.ckpt`` (an E1 base), ``zinc_kek_base.ckpt``
                         (the base the adapters sit on), ``logp_xattn_adapter.ckpt`` and
                         ``heads_kek/logp_head.ckpt`` + ``heads_kek/logp_head_size.ckpt``;
* PyComex (``pip install pycomex``) for the PyComex recipes.

A smoke run says the pipeline EXECUTES and its checks pass. It does not say the models are any good:
the molecules a two-epoch, five-step run produces are mostly invalid.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DATA = os.environ.get("DEFOG_ADAPT_HOME")
CKPTS = os.environ.get("DEFOG_ADAPT_CKPTS")

pytestmark = pytest.mark.slow


def have(*relative):
    return bool(CKPTS) and all((Path(CKPTS) / r).exists() for r in relative)


needs_data = pytest.mark.skipif(
    not DATA or not ((Path(DATA) / "zinc250k").is_dir()), reason="set DEFOG_ADAPT_HOME to the reference data")
needs_pycomex = pytest.mark.skipif(importlib.util.find_spec("pycomex") is None,
                                   reason="pip install pycomex")
needs_examples = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples/ is not in an installed wheel")


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """A private copy of examples/, so the PyComex results land in a temp directory."""
    target = tmp_path_factory.mktemp("recipes") / "examples"
    shutil.copytree(EXAMPLES, target, ignore=shutil.ignore_patterns("results", "__pycache__"))
    return target


def run(workdir, script, *args, timeout=1500):
    env = {**os.environ, "OMP_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-W", "ignore", str(workdir / script), *map(str, args)],
                          capture_output=True, text=True, timeout=timeout, env=env, cwd=workdir)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-3000:]
    return out


def ckpt(name):
    return str(Path(CKPTS) / name)


@needs_examples
@needs_data
@needs_pycomex
@pytest.mark.parametrize("dataset", ["zinc", "moses"])
def test_the_e1_recipe_trains_and_evaluates(workdir, dataset):
    out = run(workdir, "training__e1.py", "--DATASET", f"'{dataset}'", "--__TESTING__", "True")
    # The recipe's own last words, not PyComex's banner (which differs between PyComex versions).
    assert "validity (relaxed, largest fragment)" in out
    assert "Done. FCD and the other benchmark metrics are not computed here." in out


@needs_examples
@needs_data
@needs_pycomex
def test_the_e1_recipe_trains_a_conditional_model_and_keeps_the_statistics_it_needs(workdir):
    """``--CONDITION`` is the same recipe plus a condition. What it must leave behind: a model that really has
    a condition encoder, and the mean and std that standardised the labels, because the model stores neither
    and cannot be sampled without them."""
    import json

    from defog_adapt.core import DeFoGModel

    out = run(workdir, "training__e1.py", "--DATASET", "'zinc'", "--CONDITION", "'logp'", "--__TESTING__", "True")
    assert "ZINC CONDITIONAL ON logp" in out and "Done. FCD and the other benchmark metrics are not computed here." in out

    files = sorted((workdir / "results" / "training__e1").glob("*/condition.json"))
    assert len(files) == 1, files                      # the unconditional run, in the same directory, wrote none
    stats = json.loads(files[0].read_text(encoding="utf-8"))
    assert stats["property"] == "logp" and stats["property_from"] == "decoded" and stats["std"] > 0
    model = DeFoGModel.load(str(files[0].parent / "model"))
    assert model.cond_dim == 1 and model.cond_drop_prob == pytest.approx(stats["cond_drop_prob"])
    assert hasattr(model, "cond_encoder")


@needs_examples
@needs_pycomex
@pytest.mark.parametrize("args, message", [
    (("--CONDITION", "'nonsense'"), "unknown property"),
    (("--DATASET", "'moses'", "--CONDITION", "'logp'"), "ZINC only"),
])
def test_the_conditional_option_refuses_what_it_cannot_do_before_loading_anything(workdir, args, message):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-W", "ignore", str(workdir / "training__e1.py"), *args,
                           "--__TESTING__", "True"], capture_output=True, text=True, timeout=600, env=env, cwd=workdir)
    assert proc.returncode != 0 and message in (proc.stdout + proc.stderr)


@needs_examples
@needs_data
@needs_pycomex
@pytest.mark.skipif(not have("zinc_e1_seed42_kek.ckpt"), reason="needs an E1 base checkpoint")
def test_the_sanity_rl_recipe_runs_and_never_overwrites_its_base(workdir, tmp_path):
    base = ckpt("zinc_e1_seed42_kek")
    out = run(workdir, "rl_sanity__zinc.py", "--__TESTING__", "True", "--BASE_CKPT", f"'{base}'",
              "--OUT_CKPT_DIR", f"'{tmp_path / 'rl_out'}'")
    assert "vocabulary check OK" in out and "FCD and scaffold similarity are NOT computed here" in out
    assert (tmp_path / "rl_out" / "best_model.ckpt").exists()


@needs_examples
@needs_data
@needs_pycomex
@pytest.mark.skipif(not have("zinc_e1_seed42_kek.ckpt"), reason="needs an E1 base checkpoint")
def test_the_sanity_rl_recipe_refuses_to_write_into_the_base_directory(workdir):
    base = Path(ckpt("zinc_e1_seed42_kek"))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", str(workdir / "rl_sanity__zinc.py"), "--__TESTING__", "True",
         "--BASE_CKPT", f"'{base}'", "--OUT_CKPT_DIR", f"'{base.parent}'"],
        capture_output=True, text=True, timeout=600, env=env, cwd=workdir)
    assert proc.returncode != 0 and "refusing to overwrite" in (proc.stdout + proc.stderr)


@needs_examples
@needs_data
@needs_pycomex
@pytest.mark.skipif(not have("zinc_kek_base.ckpt"), reason="needs the adapter base checkpoint")
def test_the_adapter_training_recipe_trains_saves_and_evaluates(workdir):
    out = run(workdir, "adapter_training__zinc.py", "--__TESTING__", "True", "--BASE_CKPT",
              f"'{ckpt('zinc_kek_base')}'")
    assert "saved adapter ->" in out and "PROBE(w=" in out and "\nDone." in out


@needs_examples
@needs_data
@pytest.mark.skipif(not have("zinc_kek_base.ckpt"), reason="needs the adapter base checkpoint")
def test_the_property_head_script_fits_a_head_and_a_size_model(workdir, tmp_path):
    out_prefix = tmp_path / "heads" / "logp_head"
    run(workdir, "fit_property_head__zinc.py", "--base", ckpt("zinc_kek_base"), "--property", "logp",
        "--out", out_prefix, "--limit", 600, "--holdout", 100, "--epochs", 2, "--size-epochs", 3,
        "--with-size-model", timeout=900)
    from defog_adapt.core import LearnedSizeDistribution, PropertyHead

    head = PropertyHead.load(str(out_prefix) + ".ckpt")
    size = LearnedSizeDistribution.load(str(out_prefix) + "_size.ckpt")
    assert head is not None and size.property_name == "logp" and size.property_from == "decoded"


STEER_ASSETS = ("zinc_kek_base.ckpt", "logp_xattn_adapter.ckpt", "heads_kek/logp_head.ckpt",
                "heads_kek/logp_head_size.ckpt")


@needs_examples
@needs_data
@pytest.mark.skipif(not have(*STEER_ASSETS), reason="needs the logP base, adapter, head and size model")
@pytest.mark.parametrize("method", ["adapter", "fk"])
def test_the_steering_assembly_samples_and_scores(workdir, method):
    out = run(workdir, "steer_zinc.py", "--base", ckpt("zinc_kek_base"), "--property", "logp",
              "--adapter", ckpt("logp_xattn_adapter.ckpt"), "--head", ckpt("heads_kek/logp_head.ckpt"),
              "--size", ckpt("heads_kek/logp_head_size.ckpt"), "--method", method, "--target", 4.0,
              "--n", 6, "--steps", 8, "--device", "cpu")
    assert "vocabulary check OK" in out and "samples in" in out


@needs_examples
@needs_data
@pytest.mark.skipif(not have(*STEER_ASSETS), reason="needs the logP base, adapter, head and size model")
def test_the_steering_assembly_evaluates_against_real_validation_targets(workdir):
    out = run(workdir, "steer_zinc.py", "--base", ckpt("zinc_kek_base"), "--property", "logp",
              "--adapter", ckpt("logp_xattn_adapter.ckpt"), "--head", ckpt("heads_kek/logp_head.ckpt"),
              "--size", ckpt("heads_kek/logp_head_size.ckpt"), "--method", "adapter", "--evaluate", 2,
              "--n", 4, "--steps", 8, "--device", "cpu")
    assert '"total_mae"' in out and '"validity"' in out


@needs_examples
@needs_data
@needs_pycomex
@pytest.mark.skipif(not have(*STEER_ASSETS), reason="needs the logP base, adapter, head and size model")
def test_the_adapter_rl_recipe_runs_with_a_head_reward_and_a_size_model(workdir):
    out = run(workdir, "adapter_rl_finetune__zinc.py", "--__TESTING__", "True",
              "--BASE_CKPT", f"'{ckpt('zinc_kek_base')}'",
              "--ADAPTER_CKPT", f"'{ckpt('logp_xattn_adapter.ckpt')}'",
              "--HEAD_CKPT", f"'{ckpt('heads_kek/logp_head.ckpt')}'",
              "--SIZE_MODEL_CKPT", f"'{ckpt('heads_kek/logp_head_size.ckpt')}'",
              "--PROPERTY", "'logp'")
    assert "operating point: rollout/scored/probe weight 1.5" in out
    assert "saved the RL adapter" in out and "\nDone." in out
