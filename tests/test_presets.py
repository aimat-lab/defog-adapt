"""The recorded recipes (`defog_adapt.presets`) and the package version."""
import importlib.metadata
import inspect
import os
import re
from pathlib import Path

import pytest
import torch

import defog_adapt
from defog_adapt.core import DeFoGModel, Sampler
from defog_adapt.presets import E1_MODEL, E1_SAMPLING


def test_every_preset_key_is_a_real_constructor_argument():
    """A typo or a renamed argument would otherwise surface only when someone builds a model."""
    accepted = set(inspect.signature(DeFoGModel.__init__).parameters)
    assert set(E1_MODEL) <= accepted, sorted(set(E1_MODEL) - accepted)


def test_the_sampling_preset_is_accepted_by_the_sampler():
    accepted = set(inspect.signature(Sampler.__init__).parameters)
    assert set(E1_SAMPLING) <= accepted, sorted(set(E1_SAMPLING) - accepted)


def test_a_model_built_from_the_preset_records_it(small_dataloader):
    """Shrunk to a toy size and without the molecular features (they need a vocabulary), but
    every other preset value must land on the model unchanged."""
    shrink = dict(n_layers=2, hidden_dim=32, hidden_mlp_dim=64, n_heads=2, molecular_features=False)
    model = DeFoGModel.from_dataloader(small_dataloader, **{**E1_MODEL, **shrink})
    for key, want in E1_MODEL.items():
        if key not in shrink:
            assert model.hparams[key] == want, f"{key}: {model.hparams[key]!r} != {want!r}"


def test_the_preset_is_what_the_recorded_checkpoints_were_trained_with():
    """The evidence for E1_MODEL: the hyperparameters stored in the checkpoints behind the
    recorded ZINC and MOSES results. Opt-in, because those checkpoints are not distributed."""
    root = os.environ.get("DEFOG_ADAPT_CKPTS")
    paths = [Path(root) / name for name in ("zinc_e1_seed42_kek.ckpt",
                                            "moses_kek_seed44/best_model.ckpt")] if root else []
    paths = [p for p in paths if p.exists()]
    if not paths:
        pytest.skip("set DEFOG_ADAPT_CKPTS to a directory with the E1 checkpoints")
    for path in paths:
        hp = torch.load(path, map_location="cpu", weights_only=False)["hyper_parameters"]
        for key, want in E1_MODEL.items():
            assert hp[key] == want, f"{path.name}: {key} is {hp[key]!r}, the preset says {want!r}"


def test_the_package_version_is_the_installed_version_and_matches_pyproject():
    assert defog_adapt.__version__ == importlib.metadata.version("defog-adapt")
    pyproject = Path(defog_adapt.__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.exists():
        declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), flags=re.M).group(1)
        assert defog_adapt.__version__ == declared
    import defog_adapt.core as core
    assert not hasattr(core, "__version__"), "the version must have one source"
