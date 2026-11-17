"""The recorded recipes (`defog_adapt.presets`) and the package version."""
import importlib.metadata
import inspect
import os
import re
from pathlib import Path

import pytest
import torch

import defog_adapt
from defog_adapt.core import (
    AdaLNAdapter, AdapterGDPOTrainer, DeFoGModel, FeynmanKacSampler, GDPOTrainer, Sampler,
)
from defog_adapt.presets import (
    ADAPTER_ARCHITECTURE, ADAPTER_RL_TRAINER, ADAPTER_SAMPLING, ADAPTER_TRAINING, ADAPTER_WEIGHT,
    E1_MODEL, E1_SAMPLING, FK_STEERING, SANITY_RL_TRAINER,
)


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


# ============================================================ the steering / adapter / RL presets
def _params(callable_):
    return inspect.signature(callable_).parameters


def test_the_trainer_presets_are_arguments_the_trainers_accept():
    gdpo = set(_params(GDPOTrainer.__init__))
    assert set(SANITY_RL_TRAINER) <= gdpo, sorted(set(SANITY_RL_TRAINER) - gdpo)
    adapter = set(_params(AdapterGDPOTrainer.__init__)) | gdpo
    assert set(ADAPTER_RL_TRAINER) <= adapter, sorted(set(ADAPTER_RL_TRAINER) - adapter)
    assert "rollout_weight" not in ADAPTER_RL_TRAINER, \
        "rollout_weight must stay a required, per-property argument, not a default"


def test_the_trainer_presets_agree_with_the_defaults_they_restate():
    """The trainers' own defaults ARE the recorded recipe, and the presets restate them so a user
    can read the recipe in one place. If the two ever disagree, one of them is wrong."""
    gdpo = {k: v.default for k, v in _params(GDPOTrainer.__init__).items()
            if v.default is not inspect.Parameter.empty}
    for key, want in SANITY_RL_TRAINER.items():
        assert gdpo[key] == want, f"GDPOTrainer default for {key} is {gdpo[key]!r}, preset {want!r}"
    adapter = {k: v.default for k, v in _params(AdapterGDPOTrainer.__init__).items()
               if v.default is not inspect.Parameter.empty}
    for key, want in ADAPTER_RL_TRAINER.items():
        have = adapter.get(key, gdpo.get(key))
        if key in ("sample_steps", "eta", "subsample_steps", "reduction", "rollout_size"):
            continue            # set inside AdapterGDPOTrainer.__init__, not visible in the signature
        assert have == want, f"AdapterGDPOTrainer default for {key} is {have!r}, preset {want!r}"


def test_the_adapter_presets_are_arguments_the_adapter_and_sampler_accept():
    assert set(ADAPTER_ARCHITECTURE) <= set(_params(AdaLNAdapter.__init__))
    assert set(ADAPTER_SAMPLING) <= set(_params(Sampler.__init__))
    assert set(FK_STEERING) <= set(_params(FeynmanKacSampler.__init__))
    assert set(ADAPTER_TRAINING) == {"epochs", "batch_size", "lr", "grad_clip"}


def test_every_recorded_property_has_a_deployment_weight():
    assert ADAPTER_WEIGHT == {"logp": 1.5, "tpsa": 1.5, "sascore": 1.5, "qed": 2.0}


def test_the_fk_preset_is_the_dimensionless_recorded_schedule():
    assert FK_STEERING["beta"] == 20.0 and FK_STEERING["warmup_frac"] == 0.6
    assert FK_STEERING["ess_frac"] == 0.5 and FK_STEERING["resample_scheme"] == "residual"
    assert "resample_interval" not in FK_STEERING, "the interval is the sampler's own steps // 8"


def test_the_adapter_preset_rebuilds_the_recorded_adapter_exactly():
    """Opt-in: the architecture in the preset against the adapter that produced the logP result.
    Every parameter name and shape must match, which is stronger than a parameter count."""
    root = os.environ.get("DEFOG_ADAPT_CKPTS")
    base_path = Path(root) / "zinc_kek_base.ckpt" if root else None
    adapter_path = Path(root) / "logp_xattn_adapter.ckpt" if root else None
    if not (base_path and base_path.exists() and adapter_path.exists()):
        pytest.skip("set DEFOG_ADAPT_CKPTS to a directory with zinc_kek_base.ckpt and "
                    "logp_xattn_adapter.ckpt")
    base = DeFoGModel.load(str(base_path), device="cpu")
    stored = AdaLNAdapter.load(str(adapter_path), device="cpu")
    built = AdaLNAdapter.for_base(base, cond_dim=1, cond_mean=[0.0], cond_std=[1.0],
                                  **ADAPTER_ARCHITECTURE)
    shapes = lambda m: {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert shapes(built) == shapes(stored)


def test_the_sampling_and_training_presets_pin_the_recorded_values():
    """Change detectors for pure data. Each of these is a number a result was measured at, with the
    evidence in the preset's comment; editing one should be a decision, not a drive-by."""
    assert E1_SAMPLING == dict(sample_steps=500, eta=25.0, omega=0.0, time_distortion="polydec")
    assert ADAPTER_SAMPLING == dict(sample_steps=500, eta=25.0, omega=0.0, time_distortion="polydec")
    assert ADAPTER_TRAINING == dict(epochs=80, batch_size=24, lr=4e-4, grad_clip=1.0)
    assert ADAPTER_ARCHITECTURE == dict(hidden=256, time_conditioned=True, streams=("X", "E", "y"),
                                        cond_fourier=3, xattn_tokens=64, xattn_dim=128,
                                        xattn_heads=16)
