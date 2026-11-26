"""The settings the tutorials write out are the recorded ones, and every one of them is explained.

A tutorial writes its settings into the call instead of unpacking a preset, because a reader should see them. That makes a
number in a script a claim ("these are the recorded values"), and two things can then go wrong: the script drifts from the
preset it says it shows, or a setting is written out and never explained. So each number in a call is compared with its
preset (or, where there is no preset, with the default of the function the recipes use), and each settings table on a page
must have exactly one row for each setting of the call it follows, with the recorded value in it.
"""
import ast
import inspect

import pytest

from _tutorials import BY_KEY, TUTORIALS, literal_calls, settings_tables, shown_equals

pytestmark = pytest.mark.skipif(not all(t.script_path.is_file() for t in TUTORIALS),
                                reason="docs/ is not part of an installed wheel")


def check_table(table: dict, recorded: dict, where: str) -> None:
    assert set(table) == set(recorded), f"{where}: the table explains {sorted(table)}, the call has {sorted(recorded)}"
    for name, value in recorded.items():
        assert shown_equals(table[name], value), f"{where}: the table says {name} is {table[name]}, the recorded value is {value!r}"


def defaults(function) -> dict:
    return {name: p.default for name, p in inspect.signature(function).parameters.items()
            if p.default is not inspect.Parameter.empty}


# ===================================================================================================== steer
def test_steer_script_settings_are_the_recorded_presets():
    from defog_adapt.presets import ADAPTER_SAMPLING, ADAPTER_WEIGHT, FK_STEERING

    script = BY_KEY["steer"].script_text()
    (adapted,) = literal_calls(script, "AdaptedSampler")
    assert adapted == ADAPTER_SAMPLING
    (feynman_kac,) = literal_calls(script, "FeynmanKacSampler")
    assert feynman_kac == {**FK_STEERING, **ADAPTER_SAMPLING}, "Feynman-Kac's own four settings, then the sampling four"
    (branch,) = literal_calls(script, "ConditionBranch")
    assert branch == {"weight": ADAPTER_WEIGHT["logp"]}


def test_steer_tables_explain_every_setting_with_its_recorded_value():
    from defog_adapt.presets import ADAPTER_SAMPLING, ADAPTER_WEIGHT, FK_STEERING

    page = BY_KEY["steer"].page_text()
    tables = settings_tables(page)
    assert len(tables) == 2, f"the steer page has {len(tables)} settings tables, expected the sampler's and Feynman-Kac's"
    check_table(tables[0], ADAPTER_SAMPLING, "steer, sampler")
    check_table(tables[1], FK_STEERING, "steer, Feynman-Kac")
    assert f"{ADAPTER_WEIGHT['logp']} is the weight recorded for logP" in page


# ===================================================================================================== base
#: the settings of the recorded model that the small model changes: its size, and the features that need element tables
TURNED_DOWN = {"n_layers", "hidden_dim", "hidden_mlp_dim", "n_heads", "molecular_features"}
#: recorded model settings that belong to sampling, which is chosen when sampling and not built into the model
SAMPLING_SETTINGS = {"sample_steps", "eta", "omega", "sample_time_distortion"}


def test_base_model_is_the_recorded_model_with_the_size_turned_down():
    from defog_adapt.presets import E1_MODEL

    (model,) = literal_calls(BY_KEY["base"].script_text(), "from_dataloader")
    assert set(model) == set(E1_MODEL) - SAMPLING_SETTINGS, "every recorded model setting is written out, and no other"
    for name, value in model.items():
        if name in TURNED_DOWN:
            assert value != E1_MODEL[name], f"{name} is the recorded value, so the page's 'turned down' is not true"
        else:
            assert value == E1_MODEL[name], f"{name} is {value!r}, the recorded value is {E1_MODEL[name]!r}"


def test_base_sampler_cleanup_and_rewards_are_the_recorded_presets():
    from defog_adapt.domains.molecule import ring_sizes_ok
    from defog_adapt.presets import E1_SAMPLING, SANITY_RL_TRAINER

    script = BY_KEY["base"].script_text()
    (sampler,) = literal_calls(script, "Sampler")
    assert sampler == E1_SAMPLING
    (trainer,) = literal_calls(script, "GDPOTrainer")
    assert trainer == SANITY_RL_TRAINER
    reward = next(node for node in ast.walk(ast.parse(script))
                  if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "SanityReward")
    rings = [argument.value for argument in reward.args[1:]]
    assert rings == [defaults(ring_sizes_ok)["lo"], defaults(ring_sizes_ok)["hi"]], "the ring sizes the page says are 3 to 8"


def test_base_tables_explain_every_setting_with_its_recorded_value():
    from defog_adapt.presets import E1_MODEL, E1_SAMPLING, SANITY_RL_TRAINER

    tables = settings_tables(BY_KEY["base"].page_text())
    assert len(tables) == 3, f"the base page has {len(tables)} settings tables, expected the model's, the sampler's and the cleanup's"
    check_table(tables[0], {k: v for k, v in E1_MODEL.items() if k not in SAMPLING_SETTINGS}, "base, model")
    check_table(tables[1], E1_SAMPLING, "base, sampler")
    check_table(tables[2], SANITY_RL_TRAINER, "base, cleanup")


# ===================================================================================================== adapter
def test_adapter_architecture_and_training_are_the_recorded_presets():
    from defog_adapt.presets import ADAPTER_ARCHITECTURE, ADAPTER_TRAINING

    script = BY_KEY["adapter"].script_text()
    (adapter,) = literal_calls(script, "for_base")
    assert {k: adapter[k] for k in ADAPTER_ARCHITECTURE} == ADAPTER_ARCHITECTURE
    assert set(adapter) == set(ADAPTER_ARCHITECTURE) | {"cond_dim", "name", "cond_type"}
    (trainer,) = literal_calls(script, "Trainer")
    assert trainer["max_epochs"] == ADAPTER_TRAINING["epochs"]
    assert trainer["gradient_clip_val"] == ADAPTER_TRAINING["grad_clip"]
    (module,) = literal_calls(script, "AdapterModule")
    assert module["lr"] == ADAPTER_TRAINING["lr"]
    assert any(call.get("batch_size") == ADAPTER_TRAINING["batch_size"] for call in literal_calls(script, "DataLoader"))


def test_adapter_size_model_and_head_settings_are_the_defaults_the_recipes_use():
    from defog_adapt.core import PropertyHead, fit_property_head, fit_size_model

    script = BY_KEY["adapter"].script_text()
    (size,) = literal_calls(script, "fit_size_model")
    for name in ("hidden", "layers", "epochs", "lr", "batch_size"):
        assert size[name] == defaults(fit_size_model)[name], f"fit_size_model: {name}"
    (head,) = literal_calls(script, "PropertyHead")
    for name in ("hid", "layers"):
        assert head[name] == defaults(PropertyHead.__init__)[name], f"PropertyHead: {name}"
    (fit,) = literal_calls(script, "fit_property_head")
    for name in ("epochs", "lr", "batch_size", "seed"):
        assert fit[name] == defaults(fit_property_head)[name], f"fit_property_head: {name}"


def test_adapter_tables_explain_every_setting_with_its_recorded_value():
    from defog_adapt.core import PropertyHead, fit_property_head, fit_size_model
    from defog_adapt.presets import ADAPTER_ARCHITECTURE, ADAPTER_TRAINING

    tables = settings_tables(BY_KEY["adapter"].page_text())
    assert len(tables) == 4, f"the adapter page has {len(tables)} settings tables, expected architecture, training, size model, head"
    check_table(tables[0], ADAPTER_ARCHITECTURE, "adapter, architecture")
    check_table(tables[1], {"max_epochs": ADAPTER_TRAINING["epochs"], "batch_size": ADAPTER_TRAINING["batch_size"],
                            "lr": ADAPTER_TRAINING["lr"], "gradient_clip_val": ADAPTER_TRAINING["grad_clip"]},
                "adapter, training")
    size = defaults(fit_size_model)
    check_table(tables[2], {name: size[name] for name in ("hidden", "layers", "epochs", "lr", "batch_size")},
                "adapter, size model")
    head, fit = defaults(PropertyHead.__init__), defaults(fit_property_head)
    check_table(tables[3], {"hid": head["hid"], "layers": head["layers"], "epochs": fit["epochs"], "lr": fit["lr"],
                            "batch_size": fit["batch_size"]}, "adapter, head")
