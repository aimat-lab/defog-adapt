# Examples and recipes

The recipes and scripts in this folder are described on the **Recipes** page of the documentation,
[`docs/reference/recipes.md`](../docs/reference/recipes.md): what each one does and needs, the order of the pipeline, how
to run them, and what to read before trusting a number.

`quickstart_cpu.py` (every building block on a toy problem), `training__e1.py` (the base, or a model trained end to end on
a property), `rl_sanity__zinc.py` (RL on the base), `adapter_training__zinc.py` (one adapter per property),
`fit_property_head__zinc.py` (the property head and the size model), `adapter_rl_finetune__zinc.py` (RL on an adapter),
`steer_zinc.py` (the assembly) and `compare_arms__zinc.py` (the comparison behind the results figure). `_common.py` holds
their shared helpers.
