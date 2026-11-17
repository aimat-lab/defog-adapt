"""defog-adapt: frozen-base adapters, Feynman-Kac steering and RL fine-tuning for DeFoG.

The public API lives in the sub-packages:

* :mod:`defog_adapt.core`     the model, samplers, adapters, steering, RL, size models
* :mod:`defog_adapt.domains`  molecules (needs RDKit: ``pip install "defog-adapt[chem]"``)
* :mod:`defog_adapt.data`     reference splits (ZINC250k, MOSES) and vocabularies
* :mod:`defog_adapt.presets`  the recorded recipes, as plain data
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("defog-adapt")
except PackageNotFoundError:          # a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
