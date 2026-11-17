"""Where reference data lives on disk.

The research repo kept its data next to the source (``<repo>/data/...``, found as
``Path(__file__).parents[2]``). Inside an installed package that path is the interpreter's own
``site-packages`` directory: not writable, shared between projects, and wiped on upgrade. So the
data root is a per-user cache instead, and can be moved with an environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Environment variable that overrides the data root (the directory holding ``zinc250k/``,
#: ``moses/`` and ``chembl/``).
ENV_VAR = "DEFOG_ADAPT_HOME"


def data_home() -> Path:
    """``$DEFOG_ADAPT_HOME`` if set, else ``$XDG_CACHE_HOME/defog_adapt`` (``~/.cache/defog_adapt``)."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "defog_adapt"


def default_root(name: str) -> Path:
    """Directory for one dataset, e.g. ``default_root("zinc250k")``. Read at CALL time, so the
    environment variable can be set after import."""
    return data_home() / name
