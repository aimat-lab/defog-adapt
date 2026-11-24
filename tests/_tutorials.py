"""What the tutorial tests share: where the guides and their scripts are, and how to read them.

Each tutorial is a page and a script. The page shows the script in pieces (one section per step) and whole at the end; the
script is the one copy of the code. The three tutorials follow each other: the base guide writes ``base.ckpt``, the adapter
guide reads it and writes the adapter, the size model and the head, and the steer guide loads all four.
"""
import ast
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ROOT / "docs" / "guides"


@dataclass(frozen=True)
class Tutorial:
    key: str
    page: str
    script: str
    steps: int        # the numbered steps on the page
    required: int     # steps 1 to this one are required; every later step says "(optional)" in its heading
    files: tuple      # the files the script reads and writes, as its header names them

    @property
    def page_path(self) -> Path:
        return GUIDES / self.page

    @property
    def script_path(self) -> Path:
        return GUIDES / self.script

    @property
    def include_name(self) -> str:
        return f"docs/guides/{self.script}"

    def page_text(self) -> str:
        return self.page_path.read_text(encoding="utf-8")

    def script_text(self) -> str:
        return self.script_path.read_text(encoding="utf-8")


TUTORIALS = [
    Tutorial("base", "train-an-unconditional-base.md", "train_an_unconditional_base.py", steps=6, required=4,
             files=("molecules.smi", "base.ckpt")),
    Tutorial("adapter", "train-an-adapter.md", "train_an_adapter.py", steps=8, required=5,
             files=("molecules.smi", "base.ckpt", "logp_adapter.ckpt", "logp_head_size.ckpt", "logp_head.ckpt")),
    Tutorial("steer", "steer-toward-a-target.md", "steer_toward_a_target.py", steps=8, required=4,
             files=("base.ckpt", "logp_adapter.ckpt", "logp_head.ckpt", "logp_head_size.ckpt")),
]
BY_KEY = {tutorial.key: tutorial for tutorial in TUTORIALS}


def literal_calls(source: str, name: str) -> list:
    """The keyword arguments that are plain literals in each call to ``name`` (a function, or a method of that name), one
    dict per call. An argument that is computed (``condition=torch.tensor([target])``) is left out."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if called != name:
            continue
        literals = {}
        for keyword in node.keywords:
            if keyword.arg:
                try:
                    literals[keyword.arg] = ast.literal_eval(keyword.value)
                except ValueError:
                    pass
        found.append(literals)
    return found


def settings_tables(markdown: str) -> list:
    """The settings tables of a page, in order: a dict from each setting's name to the text of its 'Recorded' cell. A
    settings table is one headed ``| Setting | What it does | Recorded |``."""
    tables, current = [], None
    for line in markdown.splitlines():
        if line.startswith("| Setting |"):
            current = {}
            tables.append(current)
        elif current is not None and line.startswith("|"):
            row = re.match(r"^\| `(\w+)` \| .* \| (.+?) \|$", line)
            if row:
                current[row.group(1)] = row.group(2)
        else:
            current = None
    return tables


def shown_equals(shown: str, value) -> bool:
    """Does the text of a 'Recorded' cell say ``value``? Numbers compare as numbers (``2e-4`` is 0.0002), a list of names as
    names, anything else as text; backticks are ignored."""
    text = shown.replace("`", "").strip()
    if isinstance(value, bool):
        return text == str(value)
    if isinstance(value, (int, float)):
        try:
            return float(text) == float(value)
        except ValueError:
            return False
    if isinstance(value, (tuple, list)):
        return tuple(re.findall(r"\w+", text)) == tuple(value)
    return text == str(value)
