"""Rebuild the published diagrams: build each figure script, turn the text of its SVG into outlines (so the image does not depend
on a font being installed), and give the SVG a larger intrinsic size (the labels are 7 pt on a canvas of at most 440 pt, so at
the width of a page they would be small).

    python3 docs/assets/diagrams/src/publish.py [name ...]        # names are script names; none means all of them

It needs the author's figure toolchain ($FIGURE_SKILL) and Inkscape. Neither is published, so
the SVG files in the folder above are the artefacts to reuse; the scripts and the .vectorcraft documents next to this file
document exactly how they were made.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
SKILL = Path(os.environ["FIGURE_SKILL"]) if "FIGURE_SKILL" in os.environ else None
SCRIPTS = ["recorded_pipeline", "adapter_modulation", "feynman_kac", "steering_roles"]
SCALE = 2.0          # the published SVG is this many times larger than its canvas in pt


def figure_name(script: Path) -> str:
    return re.search(r"^NAME = '([^']+)'", script.read_text(encoding="utf-8"), flags=re.M).group(1)


def enlarge(svg: str) -> str:
    """Multiply the width and height of the root element, keeping the viewBox: the picture is the same, its default size is not."""
    def grow(match):
        return f'{match.group(1)}="{float(match.group(2)) * SCALE:g}"'
    root_end = svg.index(">", svg.index("<svg"))
    root = re.sub(r'\b(width|height)="([\d.]+)"', grow, svg[: root_end + 1])
    return root + svg[root_end + 1:]


def publish(name: str) -> None:
    script = HERE / f"{name}.py"
    figure = figure_name(script)
    subprocess.run([sys.executable, str(SKILL / "scripts" / "build.py"), str(script), "--final"], check=True)
    built = OUT / f"{figure}.svg"
    with tempfile.TemporaryDirectory() as tmp:
        outlined = Path(tmp) / "outlined.svg"
        subprocess.run(["inkscape", str(built), "--export-type=svg", "--export-text-to-path", "--export-plain-svg",
                        f"--export-filename={outlined}"], check=True, capture_output=True)
        built.write_text(enlarge(outlined.read_text(encoding="utf-8")), encoding="utf-8")
    (OUT / f"{figure}.pdf").unlink(missing_ok=True)       # the PDF is for a thesis, not for these pages
    print(f"published {built.relative_to(OUT.parent.parent)} ({built.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    if SKILL is None:
        sys.exit("set $FIGURE_SKILL to the figure toolchain (the folder with scripts/build.py)")
    for chosen in sys.argv[1:] or SCRIPTS:
        publish(chosen)
