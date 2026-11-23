# The diagrams

Four schematic diagrams on the documentation pages. They show how the pieces work; they are **not measurements**: the molecules, the
weights and the parents in them are drawn for illustration.

| diagram | page | what it says |
|---|---|---|
| `tools-overview.svg` | Concepts, overview | which tool acts when: the adapter is trained once beside a frozen base and acts at every step, Feynman-Kac steering trains nothing, RL changes the weights |
| `adapter-modulation.svg` | Concepts, Adapters | what the adapter modulates: a scale, a shift and a gate for every layer of the frozen base, and atoms reading the condition |
| `feynman-kac.svg` | Concepts, Steering | K candidates, a warm-up without resampling, then checkpoints where heavy candidates are cloned and light ones dropped |
| `steering-roles.svg` | Guides, Steer toward a target | who does what: the size model, the proposal, the ruler (the property head, which steers) and the judge (RDKit, which measures) |

## The vocabulary

It is the README banner's. A frozen model is a white trapezoid with pale blue stripes and a snowflake badge; the adapter is an
orange block with a flame; a model whose weights change carries a delta-W badge. Feynman-Kac steering is cyan and RL fine-tuning
purple. The molecules are small hand-made graphs, drawn so that no two slots in a figure show the same one (`src/vocab.py`).

Each diagram is drawn in black on a white card (the shape named `background`), so that it reads on the dark theme of the site
as well; there is no separate dark variant. `tests/test_diagrams.py` checks that the card is there and covers the canvas.

## Files

| file | what it is |
|---|---|
| `*.svg` | what the pages show: the text is outlines, so no font is needed, and each SVG is twice the size of its canvas, so its labels are readable at the width of a page |
| `src/*.py` | one script per diagram: the layout is a handful of named numbers at the top, and every position derives from them |
| `src/vocab.py` | the shared vocabulary: the frozen model, the adapter, the badges, the small molecules and the noise |
| `src/publish.py` | builds every diagram and writes the SVGs |

## Rebuilding

```bash
FIGURE_SKILL=/path/to/figure-toolchain python3 docs/assets/diagrams/src/publish.py            # all four
FIGURE_SKILL=/path/to/figure-toolchain python3 docs/assets/diagrams/src/publish.py feynman_kac # one
```

The drawing uses the author's figure toolchain (figlib and VectorCraft, found through `$FIGURE_SKILL`), which is not published,
together with Inkscape. So the SVG files are the artefacts to reuse; the scripts document exactly how they were made. The
`.vectorcraft` documents that the build writes beside the scripts are not committed: the scripts are the source.
