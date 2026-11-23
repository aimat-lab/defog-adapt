# The README banner

`banner.gif` (16.7 s, loops) is the animation at the top of the [README](../../../README.md). It replays **one
real run of the sampler**, twice, from the **same noise**:

1. the frozen DeFoG base alone turns the noise into a molecule (logP -0.3);
2. a logP adapter (the flame) docks on the frozen model and acts on every pulse that passes under it, and the same
   noise ends in a different molecule at logP 4.6 (target 4.5).

It illustrates the process. It is **not a measurement**: the pair was chosen from many for legibility.

## What is recorded, what is drawn, what was chosen

| | |
|---|---|
| **Recorded** | The discrete state at each of the 200 steps: the element of every one of the 16 nodes and the bond order of every pair, for both runs, from identical initial noise (`trajectory.json`). Every atom and bond flip on screen is a flip the sampler made, including the transient wrong ones. The step counter shows the flow time *t* the model was given at the step on screen (polydec, `t = 2s - s^2`). The logP values are RDKit Crippen values of the decoded molecules. |
| **Drawn** | Node *positions*. A graph has no coordinates, so the noise is a scatter and the finished molecule an RDKit 2D layout, and the drawing glides from one to the other (each node to the scatter slot that minimises its travel). Which frame shows which step: the 200 steps are shown at a pace that follows the number of flips, so quiet stretches are fast-forwarded. The faint time axis under the stations runs from the noise (*t* = 0) to the sample (*t* = 1), and its knob rewinds when the noise comes back. The small pulses that run from the noise, behind the model and out along the arrow stand for the model working at every step: one pulse is one model application, and the orange ring appears where the adapter acts on a pulse that passes under it (the pulse then comes out of the model orange). There are far fewer pulses than the 200 steps. The logP caption sits just above its molecule. |
| **Chosen** | The pair. 240 same-noise pairs were recorded (10 seeds of 24), 132 of them had two molecules that draw cleanly (connected, no charges, ring sizes 5 to 7, a layout with no crossing bonds), and these were ranked on: the adapter landing near the target, the base clearly lower, how few long wrong bonds survive into the second half of the run (a real transient, but one that reads as a drawing error), how early each run settles, element variety and a noise that is not all carbon. Candidate 8 of the ranked list was used. |

## Settings of the recorded run

The ZINC250k base of the recorded results (kekulized representation, structural RL) and the logP adapter (node
cross-attention, Fourier condition bands), composed in probability space at weight 1.5 for a target logP of 4.5.
16 heavy atoms, 200 steps, polydec time distortion, eta 0, omega 0; seed 0, batch index 5 of 24. These are
chosen to be readable (eta 0 keeps what is right right), not to score: the benchmark configuration is 500
steps with eta 25. The model and sampler are the ones in this package; the checkpoints are not shipped.

## Files

| file | what it is |
|---|---|
| `banner.gif` | the animation: 1600 x 431 px, 251 frames at 15 fps, about 2.2 MB |
| `trajectory.json` | the recorded run: both trajectories, the shared noise and every setting above |
| `record_trajectories.py` | `record` same-noise pairs from checkpoints and rank them, `rank` again, `export` one pair |
| `candidates.py` | the contact sheet of ranked candidate pairs, drawn the way the banner draws molecules |
| `banner.py`, `molecules.py` | one frame of the banner, and the ball-and-stick molecule drawing and layouts |
| `render_banner.py` | renders every frame and assembles the GIF with one shared palette |

## Rebuilding

```bash
python record_trajectories.py record --base BASE.ckpt --adapter LOGP_ADAPTER.ckpt --work WORK
python record_trajectories.py export --work WORK --rank 8 --out trajectory.json
python render_banner.py frames --work FRAMES && python render_banner.py gif --work FRAMES --out banner.gif
```

The drawing uses the author's figure toolchain (figlib and VectorCraft, found through `$FIGURE_SKILL`), which
is not published, together with Pillow, SciPy and ffmpeg. So the GIF and `trajectory.json` are the artefacts to
reuse; the scripts document exactly how they were made.
