# v2: a swarm that settles into a digit

A weight-shared transformer block applied 16 times to a swarm of boids in R^6.
At t=0 the boids sit on a sunflower disk in the picture plane; when the swarm
settles, their positions form an MNIST digit. See [DESIGN.md](DESIGN.md) for
the reasoning behind the initial state — it is the whole design question.

## What a boid is

A point `(x, y, c1, c2, s1, s2)`:

- `(x, y)` — picture plane, `[-1, 1]^2`. Starts on a fixed sunflower disk, so boid #37 is always
  born in the same place. That is its identity; there is no positional encoding.
- `(c1, c2)` — the class code, one of ten fixed points on a circle. Identical for every boid at t=0.
- `(s1, s2)` — the style latent, `z ~ N(0, I)` at sampling time. Identical for every boid at t=0.

The hidden coordinates are the prompt. They are part of the habitat, every boid reads them
locally, and the block is free to write to them, so they also let trajectories cross in the
picture plane and can act as a clock.

## The block

Standard attention + feed-forward, residual-scaled by 0.1, no layer norm. Heads live in the full
6-D habitat and are averaged, as in v1. Output layers are zero-initialised so an untrained swarm
stands still. `swarm.py`.

## The loss

`splat_mmd`: the final cloud is rendered onto the 28x28 grid with a Gaussian kernel, the target
image's ink is blurred with the same kernel, and the two are compared in L2 — at three kernel
widths (0.75, 1.5, 3 px), each normalised by the target's energy. That is a multi-scale Gaussian
MMD between the boid measure and the ink measure: a proper distribution metric, no Chamfer-style
clumping. Boids outside the frame get no gradient from it, so there is a small out-of-frame hinge.

## Result of the first full run

`models/final.pt` (run `runs/final`): 64 boids, 2-layer FFN (69k swarm parameters), 20 epochs, ~37 min on an M1.
Test MMD 0.125 (v2 control 0.185; a perfect ink sample scores 0.077). Rendered at 256 boids,
every class is recognisable from the prior; within-class variation is slant, width and foot
shape. Strokes are still fat and 2/3/5/8 are the softest. `vis.py` also writes style-plane
sweeps, per-class trajectory GIFs and reconstructions into the run directory.

![Samples from the prior](figures/samples.png)

![The disk folding into a digit](figures/trajectory_strip.png)

![One trajectory](figures/traj_7.gif)

## What the attention is doing

`vis_attention.py` animates the attention block ([figures/attention_7.gif](figures/attention_7.gif)): attention received per
boid, gaze arrows for a subsample, and one boid's full attention distribution, all relative to
uniform. The trained attention is close to uniform (entropy 97-99.9% of maximum, top-5 boids
carry ~7% of the mass vs 4% for uniform), so it acts as a centroid sensor rather than a
selective gaze — but a *biased* one: attention received correlates with height at -0.9 to
-0.98, so the swarm listens to its lowest boids and everyone looks at a point just below the
centre of mass. Head 1 does this from t=0, head 0 learns to do it as the swarm folds.

![Attention over time](figures/attention_strip_7.png)

## What the feed-forward block is doing

`vis_ffn.py` draws the alone-time behaviour as a vector field on the picture plane
(`ffn_fields_t0.png`, `ffn_strip_<c>.png`, [figures/ffn_7.gif](figures/ffn_7.gif)). It is an autonomous field on R^6,
not a gradient flow: the rotational share of its Jacobian is ~30% at t=0, falling to ~13% as the
swarm settles, by which point it is an almost pure contraction (negative divergence at 96% of
boids). Because the FFN also rewrites each boid's style coordinates — its largest output is in
the hidden dimensions, and the spread of hidden state across boids grows to ~1 — the effective
2-D field morphs over the 16 steps; the figures show it at the swarm-mean hidden state. At t=0
it is exact, and already legible: class 0's field *expands* where the hole goes, class 1's
streamlines converge on a single line. The alone-time move does 4x the picture-plane transport
of the flocking move at t=0 and about the same by the end — and the flocking move is a **rigid
translation**: 99% of its energy is the swarm-mean displacement at every step, which is what a
near-uniform attention has to produce. All of the shaping is done alone.

![Alone-time fields per class](figures/ffn_fields_t0.png)

![Field over time and per-boid move decomposition](figures/ffn_strip_7.png)

## Training

Class-conditional VAE. A small conv encoder maps the image to `q(s | image)`; the swarm decodes
`(class, s)`; KL to `N(0, I)` with a linear warm-up. The encoder is training-time scaffolding —
at sampling time the swarm is alone.

```bash
python train.py --run runs/final --n-boids 64 --epochs 20 --pin-class --code-radius 2 --ffn-layers 2 --hidden 256
python vis.py models/final.pt --n-boids 256       # samples.png, sweep_<c>.png, traj_<c>.gif, recon.png
python vis_attention.py models/final.pt --label 7  # what the attention block does
python vis_ffn.py models/final.pt --label 7        # what the feed-forward block does
```

`models/final.pt` is the trained model from the run described below (swarm + encoder weights
and config; `train.load_checkpoint` rebuilds it). `runs/` is gitignored.

To reproduce a figure on another machine: `pip install -r requirements.txt`, then e.g.
`python vis_ffn.py models/final.pt --label 7 --out figures_repro`. The vis scripts need no data
and no GPU (`--device cpu` works; MNIST is only downloaded by `train.py`, and by `vis.py` for
the reconstruction panel). Everything they draw is deterministic given the checkpoint.

The dynamics are permutation-equivariant and close to mean-field, so a swarm trained with 64
boids can be run with 256: `python vis.py models/final.pt --n-boids 256`. On an M1 the cost
is purely the B·H·N² attention tensor, so training small and rendering big is the way to go.

## What the first round taught (64 boids, 3 epochs each, `compare.py`)

| run | test MMD | label sensitivity | note |
|---|---|---|---|
| control (6.9k-param swarm) | 0.185 | 0.061 | squashes the disk into a blob; class barely read |
| + pinned class code, radius 2 | 0.195 | 0.064 | no effect on its own |
| + 2-layer FFN (69k params) | **0.150** | **0.108** | hollow 0s, barred 7s appear in epoch 1 |
| same, sliced-Wasserstein loss instead of MMD | 0.213 | 0.039 | trails at equal time |

For reference, a perfect 128-point sample of the ink scores 0.077. Capacity of the per-step
field was the bottleneck, not the loss and not the conditioning. The style latent does leak
class-correlated shape (a nearest-centroid classifier on the 2-D style gets ~30-35%; 0s and
1s are round vs thin), which is partly legitimate style and partly the usual CVAE prior
mismatch — worth a higher KL weight later.
