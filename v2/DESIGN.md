# v2: the swarm as a generative model

v1 (`../mnist_generative.py`) settles a swarm into a digit-ish blob. It is feeble for three
reasons, and all three are really the *same* reason: **nothing about the setup makes it a
generative model.**

1. **The initial cloud is i.i.d. uniform noise.** The swarm map is permutation-equivariant, so
   it does not see the N points, it effectively sees their *empirical measure*. N i.i.d. draws
   from a fixed base measure concentrate: the input measure is the same every time up to
   O(1/sqrt(N)) fluctuation. With N=256 the entire stochastic budget of the model is ~6% wiggle.
   There is no knob that says "a different 5".
2. **Conditioning is a 3-bit corner code on one pinned token,** so at best the model is a lookup
   table with 8 entries. Everything that varies within a class has to be absorbed as error.
3. **Chamfer loss against one sampled target cloud.** Chamfer is not a distribution metric; it is
   happy to clump points in high-density regions and leave holes, and when the target varies but
   the input doesn't, its minimizer is a smear over the class. Hence the blob.

So the v2 design question the whole thing hinges on is exactly the one you named: **what is the
initial position?** In this architecture the initial cloud has to do three jobs at once, and it
is the *only* thing that can do them, because a boid has no state other than its position:

- **identity** — a boid *is* its coordinates. Two boids at the same point are the same boid
  forever. The init is the positional encoding.
- **the prompt** — which digit.
- **the noise** — which *instance* of that digit.

That is a lot to ask of one point set, and the tension between job 1 (must be spread out and
canonical) and job 3 (must vary) is the crux.

---

## The candidates

### A. Fixed canonical sheet + global altitude latent  *(recommended)*

Work in `d_model = 4`: coordinates `(x, y, h1, h2)`. The picture is the `(x,y)` plane.

- `(x, y)` at t=0: **a fixed, maximally-even point set** — sunflower/Fibonacci disk, or a
  low-discrepancy set, identical in every run (plus tiny jitter for anti-aliasing). This is pure
  identity: boid #37 is "the one at that spot", every single time.
- `(h1, h2)` at t=0: **the same value for every boid**, equal to a global latent `z` in R^2.

The swarm starts as **a flat sheet floating at altitude z**, and the digit is whatever that
altitude makes it do. Properties:

- `z` is *global*, so it does not concentrate — this is a real generative model with a real
  latent, unlike i.i.d. noise. Diversity is bought honestly.
- Every boid can read `z` locally, with no side channel, no FiLM, no label embedding injected at
  every layer. The prompt is literally part of the habitat coordinates. The analogy survives.
- 2-D latent means the classic scrub-the-latent-plane demo: a grid of generated digits, and
  continuous morphs between them. With one latent dim (`d_model=3`, keeps v1's plottable vector
  fields) you get a single dial that sweeps through the digits — less capacity, better demo.
- Because boid identity is fixed across samples, you can color boids consistently and watch
  *the same boid* land on the upper-left serif in every 7. That's a very good picture.

The extra dimensions also do two jobs beyond carrying `z`:

- **They let trajectories cross in projection.** A weight-shared recurrent block is an
  *autonomous* ODE; in the plane its trajectories can never cross, so the flow is stuck being a
  planar diffeomorphism. Lift to 3+ dimensions and boids can fly over each other. This alone is
  a strong argument against d_model=2 and for making the extra axis genuinely dynamic rather
  than a dummy like v1's.
- **They can act as the clock** the autonomous field otherwise lacks, if the learned field
  drives them monotonically.

### B. Beacon boids — the prompt as extra tokens

Keep the working swarm generic (any spread cloud) and add `m` extra boids placed far outside
the habitat, e.g. on a ring. Their coordinates *are* the latent: `m x d` numbers. They may be
pinned in place (landmarks the swarm navigates by — stigmergy) or free to be consumed into the
digit.

This is the most on-theme option: in a transformer you prompt by *prepending tokens*, and here
that is exactly what it is. It also scales the latent without raising `d_model`. Cost: beacons
are a privileged caste of boid (especially if pinned), and v1 already did a degenerate version
of this with its single clamped condition token, so it's less of a departure.

### C. The loop — a rubber band that becomes a digit

All boids on a circle, boid identity = phase. The final cloud is a deformed closed curve, so you
can draw the swarm *connected*, and watch a rubber band wobble into a digit. Visually the best
option by a distance. Topology is not actually a problem — we only ask the ink *measure* to
match, and a loop can trace a stroke by doubling back — but strokes of "1" will need the band
folded flat, which is a lot of curvature. Best as a variant of A (the circle is just another
choice of canonical `(x,y)` set) rather than as its own design.

### D. Base measure shaped like the data

Nothing says the base measure has to be a uniform disk. Initializing from the **mean MNIST ink
measure** (a fat blob down the middle of the frame) starts the flow much closer to its target
and makes the transport far less compressive. Cheap win, slightly less pure. Orthogonal to A/B/C
— it's a choice of *which* canonical set, not of what the init means.

### E. i.i.d. noise (what v1 did)

Listed for completeness. Rejected: see the concentration argument above. Worth keeping as an
ablation precisely *because* it should visibly fail to produce diversity — that's a result.

---

## Consequences for the rest of v2

These follow from the choice above and are not independent of it.

**Loss.** Drop Chamfer. Splat the final cloud onto a 28x28 (or 56x56) grid with a differentiable
Gaussian kernel, normalize both sides to probability measures, compare with L2/L1. That is a
Gaussian MMD between measures — a proper distribution metric, no clumping pathology — it costs
O(N x 784), and it optimizes exactly the thing we care about ("the positions form a digit").
Anneal the kernel width coarse-to-fine. Optionally finish with a Sinkhorn divergence for even
spacing.

**Training the latent.** With a global `z` there are three honest routes:
- *VAE*: a small conv encoder image -> q(z|x), decode with the swarm, recon = splat loss, KL to
  N(0,I). The encoder is training-time scaffolding; at sampling time the swarm is alone. Simplest
  thing that actually works, and a 2-D `z` gives the latent-plane figure for free.
- *Adversarial on the splatted image*: sharper digits, fiddlier training.
- *OT-coupled flow matching*: sample a base cloud and a target cloud, compute a Sinkhorn/Hungarian
  coupling between them, regress the straight-line velocity. This turns the whole thing into
  supervised regression — by far the best-conditioned objective, and it produces smooth
  boid-like trajectories instead of swirl. Needs care because the recurrent field is autonomous
  and flow matching wants a time input; the extra axis can serve as the clock.

**Class conditioning.** Prefer `z = class_code + sigma * eps` with a learned per-class code, so
labelled sampling and style variation share one mechanism, and the unconditional model is the
same model with the codes marginalized.

---

## What was actually built first, and what it taught (2026-09-23)

Built: candidate **A** with `d_model = 6` — fixed sunflower disk for `(x, y)`, a fixed class code
(10 points on a circle of radius 2) in `(c1, c2)`, a 2-D style latent in `(s1, s2)`, all hidden
coordinates identical across boids at t=0. Class-code coordinates are pinned (the block never
writes to them); style coordinates are free state. Multi-scale Gaussian-splat MMD loss.
Class-conditional VAE training. See `README.md` for the numbers.

- **Capacity of the per-step field was the bottleneck.** A 6->512->6 FFN (v1's size) only ever
  learns "squash the disk into a slanted blob"; 6->256->256->6 learns class structure within one
  epoch. Everything else — pinning, code radius, sliced-Wasserstein instead of MMD — was second
  order at this stage.
- **The swarm is N-agnostic in practice**: train with 64 boids, render with 256 or 512, density
  differs by <0.5%. Attention is close to mean-field here.
- **The disk is folded like a sheet.** Colouring boids by initial position shows every stroke is
  a continuous patch of the disk; the 7 is a fold along its diagonal. The transport map is smooth.
- **Prior mismatch is visible**: a style sampled from `N(0, I)` for class 1 sometimes gives a fat
  blob, because the encoder's posterior for 1s sits in a "thin" corner of style space. Standard
  CVAE issue; a higher KL weight, or free bits, or a class-conditional prior would fix it.

## Next steps, in the order I would take them

1. More capacity in the field (3 layers or width 384) and more steps — the one lever that moved.
2. Fix the prior mismatch (KL weight ~0.05 with warm-up, or learn a per-class prior mean).
3. Then the ambitious versions: the loop init (candidate C), OT-coupled flow matching for the
   dynamics, and letting the swarm run past T=16 to see whether the digit is a fixed point.
