"""Render the candidate initial swarm configurations discussed in DESIGN.md."""

import numpy as np
import matplotlib.pyplot as plt
import torchvision

N = 256
R = 10.0
rng = np.random.default_rng(0)


def sunflower(n, radius):
    i = np.arange(n) + 0.5
    r = radius * np.sqrt(i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def grid(n, radius):
    side = int(np.ceil(np.sqrt(n)))
    c = np.linspace(-radius, radius, side)
    gy, gx = np.meshgrid(c, c, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)[:n]


def iid(n, radius):
    return rng.uniform(-radius, radius, size=(n, 2))


def loop(n, radius):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return 0.85 * radius * np.stack([np.cos(t), np.sin(t)], axis=1)


def mean_ink(n, radius):
    """Sample from the average MNIST ink measure."""
    mnist = torchvision.datasets.MNIST(root="../data", train=True, download=False)
    mean = np.zeros((28, 28))
    for i in range(2000):
        mean += np.asarray(mnist.data[i], dtype=np.float64)
    p = (mean / mean.sum()).ravel()
    idx = rng.choice(784, size=n, p=p)
    yx = np.stack([idx // 28, idx % 28], axis=1) + rng.uniform(0, 1, size=(n, 2))
    xy = np.stack([yx[:, 1], 27 - yx[:, 0]], axis=1)
    return (xy / 27 - 0.5) * 2 * radius


CANDIDATES = [
    ("E. i.i.d. uniform (v1)", iid, "tab:red"),
    ("A. sunflower disk", sunflower, "tab:blue"),
    ("A'. square grid", grid, "tab:blue"),
    ("C. loop", loop, "tab:green"),
    ("D. mean MNIST ink", mean_ink, "tab:purple"),
]

fig, axes = plt.subplots(1, len(CANDIDATES) + 1, figsize=(3.0 * (len(CANDIDATES) + 1), 3.4))
for ax, (name, fn, color) in zip(axes, CANDIDATES):
    pts = fn(N, R)
    ax.scatter(pts[:, 0], pts[:, 1], s=9, c=color, alpha=0.85, linewidths=0)
    if name.startswith("C."):
        closed = np.vstack([pts, pts[:1]])
        ax.plot(closed[:, 0], closed[:, 1], lw=0.6, c=color, alpha=0.5)
    ax.set_title(name, fontsize=10)
    ax.set_xlim(-R * 1.05, R * 1.05)
    ax.set_ylim(-R * 1.05, R * 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

# B: beacon boids on a ring outside the habitat, around a generic working cloud.
ax = axes[-1]
pts = sunflower(N, R)
ax.scatter(pts[:, 0], pts[:, 1], s=9, c="tab:blue", alpha=0.35, linewidths=0)
m = 12
t = np.linspace(0, 2 * np.pi, m, endpoint=False)
beac = 1.35 * R * np.stack([np.cos(t), np.sin(t)], axis=1)
beac += rng.normal(0, 0.9, size=beac.shape)  # the beacon coords *are* the latent
ax.scatter(beac[:, 0], beac[:, 1], s=55, c="tab:orange", marker="*", linewidths=0)
ax.set_title("B. + beacon boids (prompt)", fontsize=10)
ax.set_xlim(-R * 1.55, R * 1.55)
ax.set_ylim(-R * 1.55, R * 1.55)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

fig.suptitle("v2 candidate initial swarm configurations (N=256)", fontsize=12)
fig.tight_layout()
fig.savefig("init_gallery.png", dpi=130)
print("wrote v2/init_gallery.png")
