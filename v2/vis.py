"""Figures from a trained swarm: class-conditional samples, a latent sweep, trajectories."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from swarm import splat


def boid_colors(init_xy):
    """A fixed 2-D colormap over the initial disk, so the same boid has the same color in every figure."""
    xy = init_xy.cpu().numpy()
    r = (xy[:, 0] + 1) / 2
    g = (xy[:, 1] + 1) / 2
    return np.stack([r, g, 1 - 0.5 * (r + g) / 2], axis=1).clip(0, 1)


def scatter(ax, xy, colors, size=4):
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=colors, linewidths=0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


@torch.no_grad()
def sample_grid(swarm, path, device, per_class=8, seed=0):
    """Rows: classes. Columns: independent style draws from the prior."""
    swarm.eval()
    gen = torch.Generator().manual_seed(seed)
    num_classes = swarm.codes.shape[0]
    labels = torch.arange(num_classes).repeat_interleave(per_class).to(device)
    style = torch.randn(num_classes * per_class, swarm.d_style, generator=gen).to(device)
    final = swarm(labels, style)[..., :2].cpu().numpy()
    colors = boid_colors(swarm.init_xy)
    fig, axes = plt.subplots(num_classes, per_class, figsize=(1.3 * per_class, 1.3 * num_classes))
    for i, ax in enumerate(axes.ravel()):
        scatter(ax, final[i], colors, size=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def latent_sweep(swarm, path, device, label, extent=2.0, side=9):
    """A grid over the first two style dims for one class."""
    swarm.eval()
    lin = torch.linspace(-extent, extent, side)
    s2, s1 = torch.meshgrid(lin, lin, indexing="ij")
    style = torch.zeros(side * side, swarm.d_style)
    style[:, 0] = s1.reshape(-1)
    style[:, 1] = -s2.reshape(-1)  # top row = +extent, like an image
    labels = torch.full((side * side,), label, dtype=torch.long)
    final = swarm(labels.to(device), style.to(device))[..., :2].cpu().numpy()
    colors = boid_colors(swarm.init_xy)
    fig, axes = plt.subplots(side, side, figsize=(1.2 * side, 1.2 * side))
    for i, ax in enumerate(axes.ravel()):
        scatter(ax, final[i], colors, size=2.5)
        ax.axis("off")
    fig.suptitle(f"class {label}: style plane [-{extent}, {extent}]^2", fontsize=11)
    fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.01, right=0.99, top=0.96, bottom=0.01)
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def trajectory_gif(swarm, path, device, label, style, frames_per_step=3, fps=12):
    """One swarm run, interpolated between timesteps for smooth motion."""
    swarm.eval()
    labels = torch.tensor([label], device=device)
    style = torch.as_tensor(style, dtype=torch.float32, device=device).view(1, -1)
    _, traj = swarm(labels, style, return_traj=True)
    traj = torch.stack([t[0, :, :2] for t in traj]).cpu().numpy()  # (T+1, N, 2)
    colors = boid_colors(swarm.init_xy)
    fig, ax = plt.subplots(figsize=(4, 4))
    sc = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=14, c=colors, linewidths=0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title(f"class {label}, t = 0", fontsize=10)
    n_steps = traj.shape[0] - 1
    total = n_steps * frames_per_step + fps  # hold the final frame for a second

    def update(f):
        k = min(f / frames_per_step, n_steps)
        i, a = int(np.floor(k)), k - np.floor(k)
        pos = traj[i] if i >= n_steps else (1 - a) * traj[i] + a * traj[i + 1]
        sc.set_offsets(pos)
        title.set_text(f"class {label}, t = {k:.1f}")
        return sc, title

    anim = animation.FuncAnimation(fig, update, frames=total, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


@torch.no_grad()
def reconstructions(swarm, encoder, images, labels, path, device):
    """Target digit, its ink blurred like the loss sees it, and the swarm's reconstruction."""
    swarm.eval()
    encoder.eval()
    mu, _ = encoder(images)
    final = swarm(labels, mu)[..., :2]
    colors = boid_colors(swarm.init_xy)
    n = images.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.4 * n, 2.9))
    for i in range(n):
        axes[0, i].imshow(images[i].cpu(), cmap="gray_r", extent=(-1, 1, -1, 1))
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        scatter(axes[1, i], final[i].cpu().numpy(), colors, size=3)
    axes[0, 0].set_ylabel("target")
    axes[1, 0].set_ylabel("swarm")
    fig.subplots_adjust(wspace=0.05, hspace=0.08, left=0.03, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    from train import load_checkpoint, load_mnist

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default=None, help="output directory (default: checkpoint's directory)")
    parser.add_argument("--gif-classes", type=int, nargs="*", default=[0, 3, 7])
    parser.add_argument("--n-boids", type=int, default=None, help="render with this many boids (default: as trained)")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    swarm, encoder, cfg = load_checkpoint(args.checkpoint, device)
    if args.n_boids:
        swarm.resize(args.n_boids)
    out = Path(args.out or Path(args.checkpoint).parent)
    out.mkdir(parents=True, exist_ok=True)

    sample_grid(swarm, out / "samples.png", device, per_class=10)
    for c in range(swarm.codes.shape[0]):
        latent_sweep(swarm, out / f"sweep_{c}.png", device, label=c)
    for c in args.gif_classes:
        trajectory_gif(swarm, out / f"traj_{c}.gif", device, label=c, style=[0.6, -0.4])
    test_img, test_lab = load_mnist(train=False, device=device)
    idx = torch.arange(0, 12) * 37
    reconstructions(swarm, encoder, test_img[idx], test_lab[idx], out / "recon.png", device)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
