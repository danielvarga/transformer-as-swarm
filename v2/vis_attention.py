"""Animate what the attention block is doing while the swarm settles into a digit.

Three views per head, all on the same relative scale where 1 = uniform attention:
  - received: how much attention each boid gets (column sums of the attention matrix),
  - gaze: for a subsample of boids, an arrow to the point they are looking at
    (attention-weighted mean position),
  - focal: the full attention distribution of one boid, the most selective one.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from train import load_checkpoint

NORM = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)
CMAP = "coolwarm"


@torch.no_grad()
def run(swarm, label, style, device):
    labels = torch.tensor([label], device=device)
    style = torch.tensor([style], dtype=torch.float32, device=device)
    _, traj, attns = swarm(labels, style, return_traj=True, return_attn=True)
    pos = torch.stack([t[0, :, :2] for t in traj]).cpu().numpy()  # (T+1, N, 2)
    attn = torch.stack([a[0] for a in attns]).cpu().numpy()  # (T, H, N, N)
    return pos, attn


def frame_data(pos, attn, k):
    """Positions and attention at fractional step k (attention at step t drives the move t -> t+1)."""
    n_steps = attn.shape[0]
    i = min(int(math.floor(k)), n_steps - 1)
    a = min(k - i, 1.0)
    p = (1 - a) * pos[i] + a * pos[i + 1]
    j = min(i + 1, n_steps - 1)
    w = (1 - a) * attn[i] + a * attn[j]
    return p, w


def style_axes(ax, title=None):
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def make_animation(pos, attn, path, label, fps=12, frames_per_step=3, n_gaze=24):
    n_steps, n_heads, n, _ = attn.shape
    focal = int(np.argmin(-(attn[-1, 0] * np.log(attn[-1, 0] + 1e-12)).sum(-1)))  # most selective boid, head 0, last step
    gaze_idx = np.linspace(0, n - 1, n_gaze).astype(int)
    cols = ["attention received", f"gaze of {n_gaze} boids", f"attention paid by boid #{focal}"]

    fig, axes = plt.subplots(n_heads, 3, figsize=(9.6, 3.3 * n_heads))
    axes = np.atleast_2d(axes)
    p0, w0 = frame_data(pos, attn, 0.0)
    artists = []
    for h in range(n_heads):
        row = {}
        ax = axes[h, 0]
        row["recv"] = ax.scatter(p0[:, 0], p0[:, 1], s=16, c=w0[h].sum(0), cmap=CMAP, norm=NORM, linewidths=0)
        style_axes(ax, f"head {h}: {cols[0]}" if h == 0 else f"head {h}")

        ax = axes[h, 1]
        row["gaze_bg"] = ax.scatter(p0[:, 0], p0[:, 1], s=8, c="lightgray", linewidths=0)
        target = w0[h] @ p0
        row["gaze_q"] = ax.quiver(
            p0[gaze_idx, 0], p0[gaze_idx, 1],
            target[gaze_idx, 0] - p0[gaze_idx, 0], target[gaze_idx, 1] - p0[gaze_idx, 1],
            angles="xy", scale_units="xy", scale=1, width=0.004, color="tab:blue", alpha=0.7,
        )
        row["centroid"] = ax.plot(p0[:, 0].mean(), p0[:, 1].mean(), "x", color="black", ms=8, mew=1.5)[0]
        style_axes(ax, cols[1] if h == 0 else None)

        ax = axes[h, 2]
        row["focal"] = ax.scatter(p0[:, 0], p0[:, 1], s=16, c=w0[h][focal] * n, cmap=CMAP, norm=NORM, linewidths=0)
        row["focal_star"] = ax.plot(p0[focal, 0], p0[focal, 1], "*", color="black", ms=13, mew=0.5, mfc="yellow")[0]
        style_axes(ax, cols[2] if h == 0 else None)
        artists.append(row)

    cbar = fig.colorbar(artists[0]["recv"], ax=axes[:, [0, 2]].ravel().tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("relative to uniform attention (1 = uniform)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    suptitle = fig.suptitle("", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.9, top=0.9, bottom=0.02, wspace=0.08, hspace=0.12)

    total = n_steps * frames_per_step + fps

    def update(f):
        k = min(f / frames_per_step, n_steps)
        p, w = frame_data(pos, attn, k)
        ent = -(w * np.log(w + 1e-12)).sum(-1).mean(-1) / math.log(n)  # per head
        for h, row in enumerate(artists):
            row["recv"].set_offsets(p)
            row["recv"].set_array(w[h].sum(0))
            row["gaze_bg"].set_offsets(p)
            target = w[h] @ p
            row["gaze_q"].set_offsets(p[gaze_idx])
            row["gaze_q"].set_UVC(target[gaze_idx, 0] - p[gaze_idx, 0], target[gaze_idx, 1] - p[gaze_idx, 1])
            row["centroid"].set_data([p[:, 0].mean()], [p[:, 1].mean()])
            row["focal"].set_offsets(p)
            row["focal"].set_array(w[h][focal] * n)
            row["focal_star"].set_data([p[focal, 0]], [p[focal, 1]])
        suptitle.set_text(
            f"class {label}, t = {k:4.1f}   |   attention entropy: " + ", ".join(f"head {h} {100 * e:.1f}% of max" for h, e in enumerate(ent))
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=total, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def make_strip(pos, attn, path, label, steps=(0, 5, 10, 15)):
    """Static version of the animation for a few timesteps: rows are the six panels, columns are time."""
    n_steps, n_heads, n, _ = attn.shape
    focal = int(np.argmin(-(attn[-1, 0] * np.log(attn[-1, 0] + 1e-12)).sum(-1)))
    gaze_idx = np.linspace(0, n - 1, 24).astype(int)
    rows = [(kind, h) for kind in ("received", "gaze", "focal") for h in range(n_heads)]
    fig, axes = plt.subplots(len(rows), len(steps), figsize=(1.9 * len(steps), 1.9 * len(rows)))
    for c, t in enumerate(steps):
        p, w = pos[t], attn[min(t, n_steps - 1)]
        for r, (kind, h) in enumerate(rows):
            ax = axes[r, c]
            if kind == "received":
                ax.scatter(p[:, 0], p[:, 1], s=7, c=w[h].sum(0), cmap=CMAP, norm=NORM, linewidths=0)
            elif kind == "gaze":
                ax.scatter(p[:, 0], p[:, 1], s=4, c="lightgray", linewidths=0)
                target = w[h] @ p
                ax.quiver(p[gaze_idx, 0], p[gaze_idx, 1], target[gaze_idx, 0] - p[gaze_idx, 0], target[gaze_idx, 1] - p[gaze_idx, 1],
                          angles="xy", scale_units="xy", scale=1, width=0.006, color="tab:blue", alpha=0.7)
                ax.plot(p[:, 0].mean(), p[:, 1].mean(), "x", color="black", ms=5)
            else:
                ax.scatter(p[:, 0], p[:, 1], s=7, c=w[h][focal] * n, cmap=CMAP, norm=NORM, linewidths=0)
                ax.plot(p[focal, 0], p[focal, 1], "*", color="black", ms=8, mfc="yellow", mew=0.4)
            style_axes(ax, f"t={t}" if r == 0 else None)
            if c == 0:
                ax.set_ylabel(f"{kind}\nhead {h}", fontsize=8)
    fig.suptitle(f"class {label}: attention over time (colour: relative to uniform, 0.5-2)", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.01, wspace=0.05, hspace=0.08)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--label", type=int, default=7)
    parser.add_argument("--style", type=float, nargs=2, default=[0.6, -0.4])
    parser.add_argument("--n-boids", type=int, default=128)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    swarm, _, _ = load_checkpoint(args.checkpoint, device)
    swarm.eval().resize(args.n_boids)
    pos, attn = run(swarm, args.label, args.style, device)
    out = Path(args.out or Path(args.checkpoint).parent)

    # What correlates with being listened to?
    n = pos.shape[1]
    for t in (5, 15):
        recv = attn[t].sum(1)  # (H, N)
        p = pos[t]
        r = np.sqrt(((p - p.mean(0)) ** 2).sum(-1))
        for h in range(attn.shape[1]):
            cx, cy, cr = (np.corrcoef(recv[h], v)[0, 1] for v in (p[:, 0], p[:, 1], r))
            print(f"t={t} head {h}: received attention range {recv[h].min():.2f}-{recv[h].max():.2f}; corr with x {cx:+.2f}, y {cy:+.2f}, distance from centroid {cr:+.2f}")

    make_strip(pos, attn, out / f"attention_strip_{args.label}.png", args.label)
    make_animation(pos, attn, out / f"attention_{args.label}.gif", args.label)
    print(f"wrote {out}/attention_{args.label}.gif and attention_strip_{args.label}.png")


if __name__ == "__main__":
    main()
