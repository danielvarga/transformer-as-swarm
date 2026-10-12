"""Visualize the alone-time behaviour: the feed-forward block as a vector field on the picture plane.

The field lives on R^6. The class code is pinned, but the style coordinates are free state
that the FFN itself rewrites, so the effective 2-D field changes as the swarm's hidden state
drifts. Figures show it at the swarm-mean hidden state of each timestep, and separately
decompose each boid's actual per-step move into its flocking and alone-time parts.
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
from vis import boid_colors

DIV_CMAP = "RdBu_r"  # blue = contracting (negative divergence), red = expanding
DIV_NORM = mcolors.TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=0.15)
EXTENT = 1.1


@torch.no_grad()
def field_on_grid(swarm, hidden, g, device):
    """(x, y) part of the alone-time step on a g x g grid, hidden coordinates fixed. Returns X, Y, U, V, div."""
    xs = np.linspace(-EXTENT, EXTENT, g)
    X, Y = np.meshgrid(xs, xs)  # X[j, i] = xs[i], Y[j, i] = xs[j]
    xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32, device=device)
    h = torch.as_tensor(hidden, dtype=torch.float32, device=device).view(1, -1).expand(xy.shape[0], -1)
    f = swarm.block.field(torch.cat([xy, h], 1), swarm.update_mask)[:, :2].cpu().numpy().reshape(g, g, 2)
    U, V = f[..., 0], f[..., 1]
    dx = xs[1] - xs[0]
    div = np.gradient(U, dx, axis=1) + np.gradient(V, dx, axis=0)
    return X, Y, U, V, div


def draw_field(ax, swarm, hidden, device, g=40, stream=True, quiver_g=16, boids=None, boid_colors_=None, title=None):
    X, Y, U, V, div = field_on_grid(swarm, hidden, g, device)
    ax.imshow(div, origin="lower", extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), cmap=DIV_CMAP, norm=DIV_NORM, interpolation="bilinear")
    if stream:
        ax.streamplot(X, Y, U, V, density=1.1, color="black", linewidth=0.6, arrowsize=0.7)
    else:
        Xq, Yq, Uq, Vq, _ = field_on_grid(swarm, hidden, quiver_g, device)
        ax.quiver(Xq, Yq, Uq, Vq, angles="xy", scale_units="xy", scale=0.25, width=0.004, color="black")
    if boids is not None:
        ax.scatter(boids[:, 0], boids[:, 1], s=5, c=boid_colors_ if boid_colors_ is not None else "black", linewidths=0, zorder=3)
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


@torch.no_grad()
def class_fields_figure(swarm, style, path, device):
    """The alone-time field at t=0 for every class (exact: all boids share the hidden state then), with the final digit overlaid."""
    n_classes = swarm.codes.shape[0]
    labels = torch.arange(n_classes, device=device)
    st = torch.tensor([style] * n_classes, dtype=torch.float32, device=device)
    final = swarm(labels, st)[..., :2].cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.4))
    for c, ax in enumerate(axes.ravel()):
        hidden = torch.cat([swarm.codes[c], st[c]]).cpu().numpy()
        draw_field(ax, swarm, hidden, device, boids=final[c], title=f"class {c}")
    fig.suptitle("Alone-time field at t=0 (streamlines; blue = contracting, red = expanding), with the final swarm overlaid", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.04, hspace=0.12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def run(swarm, label, style, device):
    labels = torch.tensor([label], device=device)
    st = torch.tensor([style], dtype=torch.float32, device=device)
    traj, flocks, solos = swarm.decompose(labels, st)
    traj = torch.stack([t[0] for t in traj]).cpu().numpy()  # (T+1, N, 6)
    flocks = torch.stack([f[0, :, :2] for f in flocks]).cpu().numpy()  # (T, N, 2)
    solos = torch.stack([s[0, :, :2] for s in solos]).cpu().numpy()
    return traj, flocks, solos


def draw_moves(ax, pos, flock, solo, colors, scale=5.0, title=None):
    ax.scatter(pos[:, 0], pos[:, 1], s=7, c=colors, linewidths=0)
    ax.quiver(pos[:, 0], pos[:, 1], scale * flock[:, 0], scale * flock[:, 1], angles="xy", scale_units="xy", scale=1, width=0.004, color="tab:orange", label="flocking")
    ax.quiver(pos[:, 0], pos[:, 1], scale * solo[:, 0], scale * solo[:, 1], angles="xy", scale_units="xy", scale=1, width=0.004, color="tab:blue", label="alone-time")
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def strip_figure(swarm, traj, flocks, solos, path, label, device, steps=(0, 5, 10, 15)):
    colors = boid_colors(torch.as_tensor(traj[0, :, :2]))
    fig, axes = plt.subplots(2, len(steps), figsize=(3.0 * len(steps), 6.2))
    for c, t in enumerate(steps):
        hidden = traj[t, :, 2:].mean(0)
        spread = traj[t, :, 4:].std(0).mean()
        draw_field(axes[0, c], swarm, hidden, device, boids=traj[t, :, :2], boid_colors_=colors, title=f"t={t}: field at mean hidden state (style spread {spread:.2f})")
        draw_moves(axes[1, c], traj[t, :, :2], flocks[t], solos[t], colors, title=f"t={t}: next move x5")
    fig.suptitle(f"class {label}: the alone-time field (top; blue = contracting) morphs as the boids rewrite their own hidden state.\n"
                 "Bottom: each boid's next move, orange = flocking (attention), blue = alone-time (FFN).", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.04, hspace=0.12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_animation(swarm, traj, flocks, solos, path, label, device, fps=12, frames_per_step=3, quiver_g=16, fine_g=40):
    n_steps = flocks.shape[0]
    colors = boid_colors(torch.as_tensor(traj[0, :, :2]))
    # Precompute the field per integer step at the swarm-mean hidden state; frames interpolate.
    fields = [field_on_grid(swarm, traj[t, :, 2:].mean(0), quiver_g, device) for t in range(n_steps + 1)]
    divs = [field_on_grid(swarm, traj[t, :, 2:].mean(0), fine_g, device)[4] for t in range(n_steps + 1)]

    fig, (ax_f, ax_m) = plt.subplots(1, 2, figsize=(9.2, 4.8))
    X, Y, U, V, _ = fields[0]
    im = ax_f.imshow(divs[0], origin="lower", extent=(-EXTENT, EXTENT, -EXTENT, EXTENT), cmap=DIV_CMAP, norm=DIV_NORM, interpolation="bilinear")
    q = ax_f.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=0.25, width=0.004, color="black")
    sc_f = ax_f.scatter(traj[0, :, 0], traj[0, :, 1], s=10, c=colors, linewidths=0, zorder=3)
    for ax in (ax_f, ax_m):
        ax.set_xlim(-EXTENT, EXTENT)
        ax.set_ylim(-EXTENT, EXTENT)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    ax_f.set_title("alone-time field at the swarm-mean hidden state (arrows x4)", fontsize=9)
    cbar = fig.colorbar(im, ax=ax_f, fraction=0.04, pad=0.02)
    cbar.set_label("divergence (blue = contracting)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    sc_m = ax_m.scatter(traj[0, :, 0], traj[0, :, 1], s=10, c=colors, linewidths=0)
    scale = 5.0
    q_flock = ax_m.quiver(traj[0, :, 0], traj[0, :, 1], scale * flocks[0, :, 0], scale * flocks[0, :, 1], angles="xy", scale_units="xy", scale=1, width=0.004, color="tab:orange")
    q_solo = ax_m.quiver(traj[0, :, 0], traj[0, :, 1], scale * solos[0, :, 0], scale * solos[0, :, 1], angles="xy", scale_units="xy", scale=1, width=0.004, color="tab:blue")
    ax_m.set_title("each boid's next move x5: orange flocking, blue alone-time", fontsize=9)
    suptitle = fig.suptitle("", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.05)
    total = n_steps * frames_per_step + fps

    def update(f):
        k = min(f / frames_per_step, n_steps)
        i = min(int(math.floor(k)), n_steps - 1)
        a = min(k - i, 1.0)
        pos = (1 - a) * traj[i, :, :2] + a * traj[i + 1, :, :2]
        U = (1 - a) * fields[i][2] + a * fields[i + 1][2]
        V = (1 - a) * fields[i][3] + a * fields[i + 1][3]
        q.set_UVC(U, V)
        im.set_data((1 - a) * divs[i] + a * divs[i + 1])
        sc_f.set_offsets(pos)
        sc_m.set_offsets(pos)
        for qq, mv in ((q_flock, flocks[i]), (q_solo, solos[i])):
            qq.set_offsets(pos)
            qq.set_UVC(scale * mv[:, 0], scale * mv[:, 1])
        ratio = np.linalg.norm(solos[i], axis=1).mean() / max(np.linalg.norm(flocks[i], axis=1).mean(), 1e-9)
        suptitle.set_text(f"class {label}, t = {k:4.1f}   |   alone-time / flocking displacement = {ratio:.1f}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=total, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
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
    out = Path(args.out or Path(args.checkpoint).parent)

    class_fields_figure(swarm, args.style, out / "ffn_fields_t0.png", device)
    traj, flocks, solos = run(swarm, args.label, args.style, device)
    strip_figure(swarm, traj, flocks, solos, out / f"ffn_strip_{args.label}.png", args.label, device)
    make_animation(swarm, traj, flocks, solos, out / f"ffn_{args.label}.gif", args.label, device)
    print(f"wrote {out}/ffn_fields_t0.png, ffn_strip_{args.label}.png, ffn_{args.label}.gif")


if __name__ == "__main__":
    main()
