"""Side-by-side comparison of several runs: one sample grid per run plus a table of diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from train import evaluate, load_checkpoint, load_mnist
from vis import boid_colors, scatter


@torch.no_grad()
def diagnostics(swarm, encoder, cfg, images, labels):
    test_recon, test_kl = evaluate(swarm, encoder, images, labels, cfg)
    dev = images.device
    zero = torch.zeros(10, swarm.d_style, device=dev)
    by_label = swarm(torch.arange(10, device=dev), zero)[..., :2]
    label_sens = (by_label[1:] - by_label[:-1]).norm(dim=-1).mean().item()
    st = torch.zeros(10, swarm.d_style, device=dev)
    st[:, 0] = torch.linspace(-2, 2, 10)
    by_style = swarm(torch.zeros(10, dtype=torch.long, device=dev), st)[..., :2]
    style_sens = (by_style[1:] - by_style[:-1]).norm(dim=-1).mean().item()
    mu, _ = encoder(images)
    half = images.shape[0] // 2
    cent = torch.stack([mu[:half][labels[:half] == c].mean(0) for c in range(10)])
    leak = (torch.cdist(mu[half:], cent).argmin(1) == labels[half:]).float().mean().item()
    return dict(test_recon=test_recon, test_kl=test_kl, label_sens=label_sens, style_sens=style_sens, leak_acc=leak)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--ckpt", default="last.pt")
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--out", default="runs/compare.png")
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    images, labels = load_mnist(train=False, device=device)
    images, labels = images[:2000], labels[:2000]

    gen = torch.Generator().manual_seed(0)
    fig, axes = plt.subplots(10, args.per_class * len(args.runs), figsize=(1.1 * args.per_class * len(args.runs), 11))
    print(f"{'run':24s} {'recon':>7s} {'kl':>6s} {'label':>7s} {'style':>7s} {'leak':>6s}")
    for r, run in enumerate(args.runs):
        swarm, encoder, cfg = load_checkpoint(Path(run) / args.ckpt, device)
        swarm.eval()
        encoder.eval()
        d = diagnostics(swarm, encoder, cfg, images, labels)
        print(f"{Path(run).name:24s} {d['test_recon']:7.4f} {d['test_kl']:6.2f} {d['label_sens']:7.4f} {d['style_sens']:7.4f} {d['leak_acc']:6.3f}")
        lab = torch.arange(10).repeat_interleave(args.per_class).to(device)
        style = torch.randn(10 * args.per_class, swarm.d_style, generator=gen).to(device)
        final = swarm(lab, style)[..., :2].cpu().numpy()
        colors = boid_colors(swarm.init_xy)
        for i in range(10 * args.per_class):
            ax = axes[i // args.per_class, r * args.per_class + i % args.per_class]
            scatter(ax, final[i], colors, size=3)
            for sp in ax.spines.values():
                sp.set_linewidth(0.3)
        axes[0, r * args.per_class].set_title(Path(run).name, fontsize=8, loc="left")
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0.01, right=0.99, top=0.97, bottom=0.01)
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
