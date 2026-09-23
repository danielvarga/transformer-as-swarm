"""Train the swarm as a class-conditional VAE decoder on MNIST."""

import argparse
import json
import time
from pathlib import Path

import torch
import torchvision

from swarm import Encoder, Swarm, out_of_frame_penalty, sample_ink, sliced_wasserstein, splat_mmd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def load_mnist(train, device):
    ds = torchvision.datasets.MNIST(root=str(DATA_ROOT), train=train, download=True)
    images = ds.data.to(torch.float32).div_(255.0).to(device)  # (n, 28, 28), whole set lives on device
    labels = ds.targets.to(device)
    return images, labels


def build(cfg, device):
    swarm = Swarm(
        n_boids=cfg["n_boids"],
        d_style=cfg["d_style"],
        nhead=cfg["nhead"],
        hidden=cfg["hidden"],
        steps=cfg["steps"],
        step_scale=cfg["step_scale"],
        code_radius=cfg.get("code_radius", 1.0),
        pin_class=cfg.get("pin_class", False),
        ffn_layers=cfg.get("ffn_layers", 1),
    ).to(device)
    encoder = Encoder(cfg["d_style"]).to(device)
    return swarm, encoder


def save_checkpoint(path, swarm, encoder, cfg, epoch):
    torch.save({"swarm": swarm.state_dict(), "encoder": encoder.state_dict(), "cfg": cfg, "epoch": epoch}, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    swarm, encoder = build(ckpt["cfg"], device)
    swarm.load_state_dict(ckpt["swarm"])
    encoder.load_state_dict(ckpt["encoder"])
    return swarm, encoder, ckpt["cfg"]


def kl_to_prior(mu, logvar):
    return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(dim=-1).mean()


@torch.no_grad()
def evaluate(swarm, encoder, images, labels, cfg, batch_size=256):
    swarm.eval()
    encoder.eval()
    recon_total, kl_total, n = 0.0, 0.0, 0
    for i in range(0, images.shape[0], batch_size):
        img, lab = images[i : i + batch_size], labels[i : i + batch_size]
        mu, logvar = encoder(img)
        xy = swarm(lab, mu)[..., :2]
        recon_total += splat_mmd(xy, img, cfg["sigmas"]).item() * img.shape[0]
        kl_total += kl_to_prior(mu, logvar).item() * img.shape[0]
        n += img.shape[0]
    return recon_total / n, kl_total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/base")
    parser.add_argument("--n-boids", type=int, default=128)
    parser.add_argument("--d-style", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--ffn-layers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--step-scale", type=float, default=0.1)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.75, 1.5, 3.0])
    parser.add_argument("--mmd-weight", type=float, default=1.0)
    parser.add_argument("--swd-weight", type=float, default=0.0, help="sliced-Wasserstein weight (its scale is ~1/20 of the MMD's)")
    parser.add_argument("--n-proj", type=int, default=64)
    parser.add_argument("--code-radius", type=float, default=1.0)
    parser.add_argument("--pin-class", action="store_true", help="block never writes to the class-code coordinates")
    parser.add_argument("--beta", type=float, default=0.01, help="final KL weight")
    parser.add_argument("--beta-warmup", type=float, default=2, help="epochs of linear KL warmup")
    parser.add_argument("--frame-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = vars(args)

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))

    train_img, train_lab = load_mnist(train=True, device=device)
    test_img, test_lab = load_mnist(train=False, device=device)
    if args.max_train:
        train_img, train_lab = train_img[: args.max_train], train_lab[: args.max_train]
    test_img, test_lab = test_img[: args.max_test], test_lab[: args.max_test]

    swarm, encoder = build(cfg, device)
    params = list(swarm.parameters()) + list(encoder.parameters())
    print(f"device {device}, swarm params {sum(p.numel() for p in swarm.parameters()):,}, encoder params {sum(p.numel() for p in encoder.parameters()):,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    n_train = train_img.shape[0]
    steps_per_epoch = n_train // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1, anneal_strategy="cos"
    )

    from vis import sample_grid

    best = float("inf")
    step = 0
    for epoch in range(args.epochs):
        swarm.train()
        encoder.train()
        perm = torch.randperm(n_train, device=device)
        t0 = time.time()
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
            img, lab = train_img[idx], train_lab[idx]
            beta = args.beta * min(1.0, (epoch + i / steps_per_epoch) / max(args.beta_warmup, 1e-9))

            mu, logvar = encoder(img)
            style = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            xy = swarm(lab, style)[..., :2]
            recon = 0.0
            if args.mmd_weight:
                recon = recon + args.mmd_weight * splat_mmd(xy, img, args.sigmas)
            if args.swd_weight:
                recon = recon + args.swd_weight * sliced_wasserstein(xy, sample_ink(img, xy.shape[1]), args.n_proj)
            kl = kl_to_prior(mu, logvar)
            frame = out_of_frame_penalty(xy)
            loss = recon + beta * kl + args.frame_weight * frame

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0:
                print(
                    f"epoch {epoch} step {i}/{steps_per_epoch} recon {recon.item():.4f} kl {kl.item():.3f} "
                    f"frame {frame.item():.5f} beta {beta:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{(time.time() - t0) / (i + 1):.3f}s/it",
                    flush=True,
                )
        test_recon, test_kl = evaluate(swarm, encoder, test_img, test_lab, cfg)
        print(f"=== epoch {epoch} done in {time.time() - t0:.0f}s: test recon {test_recon:.4f} test kl {test_kl:.3f}", flush=True)
        save_checkpoint(run / "last.pt", swarm, encoder, cfg, epoch)
        if test_recon < best:
            best = test_recon
            save_checkpoint(run / "best.pt", swarm, encoder, cfg, epoch)
        sample_grid(swarm, run / f"samples_epoch{epoch:02d}.png", device)


if __name__ == "__main__":
    main()
