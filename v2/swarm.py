"""A recurrent transformer swarm whose final positions form an MNIST digit.

Each boid is a point in R^d, d = 2 (picture plane) + 2 (class code) + d_style.
At t=0 the picture-plane coordinates are a fixed sunflower disk (boid identity),
and the hidden coordinates are identical for every boid: the class code and the
style latent. The prompt is part of the habitat. One weight-shared block is
applied `steps` times; there is no positional encoding, layer norm or dropout.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sunflower(n, radius=0.9):
    """Fibonacci-spiral disk: maximally even, canonical, boid i is always at the same spot."""
    i = torch.arange(n, dtype=torch.float32) + 0.5
    r = radius * torch.sqrt(i / n)
    theta = math.pi * (1 + 5**0.5) * i
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)


def class_codes(num_classes=10, radius=1.0):
    ang = 2 * math.pi * torch.arange(num_classes, dtype=torch.float32) / num_classes
    return radius * torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)


class Block(nn.Module):
    """One timestep: a flocking move (attention) then an individual move (feed-forward).

    Heads live in the full d-dimensional habitat and are averaged rather than
    concatenated, as in v1. Output layers start at zero so the untrained swarm
    stands still.
    """

    def __init__(self, d, nhead, hidden, step_scale, ffn_layers=1):
        super().__init__()
        self.d = d
        self.nhead = nhead
        self.step_scale = step_scale
        self.qkv = nn.Linear(d, 3 * nhead * d, bias=False)
        self.out = nn.Linear(d, d)
        layers = [nn.Linear(d, hidden), nn.GELU()]
        for _ in range(ffn_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, d))
        self.ffn = nn.Sequential(*layers)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x, update_mask=None):
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.nhead, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (b, h, n, d)
        msg = F.scaled_dot_product_attention(q, k, v, scale=1.0).mean(dim=1)
        flock = self.step_scale * self.out(msg)
        x = x + (flock if update_mask is None else flock * update_mask)
        solo = self.step_scale * self.ffn(x)
        x = x + (solo if update_mask is None else solo * update_mask)
        return x


class Swarm(nn.Module):
    def __init__(
        self,
        n_boids=256,
        d_style=2,
        nhead=4,
        hidden=512,
        steps=16,
        step_scale=0.1,
        disk_radius=0.9,
        code_radius=1.0,
        num_classes=10,
        pin_class=False,
        ffn_layers=1,
    ):
        super().__init__()
        self.n_boids = n_boids
        self.d_style = d_style
        self.steps = steps
        self.d = 2 + 2 + d_style
        self.disk_radius = disk_radius
        self.register_buffer("init_xy", sunflower(n_boids, disk_radius))
        self.register_buffer("codes", class_codes(num_classes, code_radius))
        # Optionally pin the class-code coordinates: the prompt then stays legible for all 16 steps
        # instead of being overwritten by the block's own updates to the hidden dimensions.
        mask = torch.ones(self.d)
        if pin_class:
            mask[2:4] = 0.0
        self.register_buffer("update_mask", mask, persistent=False)
        self.block = Block(self.d, nhead, hidden, step_scale, ffn_layers)

    def resize(self, n_boids):
        """Change the swarm size at test time; the dynamics are permutation-equivariant and roughly mean-field."""
        self.n_boids = n_boids
        self.init_xy = sunflower(n_boids, self.disk_radius).to(self.init_xy.device)
        return self

    def initial_state(self, labels, style):
        b = labels.shape[0]
        xy = self.init_xy.unsqueeze(0).expand(b, -1, -1)
        hidden = torch.cat([self.codes[labels], style], dim=-1)
        hidden = hidden.unsqueeze(1).expand(-1, self.n_boids, -1)
        return torch.cat([xy, hidden], dim=-1)

    def forward(self, labels, style, return_traj=False):
        x = self.initial_state(labels, style)
        traj = [x]
        for _ in range(self.steps):
            x = self.block(x, self.update_mask)
            if return_traj:
                traj.append(x)
        return (x, traj) if return_traj else x


class Encoder(nn.Module):
    """Training-time scaffolding: image -> q(style | image). Not part of the swarm."""

    def __init__(self, d_style=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.GELU(),
            nn.Linear(256, 2 * d_style),
        )
        self.d_style = d_style

    def forward(self, img):
        out = self.net(img.unsqueeze(1))
        return out[:, : self.d_style], out[:, self.d_style :]


# --- rendering and loss -----------------------------------------------------------------------
# Picture plane is [-1, 1]^2; image row 0 is the top (y = +1), column 0 is the left (x = -1).


def pixel_centers(res, device=None):
    g = (torch.arange(res, device=device, dtype=torch.float32) + 0.5) / res * 2 - 1
    return g, -g  # x of columns, y of rows


def splat(points, weights, sigma_px, res=28):
    """Render weighted points as a Gaussian-blurred density on a res x res grid. (B, M, 2) -> (B, res, res)."""
    gx, gy = pixel_centers(res, points.device)
    s = sigma_px * (2.0 / res)
    kx = torch.exp(-0.5 * ((points[..., 0:1] - gx) / s) ** 2)  # (B, M, res)
    ky = torch.exp(-0.5 * ((points[..., 1:2] - gy) / s) ** 2)
    return torch.einsum("bm,bmy,bmx->byx", weights, ky, kx) / (2 * math.pi * s * s)


def blur_image(img, sigma_px):
    """Same blur as `splat`, applied to a normalized image (B, res, res) sitting on the pixel grid."""
    res = img.shape[-1]
    g, _ = pixel_centers(res, img.device)
    s = sigma_px * (2.0 / res)
    k = torch.exp(-0.5 * ((g[:, None] - g[None, :]) / s) ** 2)
    return k @ img @ k / (2 * math.pi * s * s)


def ink_measure(img):
    return img / img.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def splat_mmd(xy, img, sigmas=(0.75, 1.5, 3.0)):
    """Multi-scale Gaussian MMD between the boid cloud and the digit's ink measure.

    Both sides are blurred with the same kernel and compared in L2 on the pixel
    grid, per scale normalized by the target's energy. A proper distribution
    metric, unlike Chamfer: it cannot be fooled by clumping.
    """
    b, n, _ = xy.shape
    w = torch.full((b, n), 1.0 / n, device=xy.device)
    target = ink_measure(img)
    total = 0.0
    for s in sigmas:
        pred = splat(xy, w, s, res=img.shape[-1])
        tgt = blur_image(target, s)
        num = ((pred - tgt) ** 2).mean(dim=(-2, -1))
        den = (tgt**2).mean(dim=(-2, -1)).clamp_min(1e-12)
        total = total + (num / den).mean()
    return total / len(sigmas)


def out_of_frame_penalty(xy, margin=1.0):
    """Boids outside the frame get no gradient from the splat, so pull them back explicitly."""
    return F.relu(xy.abs() - margin).pow(2).sum(dim=-1).mean()


def sample_ink(img, n, generator=None):
    """Sample n points per image from the ink measure, jittered uniformly within their pixel. (B, res, res) -> (B, n, 2)."""
    b, res, _ = img.shape
    probs = ink_measure(img).reshape(b, -1)
    idx = torch.multinomial(probs, n, replacement=True, generator=generator)  # (B, n)
    rows, cols = idx // res, idx % res
    jitter = torch.rand(b, n, 2, device=img.device, generator=generator)
    x = -1 + (cols + jitter[..., 0]) * (2.0 / res)
    y = 1 - (rows + jitter[..., 1]) * (2.0 / res)
    return torch.stack([x, y], dim=-1)


def sliced_wasserstein(xy, target, n_proj=64):
    """Sliced W2^2 between two equal-size point sets. (B, N, 2) x (B, N, 2) -> scalar.

    Along each random direction the 1-D optimal transport is just sorting, so
    every boid gets an exact, non-vanishing gradient towards the ink no matter
    how far away it is.
    """
    theta = torch.randn(n_proj, 2, device=xy.device)
    theta = theta / theta.norm(dim=-1, keepdim=True)
    p = (xy @ theta.T).sort(dim=1).values  # (B, N, n_proj)
    q = (target @ theta.T).sort(dim=1).values
    return (p - q).pow(2).mean()
