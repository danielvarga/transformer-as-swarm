"""Export the swarm (not the encoder) from a checkpoint to weights.js for the browser app.

Weights are little-endian float32, base64-encoded, row-major as in PyTorch (out, in).
Also writes reference trajectories so the JS port can be checked against PyTorch.

    uv run --with torch python web/export_weights.py models/final.pt
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from swarm import Swarm  # noqa: E402


def b64(t):
    return base64.b64encode(t.detach().cpu().numpy().astype("<f4").tobytes()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "weights.js"))
    parser.add_argument("--ref", default=str(Path(__file__).resolve().parent / "reference.json"))
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
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
    )
    swarm.load_state_dict(ck["swarm"])
    swarm.eval()

    sd = swarm.block.state_dict()
    linears = [m for m in swarm.block.ffn if isinstance(m, torch.nn.Linear)]
    out = {
        "d": swarm.d,
        "nhead": cfg["nhead"],
        "steps": cfg["steps"],
        "step_scale": cfg["step_scale"],
        "disk_radius": swarm.disk_radius,
        "update_mask": swarm.update_mask.tolist(),
        "codes": swarm.codes.tolist(),
        "trained_n_boids": cfg["n_boids"],
        "qkv": b64(sd["qkv.weight"]),
        "out_w": b64(sd["out.weight"]),
        "out_b": b64(sd["out.bias"]),
        "ffn": [{"w": b64(m.weight), "b": b64(m.bias), "in": m.in_features, "out": m.out_features} for m in linears],
    }
    # A script rather than JSON, so index.html also works when opened straight from disk (file:// blocks fetch).
    Path(args.out).write_text("window.SWARM_WEIGHTS = " + json.dumps(out) + ";\n")

    cases = [(7, [0.5, -0.3], 64), (3, [-1.2, 0.8], 256), (0, [0.0, 0.0], 100)]
    refs = []
    with torch.no_grad():
        for label, style, n in cases:
            swarm.resize(n)
            _, traj = swarm(torch.tensor([label]), torch.tensor([style]), return_traj=True)
            refs.append({"label": label, "style": style, "n": n, "final": traj[-1][0].tolist(), "step1": traj[1][0].tolist()})
    Path(args.ref).write_text(json.dumps(refs))
    n_params = sum(p.numel() for p in swarm.block.parameters())
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KB, {n_params} params) and {args.ref}")


if __name__ == "__main__":
    main()
