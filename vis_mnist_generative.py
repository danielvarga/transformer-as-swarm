import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib.animation import FuncAnimation, PillowWriter

from mnist_generative import (
    RunConfig,
    load_model,
    make_init_sequences,
    set_seed,
    tokenize_image,
    torch_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize generations from a trained MNIST boid generator.")
    parser.add_argument("--checkpoint-path", default="model_generative.pth")
    parser.add_argument("--label", type=int, default=5, help="Digit label to generate, restricted to 0-7.")
    parser.add_argument("--sample-index", type=int, default=0, help="Which generated sample from the batch to render.")
    parser.add_argument("--num-samples", type=int, default=4, help="How many generations to sample in one batch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-set", action="store_true", help="Use the training split for the reference digit.")
    parser.add_argument("--reference-index", type=int, default=0, help="Nth example of the requested label to use as reference.")
    parser.add_argument("--output-prefix", help="Prefix for generated files. Defaults to checkpoint/label/seed-based naming.")
    parser.add_argument("--save-gif", action="store_true", help="Also save an animated GIF of the boid dynamics.")
    parser.add_argument("--frames-per-step", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--show", action="store_true", help="Open the static figure after saving it.")
    return parser.parse_args()


def normalize_for_plot(tokens):
    tokens = np.asarray(tokens)
    x = tokens[:, 1]
    y = -tokens[:, 0]
    z = tokens[:, 2] if tokens.shape[1] > 2 else np.zeros(len(tokens))
    return x, y, z


def output_prefix(args, checkpoint_path):
    if args.output_prefix:
        return Path(args.output_prefix)
    checkpoint_stem = Path(checkpoint_path).stem
    return Path(f"{checkpoint_stem}.label{args.label}.seed{args.seed}")


def make_generation_batch(config, label, num_samples):
    lengths = torch.full((num_samples,), config.fixed_boid_count, dtype=torch.long, device=torch_device)
    init_tokens = make_init_sequences(config, num_samples)
    labels = torch.full((num_samples,), label, dtype=torch.long, device=torch_device)
    return init_tokens, lengths, labels


def load_reference_tokens(label, config, train_set, reference_index):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root="./data", train=train_set, download=True, transform=transform)

    match_count = 0
    for image, image_label in dataset:
        if image_label != label:
            continue
        if match_count == reference_index:
            return tokenize_image(image, config.habitat_scaling_factor).cpu().numpy()
        match_count += 1

    raise ValueError(f"Could not find label {label} at reference index {reference_index}.")


def extract_sample_activations(layer_outputs, sample_index, boid_count):
    sample_layers = []
    for layer_output in layer_outputs:
        sample_layers.append(layer_output[sample_index, :boid_count].detach().cpu().numpy())
    return np.stack(sample_layers, axis=0)


def plot_static(sample_acts, reference_tokens, label, png_path, dpi, habitat_scale):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("Initial swarm", sample_acts[0]),
        ("Final generation", sample_acts[-1]),
        ("Reference digit", reference_tokens),
    ]

    for ax, (title, points) in zip(axes, panels):
        x, y, z = normalize_for_plot(points)
        if title == "Reference digit":
            ax.scatter(x, y, s=12, c="#444444", alpha=0.9)
        else:
            ax.scatter(x, y, s=12, c=z, cmap="coolwarm", alpha=0.85, vmin=-habitat_scale, vmax=habitat_scale)
        ax.set_title(title)
        ax.set_xlim(-habitat_scale, habitat_scale)
        ax.set_ylim(-habitat_scale, habitat_scale)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    fig.suptitle(f"Conditioned generation for digit {label}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi)
    return fig


def save_animation(sample_acts, reference_tokens, label, gif_path, frames_per_step, dpi, habitat_scale):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    gen_ax, ref_ax = axes

    ref_x, ref_y, _ = normalize_for_plot(reference_tokens)
    ref_ax.scatter(ref_x, ref_y, s=12, c="#444444", alpha=0.9)
    ref_ax.set_title("Reference digit")
    ref_ax.set_xlim(-habitat_scale, habitat_scale)
    ref_ax.set_ylim(-habitat_scale, habitat_scale)
    ref_ax.set_aspect("equal")
    ref_ax.grid(alpha=0.2)

    gen_ax.set_title(f"Generation for digit {label}")
    gen_ax.set_xlim(-habitat_scale, habitat_scale)
    gen_ax.set_ylim(-habitat_scale, habitat_scale)
    gen_ax.set_aspect("equal")
    gen_ax.grid(alpha=0.2)

    x0, y0, z0 = normalize_for_plot(sample_acts[0])
    scatter = gen_ax.scatter(x0, y0, s=12, c=z0, cmap="coolwarm", alpha=0.85, vmin=-habitat_scale, vmax=habitat_scale)

    total_frames = max((len(sample_acts) - 1) * frames_per_step, 1)

    def update(frame):
        t1 = min(frame // frames_per_step, len(sample_acts) - 1)
        t2 = min(t1 + 1, len(sample_acts) - 1)
        alpha = 0.0 if t1 == t2 else (frame % frames_per_step) / frames_per_step
        points = (1 - alpha) * sample_acts[t1] + alpha * sample_acts[t2]
        x, y, z = normalize_for_plot(points)
        scatter.set_offsets(np.column_stack([x, y]))
        scatter.set_array(z)
        gen_ax.set_title(f"Generation for digit {label} (step {t1}/{len(sample_acts) - 1})")
        return (scatter,)

    animation = FuncAnimation(fig, update, frames=total_frames, interval=60, blit=False)
    animation.save(gif_path, writer=PillowWriter(fps=20), dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    if not 0 <= args.label <= 7:
        raise ValueError("label must be in 0..7.")
    if args.sample_index < 0 or args.sample_index >= args.num_samples:
        raise ValueError("sample_index must be in [0, num_samples).")

    set_seed(args.seed)
    model, checkpoint_config = load_model(args.checkpoint_path)
    config = RunConfig(**checkpoint_config.__dict__)

    init_tokens, lengths, labels = make_generation_batch(config, args.label, args.num_samples)
    with torch.no_grad():
        layer_outputs = model(init_tokens, lengths, labels, return_all_layers=True)

    sample_acts = extract_sample_activations(layer_outputs, args.sample_index, config.fixed_boid_count)
    reference_tokens = load_reference_tokens(args.label, config, args.train_set, args.reference_index)

    prefix = output_prefix(args, args.checkpoint_path)
    png_path = prefix.with_suffix(".png")
    plot_static(
        sample_acts,
        reference_tokens,
        label=args.label,
        png_path=png_path,
        dpi=args.dpi,
        habitat_scale=config.habitat_scaling_factor,
    )
    print(f"saved {png_path}")

    if args.save_gif:
        gif_path = prefix.with_suffix(".gif")
        save_animation(
            sample_acts,
            reference_tokens,
            label=args.label,
            gif_path=gif_path,
            frames_per_step=args.frames_per_step,
            dpi=args.dpi,
            habitat_scale=config.habitat_scaling_factor,
        )
        print(f"saved {gif_path}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
