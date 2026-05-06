import argparse
import pickle
import random
import sys
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunConfig:
    recurrent: bool = True
    d_model: int = 3
    nhead: int = 10
    num_layers: int = 10
    dim_feedforward: int = 512
    habitat_scaling_factor: float = 10.0
    residual_scaling_factor: float = 0.1
    train_batch_size: int = 128
    eval_batch_size: int = 256
    lr: float = 0.003
    epoch_num: int = 80
    fixed_boid_count: int = 256
    init_mode: str = "grid"
    init_jitter: float = 0.0
    use_label_init: bool = True
    label_init_scale: float = 1.0
    z_reg_weight: float = 0.0
    out_of_bounds_reg_weight: float = 0.0
    progress_every: int = 200
    checkpoint_path: str = "model_generative.pth"
    vis_samples: int = 10
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    seed: int = 0
    skip_train: bool = False
    skip_vis: bool = False


class CustomDataset(Dataset):
    def __init__(self, target_tokens, labels):
        self.target_tokens = target_tokens
        self.labels = labels

    def __len__(self):
        return len(self.target_tokens)

    def __getitem__(self, idx):
        return self.target_tokens[idx], self.labels[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a recurrent transformer boid generator on MNIST.")
    parser.add_argument("--nonrecurrent", action="store_true", help="Disable weight sharing across layers.")
    parser.add_argument("--d-model", type=int, default=3)
    parser.add_argument("--nhead", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--habitat-scale", type=float, default=10.0)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--fixed-boid-count", type=int, default=256)
    parser.add_argument("--init-mode", choices=("random", "grid"), default="grid")
    parser.add_argument("--init-jitter", type=float, default=0.0)
    parser.add_argument("--use-label-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label-init-scale", type=float, default=1.0)
    parser.add_argument("--z-reg-weight", type=float, default=0.0)
    parser.add_argument("--out-of-bounds-reg-weight", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--checkpoint-path", default="model_generative.pth")
    parser.add_argument("--vis-samples", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-vis", action="store_true")
    args = parser.parse_args()

    config = RunConfig(
        recurrent=not args.nonrecurrent,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        habitat_scaling_factor=args.habitat_scale,
        residual_scaling_factor=args.residual_scale,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        epoch_num=args.epochs,
        fixed_boid_count=args.fixed_boid_count,
        init_mode=args.init_mode,
        init_jitter=args.init_jitter,
        use_label_init=args.use_label_init,
        label_init_scale=args.label_init_scale,
        z_reg_weight=args.z_reg_weight,
        out_of_bounds_reg_weight=args.out_of_bounds_reg_weight,
        progress_every=args.progress_every,
        checkpoint_path=args.checkpoint_path,
        vis_samples=args.vis_samples,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
        skip_train=args.skip_train,
        skip_vis=args.skip_vis,
    )
    validate_config(config)
    return config


def validate_config(config):
    if config.d_model < 3:
        raise ValueError("d_model must be at least 3 so the 3-bit label condition fits in latent space.")
    if config.nhead < 1:
        raise ValueError("nhead must be at least 1.")
    if config.num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    if config.dim_feedforward < 1:
        raise ValueError("dim_feedforward must be at least 1.")
    if config.fixed_boid_count < 1:
        raise ValueError("fixed_boid_count must be at least 1.")
    if config.init_jitter < 0:
        raise ValueError("init_jitter must be non-negative.")
    if config.label_init_scale < 0:
        raise ValueError("label_init_scale must be non-negative.")
    if config.z_reg_weight < 0 or config.out_of_bounds_reg_weight < 0:
        raise ValueError("regularization weights must be non-negative.")
    if config.train_batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("batch sizes must be at least 1.")
    if config.epoch_num < 0:
        raise ValueError("epochs must be non-negative.")
    if config.max_train_samples is not None and config.max_train_samples < 1:
        raise ValueError("max_train_samples must be positive.")
    if config.max_test_samples is not None and config.max_test_samples < 1:
        raise ValueError("max_test_samples must be positive.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labels_to_binary_targets(labels):
    if not torch.all((0 <= labels) & (labels <= 7)):
        raise ValueError("Labels must be in the range 0-7.")
    bits = torch.arange(2, -1, -1, device=labels.device)
    return (labels.unsqueeze(1) >> bits) & 1


def labels_to_condition_tokens(labels, habitat_scaling_factor):
    cond = labels_to_binary_targets(labels).to(torch.float32)
    cond = 2 * cond - 1
    return cond * habitat_scaling_factor


def tokenize_image(image, habitat_scaling_factor):
    pixels = image.squeeze()
    indices = (pixels > 0.5).nonzero(as_tuple=True)
    tokens = torch.stack(indices, dim=1).to(torch.float32)
    tokens /= 27
    tokens -= 0.5
    tokens *= 2 * habitat_scaling_factor
    return tokens


def sample_target_tokens(image, count, habitat_scaling_factor):
    tokens = tokenize_image(image, habitat_scaling_factor)
    if len(tokens) == 0:
        return (2 * torch.rand((count, 2), dtype=torch.float32) - 1) * habitat_scaling_factor
    if len(tokens) >= count:
        idx = torch.randperm(len(tokens))[:count]
    else:
        idx = torch.randint(0, len(tokens), (count,))
    return tokens[idx]


def load_mnist(config, train=True):
    transform = transforms.Compose([transforms.ToTensor()])
    print("loading dataset")
    mnist_data = torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    mnist_3bit = [(img, label) for img, label in mnist_data if label < 8]

    limit = config.max_train_samples if train else config.max_test_samples
    if limit is not None:
        mnist_3bit = mnist_3bit[:limit]

    print("tokenization")
    try:
        from tqdm.auto import tqdm

        token_iter = tqdm(mnist_3bit, desc="tokenizing", ncols=80, file=sys.stdout)
    except Exception:
        token_iter = mnist_3bit

    tokens = []
    labels = []
    for i, (img, label) in enumerate(token_iter):
        tokens.append(sample_target_tokens(img, config.fixed_boid_count, config.habitat_scaling_factor))
        labels.append(label)
        if i % config.progress_every == 0 and i > 0:
            print(f"tokenized {i}/{len(mnist_3bit)}")

    print("dataset preparation done")
    sys.stdout.flush()
    return tokens, labels


def collate_fn(batch, config):
    target_sequences, labels = zip(*batch)
    lengths = torch.full((len(target_sequences),), config.fixed_boid_count, dtype=torch.long, device=torch_device)
    padded_targets = pad_sequence(target_sequences, batch_first=True).to(torch_device)
    init_sequences = make_init_sequences(config, len(target_sequences))
    labels = torch.tensor(labels, dtype=torch.long, device=torch_device)
    return init_sequences, lengths, padded_targets, labels


def base_grid_tokens(count, habitat_scaling_factor):
    side = int(np.ceil(np.sqrt(count)))
    coords = torch.linspace(-habitat_scaling_factor, habitat_scaling_factor, side, device=torch_device)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)
    return grid[:count]


def make_init_sequences(config, batch_size):
    if config.init_mode == "random":
        init_sequences = (
            2 * torch.rand((batch_size, config.fixed_boid_count, 2), device=torch_device) - 1
        ) * config.habitat_scaling_factor
    elif config.init_mode == "grid":
        base = base_grid_tokens(config.fixed_boid_count, config.habitat_scaling_factor)
        init_sequences = base.unsqueeze(0).repeat(batch_size, 1, 1)
    else:
        raise ValueError(f"Unsupported init_mode: {config.init_mode}")

    if config.init_jitter > 0:
        init_sequences = init_sequences + config.init_jitter * torch.randn_like(init_sequences)

    return init_sequences.clamp(-config.habitat_scaling_factor, config.habitat_scaling_factor)


def create_dataloader(config, train=True, batch_size=None, shuffle=True):
    tokens, labels = load_mnist(config, train=train)
    dataset = CustomDataset(tokens, labels)
    effective_batch_size = batch_size or (config.train_batch_size if train else config.eval_batch_size)
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, config=config),
    )


class AveragedMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, batch_first=False):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.q_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(nhead)])
        self.k_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(nhead)])
        self.v_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(nhead)])
        self.batch_first = batch_first

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        scaling = float(self.d_model) ** -0.5
        head_outputs = []

        for head in range(self.nhead):
            q = self.q_projs[head](query) * scaling
            k = self.k_projs[head](key)
            v = self.v_projs[head](value)

            attn_weights = torch.bmm(q, k.transpose(1, 2))
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
            if attn_mask is not None:
                attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))

            attn_weights = torch.softmax(attn_weights, dim=-1)
            head_outputs.append(torch.bmm(attn_weights, v))

        output = torch.stack(head_outputs, dim=0).mean(dim=0)
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output, None


class ScaledTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, scaling_factor=1.0, dim_feedforward=512, **kwargs):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.self_attn = AveragedMultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.dropout = nn.Identity()
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.scaling_factor * src2
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + self.scaling_factor * src2
        return src


class MNISTBoidGenerator(nn.Module):
    def __init__(
        self,
        d_model=3,
        nhead=1,
        num_layers=10,
        recurrent=True,
        scaling_factor=0.1,
        dim_feedforward=512,
        habitat_scaling_factor=10.0,
        fixed_boid_count=256,
        use_label_init=False,
        label_init_scale=1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.recurrent = recurrent
        self.scaling_factor = scaling_factor
        self.habitat_scaling_factor = habitat_scaling_factor
        self.fixed_boid_count = fixed_boid_count
        self.use_label_init = use_label_init
        self.label_init_scale = label_init_scale
        if use_label_init:
            self.label_init = nn.Embedding(8, fixed_boid_count * 2)
            nn.init.zeros_(self.label_init.weight)
        self.encoder_layers = nn.ModuleList(
            [
                ScaledTransformerEncoderLayer(
                    d_model,
                    nhead,
                    batch_first=True,
                    scaling_factor=scaling_factor,
                    dim_feedforward=dim_feedforward,
                )
            ]
            * num_layers
            if recurrent
            else [
                ScaledTransformerEncoderLayer(
                    d_model,
                    nhead,
                    batch_first=True,
                    scaling_factor=scaling_factor,
                    dim_feedforward=dim_feedforward,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, tokens, lengths, labels, return_all_layers=False):
        if self.use_label_init:
            label_init = self.label_init(labels).view(-1, self.fixed_boid_count, 2)
            label_init = self.habitat_scaling_factor * torch.tanh(label_init)
            tokens = tokens + self.label_init_scale * label_init
            tokens = tokens.clamp(-self.habitat_scaling_factor, self.habitat_scaling_factor)

        tokens = torch.cat(
            [tokens, torch.zeros(tokens.shape[0], tokens.shape[1], self.d_model - 2, device=tokens.device)],
            dim=2,
        )

        cond = labels_to_condition_tokens(
            labels,
            habitat_scaling_factor=getattr(self, "habitat_scaling_factor", 10.0),
        ).to(tokens.device)
        cond = torch.cat(
            [cond, torch.zeros(cond.shape[0], self.d_model - 3, device=tokens.device)],
            dim=1,
        )
        cond = cond.unsqueeze(1)

        x = torch.cat([tokens, cond], dim=1)
        lengths_with_cond = lengths + 1
        max_len = x.size(1)
        attention_mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths_with_cond.unsqueeze(1)

        if return_all_layers:
            layer_outputs = [x.clone()]
            for layer in self.encoder_layers:
                x = layer(x, src_key_padding_mask=~attention_mask)
                x[:, -1, :] = cond.squeeze(1)
                layer_outputs.append(x.clone())
            return layer_outputs

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=~attention_mask)
            x[:, -1, :] = cond.squeeze(1)

        return x[:, :-1, :]


class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, lengths):
        valid = torch.arange(predictions.shape[1], device=predictions.device).unsqueeze(0) < lengths.unsqueeze(1)
        dist = torch.cdist(predictions[..., :2], targets[..., :2], p=2)

        pred_to_target = dist.masked_fill(~valid.unsqueeze(1), float("inf")).min(dim=2).values
        target_to_pred = dist.masked_fill(~valid.unsqueeze(2), float("inf")).min(dim=1).values

        normalizer = valid.sum().clamp_min(1)
        return (pred_to_target[valid].sum() + target_to_pred[valid].sum()) / normalizer


def build_model(config):
    return MNISTBoidGenerator(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        recurrent=config.recurrent,
        scaling_factor=config.residual_scaling_factor,
        dim_feedforward=config.dim_feedforward,
        habitat_scaling_factor=config.habitat_scaling_factor,
        fixed_boid_count=config.fixed_boid_count,
        use_label_init=config.use_label_init,
        label_init_scale=config.label_init_scale,
    ).to(torch_device)


def model_suffix(config):
    recurrent_str = "recurrent" if config.recurrent else "nonrecurrent"
    label_init_str = "linit" if config.use_label_init else "nolinit"
    return (
        f"gen_d{config.d_model}_b{config.num_layers}_ffwd{config.dim_feedforward}_"
        f"{config.init_mode}_{label_init_str}_{recurrent_str}"
    )


def save_checkpoint(model, config):
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
    }
    torch.save(payload, checkpoint_path)


def checkpoint_config_for_runtime(base_config, runtime_config):
    return replace(
        base_config,
        checkpoint_path=runtime_config.checkpoint_path,
        eval_batch_size=runtime_config.eval_batch_size,
        vis_samples=runtime_config.vis_samples,
        max_test_samples=runtime_config.max_test_samples,
        progress_every=runtime_config.progress_every,
        skip_train=runtime_config.skip_train,
        skip_vis=runtime_config.skip_vis,
        seed=runtime_config.seed,
    )


def config_from_legacy_model(model, checkpoint_path):
    first_layer = model.encoder_layers[0]
    return RunConfig(
        recurrent=getattr(model, "recurrent", True),
        d_model=model.d_model,
        nhead=first_layer.self_attn.nhead,
        num_layers=len(model.encoder_layers),
        dim_feedforward=first_layer.linear1.out_features,
        residual_scaling_factor=first_layer.scaling_factor,
        checkpoint_path=str(checkpoint_path),
    )


def load_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    except pickle.UnpicklingError:
        checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config_dict = checkpoint.get("config", {})
        base_config = RunConfig(**{k: v for k, v in config_dict.items() if k in RunConfig.__dataclass_fields__})
        validate_config(base_config)
        model = build_model(base_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, base_config

    if isinstance(checkpoint, MNISTBoidGenerator):
        return checkpoint.to(torch_device), config_from_legacy_model(checkpoint, checkpoint_path)

    raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")


def train_model(config):
    model = build_model(config)
    train_dataloader = create_dataloader(config, train=True, batch_size=config.train_batch_size, shuffle=True)
    test_dataloader = create_dataloader(config, train=False, batch_size=config.eval_batch_size, shuffle=False)

    criterion = ChamferLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    best_eval_loss = float("inf")

    for epoch in range(config.epoch_num):
        model.train()
        total_loss = 0.0
        try:
            from tqdm.auto import tqdm

            train_iter = tqdm(
                train_dataloader,
                desc=f"train epoch {epoch + 1}/{config.epoch_num}",
                ncols=80,
                file=sys.stdout,
            )
        except Exception:
            train_iter = train_dataloader

        for init_tokens, lengths, target_tokens, labels in train_iter:
            optimizer.zero_grad()
            output = model(init_tokens, lengths, labels)
            loss = criterion(output, target_tokens, lengths)
            if config.z_reg_weight > 0 and output.shape[-1] > 2:
                loss = loss + config.z_reg_weight * output[..., 2:].pow(2).mean()
            if config.out_of_bounds_reg_weight > 0:
                excess = (output[..., :2].abs() - config.habitat_scaling_factor).clamp_min(0)
                loss = loss + config.out_of_bounds_reg_weight * excess.pow(2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / max(len(train_dataloader), 1)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        sys.stdout.flush()
        eval_loss = evaluate_model(model, test_dataloader, criterion)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_checkpoint(model, config)
            print(f"Saved new best checkpoint with Eval Chamfer: {best_eval_loss:.4f}")
            sys.stdout.flush()

    if config.epoch_num == 0:
        save_checkpoint(model, config)
    else:
        print(f"Best Eval Chamfer: {best_eval_loss:.4f}")
        sys.stdout.flush()
    return model, test_dataloader


def evaluate_model(model, test_dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for init_tokens, lengths, target_tokens, labels in test_dataloader:
            output = model(init_tokens, lengths, labels)
            total_loss += criterion(output, target_tokens, lengths).item()
    eval_loss = total_loss / max(len(test_dataloader), 1)
    print(f"Eval Chamfer: {eval_loss:.4f}")
    sys.stdout.flush()
    return eval_loss


def main_vis(config, model=None, test_dataloader=None):
    effective_config = config
    if model is None:
        model, checkpoint_config = load_model(config.checkpoint_path)
        effective_config = checkpoint_config_for_runtime(checkpoint_config, config)

    if test_dataloader is None:
        test_dataloader = create_dataloader(
            effective_config,
            train=False,
            batch_size=effective_config.eval_batch_size,
            shuffle=False,
        )

    model.eval()
    with torch.no_grad():
        for init_tokens, lengths, target_tokens, labels in test_dataloader:
            batch_layer_outputs_padded = model(init_tokens, lengths, labels, return_all_layers=True)
            break
        else:
            raise RuntimeError("No test samples available for visualization.")

    sample_count = min(effective_config.vis_samples, len(labels))
    for sample_index in range(sample_count):
        print(f"saving {sample_index} with label {labels[sample_index].item()}")
        length = lengths[sample_index]
        layer_outputs = []
        for layer_output in batch_layer_outputs_padded:
            layer_outputs.append(layer_output[sample_index][:length].cpu().numpy())

        layer_outputs = np.array(layer_outputs)
        layer_outputs = np.transpose(layer_outputs, (1, 0, 2))
        np.save(f"acts_{model_suffix(effective_config)}_s{sample_index}.npy", layer_outputs)


def main():
    config = parse_args()
    set_seed(config.seed)

    model = None
    test_dataloader = None

    if config.skip_train:
        if not Path(config.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")
    else:
        model, test_dataloader = train_model(config)

    if not config.skip_vis:
        main_vis(config, model=model, test_dataloader=test_dataloader)


if __name__ == "__main__":
    main()
