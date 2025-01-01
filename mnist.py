import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence

torch_device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


# each boid gravitates toward a target determined by the label
class MeanL2Loss(nn.Module):
    def __init__(self, scaling_factor=1.0):
        super(MeanL2Loss, self).__init__()
        self.scaling_factor = scaling_factor

    def forward(self, predictions, labels, mask=None):
        """
        Parameters:
            predictions: Tensor of shape (batch_size, num_tokens, latent_dim)
            labels: Tensor of shape (batch_size,), turned into an L2 target for each token,
            (-scaling_factor, 0, ..., 0) for label == 0, (+scaling_factor, 0, ..., 0) for label == 1
            mask: Optional Tensor of shape (batch_size, num_tokens) with 1 for valid tokens and 0 for padding

        Returns:
            Scalar loss value (sum of L2 distances for all valid tokens)
        """

        targets = torch.zeros_like(predictions)
        targets[:, :, 0] = torch.where(labels.unsqueeze(1) == 0, -self.scaling_factor, self.scaling_factor)

        # Compute L2 distances
        l2_distances = torch.norm(predictions - targets, p=2, dim=-1)  # Shape: (batch_size, num_tokens)

        # Apply mask if provided
        if mask is not None:
            l2_distances = l2_distances * mask  # Zero out padded token distances

        # Sum all L2 distances
        loss = l2_distances.mean()
        return loss


def create_dataloader(train=True, batch_size=32, shuffle=True):
    tokens, labels = load_mnist(train=train)
    dataset = CustomDataset(tokens, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


# Load and tokenize MNIST dataset
def load_mnist(train=True):
    l1 = 5 ; l2 = 6
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    mnist_01 = [(img, label) for img, label in mnist_data if label in {l1, l2}]

    def tokenize(image):
        pixels = image.squeeze().numpy()
        tokens = [(y, x) for x in range(28) for y in range(28) if pixels[x, y] > 0.5]
        tokens = torch.tensor(tokens, dtype=torch.float32).to(torch_device)
        tokens /= 27
        tokens -= 0.5
        tokens *= 20 # in (-10, 10)
        return tokens

    tokens = [tokenize(img) for img, _ in mnist_01]
    labels = torch.tensor([0 if label == l1 else 1 for _, label in mnist_01], dtype=torch.long).to(torch_device)
    return tokens, labels


def vis_boids():
    import matplotlib.pyplot as plt
    tokens, labels = load_mnist(train=False)
    t = tokens[0].numpy()

    plt.scatter(t[:, 0], t[:, 1])
    plt.scatter([0, 27, 0, 27], [0, 0, 27, 27])
    plt.show()


# vis_boids() ; exit()


class ScaledTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, scaling_factor=1.0, dim_feedforward=512, **kwargs):
        super().__init__(d_model, nhead, dim_feedforward=dim_feedforward, **kwargs)
        self.scaling_factor = scaling_factor  # Store the scaling factor

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention with residual connection and scaling
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.scaling_factor * self.dropout1(src2)
        if hasattr(self, 'norm1') and self.norm1 is not None:  # Handle custom norm
            src = self.norm1(src)

        # Feedforward with residual connection and scaling
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.scaling_factor * self.dropout2(src2)
        if hasattr(self, 'norm2') and self.norm2 is not None:  # Handle custom norm
            src = self.norm2(src)

        return src


class MNISTTransformer(nn.Module):
    def __init__(self, d_model=3, nhead=1, num_layers=10):
        super().__init__()
        self.d_model = d_model

        recurrent = True
        if recurrent:
            self.encoder_layers = nn.ModuleList([ScaledTransformerEncoderLayer(d_model, nhead, batch_first=True, scaling_factor=0.1)] * num_layers)
        else:
            self.encoder_layers = nn.ModuleList(
                [
                    # nn.TransformerEncoderLayer(d_model, nhead, batch_first=True) for _ in range(num_layers)
                    ScaledTransformerEncoderLayer(d_model, nhead, batch_first=True, scaling_factor=0.1) for _ in range(num_layers)
                ]
            )

        # in 3D, layernorm is not good, it forces the tokens onto a circle.
        for encoder_layer in self.encoder_layers:
            encoder_layer.norm1 = nn.Identity()
            encoder_layer.norm2 = nn.Identity()
        model = self.encoder_layers[0]
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params = }")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{trainable_params = }")
        for p in model.parameters():
            print(p.numel(), p.shape)


    def forward(self, tokens, lengths, return_all_layers=False):
        # Zero-pad the tokens to match d_model
        tokens = torch.cat([tokens, torch.zeros(tokens.shape[0], tokens.shape[1], self.d_model - 2, device=tokens.device)], dim=2)

        # Create attention mask (1 for valid tokens, 0 for padding)
        max_len = tokens.size(1)
        attention_mask = torch.arange(max_len, device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)

        x = tokens

        if return_all_layers:
            layer_outputs = []
            layer_outputs.append(x.clone())
            for layer in self.encoder_layers:
                x = layer(x, src_key_padding_mask=~attention_mask)
                layer_outputs.append(x.clone())
            return layer_outputs

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=~attention_mask)

        return x

        # Mask the mean calculation to ignore padding
        mean_token = (x * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return mean_token[:, :2]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], device=torch_device)  # Original lengths
    padded_sequences = pad_sequence(sequences, batch_first=True)  # Pad sequences
    labels = torch.tensor(labels, device=torch_device)  # Convert labels to tensor
    return padded_sequences, lengths, labels


# Training loop
def train_model():
    model = MNISTTransformer().to(torch_device)

    train_dataloader = create_dataloader(train=True, batch_size=32, shuffle=True)
    test_dataloader = create_dataloader(train=False, batch_size=1000, shuffle=False)

    criterion = MeanL2Loss(scaling_factor=10.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(20):
        total_loss = 0
        for batch_tokens, batch_lengths, batch_labels in train_dataloader:
            optimizer.zero_grad()
            output = model(batch_tokens, batch_lengths)

            # Create attention mask (1 for valid tokens, 0 for padding)
            max_len = batch_tokens.size(1)
            attention_mask = torch.arange(max_len, device=batch_tokens.device).unsqueeze(0) < batch_lengths.unsqueeze(1)
            loss = criterion(output, batch_labels, mask=attention_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")
        evaluate_model(model, test_dataloader)
    torch.save(model, "model.pth")
    return model


# Evaluation loop
def evaluate_model(model, test_dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_tokens, batch_lengths, batch_labels in test_dataloader:
            output = model(batch_tokens, batch_lengths)
            # predicted = torch.argmax(output, dim=1)

            max_len = batch_tokens.size(1)
            attention_mask = torch.arange(max_len, device=batch_tokens.device).unsqueeze(0) < batch_lengths.unsqueeze(1)
            mean_token = (output * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            predicted = mean_token[..., 0] > 0

            correct += (predicted == batch_labels).sum().item()
            total += len(batch_labels)

    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy:.4f}")


def main_vis():
    model = torch.load("model.pth")
    test_dataloader = create_dataloader(train=False, batch_size=1000, shuffle=False)

    with torch.no_grad():
        for batch_tokens, batch_lengths, batch_labels in test_dataloader:
            batch_layer_outputs_padded = model(batch_tokens, batch_lengths, return_all_layers=True)
            break

    for sample_index in range(10):
        print(f"saving {sample_index} with label {batch_labels[sample_index]}")
        length = batch_lengths[sample_index]
        layer_outputs = []
        for layer_output in batch_layer_outputs_padded:
            layer_outputs.append(layer_output[sample_index][:length].cpu().numpy())

        layer_outputs = np.array(layer_outputs)
        layer_outputs = np.transpose(layer_outputs, (1, 0, 2))
        num_tokens, num_transformer_blocks_plus_1, latent_dim = layer_outputs.shape
        np.save(f"acts08_d{latent_dim}_b{num_transformer_blocks_plus_1 - 1}_s{sample_index}_recurrent.npy", layer_outputs)


if __name__ == "__main__":
    model = train_model()
    main_vis() ; exit()
