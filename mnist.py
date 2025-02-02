import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

torch_device = "cuda" if torch.cuda.is_available() else "cpu"


BINARY = (2, 3)
RECURRENT = True
D_MODEL = 2
NHEAD = 5
NUM_LAYERS = 5 # number of timesteps when interpreted as swarm simulation
DIM_FEEDFORWARD = 50
HABITAT_SCALING_FACTOR = 10
RESIDUAL_SCALING_FACTOR = 0.1
SEPARATION_STRENGTH = 0.01
TRAIN_BATCH_SIZE = 256
LR = 0.005
EPOCH_NUM = 60


class CustomDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]



def labels_to_binary_targets(labels):
    """
    Converts MNIST labels (0–7) into 3-bit binary targets.

    Args:
        labels (torch.Tensor): A 1D tensor of MNIST labels (restricted to 0–7).

    Returns:
        torch.Tensor: A 2D tensor of shape (batch_size, 3), where each row is a
                      3-bit binary representation of the corresponding label.
    """
    # TODO HACK HACK HACK
    if not torch.all((0 <= labels) & (labels <= 3)):
        raise ValueError("Labels must be in the range 0–7.")

    if not torch.all((0 <= labels) & (labels <= 2)):
        raise ValueError("Labels must be in the range 0–1.")
    labels = labels * 3 # because we want them in opposite corners

    # Convert labels to binary and return as a 3D tensor
    return (labels.unsqueeze(1) >> torch.arange(1, -1, -1).to(torch_device)) & 1


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

        targets = torch.zeros((predictions.shape[0], predictions.shape[2])).to(predictions.dtype).to(torch_device)
        # targets[:, :, 0] = torch.where(labels.unsqueeze(1) == 0, -self.scaling_factor, self.scaling_factor)
        targets[:, :2] = self.scaling_factor * (2 * labels_to_binary_targets(labels).to(predictions.dtype) - 1)

        # Compute L2 distances
        l2_distances = torch.norm(predictions - targets.unsqueeze(1), p=2, dim=-1)  # Shape: (batch_size, num_tokens, latent_dim)

        # Apply mask if provided
        if mask is not None:
            l2_distances = l2_distances * mask  # Zero out padded token distances

        # Sum all L2 distances
        loss = l2_distances.mean()
        return loss


def create_dataloader(train=True, batch_size=32, shuffle=True, binary=None):
    tokens, labels = load_mnist(train=train, binary=binary)
    dataset = CustomDataset(tokens, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def tokenize_image(image):
    image = image.to(torch_device)
    pixels = image.squeeze()

    # Perform the thresholding and get indices directly on the GPU
    indices = (pixels > 0.5).nonzero(as_tuple=True)

    # Stack the indices into (y, x) pairs
    tokens = torch.stack(indices, dim=1).float()

    # Normalize the tokens directly on the GPU
    tokens /= 27
    tokens -= 0.5
    tokens *= 2 * HABITAT_SCALING_FACTOR  # In range (-HABITAT_SCALING_FACTOR, HABITAT_SCALING_FACTOR)

    return tokens

'''
def tokenize_image(image):
    pixels = image.squeeze().numpy()
    indices = np.where(pixels > 0.5)
    tokens = list(zip(indices[0], indices[1]))
    tokens = torch.tensor(tokens, dtype=torch.float32).to(torch_device)
    tokens /= 27
    tokens -= 0.5
    tokens *= 20 # in (-10, 10)
    return tokens
'''

def load_mnist(train=True, binary=None):
    transform = transforms.Compose([transforms.ToTensor()])
    print("loading dataset")
    mnist_data = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    mnist_3bit = [(img, label) for img, label in mnist_data if label < 8]

    if binary is not None:
        l1, l2 = binary
        print(f"choosing: filtering classification task to binary {l1} vs {l2}, output 0-1")
        mnist_3bit = [(img, 0 if label==l1 else 1) for img, label in mnist_data if label in {l1, l2}]
    else:
        print(f"choosing: 8-way classification")

    print("tokenization")
    tokens = [tokenize_image(img) for img, _ in mnist_3bit]
    labels = torch.tensor([label for _, label in mnist_3bit], dtype=torch.long).to(torch_device)
    print("dataset preparation done")
    sys.stdout.flush()
    return tokens, labels


def vis_boids():
    tokens, labels = load_mnist(train=False)
    t = tokens[0].numpy()

    plt.scatter(t[:, 0], t[:, 1])
    plt.scatter([0, 27, 0, 27], [0, 0, 27, 27])
    plt.show()


# vis_boids() ; exit()


class AveragedMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, batch_first=False):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        
        # Each head gets its own set of projection matrices
        self.q_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(nhead)])
        self.k_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(nhead)])
        self.v_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(nhead)])
        
        self.batch_first = batch_first

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, seq_len, _ = query.shape
        scaling = float(self.d_model) ** -0.5
        
        # Store outputs from each head
        head_outputs = []
        
        for head in range(self.nhead):
            q = self.q_projs[head](query) * scaling
            k = self.k_projs[head](key)
            v = self.v_projs[head](value)
            
            # Compute attention scores
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1), float('-inf'))
            
            if attn_mask is not None:
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            
            attn_weights = torch.softmax(attn_weights, dim=-1)
            head_output = torch.bmm(attn_weights, v)
            head_outputs.append(head_output)
        
        # Average the outputs from all heads
        output = torch.stack(head_outputs, dim=0).mean(dim=0)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, None


class ScaledTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, scaling_factor=1.0, dim_feedforward=512, l2_penalty=0.00001, **kwargs):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.l2_penalty = l2_penalty
        # Replace the default attention with our custom attention that averages heads
        self.self_attn = AveragedMultiheadAttention(d_model, nhead, batch_first=True)
        # Remove normalization and dropout as per the original
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        # in our continuous dynamics, dropout hurts performance
        self.dropout = nn.Identity()
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        # Add feedforward network components
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention with residual connection and scaling
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        
        # Use getattr to provide a default value if attribute doesn't exist
        l2_penalty = getattr(self, 'l2_penalty', 0.01)
        attn_l2_loss = l2_penalty * torch.norm(src2, p=2)
        
        src = src + self.scaling_factor * src2

        # Feedforward with residual connection and scaling
        src2 = self.linear2(self.activation(self.linear1(src)))
        
        # Calculate L2 penalty for feedforward output
        ffn_l2_loss = l2_penalty * torch.norm(src2, p=2)
        
        src = src + self.scaling_factor * src2

        # Store the total L2 loss in the layer
        self.layer_l2_loss = attn_l2_loss + ffn_l2_loss

        if SEPARATION_STRENGTH != 0:
            src += HABITAT_SCALING_FACTOR * SEPARATION_STRENGTH * boid_separation(src, src_key_padding_mask, 1, separation_weight=1.0, eps=1e-6)

        return src


class MNISTTransformer(nn.Module):
    def __init__(self, d_model=3, nhead=1, num_layers=10, recurrent=True, scaling_factor=0.1):
        super().__init__()
        self.d_model = d_model
        if recurrent:
            self.encoder_layers = nn.ModuleList([ScaledTransformerEncoderLayer(d_model, nhead,
                batch_first=True, scaling_factor=scaling_factor,
                dim_feedforward=DIM_FEEDFORWARD)] * num_layers)
        else:
            self.encoder_layers = nn.ModuleList(
                [
                    ScaledTransformerEncoderLayer(d_model, nhead,
                        batch_first=True, scaling_factor=scaling_factor,
                        dim_feedforward=DIM_FEEDFORWARD) for _ in range(num_layers)
                ]
            )

        model = self.encoder_layers[0]
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params = }")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{trainable_params = }")
        for p in model.parameters():
            print(p.numel(), p.shape)
        print("-----")

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
def train_model(model, train_dataloader, test_dataloader):
    criterion = MeanL2Loss(scaling_factor=HABITAT_SCALING_FACTOR)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCH_NUM):
        total_loss = 0
        for batch_tokens, batch_lengths, batch_labels in train_dataloader:
            optimizer.zero_grad()
            output = model(batch_tokens, batch_lengths)

            # Create attention mask (1 for valid tokens, 0 for padding)
            max_len = batch_tokens.size(1)
            attention_mask = torch.arange(max_len, device=batch_tokens.device).unsqueeze(0) < batch_lengths.unsqueeze(1)
            
            # Calculate main loss
            main_loss = criterion(output, batch_labels, mask=attention_mask)
            
            # Add L2 penalties from all transformer layers
            l2_loss = sum(layer.layer_l2_loss for layer in model.encoder_layers)
            
            # Total loss is the main loss plus the L2 penalties
            loss = main_loss + l2_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")
        sys.stdout.flush()
        evaluate_model(model, test_dataloader)
    torch.save(model, "model.pth")
    return model


def classifier_07(predictions, scaling_factor):
    # Generate the 8 vertices of the unit cube
    cube_vertices = torch.tensor(
        [[(i >> 1) & 1, i & 1] for i in range(4)],
        dtype=predictions.dtype,
        device=predictions.device
    )

    # Compute the distances between predictions and cube vertices
    distances = torch.cdist(predictions[..., :2], cube_vertices, p=2)

    # Find the index of the closest vertex for each prediction
    predicted_labels = torch.argmin(distances, dim=1)
    return predicted_labels


def model_suffix():
    labels_str = "".join(map(str, BINARY)) if BINARY is not None else "all8"
    recurrent_str = "recurrent" if RECURRENT else "nonrecurrent"
    return f"{labels_str}_d{D_MODEL}_b{NUM_LAYERS}_{recurrent_str}"


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

            # original binary:
            # predicted = mean_token[..., 0] > 0

            # labels 0 to 7 correspond to vertices of the {-10, 10}^3 cube.
            predicted = classifier_07(mean_token, scaling_factor=HABITAT_SCALING_FACTOR)

            assert BINARY is not None
            # TODO HACK HACK HACK
            # 0 and 3 are the labels so that boids gather in opposing corners
            predicted //= 3

            correct += (predicted == batch_labels).sum().item()
            total += len(batch_labels)

    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    sys.stdout.flush()


def middle_of_animation_grid(timestep, batch_layer_outputs_padded, batch_lengths, batch_labels):
    batch_layer_outputs_padded = batch_layer_outputs_padded[timestep]
    batch_layer_outputs_padded = batch_layer_outputs_padded[:100]
    batch_lengths = batch_lengths[:100]
    batch_labels = batch_labels[:100]
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = axes.flatten()
    
    # Plot first 100 samples
    for idx, (outputs, length, label) in enumerate(zip(batch_layer_outputs_padded, batch_lengths, batch_labels)):
        if idx >= 100:  # Only plot first 100 samples
            break
        
        ax = axes[idx]
        # Ensure outputs are on the CPU for plotting
        outputs_cpu = outputs.cpu() if outputs.is_cuda else outputs
        
        # Set x-axis limit (as defined by the plot limits)
        x_lim = HABITAT_SCALING_FACTOR * 1.2
        # Now, use the x coordinate of each token to define a left-to-right gradient.
        # Normalize the x value to be within [0, 1] based on the known x-axis limits.
        colors = ((outputs_cpu[:length, 1] + x_lim) / (2 * x_lim)).numpy()
        
        # Get x and y coordinates (convert to numpy arrays)
        x_vals = outputs_cpu[:length, 1].numpy()
        y_vals = (-outputs_cpu[:length, 0]).numpy()
        
        ax.scatter(x_vals, y_vals, c=colors, cmap='viridis', alpha=0.5)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-x_lim, x_lim)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Adjusting the label text (if needed)
        label = label + 2
        ax.set_title(f'{label}')
    
    plt.tight_layout()
    plt.savefig(f"step_{timestep}.png")



def main_vis():
    # model_filename = "model." + model_suffix() + ".pth"
    model_filename = "model.23_d2_b10_recurrent_multihead10.pth"
    model_filename = "model.23_d2_b5_recurrent_multihead1_ffwd10.pth"
    model_filename = "model_separation2.pth"
    model = torch.load(model_filename, map_location=torch_device)

    test_dataloader = create_dataloader(train=False, batch_size=1000, shuffle=False, binary=BINARY)

    with torch.no_grad():
        for batch_tokens, batch_lengths, batch_labels in test_dataloader:
            batch_layer_outputs_padded = model(batch_tokens, batch_lengths, return_all_layers=True)
            break


    for timestep in range(len(batch_layer_outputs_padded)):
        middle_of_animation_grid(timestep, batch_layer_outputs_padded, batch_lengths, batch_labels)

    exit()
    for sample_index in range(10):
        print(f"saving {sample_index} with label {batch_labels[sample_index]}")
        length = batch_lengths[sample_index]
        layer_outputs = []
        for layer_output in batch_layer_outputs_padded:
            layer_outputs.append(layer_output[sample_index][:length].cpu().numpy())

        layer_outputs = np.array(layer_outputs)
        layer_outputs = np.transpose(layer_outputs, (1, 0, 2))
        num_tokens, num_transformer_blocks_plus_1, latent_dim = layer_outputs.shape
        np.save("acts_" +  model_suffix() + f"_s{sample_index}.npy", layer_outputs)


def boid_separation(positions, src_key_padding_mask, separation_distance, separation_weight=1.0, eps=1e-6):
    """
    Compute differentiable separation force for boids over a batch of scenes.

    Args:
        positions (torch.Tensor): Tensor of shape (B, N, D) containing positions of boids,
                                  where B is the batch size, N is the maximum number of birds
                                  (with padding for samples with fewer birds), and D is the dimension.
        batch_lengths (torch.Tensor): 1D tensor of length B indicating the number of valid birds
                                      in each batch sample.
        separation_distance (float): Distance threshold within which boids repel each other.
        separation_weight (float): Scaling factor for the separation force.
        eps (float): Small constant used to avoid division by zero.

    Returns:
        torch.Tensor: Tensor of shape (B, N, D) with computed separation forces. For padded
                      (invalid) bird positions, the force is zero.
    """
    B, N, D = positions.shape
    # Compute pairwise differences: shape (B, N, N, D)
    pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
    
    # Compute pairwise distances: shape (B, N, N)
    distances = torch.norm(pos_diff, dim=-1)
    
    # Create a pairwise mask: both boids in a pair must be valid
    valid_pair_mask = src_key_padding_mask.unsqueeze(1) & src_key_padding_mask.unsqueeze(2)  # shape (B, N, N)
    
    # Exclude self interactions by zeroing out the diagonal
    diag_mask = torch.eye(N, dtype=torch.bool, device=positions.device).unsqueeze(0)  # shape (1, N, N)
    
    # Final mask: valid pairs (non-self) with distance > 0 and within separation_distance
    mask = valid_pair_mask & (~diag_mask) & (distances > 0) & (distances < separation_distance)
    mask_expanded = mask.unsqueeze(-1).float()  # shape (B, N, N, 1)
    
    # Compute normalized repulsion vectors while avoiding division by zero with eps
    repulsion = pos_diff / (distances.unsqueeze(-1) + eps)  # shape (B, N, N, D)
    
    # Only include contributions from valid neighbors
    repulsion = repulsion * mask_expanded
    
    # Sum repulsion contributions from neighbors for each boid, then apply the separation weight
    separation_force = repulsion.sum(dim=2) * separation_weight  # shape (B, N, D)
    
    # Zero-out forces for padded (invalid) boid positions
    separation_force = separation_force * src_key_padding_mask.unsqueeze(-1).float()
    
    return separation_force


# Training loop
def main_train():
    model = MNISTTransformer(
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        recurrent=RECURRENT, scaling_factor=RESIDUAL_SCALING_FACTOR).to(torch_device)

    train_dataloader = create_dataloader(train=True, batch_size=TRAIN_BATCH_SIZE, shuffle=True, binary=BINARY)
    test_dataloader = create_dataloader(train=False, batch_size=1000, shuffle=False, binary=BINARY)

    train_model(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    # model = main_train() ; exit()
    main_vis()

