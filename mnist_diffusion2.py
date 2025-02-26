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
NUM_LAYERS = 1 # number of timesteps when interpreted as swarm simulation
DIM_FEEDFORWARD = 50
HABITAT_SCALING_FACTOR = 10
RESIDUAL_SCALING_FACTOR = 0.1
SEPARATION_STRENGTH = 1.0
TRAIN_BATCH_SIZE = 256
LR = 0.01
EPOCH_NUM = 20


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
    def __init__(self, d_model, nhead, scaling_factor=1.0, dim_feedforward=512, l2_penalty=0.00002, **kwargs):
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

        attn_l2_loss = self.l2_penalty * torch.norm(src2, p=2)

        src = src + self.scaling_factor * src2

        # Feedforward with residual connection and scaling
        src2 = self.linear2(self.activation(self.linear1(src)))

        # Calculate L2 penalty for feedforward output
        ffn_l2_loss = self.l2_penalty * torch.norm(src2, p=2)

        src = src + self.scaling_factor * src2

        # Store the total L2 loss in the layer
        self.layer_l2_loss = attn_l2_loss + ffn_l2_loss

        if SEPARATION_STRENGTH != 0:
            # src += HABITAT_SCALING_FACTOR * SEPARATION_STRENGTH * boid_separation(src, src_key_padding_mask, 1, separation_weight=1.0, eps=1e-6)

            # ~ aka logical not because this one expects true for the alive birds.
            self.layer_l2_loss += SEPARATION_STRENGTH * covariance_loss(src, ~ src_key_padding_mask)

        return src


def covariance_loss(points, mask):
    """
    Compute a loss that penalizes the deviation of the covariance matrix of each point cloud from the identity matrix.

    Args:
        points (torch.Tensor): Tensor of shape (B, N, 2) containing point cloud coordinates.
        mask (torch.Tensor): Tensor of shape (B, N) with 1 for valid points and 0 for padded ones.

    Returns:
        torch.Tensor: The computed covariance loss (a scalar).
    """
    B, N, _ = points.shape

    # this scaling to plusminus 2 is needed to ensure that the minimal loss
    # is attained roughly when a quarter of the habitat is occupied.
    points = points / HABITAT_SCALING_FACTOR * 2

    # Expand mask to match the last dimension of points
    mask_expanded = mask.unsqueeze(-1)  # shape (B, N, 1)

    # Compute the number of valid points per batch element
    valid_counts = mask.sum(dim=1, keepdim=True)  # shape (B, 1)

    # Compute the mean of the valid points for each cloud.
    # Use masking to ignore padded points.
    mean = (points * mask_expanded).sum(dim=1) / valid_counts  # shape (B, 2)

    # Center the points by subtracting the mean.
    # Broadcasting takes care of subtracting the mean from every point.
    centered = points - mean.unsqueeze(1)  # shape (B, N, 2)

    # Zero out the contributions from padded points.
    centered = centered * mask_expanded  # shape (B, N, 2)

    # Compute the covariance matrix for each point cloud.
    # For each batch element: cov = (X^T X) / valid_count, where X is (N, 2)
    # First, we need to ensure valid_counts is of shape (B, 1, 1) for proper broadcasting.
    valid_counts = valid_counts.view(B, 1, 1)
    cov = torch.bmm(centered.transpose(1, 2), centered) / valid_counts  # shape (B, 2, 2)

    # Create an identity matrix of shape (B, 2, 2)
    identity = torch.eye(2, device=points.device).unsqueeze(0).expand(B, 2, 2)

    # Compute the difference between the covariance and the identity.
    diff = cov - identity

    # Compute the loss as the squared Frobenius norm of the difference,
    # then average over the batch.
    loss = torch.mean(torch.sum(diff * diff, dim=(1, 2)))
    return loss


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


def model_suffix():
    labels_str = "".join(map(str, BINARY)) if BINARY is not None else "all8"
    recurrent_str = "recurrent" if RECURRENT else "nonrecurrent"
    return f"{labels_str}_d{D_MODEL}_b{NUM_LAYERS}_{recurrent_str}"



def middle_of_animation_grid(timestep, batch_layer_outputs_padded, batch_lengths, batch_labels):
    grid_size = 10

    assert len(batch_layer_outputs_padded) == grid_size ** 2
    assert len(batch_lengths) == grid_size ** 2
    assert len(batch_labels) == grid_size ** 2
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()

    for idx, (outputs, length, label) in enumerate(zip(batch_layer_outputs_padded, batch_lengths, batch_labels)):
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
        # label = label + 2
        # ax.set_title(f'{label}')

    plt.tight_layout()
    plt.savefig(f"step_{timestep}.png")


def main_vis():
    # model_filename = "model." + model_suffix() + ".pth"
    model_filename = "model.23_d2_b10_recurrent_multihead10.pth"
    model_filename = "model.23_d2_b5_recurrent_multihead1_ffwd10.pth"
    model_filename = "model.pth"
    model = torch.load(model_filename, map_location=torch_device)

    test_dataloader = create_dataloader(train=False, batch_size=1000, shuffle=False, binary=BINARY)

    vis(model, test_dataloader)


def filter_for_vis(batch_outputs, batch_lengths, batch_labels, k=50):
    # Create boolean masks for each label
    mask_A = (batch_labels == 0)
    mask_B = (batch_labels == 1)

    # Get the indices where each mask is True
    indices_A = torch.nonzero(mask_A, as_tuple=True)[0]
    indices_B = torch.nonzero(mask_B, as_tuple=True)[0]

    # Select the first k indices for each label.
    # (Make sure there are at least k elements for each label in your data)
    selected_indices_A = indices_A[:k]
    selected_indices_B = indices_B[:k]

    # Concatenate the indices so that first k come from label A and the next k come from label B
    selected_indices = torch.cat((selected_indices_A, selected_indices_B), dim=0)

    # Now index into your original tensors
    filtered_batch_outputs = batch_outputs[selected_indices]
    filtered_batch_lengths = batch_lengths[selected_indices]
    filtered_batch_labels = batch_labels[selected_indices]

    return filtered_batch_outputs, filtered_batch_lengths, filtered_batch_labels


def vis(model, test_dataloader):
    with torch.no_grad():
        for batch_tokens, batch_lengths, batch_labels in test_dataloader:
            batch_layer_outputs_padded = model(batch_tokens, batch_lengths, return_all_layers=True)
            break

    for timestep in range(len(batch_layer_outputs_padded)):
        filtered_batch_outputs, filtered_batch_lengths, filtered_batch_labels = filter_for_vis(batch_layer_outputs_padded[timestep], batch_lengths, batch_labels, k=50)
        middle_of_animation_grid(timestep, filtered_batch_outputs, filtered_batch_lengths, filtered_batch_labels)

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




class InterpolationDataset(Dataset):
    """
    Dataset to create (x, y) pairs from interpolation sequences where:
    x = interpolations[r, t]
    y = interpolations[r, t+1]
    
    r = random sample index
    t = random time step (ensuring t+1 exists)
    """
    def __init__(self, interpolations):
        """
        Args:
            interpolations: Tensor of shape [num_samples, num_steps, num_points, 2]
        """
        self.interpolations = interpolations
        self.num_samples = interpolations.shape[0]
        self.num_steps = interpolations.shape[1]
        
    def __len__(self):
        # Number of possible pairs is num_samples * (num_steps - 1)
        return self.num_samples * (self.num_steps - 1)
    
    def __getitem__(self, idx):
        # Convert flat index to (sample_idx, step_idx)
        sample_idx = idx // (self.num_steps - 1)
        step_idx = idx % (self.num_steps - 1)
        
        # Get the consecutive point clouds
        x = self.interpolations[sample_idx, step_idx]
        y = self.interpolations[sample_idx, step_idx + 1]
        
        return x, y

def create_interpolation_dataloader(interpolations, batch_size=32, shuffle=True):
    """
    Creates a dataloader for training with interpolation pairs.
    
    Args:
        interpolations: Tensor of shape [num_samples, num_steps, num_points, 2]
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader object with (x, y) pairs where x and y are consecutive point clouds
    """
    dataset = InterpolationDataset(interpolations)
    # No need for a custom collate_fn since all point clouds have the same number of points
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def random_interpolation_pairs(interpolations, batch_size=32):
    """
    Alternative way to generate random (x, y) pairs directly without using a DataLoader.
    This can be useful for quick experimentation.
    
    Args:
        interpolations: Tensor of shape [num_samples, num_steps, num_points, 2]
        batch_size: Number of random pairs to generate
        
    Returns:
        Two tensors x and y containing the input and target point clouds
    """
    num_samples = interpolations.shape[0]
    num_steps = interpolations.shape[1]
    
    # Sample random indices
    sample_indices = torch.randint(0, num_samples, (batch_size,), device=interpolations.device)
    # Ensure t+1 exists by limiting step_indices to num_steps-2
    step_indices = torch.randint(0, num_steps-1, (batch_size,), device=interpolations.device)
    
    # Get the sampled point clouds
    x = interpolations[sample_indices, step_indices]
    y = interpolations[sample_indices, step_indices + 1]
    
    return x, y



# Training loop
def main_train():
    # Load the saved interpolations
    interpolations_dataset = torch.load("mnist_interpolations.pt")
    interpolations = interpolations_dataset["interpolations"]
    labels = interpolations_dataset["labels"]
    
    print(f"Loaded interpolations: {interpolations.shape}")
    print(f"Loaded labels: {labels.shape}")
    
    # Create dataloader for training
    train_dataloader = create_interpolation_dataloader(
        interpolations, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    
    # Create model
    model = MNISTTransformer(
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        recurrent=RECURRENT, scaling_factor=RESIDUAL_SCALING_FACTOR).to(torch_device)
    
    # Train the model
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    for epoch in range(EPOCH_NUM):
        total_loss = 0
        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(torch_device)
            batch_y = batch_y.to(torch_device)
            
            # Assuming all point clouds have the same number of points
            batch_size = batch_x.shape[0]
            num_points = batch_x.shape[1]
            lengths = torch.full((batch_size,), num_points, device=torch_device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch_x, lengths)
            
            # Calculate loss (using MSE or your own loss function)
            loss = torch.nn.functional.mse_loss(output, batch_y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{EPOCH_NUM}, Loss: {total_loss / len(train_dataloader):.6f}")
    
    # Save the trained model
    torch.save(model, "diffusion_model.pth")
    
    return model


def generate_mnist_digits(model, num_samples=16, num_points=200, num_steps=20, scale=2.0):
    """
    Generate MNIST digits using the trained diffusion model.
    
    Args:
        model: The trained diffusion model
        num_samples: Number of digits to generate
        num_points: Number of points in each digit
        num_steps: Number of diffusion steps to run
        scale: Initial scale for the random normal distribution
        
    Returns:
        generated_digits: Final point clouds
        all_steps: All intermediate steps for visualization
    """
    device = next(model.parameters()).device
    
    # Initialize random point clouds with normal distribution
    point_clouds = torch.randn(num_samples, num_points, 2, device=device) * scale
    
    # All points are valid (no padding)
    lengths = torch.full((num_samples,), num_points, device=device)
    
    # Store all steps including the initial state
    all_steps = [point_clouds.clone()]
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for step in range(num_steps):
            # Apply the model to update positions
            updated_clouds = model(point_clouds, lengths)
            all_steps.append(updated_clouds.clone())
            point_clouds = updated_clouds
    
    return point_clouds, all_steps

def visualize_generated_digits(generated_digits, all_steps=None, grid_size=None):
    """
    Visualize the generated MNIST digits.
    
    Args:
        generated_digits: Tensor of shape [num_samples, num_points, 2] with final point clouds
        all_steps: Tensor with all intermediate steps (optional)
        grid_size: Size of the visualization grid (default: square root of num_samples)
    """
    num_samples = generated_digits.shape[0]
    
    # Determine grid size
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create a figure for the final results
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    # Ensure tensor is on CPU for plotting
    if generated_digits.is_cuda:
        generated_digits = generated_digits.cpu()
    
    # Set the visualization limits based on HABITAT_SCALING_FACTOR from the model
    x_lim = HABITAT_SCALING_FACTOR * 1.2
    
    for i in range(min(num_samples, len(axes))):
        ax = axes[i]
        points = generated_digits[i]
        
        # Use a color gradient based on x-coordinate (similar to existing visualization)
        colors = ((points[:, 1] + x_lim) / (2 * x_lim)).numpy()
        
        ax.scatter(points[:, 1], -points[:, 0], c=colors, cmap='viridis', alpha=0.6)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-x_lim, x_lim)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig("generated_digits.png")
    plt.close()
    
    # If all_steps is provided, create a sequence of images showing generation process
    if all_steps is not None:
        if isinstance(all_steps, list):
            num_steps = len(all_steps)
        else:
            if all_steps.is_cuda:
                all_steps = [step.cpu() for step in all_steps]
            num_steps = len(all_steps)
        
        # Save individual frames for each step
        for step_idx in range(num_steps):
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            axes = axes.flatten()
            
            step_data = all_steps[step_idx]
            
            for i in range(min(num_samples, len(axes))):
                ax = axes[i]
                points = step_data[i]
                
                colors = ((points[:, 1] + x_lim) / (2 * x_lim)).numpy()
                
                ax.scatter(points[:, 1], -points[:, 0], c=colors, cmap='viridis', alpha=0.6)
                ax.set_xlim(-x_lim, x_lim)
                ax.set_ylim(-x_lim, x_lim)
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            plt.savefig(f"generation_step_{step_idx:02d}.png")
            plt.close()

def main_generate():
    """Generate MNIST digits using the trained model."""
    # Load the trained model
    model = torch.load("diffusion_model.pth", map_location=torch_device)
    
    # Generate MNIST digits
    num_samples = 16  # 4x4 grid
    num_points = 200  # Number of points per digit
    num_steps = 20    # Number of diffusion steps
    
    print("Generating MNIST digits...")
    generated_digits, all_steps = generate_mnist_digits(
        model, 
        num_samples=num_samples, 
        num_points=num_points,
        num_steps=num_steps, 
        scale=2.0
    )
    
    # Visualize the results
    print("Visualizing results...")
    visualize_generated_digits(generated_digits, all_steps, grid_size=4)
    
    print("Generation complete. Check the output images.")

if __name__ == "__main__":
    # main_train()
    # main_vis()
    main_generate()
