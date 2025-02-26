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

HABITAT_SCALING_FACTOR = 10

class CustomDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


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
    tokens = same_size(tokens)
    return tokens, labels


def same_size(tokens):
    N = 200
    tokens2 = []
    for t in tokens:
        if len(t) > N:
            indices = torch.randperm(len(t))[:N]
            t = t[indices]
        elif len(t) < N:
            indices = torch.randint(len(t), (N,))
            t = t[indices]
        noise = torch.randn_like(t) * (HABITAT_SCALING_FACTOR/80)
        t = t + noise
        tokens2.append(t)

    return tokens2

def vis_boids():


    plt.scatter(t[:, 1], -t[:, 0])
    plt.scatter([-HABITAT_SCALING_FACTOR, -HABITAT_SCALING_FACTOR, HABITAT_SCALING_FACTOR, HABITAT_SCALING_FACTOR],
                [-HABITAT_SCALING_FACTOR,  HABITAT_SCALING_FACTOR, -HABITAT_SCALING_FACTOR, HABITAT_SCALING_FACTOR])
    plt.show()


def sinkhorn_transport(source, target, epsilon=0.01, num_iterations=100, return_plan=False):
    """
    Performs Sinkhorn's algorithm for optimal transport between source and target point clouds.
    With numerical stability improvements.
    """
    # Compute cost matrix (squared Euclidean distances)
    C = torch.cdist(source, target, p=2).pow(2)
    
    # Apply exponential to get the kernel matrix with numerical stability
    # K_ij = exp(-C_ij / epsilon)
    K = torch.exp(-C / epsilon)
    
    # Handle numerical issues - replace any NaN or Inf values
    K = torch.nan_to_num(K, nan=1e-10, posinf=1e10, neginf=1e-10)
    
    # Initialize uniform weights for source and target distributions
    a = torch.ones(source.shape[0], device=source.device) / source.shape[0]
    b = torch.ones(target.shape[0], device=target.device) / target.shape[0]
    
    # Stabilized Sinkhorn iterations
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    
    for _ in range(num_iterations):
        # Prevent division by zero with a small epsilon
        u_new = a / (K @ v + 1e-10)
        v_new = b / (K.T @ u + 1e-10)
        
        # Clip values to prevent numerical issues
        u = torch.clamp(u_new, min=1e-10, max=1e10)
        v = torch.clamp(v_new, min=1e-10, max=1e10)
        
        # Check if we have NaN values and break early if needed
        if torch.isnan(u).any() or torch.isnan(v).any():
            print("NaN detected in Sinkhorn iteration, using last valid values")
            break
    
    # Compute transport plan with numerical stability
    # We'll use element-wise operations instead of matrix diagonal products
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    
    # Handle any remaining numerical issues
    P = torch.nan_to_num(P, nan=1e-10)
    
    if return_plan:
        return P
    else:
        # Transport the source points toward target
        row_sums = P.sum(dim=1, keepdim=True)
        # Avoid division by zero
        row_sums = torch.clamp(row_sums, min=1e-10)
        normalized_P = P / row_sums
        transported_source = normalized_P @ target
        return transported_source

def interpolate_with_sinkhorn(source, target, steps=10, epsilon=0.05):
    """
    Creates an interpolation between source and target point clouds using Sinkhorn transport.
    """
    interpolated = [source]
    current = source.clone()
    
    for i in range(steps):
        # Calculate transported points
        transported = sinkhorn_transport(current, target, epsilon=epsilon, num_iterations=5)
        
        # Check for NaN values
        if torch.isnan(transported).any():
            print(f"NaN detected in step {i+1}, stopping interpolation")
            break
            
        # Linear interpolation between current and transported points
        alpha = (i + 1) / steps
        current = (1 - alpha) * source + alpha * transported
        interpolated.append(current)
    
    return interpolated



def create_interpolation(target_point_cloud):
    # Create a random normal source point cloud with the same number of points
    N = len(target_point_cloud)
    
    # Use a smaller scaling factor for numerical stability
    source_point_cloud = torch.randn(N, 2, device=torch_device) * (HABITAT_SCALING_FACTOR/5)
    
    # For better numerical stability, ensure both point clouds have similar scales
    target_std = target_point_cloud.std()
    source_std = source_point_cloud.std()
    source_point_cloud = source_point_cloud * (target_std / source_std)
    
    # Generate interpolation steps
    steps = 20
    interpolated = interpolate_with_sinkhorn(source_point_cloud, target_point_cloud, steps=steps, epsilon=.2)
    
    return interpolated

def create_interpolation_from_tokens(tokens):
    interpolations = []
    for i, token in enumerate(tokens):
        if i % 100 == 0:
            print(f"Creating interpolation for token {i}")
        # Get list of interpolated point clouds
        interp_sequence = create_interpolation(token)
        # Convert the list of tensors to a single stacked tensor
        interp_tensor = torch.stack(interp_sequence)
        interpolations.append(interp_tensor)
    return torch.stack(interpolations)

def main_create_interpolations():
    tokens, labels = load_mnist(train=True)
    interpolations = create_interpolation_from_tokens(tokens)
    # Save interpolations to file
    torch.save({
        'interpolations': interpolations,
        'labels': labels
    }, 'mnist_interpolations.pt')
    print("Saved interpolations to mnist_interpolations.pt")

main_create_interpolations()

def demo_sinkhorn_interpolation():
    """
    Demonstrates the Sinkhorn interpolation on MNIST data.
    """
    # Load a single MNIST digit
    tokens, labels = load_mnist(train=True)
    target_point_cloud = tokens[2]  # Get the first digit
    interpolated = create_interpolation(target_point_cloud)

    # Visualize the interpolation
    # Calculate the correct grid dimensions based on the actual number of interpolated points
    num_points = len(interpolated)
    cols = (num_points + 1) // 2  # Ceiling division to ensure we have enough columns
    
    fig, axes = plt.subplots(2, cols, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, points in enumerate(interpolated):
        if i >= len(axes):  # Safety check to avoid index errors
            print(f"Warning: Not enough subplots for all {num_points} interpolation steps")
            break
            
        if torch.isnan(points).any():
            print(f"NaN values detected in visualization step {i}")
            continue
            
        ax = axes[i]
        points_np = points.cpu().numpy()
        ax.scatter(points_np[:, 1], -points_np[:, 0], s=5, alpha=0.6)
        ax.set_xlim(-HABITAT_SCALING_FACTOR, HABITAT_SCALING_FACTOR)
        ax.set_ylim(-HABITAT_SCALING_FACTOR, HABITAT_SCALING_FACTOR)
        ax.set_title(f"Step {i}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any unused subplots
    for i in range(num_points, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

    

# Uncomment to run the Sinkhorn interpolation demo
# demo_sinkhorn_interpolation()


# vis_boids()

