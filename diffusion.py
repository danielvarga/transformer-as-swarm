import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Device configuration
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters (adjust as desired)
BINARY = (2, 3)
D_MODEL = 2       # for positions, we keep the 2 coordinates; time conditioning is added to these coordinates
NHEAD = 5
NUM_LAYERS = 10
DIM_FEEDFORWARD = 50
HABITAT_SCALING_FACTOR = 10
RESIDUAL_SCALING_FACTOR = 0.1
TRAIN_BATCH_SIZE = 256
LR = 0.005
EPOCH_NUM = 20

# Diffusion schedule hyperparameters:
T = 1000         # number of diffusion steps
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T).to(torch_device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # shape: (T,)

##############################
# --- Data & Tokenization --- 
##############################

class CustomDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]

def tokenize_image(image):
    image = image.to(torch_device)
    pixels = image.squeeze()
    # Get indices of pixels greater than 0.5 (assumed “on” pixels)
    indices = (pixels > 0.5).nonzero(as_tuple=True)
    tokens = torch.stack(indices, dim=1).float()  # shape: (num_tokens, 2)
    # Normalize tokens into a fixed range
    tokens /= 27
    tokens -= 0.5
    tokens *= 2 * HABITAT_SCALING_FACTOR  # now roughly in (-HABITAT_SCALING_FACTOR, HABITAT_SCALING_FACTOR)
    return tokens

def load_mnist(train=True, binary=None):
    transform = transforms.Compose([transforms.ToTensor()])
    print("Loading MNIST dataset")
    mnist_data = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    if binary is not None:
        l1, l2 = binary
        print(f"Filtering to binary {l1} vs {l2}")
        mnist_filtered = [(img, 0 if label == l1 else 1) for img, label in mnist_data if label in {l1, l2}]
    else:
        mnist_filtered = [(img, label) for img, label in mnist_data]
    tokens = [tokenize_image(img) for img, _ in mnist_filtered]
    labels = torch.tensor([label for _, label in mnist_filtered], dtype=torch.long).to(torch_device)
    print("Dataset loaded.")
    return tokens, labels

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], device=torch_device)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels, device=torch_device)
    return padded_sequences, lengths, labels

def create_dataloader(train=True, batch_size=32, shuffle=True, binary=None):
    tokens, labels = load_mnist(train=train, binary=binary)
    dataset = CustomDataset(tokens, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

###################################
# --- Diffusion Model Components ---
###################################

# 1. Time Embedding: embeds a scalar diffusion timestep into a vector
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t):
        # t: (batch_size,) -> output: (batch_size, embed_dim)
        return self.embed(t.unsqueeze(-1).float())

# 2. Modified Transformer for Diffusion:
#    It accepts noised boid positions and conditions on the diffusion timestep.
class MNISTDiffusionTransformer(nn.Module):
    def __init__(self, d_model=2, nhead=1, num_layers=10, scaling_factor=0.1):
        super().__init__()
        self.d_model = d_model
        self.time_embedding = TimeEmbedding(d_model)
        # We use a simple transformer encoder (here reusing a scaled transformer block)
        self.encoder_layers = nn.ModuleList([
            ScaledTransformerEncoderLayer(d_model, nhead,
                                          batch_first=True,
                                          scaling_factor=scaling_factor,
                                          dim_feedforward=DIM_FEEDFORWARD)
            for _ in range(num_layers)
        ])
    
    def forward(self, tokens, lengths, t):
        # tokens: (B, N, 2) -- noised positions
        # t: (B,) diffusion timestep as a float
        # Compute time embeddings and add them to every token
        time_embed = self.time_embedding(t)  # (B, d_model)
        time_embed_expanded = time_embed.unsqueeze(1)  # (B, 1, d_model)
        
        # In this diffusion formulation we assume d_model == 2 so that the tokens
        # already have the correct shape. (If d_model > 2, you might need to pad tokens.)
        x = tokens + time_embed_expanded  # simple additive conditioning
        
        # Build attention mask from lengths
        max_len = x.size(1)
        attention_mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=~attention_mask)
        return x  # The network will be trained to predict the noise added

# (Re-use the original ScaledTransformerEncoderLayer and AveragedMultiheadAttention definitions)
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

        batch_size, seq_len, _ = query.shape
        scaling = float(self.d_model) ** -0.5
        head_outputs = []
        for head in range(self.nhead):
            q = self.q_projs[head](query) * scaling
            k = self.k_projs[head](key)
            v = self.v_projs[head](value)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1), float('-inf'))
            if attn_mask is not None:
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            head_output = torch.bmm(attn_weights, v)
            head_outputs.append(head_output)
        output = torch.stack(head_outputs, dim=0).mean(dim=0)
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output, None

class ScaledTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, scaling_factor=1.0, dim_feedforward=512, l2_penalty=0.00002, **kwargs):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.l2_penalty = l2_penalty
        self.self_attn = AveragedMultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        attn_l2_loss = self.l2_penalty * torch.norm(src2, p=2)
        src = src + self.scaling_factor * src2
        src2 = self.linear2(self.activation(self.linear1(src)))
        ffn_l2_loss = self.l2_penalty * torch.norm(src2, p=2)
        src = src + self.scaling_factor * src2
        self.layer_l2_loss = attn_l2_loss + ffn_l2_loss
        return src

#####################################
# --- Diffusion Training & Sampling ---
#####################################

def get_noise_scales(t_index):
    # Given an integer timestep t_index, return sqrt(alpha_bar) and sqrt(1 - alpha_bar)
    alpha_bar = alphas_cumprod[t_index]
    return math.sqrt(alpha_bar.item()), math.sqrt(1 - alpha_bar.item())

def train_diffusion_model(model, dataloader, optimizer, num_epochs=EPOCH_NUM):
    mse_loss = nn.MSELoss()
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_tokens, batch_lengths, batch_labels in dataloader:
            optimizer.zero_grad()
            B = batch_tokens.size(0)
            # Sample a random diffusion timestep t for each sample in the batch (using 0-indexing)
            t = torch.randint(0, T, (B,), device=torch_device).float()
            
            # For each sample, compute the coefficients from the schedule:
            sqrt_alpha_list = []
            sqrt_one_minus_alpha_list = []
            for ti in t:
                sa, soa = get_noise_scales(int(ti.item()))
                sqrt_alpha_list.append(sa)
                sqrt_one_minus_alpha_list.append(soa)
            # Reshape to (B, 1, 1) for broadcasting over tokens:
            sqrt_alpha = torch.tensor(sqrt_alpha_list, device=torch_device).view(B, 1, 1)
            sqrt_one_minus_alpha = torch.tensor(sqrt_one_minus_alpha_list, device=torch_device).view(B, 1, 1)
            
            # Note: batch_tokens has shape (B, N, 2) representing the clean boid positions.
            noise = torch.randn_like(batch_tokens)
            # Create noised input:
            x_t = sqrt_alpha * batch_tokens + sqrt_one_minus_alpha * noise
            
            # Forward pass: condition on the noised positions and the timestep.
            pred_noise = model(x_t, batch_lengths, t)
            
            loss = mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Diffusion Loss: {total_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), "diffusion_model.pth")
    return model


def sample(model, lengths, num_steps=T):
    model.eval()
    B = lengths.size(0)
    max_tokens = lengths.max().item()
    x = torch.randn(B, max_tokens, 2, device=torch_device)
    with torch.no_grad():
        for t_step in reversed(range(num_steps)):
            t_tensor = torch.full((B,), t_step, device=torch_device).float()
            pred_noise = model(x, lengths, t_tensor)
            alpha_bar = alphas_cumprod[t_step]
            # Clamp the square root to avoid division by a very small number.
            sqrt_alpha_bar = max(math.sqrt(alpha_bar.item()), 1e-3)
            sqrt_one_minus_alpha_bar = math.sqrt(1 - alpha_bar.item())
            x = (x - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
            if t_step > 0:
                x = x + torch.randn_like(x) * math.sqrt(betas[t_step].item())
            if t_step == num_steps - 3:
                break
    return x

#####################################
# --- Main Function to Put It Together ---
#####################################

def main():
    # Create dataloaders for training and testing.
    train_dataloader = create_dataloader(train=True, batch_size=TRAIN_BATCH_SIZE, shuffle=True, binary=BINARY)
    test_dataloader = create_dataloader(train=False, batch_size=64, shuffle=True, binary=BINARY)

    # Instantiate the diffusion model and optimizer.
    model = MNISTDiffusionTransformer(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        scaling_factor=RESIDUAL_SCALING_FACTOR
    ).to(torch_device)

    model.load_state_dict(torch.load("diffusion_model.pth", map_location=torch_device))


    '''
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Train the diffusion model.
    print("Starting training...")
    train_diffusion_model(model, train_dataloader, optimizer, num_epochs=EPOCH_NUM)
    print("Training complete.")
    '''

    # Use the trained model to sample boid positions.
    # Here, we use one batch from the test dataloader to get the sequence lengths.
    batch_tokens, batch_lengths, batch_labels = next(iter(test_dataloader))
    sampled_boids = sample(model, batch_lengths, num_steps=T)
    
    # Visualize the first sample (only plot the first sample's valid tokens).
    sample0 = sampled_boids[0, :batch_lengths[0]].cpu().detach().numpy()
    print(sample0[:, 0].min(), sample0[:, 0].max(), sample0[:, 1].min(), sample0[:, 1].max())
    print(sample0)
    plt.figure(figsize=(6, 6))
    plt.scatter(sample0[:, 0], sample0[:, 1], s=10, alpha=0.7)
    plt.title("Sampled Boid Arrangement (Diffused Digit)")
    plt.xlim(-HABITAT_SCALING_FACTOR*1.2, HABITAT_SCALING_FACTOR*1.2)
    plt.ylim(-HABITAT_SCALING_FACTOR*1.2, HABITAT_SCALING_FACTOR*1.2)
    plt.savefig("vis.png")
    plt.show()

if __name__ == "__main__":
    main()
