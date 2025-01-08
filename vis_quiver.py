import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from mnist import MNISTTransformer, ScaledTransformerEncoderLayer

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load("model.pth").to(torch_device)


def ffwd(x):
    m = model.encoder_layers[0]
    return m.linear2(nn.ReLU()(m.linear1(x)))


# Generate the grid
x_range = np.linspace(-10, 10, 10)  # Adjust resolution if needed
y_range = np.linspace(-10, 10, 10)
z_range = np.linspace(-10, 10, 10)
X, Y, Z = np.meshgrid(x_range, y_range, z_range)

# Flatten the grid for easier processing
grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(torch_device)

# Compute model outputs
with torch.no_grad():
    outputs = ffwd(grid_tensor)
vectors = outputs.cpu().numpy() # - grid_points  # Compute vectors (change this logic as needed)

# Reshape vectors to match the grid
U, V, W = vectors[:, 0].reshape(X.shape), vectors[:, 1].reshape(Y.shape), vectors[:, 2].reshape(Z.shape)

# Normalize to [0, 1] range for RGB colors
R = (U_norm - U_norm.min()) / (U_norm.max() - U_norm.min())
G = (V_norm - V_norm.min()) / (V_norm.max() - V_norm.min())
B = (W_norm - W_norm.min()) / (W_norm.max() - W_norm.min())
colors = np.stack([R, G, B], axis=-1).reshape(-1, 3)

# Plot the vector field
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W, length=0.5, normalize=True, color=colors)

# Set labels and show plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
