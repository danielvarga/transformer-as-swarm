import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from mnist import MNISTTransformer, ScaledTransformerEncoderLayer

torch_device = "cuda" if torch.cuda.is_available() else "cpu"


# model_filename = "model.all8_d3_b10_ffwd512_recurrent.pth"
model_filename = "model.56_d3_b10_ffwd16_recurrent.pth"
model = torch.load(model_filename, map_location=torch.device('cpu')).to(torch_device)


def ffwd(x):
    m = model.encoder_layers[0]
    return m.linear2(nn.ReLU()(m.linear1(x)))


def attention(x, kind):
    mha = model.encoder_layers[0].self_attn
    mha.in_proj_weight.shape[0] == 3 * mha.in_proj_weight.shape[1]
    embed_dim = mha.in_proj_weight.shape[1]

    if kind == 'query':
        offset = 0
    elif kind == 'key':
        offset = embed_dim
    elif kind == 'value':
        offset = 2 * embed_dim
    else:
        assert kind in ('query', 'key', 'value')

    weight = mha.in_proj_weight[offset: offset + embed_dim, :]
    bias = mha.in_proj_bias[offset: offset + embed_dim]
    output = torch.nn.functional.linear(x, weight, bias)
    return output


# Generate the grid
x_range = np.linspace(-10, 10, 6)  # Adjust resolution if needed
y_range = np.linspace(-10, 10, 6)
z_range = np.linspace(-10, 10, 6)
X, Y, Z = np.meshgrid(x_range, y_range, z_range)

# Flatten the grid for easier processing
grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(torch_device)

def show_quiver(vectors, color):

    # Reshape vectors to match the grid
    U, V, W = vectors[:, 0].reshape(X.shape), vectors[:, 1].reshape(Y.shape), vectors[:, 2].reshape(Z.shape)

    '''
    # Normalize vector magnitudes for RGB mapping
    magnitude = np.sqrt(U**2 + V**2 + W**2)
    magnitude_max = magnitude.max() if magnitude.max() > 0 else 1
    U_norm, V_norm, W_norm = U / magnitude_max, V / magnitude_max, W / magnitude_max

    # Normalize to [0, 1] range for RGB colors
    R = (U_norm - U_norm.min()) / (U_norm.max() - U_norm.min())
    G = (V_norm - V_norm.min()) / (V_norm.max() - V_norm.min())
    B = (W_norm - W_norm.min()) / (W_norm.max() - W_norm.min())
    colors = np.stack([R, G, B], axis=-1).reshape(-1, 3)
    '''

    # Plot the vector field
    ax.quiver(X, Y, Z, U, V, W, color=color)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set labels and show plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


color_mapping = [('key', 'green'), ('query', 'blue'), ('value', 'red'), ('feedforward', 'black')]

for kind, color in color_mapping:
    with torch.no_grad():
        if kind == 'feedforward':
            vectors = ffwd(grid_tensor)
            vectors /= 30
        else:
            vectors = attention(grid_tensor, kind=kind).cpu().numpy()
            vectors /= 15
    show_quiver(vectors, color)
    print(kind, color)

# Create proxy artists for the legend
legend_handles = [plt.Line2D([0], [0], color=color, lw=2, label=name) for name, color in color_mapping]

# Add the legend to the plot
ax.legend(handles=legend_handles, loc='upper left')

plt.savefig("quiver.pdf")
plt.show()
