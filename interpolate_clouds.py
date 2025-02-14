import torch
from geomloss import SamplesLoss
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Create two simple point clouds
# Cloud 1: Points in a circle
n_points = 100
theta = torch.linspace(0, 2*np.pi, n_points)
r = 1.0
cloud1 = torch.stack([
    r * torch.cos(theta),
    r * torch.sin(theta)
], dim=1)

# Cloud 2: Points in a square
side_length = 2.0
t = torch.linspace(-side_length/2, side_length/2, int(np.sqrt(n_points)))
x, y = torch.meshgrid(t, t, indexing='ij')
cloud2 = torch.stack([x.flatten(), y.flatten()], dim=1)[:n_points]

# Convert to float32 and add batch dimension
cloud1 = cloud1.float().unsqueeze(0)
cloud2 = cloud2.float().unsqueeze(0)

# Initialize Sinkhorn loss
loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)

# Function to interpolate between clouds
def interpolate_clouds(cloud1, cloud2, t):
    # Compute the optimal transport plan
    return (1-t) * cloud1 + t * cloud2

# Create figure and axis for animation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_aspect('equal')

# Initialize scatter plot
scatter = ax.scatter([], [], c='blue', alpha=0.6, s=20)

# Animation update function
def update(frame):
    t = frame / 100  # This will give us 100 frames from 0 to 1
    interpolated = interpolate_clouds(cloud1, cloud2, t)
    scatter.set_offsets(interpolated[0].numpy())
    ax.set_title(f't = {t:.2f}')
    return scatter,

# Create animation
anim = animation.FuncAnimation(
    fig, update, frames=101,  # 101 frames for smooth 0 to 1 transition
    interval=50,  # 50ms between frames
    blit=True,
    repeat=True
)

plt.tight_layout()

# Save animation as GIF (optional)
anim.save('point_cloud_interpolation.gif', writer='pillow')

plt.show() 