import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


filename = "layer_outputs_sample2_no_layernorm.npy"
filename = "layer_outputs_sample0_no_layernorm_scale0.01.npy"
filename = "layer_outputs_sample2_no_layernorm_scale0.1_01loss.npy"
filename = "layer_outputs_d2_sample0_no_layernorm_scale0.1_01loss.npy"
filename = "layer_outputs_d2_b10_sample0_no_layernorm_scale0.1_01loss.npy"
sample_id = 2 ; filename = f"acts_d2_b10_s{sample_id}.npy"
sample_id = 2 ; filename = f"acts_d2_b4_s{sample_id}_recurrent.npy"



sample_id = sys.argv[1]
filename = f"acts08_d3_b10_s{sample_id}_recurrent.npy"
filename = f"acts56_d3_b10_ffwd512_recurrent_s{sample_id}.npy"


whole_activations = np.load(filename)

# so that vis_boids.py code is directly reusable:
whole_activations = np.transpose(whole_activations, (1, 0, 2))

# it is mirrored for some reason, we mirror it back.
# TODO remove this back-and-forth.
whole_activations[:, :, 1] = - whole_activations[:, :, 1]
acts = whole_activations


print(acts.shape)

num_blocks, num_tokens, latent_dimension = whole_activations.shape
assert latent_dimension in (2, 3)


'''
for block in range(num_blocks):
    fig = plt.figure()
    if latent_dimension == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(acts[block, :, 0], acts[block, :, 1], acts[block, :, 2])
    else:
        ax = fig.add_subplot(111)
        ax.scatter(acts[block, :, 0], acts[block, :, 1])
    plt.show()
'''


num_timesteps = num_blocks
num_frames_per_step = 15  # Number of frames for each tween


# this was supposed to be the 2d version of the code coming after it,
# but it was never completely finished.
'''
# Prepare the figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111)


# ax.set_xlim(-20, 20)
# ax.set_ylim(-20, 20)

# Initial scatterplot
scatter = ax.scatter([], [], s=50)


def update(frame):
    # Calculate current and next timestep indices
    t1 = frame // num_frames_per_step
    t2 = min(t1 + 1, num_timesteps - 1)
    
    # Interpolation factor
    alpha = (frame % num_frames_per_step) / num_frames_per_step
    
    # Interpolate positions
    current_positions = (1 - alpha) * acts[t1, :, :] + alpha * acts[t2, :, :]
    
    # Update scatterplot
    scatter._offsets = (
        current_positions[:, 0],
        current_positions[:, 1]
    )
    return scatter,

# Total number of frames
total_frames = (num_timesteps - 1) * num_frames_per_step

# Create the animation
ani = FuncAnimation(
    fig, update, frames=total_frames, interval=50, blit=False
)


ani.save('vis_mnist_boids.mp4', writer='ffmpeg') ; exit()


# Save or display the animation
# To display the animation in a Jupyter notebook, use %matplotlib inline
plt.show()
'''


# Prepare the figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=60, azim=-45)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# Initial scatterplot
scatter = ax.scatter([], [], [], s=20)


def update(frame):
    # Calculate current and next timestep indices
    t1 = frame // num_frames_per_step
    t2 = min(t1 + 1, num_timesteps - 1)
    
    # Interpolation factor
    alpha = (frame % num_frames_per_step) / num_frames_per_step
    
    # Interpolate positions
    current_positions = (1 - alpha) * acts[t1, :, :] + alpha * acts[t2, :, :]
    
    # Update scatterplot
    scatter._offsets3d = (
        current_positions[:, 0], 
        current_positions[:, 1], 
        current_positions[:, 2]
    )
    return scatter,

# Total number of frames
total_frames = (num_timesteps - 1) * num_frames_per_step

# Create the animation
ani = FuncAnimation(
    fig, update, frames=total_frames, interval=50, blit=False
)


output_filename = "vis_mnist_boids." + filename.replace(".npy", ".mp4")
ani.save(output_filename, writer='ffmpeg') ; exit()


# Save or display the animation
# To display the animation in a Jupyter notebook, use %matplotlib inline
plt.show()



exit()

do_3d = True

fig = plt.figure()
if do_3d:
    ax = fig.add_subplot(111, projection='3d')
else:
    ax = fig.add_subplot(111)


for boid in range(num_tokens):
    if do_3d:
        ax.plot(acts[:, boid, 0], acts[:, boid, 1], acts[:, boid, 2])
        ax.scatter(acts[:, boid, 0], acts[:, boid, 1], acts[:, boid, 2])
        ax.scatter(acts[:1, boid, 0], acts[:1, boid, 1], acts[:1, boid, 2], s=40, c='r')
    else:
        ax.plot(acts[:, boid, 0], acts[:, boid, 1])
        ax.scatter(acts[:, boid, 0], acts[:, boid, 1])
        ax.scatter(acts[:1, boid, 0], acts[:1, boid, 1], s=400, c='r')

plt.show()
