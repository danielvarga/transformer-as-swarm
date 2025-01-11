# Transformer as swarm

A recurrent transformer operating in a 3D latent space, solving MNIST classification tasks.
Can also be interpreted as swarm intelligence.

In the following examples, a binary classification task is solved. Initially, the members of the swarm are placed at the foreground pixels of an MNIST image.
If the swarm settles left, the image is a 5; if right, it's a 6.

- [Video: swarm of boids classifying digit 6](https://static.renyi.hu/ai-shared/daniel/boids/vis_mnist_boids.acts56_d3_b10_ffwd512_recurrent_s1.mp4)
- [Video: swarm of boids classifying digit 5](https://static.renyi.hu/ai-shared/daniel/boids/vis_mnist_boids.acts56_d3_b10_ffwd512_recurrent_s2.mp4)

The interaction between the boids is determined by an attention block with 36 parameters, and isolated boid behavior is determined by a
feed-forward block with 3600 parameters.



# Transformer-Boid Swarm Analogy

The motivation behind the toy model above was an educational analogy between transformers and swarm simulations. We now present this analogy. The concepts align like this:

- token = boid (bird-like objects, the individual members of the swarm)
- transformer embedding space = boid habitat
- attention block = implements flocking behavior
- feedforward block = implements boid behavior when not interacting with other boids

- The boids do not have internal state / memory. They only encode information by their positions.
- Each sequentially executed transformer block corresponds to a timestep of the swarm simulation.
- Residual connections guarantee that boid movement is incremental.


## Actions of the boids (Tokens)

At every timestep, each boid performs two actions one after another:

1. Flocking movement, which is determined by the positions of other boids. (Implemented by the attention block.)
2. Individual movement, which is determined solely by the position of the boid. (Implemented by the feedforward block.)

For the purposes of the analogy, let us imagine the boid habitat as a massive spherical dome.
For the flocking (attention) move, each boid performs two main actions:

1. **Look at a Point on the Dome**:
   - The boid turns its gaze towards a specific point on the spherical dome, which represents the **query** in the transformer.

2. **Project a Message to a Point on the Dome**:
   - The boid sends a message to a specific point on the dome.
     - The **position** of the message corresponds to the **key**.
     - The **content** of the message ("move in this direction") corresponds to the **value**.

- **Query**: The direction the boid looks (its focus of attention).
- **Key**: The point where the boid projects its message.
- **Value**: The content of the message, an instruction on which direction to move.

- Each boid processes all the messages it receives from its field of vision, and acts accordingly.

The analogy only considers single-head attention, but it is easy to extend it to multi-head attention.
A more fundamental issue with the analogy is that in traditional swarm simulations, behavior does not depend on the timestep.
This motivated us to consider the **recurrent** transformers above, as the missing link between transformers and swarms.
