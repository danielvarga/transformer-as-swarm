
# Transformer as swarm

A recurrent transformer operating in a 3D latent space, solving MNIST classification tasks.
Can also be interpreted as a form of swarm intelligence.

In the following examples, a binary classification task is solved. Initially, the members of the swarm are placed at the foreground pixels of an MNIST image.
If the swarm settles left, the image is a 5; if right, it's a 6.

- [Video: swarm of boids classifying digit 6](https://static.renyi.hu/ai-shared/daniel/boids/vis_mnist_boids.acts56_d3_b10_ffwd512_recurrent_s1.mp4)
- [Video: swarm of boids classifying digit 5](https://static.renyi.hu/ai-shared/daniel/boids/vis_mnist_boids.acts56_d3_b10_ffwd512_recurrent_s2.mp4)

The interaction between the boids is determined by an attention block with 36 parameters, and isolated boid behavior is determined by a
feed-forward block with 3600 parameters.


## Transformer - boid swarm analogy

The motivation behind the toy model above was an educational analogy between transformers and swarm simulations. We now present this analogy. The concepts align like this:

- token = boid (bird-like object, the individual member of the swarm)
- transformer embedding space = boid habitat
- attention block = implements flocking behavior
- feedforward block = implements boid behavior when not interacting with other boids

Furthermore,

- The boids do not have internal state / memory. They only encode information by their positions.
- Each sequentially executed transformer block corresponds to a timestep of the swarm simulation.
- Residual connections guarantee that boid movement is incremental.


### Boid behavior

At every timestep, each boid performs two actions one after another:

1. Flocking movement, which is determined by the positions of other boids. (Implemented by the attention block.)
2. Individual movement, which is determined solely by the position of the boid. (Implemented by the feedforward block.)

For the purposes of the analogy, let us imagine the boid habitat as a massive spherical dome.
For the flocking (attention) move, each boid performs two main actions:

1. **Look at a point on the dome**:
   - The boid turns its gaze towards a specific point on the spherical dome, which represents the **query** in the transformer.

2. **Project a message to a point on the dome**:
   - The boid sends a message to a specific point on the dome.
     - The **position** of the message corresponds to the **key**.
     - The **content** of the message ("move in this direction") corresponds to the **value**. It is itself a point on the dome.

- **Query**: The direction the boid looks (its focus of attention).
- **Key**: The point where the boid projects its message.
- **Value**: The content of the message, an instruction on which direction to move.

Each boid processes all the messages it receives from its field of vision, and acts accordingly.

The analogy only considers single-head attention, but it is easy to extend it to multi-head attention.
A more fundamental issue with the analogy is that in traditional swarm simulations, behavior does not depend on the timestep.
This motivated us to consider the **recurrent** transformers above, as the missing link between transformers and swarm simulations.


## Back to the toy model

We now circle back to presenting our toy model. It is defined by a single-head transformer block that is recurrently applied 10 times.
(As an alternative interpretation, as a 10-block transformer with full weight-sharing between blocks.)
It does not employ positional encoding, layer normalization or dropout. To facilitate smooth incremental movement, output is scaled by
a factor of 0.1 before the residual is added to the token embedding. (Embedding is boid position, residual is boid speed.)

The behavior is fully determined by four vector fields. For the above model solving binary classification of MNIST 5 and 6 digits,
the vector fields look like this:

![Quiver plot of vector fields](quiver.png "Quiver plot of vector fields")

The 10-block recurrent (weight-shared) model achieves a binary classification accuracy of 96.43%. For comparison, a single transformer block
achieves 89.68%, two non-recurrent blocks achieve 96.97%, two recurrent blocks achieve 95.03%.

Needless to say, we do not promote these models as relevant in practical applications. Their performance is limited
both by their recurrent nature and their low embedding dimension. A 100-dimensional, 10-block, non-recurrent,
but otherwise identical model achieves 99.35% classification accuracy.
