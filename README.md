# Transformer as swarm

A recurrent transformer operating in a 3D latent space, solving MNIST classification tasks.
Can also be interpreted as swarm intelligence.

In the following examples, a binary classification task is solved. Initially, the swarm is placed at the foreground pixels of an MNIST image.
If the swarm settles left, the image is a 5; if right, it's a 6.

- [Video: swarm of boids classifying digit 6](https://static.renyi.hu/ai-shared/daniel/boids/vis_mnist_boids.acts56_d3_b10_ffwd512_recurrent_s1.mp4)
- [Video: swarm of boids classifying digit 5](https://static.renyi.hu/ai-shared/daniel/boids/vis_mnist_boids.acts56_d3_b10_ffwd512_recurrent_s2.mp4)

The interaction between the boids is determined by an attention block with 36 weights, and isolated boid behavior is determined by a
feed-forward block of 3600 parameters.
