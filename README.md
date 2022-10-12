# Variational Autoencoder example


The colab notebook has an implementation of a simple VAE for fashion-MNIST. This reproduces some of the figure in Kevin Murphy's recent book "Probabilistic Machine Learning: an Introduction" (page ..).

- Uses [Flax](https://flax.readthedocs.io/en/latest/), a Deep Learning framework built with [JAX](https://jax.readthedocs.io/en/latest/)
- Defines a simple VAE using a MLP as encoders and decoders
- optimises ELBO using Adam. This is done manually rather than using Flax's helper functions (such as [TrainState](https://flax.readthedocs.io/en/latest/_modules/flax/training/train_state.html))
- Looks at some reconstructed images
- Generates new images by sampling from the latent space and running the traine decoder
- Interpolates two images from the training set. This works as the latent space is relatively flat (see page xxx in Kevin Murphy's book.)
