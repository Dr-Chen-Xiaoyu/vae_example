# Variational Autoencoder example


The [VAE_fashion_MNIST](VAE_fashion_MNIST.ipynb) notebook has an implementation of a simple VAE for fashion-MNIST. You can run it in directly in colab [here](https://colab.research.google.com/github/jeremiecoullon/vae_example/blob/main/VAE_fashion_MNIST.ipynb).

This reproduces some of the figures in the VAE section of Kevin Murphy's recent book ["Probabilistic Machine Learning: an Introduction"](https://probml.github.io/pml-book/book1.html) (section 20.3.5).

- Uses [Flax](https://flax.readthedocs.io/en/latest/), a Deep Learning framework built with [JAX](https://jax.readthedocs.io/en/latest/)
- Defines a simple VAE using a MLP as encoders and decoders
- optimises ELBO using Adam. This is done manually rather than using Flax's helper functions (such as [TrainState](https://flax.readthedocs.io/en/latest/_modules/flax/training/train_state.html))
- Looks at some reconstructed images
- Generates new images by sampling from the latent space and running the traine decoder
- Interpolates two images from the training set. This works as the latent space in VAEs is fairly flat (see section 20.3.5 in Kevin Murphy's book)
