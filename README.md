## Wasserstein GAN with Gradient Penalty (WGAN-GP)

This repository contains Python code for implementing a Wasserstein Generative Adversarial Network (WGAN) with Gradient Penalty (WGAN-GP) using PyTorch.

### Description

The WGAN-GP is an extension of the original WGAN that improves training stability and addresses issues like mode collapse. It achieves this by penalizing the norm of the gradient of the critic's output with respect to its input data, which encourages smoother critic functions and helps prevent mode collapse.

### Files

#### discriminator.py

The `discriminator.py` file contains the implementation of the discriminator network used in the WGAN-GP model. It consists of convolutional layers followed by leaky ReLU activations.

#### generator.py

The `generator.py` file contains the implementation of the generator network used in the WGAN-GP model. It consists of convolutional blocks and residual blocks to generate realistic images.

### Usage

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script to train the WGAN-GP model.

### Dataset

The dataset used for training the WGAN-GP model is not provided in this repository. You need to prepare your own dataset of images for training.

### Model Architecture

The WGAN-GP model consists of a generator and a discriminator. The generator generates fake images from random noise, while the discriminator evaluates the realism of the generated images. The training process involves optimizing the generator and discriminator networks simultaneously in a minimax game.

### References

- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [Gradient Penalty](https://arxiv.org/abs/1704.00028)
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

