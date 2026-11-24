# Generative Models: Beta-VAE, GANs, CycleGAN and WGAN-GP

Project 3 of CS 437 Deep Learning, LUMS, Spring 2026. Four notebooks on how generative models trade off sample quality, disentanglement and stability.

## Parts

| Part | Notebook | What it covers |
| --- | --- | --- |
| 1A. Beta-VAE on CelebA | [`1a_beta_vae_celeba.ipynb`](notebooks/1a_beta_vae_celeba.ipynb) ([nbviewer](https://nbviewer.org/github/SameerRahil/Deep-Learning-Generative-Models/blob/main/notebooks/1a_beta_vae_celeba.ipynb)) | Beta-VAE architecture and loss, a latent-size sweep (5, 10, 15) at beta = 5, a beta sweep (1, 5, 10, 50) at latent size 15, and latent traversals to see which factors each dimension controls. |
| 1B. GANs and latent interpolation | [`1b_gan_latent_interpolation.ipynb`](notebooks/1b_gan_latent_interpolation.ipynb) ([nbviewer](https://nbviewer.org/github/SameerRahil/Deep-Learning-Generative-Models/blob/main/notebooks/1b_gan_latent_interpolation.ipynb)) | A GAN trained on MNIST, interpolation through its latent space, the same with a pre-trained CelebA generator (3.59M parameters), and latent vector arithmetic. |
| 1C. CycleGAN (horse2zebra) | [`1c_cyclegan_horse2zebra.ipynb`](notebooks/1c_cyclegan_horse2zebra.ipynb) ([nbviewer](https://nbviewer.org/github/SameerRahil/Deep-Learning-Generative-Models/blob/main/notebooks/1c_cyclegan_horse2zebra.ipynb)) | Unpaired image translation with a pre-trained CycleGAN: ablation experiments, fine-tuning, loss curves, and evaluation with cycle reconstruction error and FID. |
| 1D. DCGAN vs WGAN-GP | [`1d_dcgan_vs_wgan_gp.ipynb`](notebooks/1d_dcgan_vs_wgan_gp.ipynb) ([nbviewer](https://nbviewer.org/github/SameerRahil/Deep-Learning-Generative-Models/blob/main/notebooks/1d_dcgan_vs_wgan_gp.ipynb)) | DCGAN and WGAN-GP (critic with gradient penalty) on Fashion-MNIST and CIFAR-10, comparing training curves and samples to study mode collapse. |

If GitHub cannot display a notebook, the nbviewer link next to it opens the same file.

## Results

**Beta-VAE on CelebA (validation loss, latent size 15):**

| Beta | Total | KL divergence | Reconstruction |
| --- | --- | --- | --- |
| 1 | 258.70 | 30.55 | 228.15 |
| 5 | 354.31 | 18.09 | 263.86 |
| 10 | 427.90 | 11.59 | 312.03 |
| 50 | 654.20 | 2.93 | 507.56 |

Raising beta squeezes the KL term (a more compressed, more disentangled code) at the cost of reconstruction. At beta = 5, latent size 15 reconstructed best (268.07, against 305.91 at 10 and 382.91 at 5).

**CycleGAN FID (lower is better):** ablation 1: 156.32, ablation 2: 173.81, fine-tuned: 143.56.

## Checkpoints

[`checkpoints/`](checkpoints) holds the six trained beta-VAE models (named by beta and latent size) and the MNIST GAN generator, each about 6 MB.

## Stack

Python, PyTorch, torchvision, torch-fidelity, Matplotlib.

## Running the notebooks

The notebooks were run on Google Colab or Kaggle with a GPU. Open one in Colab, set the runtime to GPU, and run the cells in order. MNIST, Fashion-MNIST and CIFAR-10 download through torchvision. CelebA and horse2zebra are loaded inside the notebooks (Google Drive or Kaggle).

## About

Coursework for CS 437 Deep Learning, LUMS, Spring 2026, shared as part of my portfolio (course grade: A+). The notebooks keep the original handout text and my implementations, answers and outputs.

If you are taking this course, please do your own work.

**Sameer Rahil**, BS Computer Science, LUMS. More projects at [github.com/SameerRahil](https://github.com/SameerRahil).
