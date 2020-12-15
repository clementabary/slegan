# Few-shot Adversarial Image Synthesis with SLE-GAN

PyTorch Lightning-based implementation of ICLR2021's "[Towards faster and stabilized GAN training for high-fidelity few-shot image synthesis](https://openreview.net/pdf?id=1Fqg133qRaI)" (unofficial).

The generator needs to compare favorably to StyleGAN2 with latest model configuration and differentiable data augmentation for best few-shot training performance.

##### Build a strong baseline for G :
- [x] spectral normalization (over D or G)
- [x] exponential-moving-average optimization on G
- [x] differentiable augmentation (over D)
- [x] GLU instead of ReLU in G

##### Add the two proposed techniques :
- [x] skip-layer excitation module
- [x] self-supervised discriminator

##### Refinements (not in paper but in official code) :
- [x] LPIPS-VGG perceptual loss for reconstruction
- [x] Label smoothing in hinge loss
- [x] Noise injection layer
- [x] Swish activation in SLE blocks
- [x] Auxiliary 128-sized layer output

##### Miscellaneous to-do :
- [ ] Add FID tracking (every 10k iterations)
- [ ] Add sampling with truncation
- [ ] Add interpolation tools
- [ ] Add style mixing pipeline

##### First samples

The following image grid of size 1024 has been generated at the 65k-th iteration (on a "one day - one P100" basis) with the main configuration as is.

![](samples/flowers_0.png)
![](samples/flowers_1.png)
![](samples/flowers_2.png)
![](samples/flowers_3.png)
