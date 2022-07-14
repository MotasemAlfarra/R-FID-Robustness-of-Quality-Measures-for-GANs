# R-FID-Robustness-of-Quality-Measures-for-GANs
This is the official repo for the ECCV paper: "On the Robustness of Quality Measures for GANs"

This work was accpeted to ECCV 2022.

Preprint: https://arxiv.org/pdf/2201.13019.pdf

(This repo is still under construction. We are actively updating it.)

![plot](./pull.png)

# Environment Installation

To reproduce the experiments of our paper, first you need to install the environment through running the following line.

`conda env create -f env.yml`

Then, activate the envrionment through running

`conda activate attack_gan_metrics`

# Pixel Attacks on Inception Score
Pending

# Pixel Attacks on FID
Pending

# Latent Attacks on FID
Pending

# Download Pretrained Models
To access the adversarially trained trained inception models, please download them from:
https://drive.google.com/file/d/1-AilCTTcnz2iCttHt5MP1N5upZO37VFz/view?usp=sharing
where the zip file contain three models. `ffhq.pkl` is pretrained GAN on FFHQ that was used in our experiments.
`kappa_64.pth.tar` and `kappa_128.pth.tar` are adversarially trained InceptionV3 models that are trained with `$\ell_\infty$` PGD adversarial training.
