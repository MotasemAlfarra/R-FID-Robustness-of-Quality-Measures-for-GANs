# R-FID-Robustness-of-Quality-Measures-for-GANs
This is the official repo for the ECCV paper: *"On the Robustness of Quality Measures for GANs"*, which was accepted to ECCV 2022.

Preprint available [here](https://arxiv.org/pdf/2201.13019.pdf).

![plot](./pull.png)

If you find our work useful, please cite it appropriately as:

```
@article{alfarra2022robustness,
  title={On the Robustness of Quality Measures for GANs},
  author={Alfarra, Motasem and P{\'e}rez, Juan C. and Fr{\"u}hst{\"u}ck, Anna and Torr, Philip HS and Wonka, Peter and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2201.13019},
  year={2022}
}
```

# Environment Installation

To reproduce the experiments of our paper, first you need to install the environment by running the following line:

`conda env create -f env.yml`

Then, activate the environment by running

`conda activate attack_gan_metrics`

# Pixel Attacks on Inception Score
To run pixel attacks on the Inception Score (IS), run

`python main_is.py`

However, you will need to pass the arguments for the experiment you want to run.
To run the optimization that generates good-looking images with bad scores, you need to set the following arguments:

- `--dataset` : Either `cifar10` or `imagenet`
- `--dataset-path` : the path to the directory where the dataset is located
- `--dataset-split` : either `train` or `val`
- `--eps` : The allowed perturbation budget per image

On the other hand, to create a random dataset (images with noise) but good scores, you would need to set the following arguments:
First, keep `--eps` as `None`

- `--num-instances` : Number of images in that dataset
- `--resolution` : The resolution of the noise images

# Pixel Attacks on FID
To run pixel attacks on Fréchet Inception Distance (FID), run

`python main_fid.py`

Please follow the same setup as before, but include the following:

- `--real-dataset-path` path to the real dataset the FID will be computed against

# Computation of R-FID
To compute the Robust version of FID, noted as R-FID, run the following line

`python main_fid.py`

with passing the following arguments:

- `--real-dataset-path` path to the real dataset the FID will be computed against
- `--evaluate-path` path to the dataset that needs to be evaluated with R-FID
- `--robust-inception-path` path to the robustly trained inception model. We provide two robustly trained inception models on ImageNet.

# Latent Attacks on FID
Pending

# Download Pretrained Models
To access the adversarially-trained Inception models, please download them from:
[this link](https://drive.google.com/file/d/1-AilCTTcnz2iCttHt5MP1N5upZO37VFz/view?usp=sharing),
where the zip file contains three models: (1) `ffhq.pkl` is an FFHQ pretrained GAN that was used in our experiments, (2)
`kappa_64.pth.tar` and (3) `kappa_128.pth.tar` are adversarially-trained InceptionV3 models that are trained with `$\ell_\infty$` PGD adversarial training.
