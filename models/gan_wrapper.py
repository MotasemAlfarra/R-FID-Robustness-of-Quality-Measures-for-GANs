import torch
import pickle 
import os.path as osp
import torch.nn as nn
from pathlib import Path


home = str(Path.home())
global_path = osp.join(home, 'attack_gan_metrrics/models') # '/home/alfarrm/attack_gan_metrrics/models'
paths = {
    'dog': global_path + '/pretrained_gans/afhqdog.pkl',
    'cat': global_path + '/pretrained_gans/afhqcat.pkl',
    'metfaces': global_path + '/pretrained_gans/metfaces.pkl',
    'ffhq': global_path + '/pretrained_gans/ffhq.pkl'
}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class StyleGAN(nn.Module):
    def __init__(self, dataset, truncation=0.5, noise_mode = 'const'):
        super().__init__()

        print("Loading saved StyleGANv2 for ", dataset)
        checkpoint_path = paths[dataset]
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.Unpickler(f).load()

        self.model = checkpoint['G_ema'].to(device)
        self.trunction = truncation
        self.noise_mode = noise_mode

        print("GAN is loaded!")

    def forward(self, z, input_is_latent=False):
        if not input_is_latent:
            z = self.model.mapping(z, None, truncation_psi=self.trunction)
        
        return self.model.synthesis(z, noise_mode=self.noise_mode)

class GAN_Wrapper(nn.Module):
    """
    A Gan combined with a classifier to 
    compute embeddings or logits for IS or FID
    """
    def __init__(self, gan, inception):
        super().__init__()
        
        self.gan = gan
        self.inception = inception

    def forward(self, x, input_is_latent=False, logits=False):
        images = self.gan(x, input_is_latent)
        # The output of the GAN is assumed to be in [-1, 1]
        images = (images + 1)/2
        return self.inception(images, logits)
