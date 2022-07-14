import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.all_datasets import Fake_Dataset 
# from torchvision.transforms.functional import to_pil_image
import PIL.Image
from utils.frechet_distance import FrechetDistance, MultivariateNormal
from PIL import Image
import numpy as np

class attack_gan_fid(object):
    def __init__(self, model, lr=0.02, steps=100,
                  batch_size=128, device='cuda'):
        super().__init__()

        self.steps = steps
        self.device = device
        self.lr = lr
        # self.threshold = threshold
        self.model = model.to(self.device)
        self.batch_size = batch_size
    
    def run_attack(self, real_dataset, eps=None, save_path='./dataset', optimize_w=False, num_save_samples=100):
        """
        This attack returns set of embeddings and latents 
        """
        self.real_dataset = real_dataset
        fake_latents, fake_embeddings = [], []
        num_instances = real_dataset.shape[0]

        for i in tqdm(range(num_instances//self.batch_size)):
            start = i*self.batch_size
            end = min((i+1)*self.batch_size, num_instances)

            real_batch = real_dataset[start:end].to(self.device)

            if optimize_w:
                if eps is None: # Running the optimization in the latent space w
                    new_latents, new_embeddings = self.attack_nearest_batch_w(real_batch.shape[0])
                else:
                    new_latents, new_embeddings = self.attack_nearest_batch_con_w(real_batch.shape[0])
            else: # Running the optimization in the latent space z
                if eps is None: # Unconstrained optimization for minimizing FID based on NN
                    new_latents, new_embeddings = self.attack_nearest_batch(real_batch.shape[0])
                else:           # constrained optimization for maximizing FID
                    new_latents, new_embeddings = self.attack_batch_con(real_batch)
           
            fake_latents.append(new_latents)
            fake_embeddings.append(new_embeddings)

        fake_latents = torch.stack(fake_latents, dim=0).reshape(num_instances, -1)
        fake_embeddings = torch.stack(fake_embeddings, dim=0).reshape(num_instances, -1)

        latent_path = f'{save_path}/latents_w.pth' if optimize_w else f'{save_path}/latents.pth'
        torch.save(fake_latents, latent_path)
        torch.save(fake_embeddings, f'{save_path}/embeddings.pth')
        
        self.save_batch(fake_latents, save_path, num_save_samples, optimize_w)
        return fake_embeddings


    def save_batch(self, fake_latents,save_path, num_save_samples, input_is_latent):
        tot = fake_latents.shape[0]
        idx =  np.random.randint(0, tot, num_save_samples)
        import os
        os.makedirs(save_path + '/fake_images', exist_ok=True)

        for i in range(num_save_samples):
            
            if input_is_latent:
                to_forward = fake_latents[idx[i]].reshape(1, 18, 512)
            else:
                to_forward = fake_latents[idx[i]].unsqueeze(0)

            batch = self.model.gan(to_forward.to(self.device), input_is_latent).cpu()
            img = (batch.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].numpy(), 'RGB').save(save_path + '/fake_images/{}.png'.format(i))
        return

    def attack_batch(self, real_batch):
        BS = real_batch.shape[0]
        fake_batch = torch.randn((BS, 512), device=self.device, requires_grad=True)
        # optimizer = torch.optim.SGD([fake_batch], lr=self.lr, momentum=0.9)
        optimizer = torch.optim.Adam([fake_batch], lr=self.lr)
        for _ in range(self.steps):
            optimizer.zero_grad()
            loss = (torch.norm(self.model(fake_batch)[0] - real_batch))/BS
            loss.backward()
            optimizer.step()
        
        embeddings = self.model(fake_batch)[0].detach().cpu().reshape(BS, -1)
        fake_batch = fake_batch.detach().cpu().reshape(BS, -1)
        return fake_batch, embeddings
    
    def attack_nearest_batch(self, BS):
        fake_batch = torch.randn((BS, 512), device=self.device, requires_grad=True)
        # optimizer = torch.optim.SGD([fake_batch], lr=self.lr, momentum=0.9)
        mse = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([fake_batch], lr=self.lr)
        for _ in range(self.steps):
            optimizer.zero_grad()
            embs = self.model(fake_batch, logits=False)[0].reshape(BS, -1)
            dist = torch.cdist(embs.detach().to("cpu"), self.real_dataset)
            indx = torch.argmin(dist, dim=1)
            targets = self.real_dataset[indx]
            # import pdb
            # pdb.set_trace()
            # loss = (torch.norm(embs - targets.to(self.device)))/BS
            loss = mse(embs, targets.to(self.device))
            # print(loss.item())
            loss.backward()
            optimizer.step()
        
        embeddings = self.model(fake_batch)[0].detach().cpu().reshape(BS, -1)
        fake_batch = fake_batch.detach().cpu().reshape(BS, -1)
        return fake_batch, embeddings

    def attack_batch_con(self, real_batch):
        BS = real_batch.shape[0]
        mse = torch.nn.MSELoss()
        delta = torch.randn((BS, 512), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        for _ in range(self.steps):
            # loss = -1*(torch.norm(self.model(delta)[0] - real_batch))/BS
            loss = -1*mse(self.model(delta)[0], real_batch)
            loss.backward()
            optimizer.step()
        
        embeddings = self.model(delta)[0].detach().cpu().reshape(BS, -1)
        delta = delta.detach().cpu().reshape(BS, -1)
        return delta, embeddings

    def attack_nearest_batch_w(self, BS):
        # latents = torch.randn((BS, 512), device=self.device)
        # print("Getting the latents w for initialization")
        fake_batch = self.model.gan.model.mapping(torch.randn((BS, 512), device=self.device),
                                             None, truncation_psi=self.model.gan.trunction)
        fake_batch.requires_grad_(True)
        optimizer = torch.optim.RMSprop([fake_batch], lr=self.lr)
        mse = torch.nn.MSELoss()
        # print("starting the optimization")
        # optimizer = torch.optim.Adam([fake_batch], lr=self.lr)
        for _ in range(self.steps):
            optimizer.zero_grad()
            embs = self.model(fake_batch, input_is_latent=True, logits=False)[0].reshape(BS, -1)
            dist = torch.cdist(embs.detach().to("cpu"), self.real_dataset)
            indx = torch.argmin(dist, dim=1)
            targets = self.real_dataset[indx]
            # import pdb
            # pdb.set_trace()
            loss = mse(embs, targets.to(self.device))
            # print(loss.item())
            loss.backward()
            optimizer.step()
        
        embeddings = self.model(fake_batch, input_is_latent=True, logits=False)[0].detach().cpu().reshape(BS, -1)
        fake_batch = fake_batch.detach().cpu().reshape(BS, -1)
        return fake_batch, embeddings

    def attack_nearest_batch_con_w(self, BS):
        # latents = torch.randn((BS, 512), device=self.device)
        # print("Getting the latents w for initialization")
        fake_batch = self.model.gan.model.mapping(torch.randn((BS, 512), device=self.device),
                                             None, truncation_psi=self.model.gan.trunction)
        fake_batch.requires_grad_(True)
        optimizer = torch.optim.SGD([fake_batch], lr=self.lr, momentum=0.9)
        mse = torch.nn.MSELoss()
        # print("starting the optimization")
        # optimizer = torch.optim.Adam([fake_batch], lr=self.lr)
        for _ in range(self.steps):
            optimizer.zero_grad()
            embs = self.model(fake_batch, input_is_latent=True, logits=False)[0].reshape(BS, -1)
            dist = torch.cdist(embs.detach().to("cpu"), self.real_dataset)
            indx = torch.argmin(dist, dim=1)
            targets = self.real_dataset[indx]
            # import pdb
            # pdb.set_trace()
            loss = -1*mse(embs, targets.to(self.device))
            # print(loss.item())
            loss.backward()
            optimizer.step()
        
        embeddings = self.model(fake_batch, input_is_latent=True, logits=False)[0].detach().cpu().reshape(BS, -1)
        fake_batch = fake_batch.detach().cpu().reshape(BS, -1)
        return fake_batch, embeddings

    def compute_fid(self, embedding1, embedding2, return_parts=False):
        fid = FrechetDistance()
        dataset1_distribution = MultivariateNormal(embedding1.shape[1])
        dataset2_distribution = MultivariateNormal(embedding2.shape[1])
        # print(embedding1.shape, embedding2.shape)
        dataset1_distribution(embedding1.cpu())
        dataset2_distribution(embedding2.cpu())
        return fid(dataset1_distribution, dataset2_distribution, 
            return_parts=return_parts)
