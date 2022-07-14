import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.all_datasets import Fake_Dataset 
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from utils.frechet_distance import FrechetDistance, MultivariateNormal

class attack_fid(object):
    def __init__(self, model, lr=0.02, steps=100, threshold=0.9, batch_size=128, device='cuda'):
        super().__init__()

        self.steps = steps
        self.device = device
        self.lr = lr
        self.threshold = threshold
        self.model = model.to(self.device)
        self.batch_size = batch_size
    
    def run_attack(self, fake_dataset, real_dataset, eps=None, save_path='./dataset'):
        """
        The attack produces set of samples that are predicted with high confidence,
        and number of samples are distributed equally with respect of classes.
        """
        datalodaer_f = DataLoader(fake_dataset, self.batch_size, shuffle=False, drop_last=False)
        datalodaer_r = DataLoader(real_dataset, self.batch_size, shuffle=False, drop_last=False)
        datalodaer_f, datalodaer_r = iter(datalodaer_f), iter(datalodaer_r)
        for i in tqdm(range(len(datalodaer_f))):
            idx_f, fake_batch, _ = next(datalodaer_f) #I am taking the indices from this guy
            _, real_batch, label = next(datalodaer_r) #and saving them in the labels of this guy
            fake_batch, real_batch = fake_batch.to(self.device), real_batch.to(self.device)

            if eps is None: # Unconstrained optimization for noise dataset
                new_batch = self.attack_batch(fake_batch, real_batch)
            else: # constrained optimization for real dataset
                new_batch = self.attack_batch_con(fake_batch, real_batch, eps)
           
            self.save_batch(new_batch, idx_f, label, save_path)

        new_dataset = Fake_Dataset(save_path + '/fake_images')
        return new_dataset

    def save_batch(self, batch, idx, label, path):
        for i, j in enumerate(label):
            im = to_pil_image(batch[i])
            im.save(path + '/fake_images/{}/{}.png'.format(j.item(), idx[i].item()))
        return

    def attack_batch(self, fake_batch, real_batch):
        fake_batch.requires_grad_(True)
        optimizer = torch.optim.SGD([fake_batch], lr=self.lr, momentum=0.9)
        # optimizer = torch.optim.LBFGS([fake_batch], lr=self.lr)

        for _ in range(self.steps):
            # def closure():
            optimizer.zero_grad()
            loss = ( torch.norm( self.model(fake_batch)[0] - self.model(real_batch)[0] ) )**2
            # print(loss.item())
        # if loss.item() < self.threshold:
        #     break
            loss.backward()
                # return loss
            # optimizer.step(closure)
            optimizer.step()
            # Projecting into the input domain
            fake_batch.data.clamp_(0, 1)
            # with torch.no_grad():
            #     temp = ( torch.norm( self.model(fake_batch)[0] - self.model(real_batch)[0] ) )**2
            #     print(temp.item())
            print(loss.item())
        # fake_batch.requires_grad_(False)
        return fake_batch.detach().cpu()

    def attack_batch_con(self, fake_batch, real_batch, eps):
        delta = torch.randn_like(fake_batch, requires_grad=True)*0.001
        delta.data.clamp_(-eps, eps)
        for _ in range(self.steps):
            loss = ( torch.norm( self.model(fake_batch + delta)[0] - self.model(real_batch)[0] ) )**2

            # if loss.item() > self.threshold:
            #     break
            grad = torch.autograd.grad(loss, delta)[0].sign()
            delta.data += self.lr * grad

            # Projecting into [-eps, eps]
            delta.data.clamp_(-eps, eps)
            
        new_batch = (fake_batch + delta.detach()).clamp_(0, 1)
        return new_batch.cpu()
    
    def compute_dataset_distribution(self, dataset):
        dataset_distribution = None
        loader = DataLoader(dataset, self.batch_size, shuffle=False, drop_last=False, num_workers=16)
        for _, batch, _ in tqdm(loader):
            with torch.no_grad():
                features = self.model(batch.to(self.device))[0].reshape(self.batch_size, -1)
            if dataset_distribution is None:
                dataset_distribution = MultivariateNormal(features.shape[-1]).to(self.device)
            dataset_distribution(features)
        return dataset_distribution

    def compute_fid(self, dataset1, dataset2):
        fid = FrechetDistance()
        dataset1_distribution = self.compute_dataset_distribution(dataset1)
        dataset2_distribution = self.compute_dataset_distribution(dataset2)
        return fid(dataset1_distribution, dataset2_distribution)


# # Experiment on adding noise to one distribution
#     def compute_dataset_distribution(self, dataset, sigma=0.0):
#         dataset_distribution = None
#         loader = DataLoader(dataset, self.batch_size, shuffle=False, drop_last=False, num_workers=16)
#         for _, batch, _ in tqdm(loader):
#             with torch.no_grad():
#                 features = self.model(batch.to(self.device) + sigma * torch.randn_like(batch, device=self.device))[0].reshape(self.batch_size, -1)
#             if dataset_distribution is None:
#                 dataset_distribution = MultivariateNormal(features.shape[-1]).to(self.device)
#             dataset_distribution(features)
#         return dataset_distribution

#     def compute_fid(self, dataset1, dataset2):
#         fid = FrechetDistance()
#         dataset1_distribution = self.compute_dataset_distribution(dataset1)
#         dataset2_distribution = self.compute_dataset_distribution(dataset2, sigma=0.4)
#         return fid(dataset1_distribution, dataset2_distribution)

# # Experiment on blurring the distribution
    # def compute_dataset_distribution(self, dataset, blur=False):
    #     dataset_distribution = None
    #     loader = DataLoader(dataset, self.batch_size, shuffle=False, drop_last=False, num_workers=16)
    #     if blur:
    #         import torchvision
    #         blur_layer = torchvision.transforms.GaussianBlur(11, sigma=0.1)
    #     for _, batch, _ in tqdm(loader):
    #         with torch.no_grad():
    #             if blur:
    #                 batch = blur_layer(batch.to(self.device))
    #             features = self.model(batch.to(self.device))[0].reshape(self.batch_size, -1)
    #         if dataset_distribution is None:
    #             dataset_distribution = MultivariateNormal(features.shape[-1]).to(self.device)
    #         dataset_distribution(features)
    #     return dataset_distribution

    # def compute_fid(self, dataset1, dataset2):
    #     fid = FrechetDistance()
    #     dataset1_distribution = self.compute_dataset_distribution(dataset1)
    #     dataset2_distribution = self.compute_dataset_distribution(dataset2, blur=True)
    #     return fid(dataset1_distribution, dataset2_distribution)