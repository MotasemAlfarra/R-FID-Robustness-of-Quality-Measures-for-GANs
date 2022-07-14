import torch
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy, softmax, kl_div
from utils.all_datasets import Fake_Dataset 
from PIL import Image
from torchvision.transforms.functional import to_pil_image

class attack_inception_score(object):
    def __init__(self, model, lr=0.02, steps=100, threshold=0.9, batch_size=128, device='cuda'):
        super().__init__()

        self.steps = steps
        self.device = device
        self.lr = lr
        self.threshold = threshold
        self.model = model.to(self.device)
        self.batch_size = batch_size
    
    def run_attack(self, dataset, eps=None, save_path='./dataset'):
        """
        The attack produces set of samples that are predicted with high confidence,
        and number of samples are distributed equally with respect of classes.
        """
        datalodaer = DataLoader(dataset, self.batch_size, shuffle=False, drop_last=False, num_workers=16)
        # idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        for idx, batch, label in tqdm(datalodaer):
            batch, label = batch.to(self.device), label.to(self.device)

            if eps is None: # maximize the confidence with respect label
                new_batch = self.attack_batch(batch, label)

            else: # minimizes the confidence to the all labels label.
                new_batch = self.attack_batch_con(batch, eps)
           
            # self.save_batch(new_batch, idx, [idx_to_class[l] for l in label], save_path)
            self.save_batch(new_batch, idx, label, save_path)

        
        new_dataset = Fake_Dataset(save_path + '/fake_images')
        return new_dataset

    def save_batch(self, batch, idx, label, path):
        for i, j in enumerate(label):
            im = to_pil_image(batch[i])
            im.save(path + '/fake_images/{}/{}.png'.format(j.item(), idx[i].item()))
        return

    def attack_batch(self, batch, label):
        batch.requires_grad_(True)
        optimizer = torch.optim.SGD([batch], lr=self.lr)

        for _ in range(self.steps):
            optimizer.zero_grad()
            output = self.model(batch, True)
            if softmax(output, 1)[torch.arange(batch.shape[0]), label].min().item() > self.threshold:
                break

            loss = cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            
            # Projecting into the input domain
            batch.data.clamp_(0, 1)
            

        batch.requires_grad_(False)
        return batch.cpu()

    def attack_batch_con(self, batch, eps):
        delta = torch.randn_like(batch, requires_grad=True)*0.001
        delta.data.clamp_(-eps, eps)
        for _ in range(self.steps):
            output = self.model(batch + delta, True)
            if softmax(output, 1).max().item() < self.threshold:
                break

            label = output.argmax(1)
            loss = cross_entropy(output, label)
            grad = torch.autograd.grad(loss, delta)[0].sign()
            delta.data += self.lr * grad

            # Projecting into [-eps, eps]
            delta.data.clamp_(-eps, eps)
            

        # delta.requires_grad_(False)
        new_batch = (batch + delta.detach()).clamp_(0, 1)
        return new_batch.cpu()

    def compute_probs(self, dataset):
        datalodaer = DataLoader(dataset, self.batch_size, shuffle=False, drop_last=False, num_workers=16)
        probs = None
        for idx, batch, _ in tqdm(datalodaer):
            output = softmax(self.model(batch.to(self.device), True), 1)
            if probs is None:
                probs = torch.zeros((len(dataset), output.shape[-1]))
            probs[idx] = output.cpu()
        return probs

    def compute_acc(self, dataset):
        datalodaer = DataLoader(dataset, self.batch_size, shuffle=False, drop_last=False, num_workers=16)
        total, correct = 0.0, 0.0
        for idx, batch, label in datalodaer:
            with torch.no_grad():
                output = softmax(self.model(batch.to(self.device), True), 1)
            correct += (output.cpu().argmax(1) == label).sum().item()
            total += len(idx)
        return 100*correct/total

    def compute_inception_score(self, dataset, splits = 10):
        """ Taken from 
        https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
        and modified
        """
        eps = 1e-7 #To avoid nans
        probs = self.compute_probs(dataset)
        scores = []
        for i in range(splits):
            part = probs[
                (i * probs.shape[0] // splits):
                ((i + 1) * probs.shape[0] // splits), :]
            kl = part * (
                torch.log(part + eps) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0) + eps))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
            
        scores = torch.stack(scores)
        inception_score = torch.mean(scores).cpu().item()
        std = torch.std(scores).cpu().item() if splits > 1 else 0.0
        
        del probs, scores
        return inception_score, std
