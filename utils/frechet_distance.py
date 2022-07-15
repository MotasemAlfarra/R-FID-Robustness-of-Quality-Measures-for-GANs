"""FrechetDistance"""
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
import torchvision
from tqdm import tqdm


class FrechetDistance:
    """Frechet's distance between two multi-variate Gaussians
    https://www.sciencedirect.com/science/article/pii/0047259X8290077X
    """
    def __init__(self, double=True, num_iterations=20, eps=1e-12):
        self.eps = eps
        self.double = double
        self.num_iterations = num_iterations

    def __call__(self, normal1, normal2, return_parts=False):
        # make sure that both of them are unbiased set the same
        mu1, sigma1 = normal1.mean, normal1.covariance_matrix
        mu2, sigma2 = normal2.mean, normal2.covariance_matrix
        return self.compute(mu1, sigma1, mu2, sigma2, return_parts=return_parts)

    def compute(self, mu1, sigma1, mu2, sigma2, return_parts=False):
        """Compute Frechet's distance between two multi-variate Gaussians
        https://gist.github.com/ModarTensai/185ca53b35b012c7fe781e4c567378a6
        """
        norm_2 = (mu1 - mu2).norm(2, dim=-1).pow(2)
        trace1 = sigma1.diagonal(0, -1, -2).sum(-1)
        trace2 = sigma2.diagonal(0, -1, -2).sum(-1)
        # sigma3 = self.psd_matrix_sqrt(sigma1 @ sigma2)
        # if torch.isnan(sigma3).any():
        sigma3 = torch.from_numpy(sqrtm((sigma1 @ sigma2).cpu().numpy()))
        sigma3 = sigma3.real if 'complex' in str(sigma3.dtype) else sigma3
        trace3 = sigma3.diagonal(0, -1, -2).sum(-1)
        if return_parts:
            return norm_2, torch.relu(trace1 + trace2 - 2 * trace3.to(trace1.device))
        else:
            return norm_2 + torch.relu(trace1 + trace2 - 2 * trace3.to(trace1.device))

    def psd_matrix_sqrt(self, matrix):
        """Compute the square root of a PSD matrix using Newton's method
        https://gist.github.com/ModarTensai/7c4aeb3d75bf1e0ab99b24cf2b3b37a3
        """
        dtype = matrix.dtype
        if self.double:
            matrix = matrix.double()
        norm = matrix.norm(dim=[-2, -1], keepdim=True).clamp_min_(self.eps)
        matrix = matrix / norm

        def mul_diag_add(inputs, scale=-0.5, diag=1.5):
            # multiply by a scalar then add a scalar to the diagonal
            inputs.mul_(scale).diagonal(0, -1, -2).add_(diag)
            return inputs

        other = mul_diag_add(matrix.clone())  # avoid inplace
        matrix = matrix @ other
        for i in range(1, self.num_iterations):
            temp = mul_diag_add(other @ matrix)
            matrix = matrix @ temp
            if i + 1 < self.num_iterations:  # skip last step
                other = temp @ other
        return (matrix * norm.sqrt()).to(dtype)


class MultivariateNormal(nn.Module):
    """Multivariate normal (also called Gaussian) distribution
    https://gist.github.com/ModarTensai/185ca53b35b012c7fe781e4c567378a6
    """
    def __init__(self, feature_size, unbiased=True):
        super().__init__()
        self.count = 0
        self.unbiased = bool(unbiased)
        self.feature_size = feature_size
        mean = torch.zeros(self.feature_size)
        mass = torch.zeros(self.feature_size, self.feature_size)
        self.register_buffer('mean', mean)
        self.register_buffer('mass', mass)

    @property
    def factor(self):
        """Get the normalization factor"""
        return 1 / (self.count - int(bool(self.unbiased)))

    @property
    def covariance_matrix(self):
        """Get the covariance matrix"""
        return self.mass * self.factor

    @property
    def variance(self):
        """Get the variance."""
        return self.mass.diag() * self.factor

    def forward(self, batch):
        """Perform the forward pass (only update in training mode)"""
        mean, covariance, count = self.get_stats(batch, self.unbiased)
        if self.training:
            self.stats_update(mean, covariance, count, self.unbiased)
        return mean, covariance

    @staticmethod
    def get_stats(batch, unbiased=True):
        """Compute the statistics of a batch
        https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
        """
        assert 1 <= batch.ndim <= 2
        if batch.ndim == 1:
            batch.unsqueeze(0)
        count = batch.shape[0]
        mean = batch.mean(0)
        if count == 1:
            covariance = None
        else:
            batch = batch - mean
            factor = 1 / (count - int(bool(unbiased)))
            covariance = factor * batch.t().conj() @ batch
        return mean, covariance, count

    def stats_update(self, mean, covariance, count, unbiased=None):
        """Update the model given batch statistics
        https://gist.github.com/ModarTensai/dc95444faf3624ed979b4d0b2088fdf1
        """
        diff1 = mean - self.mean
        self.mean += diff1 * (count / (self.count + count))
        diff2 = mean - self.mean
        mass = diff1[:, None].conj() @ diff2[None, :]
        if count > 1:
            mass += covariance
            mass *= count
            if unbiased is None:
                unbiased = self.unbiased
            if unbiased:
                mass -= covariance
        self.mass += mass
        self.count += count


