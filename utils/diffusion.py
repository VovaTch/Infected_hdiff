from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
import numpy as np


SCHEDULER_LIST = ['cosine']


def cosine_schedule(timesteps: int, s: float=0.008):
    """
    Cosine scheduler works better than the linear one
    """
    
    time_vector = torch.linspace(0, timesteps - 1, timesteps)
    return torch.cos((time_vector / timesteps + s) / (1 + s) * np.pi / 2) ** 2


def get_index_from_list(vals: torch.Tensor, t: torch.Tensor, x_shape: List[int]):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@dataclass
class DiffusionConstants:
    """
    Dataclass for carrying the essential constants for the diffusion model.
    """
    
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    
    def __init__(self, timesteps: int, scheduler: str='cosine'):
        
        self.T = timesteps
        assert scheduler in SCHEDULER_LIST, f'The scheduler must be one from the scheduler list {SCHEDULER_LIST}'
        
        if scheduler == 'cosine':
            
            self.alphas_cumprod = cosine_schedule(timesteps)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
            self.betas = 1. - torch.div(self.alphas_cumprod / self.alphas_cumprod_prev)
            self.alphas = 1. - self.betas
            
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        
def forward_diffusion_sample(x_0: torch.Tensor, 
                             t: torch.Tensor, 
                             diffusion_constants: DiffusionConstants, 
                             device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(diffusion_constants.sqrt_alphas_cumprod, 
                                                t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(diffusion_constants.sqrt_one_minus_alphas_cumprod, 
                                                          t, x_0.shape)
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
        
            
