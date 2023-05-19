from abc import abstractmethod
from typing import List, Optional

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from utils.diffusion import get_index_from_list, DiffusionConstants
from loss import TotalLoss

class BaseNetwork(pl.LightningModule):
    
    def __init__(self,
                 learning_rate: float,
                 weight_decay: float,
                 batch_size: int,
                 epochs: int,
                 scheduler_type: str,
                 steps_per_epoch: int=500,
                 loss_obj: TotalLoss=None,
                 data_multiplier: float=1,
                 **kwargs) -> pl.LightningModule:
        
        super().__init__()
        
        # Initialize variables
        self.cfg = kwargs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.scheduler_name = scheduler_type
        self.steps_per_epoch = steps_per_epoch
        self.loss_obj = loss_obj
        self.data_multiplier = data_multiplier
        
        # Optimizers
        assert scheduler_type in ['none', 'one_cycle_lr', 'reduce_on_platou'] # TODO fix typo, program the schedulers in
        self.scheduler_type = scheduler_type
            
            
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        if self.scheduler_type == 'one_cycle_lr':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, 
                                                            epochs=self.epochs,
                                                            steps_per_epoch=self.steps_per_epoch)
        elif self.scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1e6)
            
        elif self.scheduler_type == 'none':
            return [optimizer]
            
        scheduler_settings= {'scheduler': scheduler,
                             'interval': 'step',
                             'monitor': 'Training reconstruction loss',
                             'frequency': 1} # Set the scheduler to step every batch
        
        return [optimizer], [scheduler_settings]
    
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError('A forward method should be implemented')
    
    
class BaseDiffusionModel(BaseNetwork):
    """
    Base class for a diffusion model, contains the methods that diffusion models require.
    """
    
    def __init__(self, 
                 scheduler: str,
                 num_steps: int,
                 **kwargs) -> pl.LightningModule:
        
        # Initialize variables
        super().__init__(**kwargs)
        self.scheduler = scheduler
        self.num_steps = num_steps
        self.diffusion_constants = DiffusionConstants(self.num_steps, scheduler=scheduler)
        
        
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: List[torch.Tensor]=None):
        """
        Forward method for the diffusion, must address both the time embedding and the conditionals
        """
        pass
    
    
    @abstractmethod
    def _right_pad_if_necessary(self, x: torch.Tensor):
        """
        Pad x if necessary; can also make it a transparent method if needed
        """
        pass
    
    
    @torch.no_grad()
    def sample_timestep(self, 
                        x: torch.Tensor, 
                        t: torch.Tensor,
                        conditional_list: Optional[List[torch.Tensor]]=None,
                        verbose: bool=False):
        """
        Calls the model to predict the noise in the sound sample and returns 
        the denoised sound sample. 
        Applies noise to this sound sample, if we are not in the last step yet.
        """
        
        betas_t = get_index_from_list(self.diffusion_constants.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.diffusion_constants.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.diffusion_constants.sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        noise_pred = self(x, t, conditional_list)
        x = self._right_pad_if_necessary(x.transpose(1, 2)).transpose(1, 2)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = get_index_from_list(self.diffusion_constants.posterior_variance, t, x.shape)
        posterior_variance_t[t == 0] = 0
        
        # Show a bunch of plots and prints
        if verbose:
        
            print(sqrt_recip_alphas_t)
            print(betas_t)
            print(sqrt_one_minus_alphas_cumprod_t)
            
            plt.figure(figsize=(25, 5))
            plt.plot((x[0, ...])[0, ...].squeeze(0).cpu().detach().numpy())
            plt.show()
            
            plt.figure(figsize=(25, 5))
            plt.plot((self(x, t, conditional_list)[0, ...]).squeeze(0).cpu().detach().numpy())
            plt.show()
            
            plt.figure(figsize=(25, 5))
            plt.plot((x[0, ...] - self(x, t, conditional_list)[0, ...]).squeeze(0).cpu().detach().numpy())
            plt.show()
        
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
    
    @torch.no_grad()
    def denoise(self, noisy_input: torch.Tensor, conditionals: Optional[List[torch.Tensor]]=None, show_process_plots: bool=False):
        """
        The main denoising method. Expects to get a BS x 1 x Length input, will output a denoised music sample.

        Args:
            noisy_input (torch.Tensor): the input, can be noise, can be noisy music. The model should handle both.
        """
        
        multiplied_noisy_input = self.data_multiplier * noisy_input
        multiplied_conditionals = [cond * self.data_multiplier for cond in conditionals]
        
        running_slice = multiplied_noisy_input.clone()
        batch_size = noisy_input.shape[0]
        for time_step in reversed(range(self.num_steps)):

            time_input = torch.tensor([time_step for _ in range(batch_size)]).to(self.device)
            running_slice = self.sample_timestep(running_slice, time_input, multiplied_conditionals)
            
            if show_process_plots:
                plt.figure(figsize=(25, 5))
                plt.ylim((-1.1, 1.1))
                plt.plot(running_slice[0, ...].squeeze(0).cpu().detach().numpy())
                plt.show()
                    
        return running_slice / self.data_multiplier
    
    
    
    
    
    
        