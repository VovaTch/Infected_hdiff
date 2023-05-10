import torch
import pytorch_lightning as pl


class BaseNetwork(pl.LightningModule):
    
    def __init__(self,
                 learning_rate: float,
                 weight_decay: float,
                 batch_size: int,
                 epochs: int,
                 scheduler_type: str,
                 steps_per_epoch: int=500,
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