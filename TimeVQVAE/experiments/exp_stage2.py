import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import pytorch_lightning as pl

from generators.maskgit import MaskGIT
from generators.maskgit_consecutive import MaskGIT_consecutive
from generators.maskgit_wsm import MaskGIT_wsm
from generators.maskgit_consecutive_wsm import MaskGIT_consecutive_wsm
from generators.sample import*
from utils import linear_warmup_cosine_annealingLR


class ExpStage2(pl.LightningModule):
    def __init__(self,
                 dataset_name: str,
                 in_channels:int,
                 input_length: int,
                 config: dict,
                 n_classes: int,
                 use_weather_states:bool=False,
                 consecutive:bool=False,
                 weather_states_mask=False,
                 ):
        super().__init__()
        self.n_classes = n_classes
        self.config = config
        self.use_weather_states = use_weather_states
        self.consecutive = consecutive
        if consecutive:
            if weather_states_mask:
                self.maskgit = MaskGIT_consecutive_wsm(dataset_name, in_channels, input_length, **config['MaskGIT'], config=config, n_classes=n_classes)
            else: 
                self.maskgit = MaskGIT_consecutive(dataset_name, in_channels, input_length, **config['MaskGIT'], config=config, n_classes=n_classes)
        else:
            if weather_states_mask:
                self.maskgit = MaskGIT_wsm(dataset_name, in_channels, input_length, **config['MaskGIT'], config=config, n_classes=n_classes)
            else:
                self.maskgit = MaskGIT(dataset_name, in_channels, input_length, **config['MaskGIT'], config=config, n_classes=n_classes)

    def training_step(self, batch, batch_idx):
        if self.use_weather_states:
            x, y, ws = batch
            mask_pred_loss = self.maskgit(x, y, ws)
        else:
            x, y = batch
            mask_pred_loss = self.maskgit(x, y)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        self.log('global_step', self.global_step)
        loss_hist = {'loss': mask_pred_loss,
                     'mask_pred_loss': mask_pred_loss,
                     }
        for k in loss_hist.keys():
            self.log(f'train/{k}', loss_hist[k])

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        if self.use_weather_states:
            x, y, ws = batch
            mask_pred_loss = self.maskgit(x, y, ws)
        else:
            x, y = batch
            mask_pred_loss = self.maskgit(x, y)

        # log
        self.log('global_step', self.global_step)
        loss_hist = {'loss': mask_pred_loss,
                     'mask_pred_loss': mask_pred_loss,
                     }
        for k in loss_hist.keys():
            self.log(f'val/{k}', loss_hist[k])
        
        # maskgit sampling & evaluation
        if batch_idx == 0 and (self.training == False) and (self.use_weather_states==False):
            print('computing evaluation metrices...')
            self.maskgit.eval()

            n_samples = 64
            xhat = unconditional_sample(self.maskgit, n_samples, x.device) 

            self._visualize_generated_timeseries(xhat)

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config['exp_params']['lr'])
        scheduler = linear_warmup_cosine_annealingLR(opt, self.config['trainer_params']['max_steps']['stage2'], 
                                                     self.config['exp_params']['linear_warmup_rate'], 
                                                     min_lr=self.config['exp_params']['min_lr'])
        return {'optimizer': opt, 'lr_scheduler': scheduler}

    def _visualize_generated_timeseries(self, xhat):
        b = 0        
        n_rows = 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(4, 2*n_rows))
        fig.suptitle(f'step-{self.global_step} \n unconditional sampling')
        axes = axes.flatten()
        axes[0].set_title(r'$\hat{x}$ (Easterly)')
        axes[0].plot(xhat[b,0,:])
        axes[1].set_title(r'$\hat{x}$ (Northerly)')
        axes[1].plot(xhat[b,1,:])

        plt.tight_layout()
        self.logger.log_image(key='generated sample', images=[wandb.Image(plt),])
        plt.close()