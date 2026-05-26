import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import pytorch_lightning as pl

from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantization import VectorQuantize
from utils import quantize, linear_warmup_cosine_annealingLR


class ExpStage1(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 input_length: int,
                 config: dict):
        """
        :param input_length: length of input time series
        :param config: configs/config.yaml
        :param n_train_samples: number of training samples
        """
        super().__init__()
        self.input_length = input_length
        self.config = config

        init_dim = config['encoder']['init_dim']
        hid_dim = config['encoder']['hid_dim']

        # encoder
        self.encoder = VQVAEEncoder(init_dim, hid_dim, in_channels, config['encoder']['n_resnet_blocks'])

        # quantizer
        # the last layer of encoder and the first layer of decoder are included in the VectorQuantize
        self.vq_model = VectorQuantize(hid_dim, config['VQ-VAE']['codebook_sizes'], **config['VQ-VAE'])

        # decoder
        self.decoder = VQVAEDecoder(hid_dim, in_channels, config['decoder']['n_resnet_blocks'], input_length)

    def forward(self, batch, batch_idx, return_x_rec:bool=False, return_x_vq:bool=False):
        """
        :param x: input time series (b c l)
        """
        x, y = batch
        x = x.float()

        recons_loss = 0.
        vq_losses = None
        perplexities = 0.
        
        # Encode
        z = self.encoder(x)
        z_q, s, vq_loss, perplexity = quantize(z, self.vq_model)
        xhat = self.decoder(z_q)  # (b c l)

        if return_x_rec:
            x_rec = xhat # (b c l)
            return x_rec  # (b c l)
            
        if return_x_vq:
            x_rec = xhat # (b c l)
            return x_rec, z_q, s

        recons_loss = F.mse_loss(x, xhat)
        perplexities = perplexity
        vq_losses = vq_loss


        # plot `x` and `xhat`
        if not self.training and batch_idx == 0:
            b = np.random.randint(0, x.shape[0])

            alpha = 0.6
            n_rows = 2
            fig, axes = plt.subplots(n_rows, 1, figsize=(4, 2*n_rows))
            plt.suptitle(f'step-{self.global_step} \n (blue:GT, orange:reconstructed)')
            axes[0].plot(x[b, 0].cpu(), alpha=alpha)
            axes[0].plot(xhat[b, 0].detach().cpu(), alpha=alpha)
            axes[0].set_title(r'$x$ (Easterly)')
            # axes[0].set_ylim(-4, 4)

            axes[1].plot(x[b, 1].cpu(), alpha=alpha)
            axes[1].plot(xhat[b, 1].detach().cpu(), alpha=alpha)
            axes[1].set_title(r'$x$ (Northerly)')
            # axes[1].set_ylim(-4, 4)

            plt.tight_layout()
            wandb.log({"x vs x_rec (val)": wandb.Image(plt)})
            plt.close()

        return recons_loss, vq_losses, perplexities

    def training_step(self, batch, batch_idx):
        recons_loss, vq_losses, perplexities = self.forward(batch, batch_idx)
        loss = recons_loss + vq_losses['loss']

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss,
                     'vq_loss': vq_losses['loss'],
                     'commit_loss': vq_losses['commit_loss'],
                     'perplexity': perplexities,
                     }
        
        # log
        self.log('global_step', self.global_step)
        for k in loss_hist.keys():
            self.log(f'train/{k}', loss_hist[k])

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        recons_loss, vq_losses, perplexities = self.forward(batch, batch_idx)
        loss = recons_loss + vq_losses['loss']

        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss,
                     'vq_loss': vq_losses['loss'],
                     'commit_loss': vq_losses['commit_loss'],
                     'perplexity': perplexities,
                     }
        
        # log
        self.log('global_step', self.global_step)
        for k in loss_hist.keys():
            self.log(f'val/{k}', loss_hist[k])

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config['exp_params']['lr'])
        scheduler = linear_warmup_cosine_annealingLR(opt, self.config['trainer_params']['max_steps']['stage1'], self.config['exp_params']['linear_warmup_rate'], min_lr=self.config['exp_params']['min_lr'])
        return {'optimizer': opt, 'lr_scheduler': scheduler}
    

