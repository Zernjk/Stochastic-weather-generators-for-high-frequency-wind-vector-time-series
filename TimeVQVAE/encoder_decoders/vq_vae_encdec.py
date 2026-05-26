"""
reference: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
from utils import SnakeActivation


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout:float=0.):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels
        
        kernel_size = (1, 3)
        padding = (0, 1)

        layers = [
            SnakeActivation(in_channels, 2), 
            weight_norm(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding, groups=in_channels)),
            weight_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=1)),
            SnakeActivation(out_channels, 2), 
            weight_norm(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding, groups=mid_channels)),
            weight_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=1)),
            nn.Dropout(dropout)
        ]
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.proj(x) + self.convs(x)


class VQVAEEncBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout:float=0.
                 ):
        super().__init__()
        
        kernel_size = (1, 3) 
        padding = (0, 1) 

        self.block = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding, padding_mode='replicate', groups=in_channels)),
            weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)),
            SnakeActivation(out_channels, 2), 
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEDecBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout:float=0.
                 ):
        super().__init__()
        
        kernel_size = (1, 4) 
        padding = (0, 1)

        self.block = nn.Sequential(
            weight_norm(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding, groups=in_channels)),
            weight_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1)),
            SnakeActivation(out_channels, 2), 
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEEncoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 init_dim:int,
                 hid_dim: int,
                 num_channels: int,
                 n_resnet_blocks: int,
                 dropout:float=0.3,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param n_resnet_blocks: number of ResNet blocks
        :param bn: use of BatchNorm
        :param kwargs:
        """
        super().__init__()

        d = init_dim
        enc_layers = [VQVAEEncBlock(num_channels, d),]
        d *= 2
        
        enc_layers.append(ResBlock(d//2, hid_dim, dropout=dropout))
        self.encoder = nn.Sequential(*enc_layers)

        self.is_num_tokens_updated = False
        self.register_buffer('num_tokens', torch.tensor(0))
        self.register_buffer('H_prime', torch.tensor(0))
        self.register_buffer('W_prime', torch.tensor(0))
    
    def forward(self, x):
        """
        :param x: (b c l)
        """
        x = x.unsqueeze(2)  #(b, c, 1, l)

        out = self.encoder(x)  # (b hid h=1 w)

        # out = F.normalize(out, dim=1)  # following hilcodec, if using EuclideanCodebook -> normalize=False

        if not self.is_num_tokens_updated:
            self.H_prime = torch.tensor(out.shape[2])
            self.W_prime = torch.tensor(out.shape[3])
            self.num_tokens = self.H_prime * self.W_prime
            self.is_num_tokens_updated = True
        return out


class VQVAEDecoder(nn.Module):
    def __init__(self,
                 hid_dim: int,
                 num_channels: int,
                 n_resnet_blocks: int,
                 input_length:int,
                 dropout: float = 0.3,
                 **kwargs):
        """
        Decoder that keeps resolution fixed (no upsampling).
        
        :param hid_dim: hidden dim from encoder output
        :param num_channels: final output channels (e.g. 2 for wind speed components)
        :param n_resnet_blocks: number of residual blocks
        :param frequency_indepence: kernel shape config
        :param dropout: dropout rate
        """
        super().__init__()

        kernel_size = (1, 3)
        padding = (0, 1) 

        layers = []

        # Residual blocks
        for _ in range(n_resnet_blocks):
            layers.append(ResBlock(hid_dim, hid_dim, dropout=dropout))

        # Final conv mapping to output channels
        layers.append(nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, kernel_size=kernel_size, padding=padding, groups=hid_dim),
            nn.Conv2d(hid_dim, num_channels, kernel_size=1)
        ))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z_q):
        """
        :param z_q: (B, hid_dim, 1, L)
        :return: (B, num_channels=C, L)
        """
        out = self.decoder(z_q)         # (B, num_channels=C, 1, L)
        out = out.squeeze(2)            # remove height dim → (B, num_channels=C, L)

        return out
