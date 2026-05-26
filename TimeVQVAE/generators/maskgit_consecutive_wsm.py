import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import tempfile
from typing import Union

from einops import repeat, rearrange
from typing import Callable
from generators.bidirectional_transformer import BidirectionalTransformer, BidirectionalTransformer_wsm

from encoder_decoders.vq_vae_encdec import VQVAEEncoder
from vector_quantization.vq import VectorQuantize

from experiments.exp_stage1 import ExpStage1
from utils import freeze, quantize



class MaskGIT_consecutive_wsm(nn.Module):
    """
    ref: https://github.com/dome272/MaskGIT-pytorch/blob/cff485ad3a14b6ed5f3aa966e045ea2bc8c68ad8/transformer.py#L11
    """

    def __init__(self,
                 dataset_name: str,
                 in_channels:int,
                 input_length: int,
                 choice_temperatures: dict,
                 T: dict,
                 config: dict,
                 n_classes: int,
                 **kwargs):
        super().__init__()
        self.choice_temperature = choice_temperatures
        self.T = T
        self.config = config
        self.n_classes = n_classes

        self.mask_token_ids = config['VQ-VAE']['codebook_sizes']
        self.mask_token_ids_ws = 16
        self.gamma = self.gamma_func("cosine")
        self.connect_period = config['MaskGIT']['connect_period']

        # load the staeg1 model
        self.stage1 = ExpStage1.load_from_checkpoint(os.path.join('saved_models', f'stage1-{dataset_name}.ckpt'), 
                                                     in_channels=in_channels,
                                                     input_length=input_length, 
                                                     config=config,
                                                     map_location='cpu')
        freeze(self.stage1)
        self.stage1.eval()

        self.encoder = self.stage1.encoder
        self.decoder = self.stage1.decoder
        self.vq_model = self.stage1.vq_model
        

        # token lengths
        self.num_tokens = self.encoder.num_tokens.item()

        # latent space dim
        self.H_prime, self.W_prime = self.encoder.H_prime.item(), self.encoder.W_prime.item()

        emb_dim = self.config['encoder']['hid_dim']

        # transformers / prior models
        self.transformer = BidirectionalTransformer_wsm(self.num_tokens+self.connect_period,
                                                    config['VQ-VAE']['codebook_sizes'],
                                                    emb_dim,
                                                    **config['MaskGIT']['prior_model'],
                                                    n_classes=n_classes,
                                                    )

    def load(self, model, dirname, fname):
        """
        model: instance
        path_to_saved_model_fname: path to the ckpt file (i.e., trained model)
        """
        try:
            model.load_state_dict(torch.load(dirname.joinpath(fname)))
        except FileNotFoundError:
            dirname = Path(tempfile.gettempdir())
            model.load_state_dict(torch.load(dirname.joinpath(fname)))



    @torch.no_grad()
    def encode_to_z_q(self, x, encoder: VQVAEEncoder, vq_model: VectorQuantize, svq_temp:Union[float,None]=None):
        """
        encode x to zq

        x: (b c l)
        """
        x = x.float()
        z = encoder(x)
        zq, s, _, _ = quantize(z, vq_model, svq_temp=svq_temp)  # (b c h w), (b (h w) h), ...
        return zq, s
    
    def masked_prediction(self, transformer, class_condition, s_in, ws_in):
        """
        masked prediction with classifier-free guidance
        """
        if isinstance(class_condition, type(None)):
            # unconditional 
            logits, logits_ws = transformer(s_in, class_condition=None, ws_M=ws_in)  # (b n k)
        else:
            # class-conditional
            logits, logits_ws = transformer(s_in, class_condition=class_condition, ws_M=ws_in)  # (b n k)
           
        return logits, logits_ws
        

    def forward(self, x, y, weather_states=None):
        """
        x: (B, C, 2*L) 2days data 
        y: (B, 1)
        weather_states: (B, 2L) int weather states

        ref [https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py]
        """
        self.encoder.eval()
        self.vq_model.eval()

        device = x.device

        L = int(x.shape[2] / 2)
        x_before = x[:, :, :L]
        x_after = x[:, :, L:]

        _, s_before = self.encode_to_z_q(x_before, self.encoder, self.vq_model)  # (b n)

        s_before = s_before[:, -self.connect_period:] # (b, connect_period)

        _, s_after = self.encode_to_z_q(x_after, self.encoder, self.vq_model)  # (b n)

        s = torch.cat((s_before, s_after), dim=1) # (b n+connect_period)

        weather_ids = weather_states.long()
        ws_before = weather_ids[:, :L]
        ws_after = weather_ids[:, L:]
        ws_before = ws_before[:, -self.connect_period:] # (b, connect_period)
        ws = torch.cat((ws_before, ws_after), dim=1) # (b n+connect_period)

        # mask tokens
        s_M, ws_M, mask = self._randomly_mask_tokens(s, self.mask_token_ids, ws, self.mask_token_ids_ws, device, k=self.connect_period)  # (b n+connect_period), (b n+connect_period) where 0 for masking and 1 for un-masking

       # prediction
        logits, logits_ws = self.masked_prediction(self.transformer, y, s_M, ws_M)  # (b n+connect k)
        
        # maksed prediction loss on s
        logits_on_mask = logits[~mask]  # (bm k) where m < n
        s_on_mask = s[~mask]  # (bm) where m < n
        mask_pred_loss1 = F.cross_entropy(logits_on_mask.float(), s_on_mask.long())

        # mask prediction loss on ws
        logits_on_mask_ws = logits_ws[~mask]  # (bm k) where m < n
        ws_on_mask = ws[~mask]  # (bm) where m < n
        mask_pred_loss2 = F.cross_entropy(logits_on_mask_ws.float(), ws_on_mask.long())

        mask_pred_loss = mask_pred_loss1 + mask_pred_loss2

        return mask_pred_loss

    def _randomly_mask_tokens(self, s, mask_token_id, ws, mask_token_id_ws, device, k=0):
        """
        s: token set
        k: number of initial tokens to keep unmasked (default=0)
        """
        b, n = s.shape
        
        # sample masking indices
        ratio = np.random.uniform(0, 1, (b,))  # (b,)
        n_unmasks = np.floor(self.gamma(ratio) * (n-k)) + k # (b,)
        n_unmasks = np.clip(n_unmasks, a_min=k, a_max=n-1).astype(int)  # ensures that there's at least one masked token beyond first k
        rand = torch.rand((b, n), device=device)  # (b n)
        mask = torch.zeros((b, n), dtype=torch.bool, device=device)  # (b n) initially all masked

        # Force first k tokens to remain unmasked
        if k > 0:
            mask[:, :k] = True

            # Randomly unmask remaining tokens
        for i in range(b):
            # Only consider positions >= k for masking
            remaining_positions = rand[i, k:]  # (n - k,)
            # Select top (n_unmasks[i] - k) positions to unmask (since first k are already unmasked)
            n_to_unmask = n_unmasks[i] - k
            if n_to_unmask > 0:
                _, top_indices = remaining_positions.topk(n_to_unmask, dim=-1)
                # Offset indices by k to get the correct positions in the full sequence
                mask[i, k + top_indices] = True

        # Apply masking
        masked_indices = mask_token_id * torch.ones((b, n), device=device)  # (b, n)
        s_M = torch.where(mask, s, masked_indices)  # More readable than mask * s + (~mask) * masked_indices

        # mask the token set
        masked_indices_ws = mask_token_id_ws * torch.ones((b, n), device=device)  # (b n)
        ws_M = mask * ws + (~mask) * masked_indices_ws  # (b n); `~` reverses bool-typed data

        return s_M.long(), ws_M.long(), mask


    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def create_input_tokens_normal(self, num, num_tokens, mask_token_ids, device):
        """
        returns masked tokens
        """
        blank_tokens = torch.ones((num, num_tokens), device=device)
        masked_tokens = mask_token_ids * blank_tokens
        return masked_tokens.to(torch.int64)
    
    def mask_by_random_topk(self, mask_len, probs, probs_ws, temperature=1.0, device='cpu'):
        """
        mask_len: (b 1)
        probs: (b n); also for the confidence scores

        This version keeps `mask_len` exactly.
        """
        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            """
            Gumbel max trick: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
            """
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))
        
        probs_use = 0.5*(probs + probs_ws)

        confidence = torch.log(probs_use + 1e-5) + temperature * gumbel_noise(probs_use).to(device)  # Gumbel max trick; 1e-5 for numerical stability; (b n)
        mask_len_unique = int(mask_len.unique().item())
        masking_ind = torch.topk(confidence, k=mask_len_unique, dim=-1, largest=False).indices  # (b k)
        masking = torch.zeros_like(confidence).to(device)  # (b n)
        for i in range(masking_ind.shape[0]):
            masking[i, masking_ind[i].long()] = 1.
        masking = masking.bool()
        return masking

  

    def first_pass(self,
                   s: torch.Tensor,
                   ws: torch.Tensor,
                   unknown_number_in_the_beginning,
                   class_condition: Union[torch.Tensor, None],
                   gamma: Callable,
                   device,
                   k=0, 
                   s_before=None,
                   ws_before=None):
        """
        Modified to preserve first k tokens throughout decoding.
        """

        for t in range(self.T):

            logits, logits_ws = self.masked_prediction(self.transformer, class_condition, s, ws)  # (b n+k K)

            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()  # (b n+k)
            unknown_map = (s == self.mask_token_ids)  # which tokens need to be sampled; (b n+k)
            # Never modify first k tokens if s_before is provided
            if s_before is not None and k > 0:
                unknown_map[:, :k] = False  # Protect first k tokens from being modified
                s[:, :k] = s_before[:, -k:]  # Reset first k tokens to original values
            sampled_ids = torch.where(unknown_map, sampled_ids, s)  # keep the previously-sampled tokens; (b n+k)


            sampled_ids_ws = torch.distributions.categorical.Categorical(logits=logits_ws).sample()  # (b n+k)
            # Never modify first k tokens if s_before is provided
            if ws_before is not None and k > 0:
                ws[:, :k] = ws_before[:, -k:]  # Reset first k tokens to original values
            sampled_ids_ws = torch.where(unknown_map, sampled_ids_ws, ws)  # keep the previously-sampled tokens; (b n+k)



            # create masking according to `t`
            ratio = 1. * (t + 1) / self.T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs; (b n K)
            selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze()  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b n)
            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # assign inf probability to the previously-selected tokens; (b n)

            probs_ws = F.softmax(logits_ws, dim=-1)  # convert logits into probs; (b n K)
            selected_probs_ws = torch.gather(probs_ws, dim=-1, index=sampled_ids_ws.unsqueeze(-1)).squeeze()  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b n)
            selected_probs_ws = torch.where(unknown_map, selected_probs_ws, _CONFIDENCE_OF_KNOWN_TOKENS) 

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(mask_len, min=0.)  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, selected_probs_ws, temperature=self.choice_temperature * (1. - ratio), device=device)

            # Ensure we never mask the first k tokens
            if k > 0:
                masking[:, :k] = False

            # Masks tokens with lower confidence.
            s = torch.where(masking, self.mask_token_ids, sampled_ids)  # (b n+k)
            ws = torch.where(masking, self.mask_token_ids_ws, sampled_ids_ws) # (b n+k)

        return s, ws


    @torch.no_grad()
    def iterative_decoding(self, num=1, mode="cosine", class_index=None, device='cpu', k=0, s_before=None, ws_before=None):
        """
        It performs the iterative decoding and samples token indices.
        :param num: number of samples
        :return: sampled token indices
        Modified to preserve first k tokens of s from input s_before.
        :param k: Number of initial tokens to preserve (default=0).
        :param s_before: Original s input to preserve first k tokens from s_beore
        """
        s = self.create_input_tokens_normal(num, self.num_tokens, self.mask_token_ids, device)  # (b n)
        s_before = s_before.expand(num, -1)
        s = torch.cat((s_before[:, -k:], s), dim=1) #(b n+k)

        ws = self.create_input_tokens_normal(num, self.num_tokens, self.mask_token_ids_ws, device)  # (b n)
        ws_before = ws_before.expand(num, -1)
        ws = torch.cat((ws_before[:, -k:], ws), dim=1) #(b n+k)

        unknown_number_in_the_beginning = torch.sum(s == self.mask_token_ids, dim=-1)  # (b,)
        gamma = self.gamma_func(mode)
        class_condition = repeat(torch.Tensor([class_index]).int().to(device), 'i -> b i', b=num) if class_index != None else None  # (b 1)

        s, ws = self.first_pass(s, ws, unknown_number_in_the_beginning, class_condition, gamma, device, k=k, s_before=s_before, ws_before=ws_before)
        
        # select the after s
        s = s[:, k:] #(b n)
        ws = ws[:, k:]
        
        return s, ws

    def decode_token_ind_to_timeseries(self, 
                                       s: torch.Tensor, 
                                       return_representations: bool = False):
        """
        It takes token embedding indices and decodes them to time series.
        :param s: token embedding index
        :param return_representations:
        :return:
        """
        self.eval()

        vq_model = self.vq_model
        decoder = self.decoder

        zq = F.embedding(s, vq_model._codebook.embed)  # (b n d)
        zq = vq_model.project_out(zq)  # (b n c)
      
        zq = rearrange(zq, 'b n c -> b c n')  # (b c n) == (b c (h w))
        H_prime = self.H_prime
        W_prime = self.W_prime
        zq = rearrange(zq, 'b c (h w) -> b c h w', h=H_prime, w=W_prime)

        xhat = decoder(zq)

        if return_representations:
            return xhat, zq, s
        else:
            return xhat