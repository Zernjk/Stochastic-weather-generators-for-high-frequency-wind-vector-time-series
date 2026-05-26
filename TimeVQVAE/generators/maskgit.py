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
from generators.bidirectional_transformer import BidirectionalTransformer

from encoder_decoders.vq_vae_encdec import VQVAEEncoder
from vector_quantization.vq import VectorQuantize

from experiments.exp_stage1 import ExpStage1
from utils import freeze, quantize


class MaskGIT(nn.Module):
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
        self.gamma = self.gamma_func("cosine")

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

        # transformers / prior models
        emb_dim = self.config['encoder']['hid_dim']

        # # 16 possible 4-bit patterns (from weather)
        # if in_channels == 2:
        #     self.weather_embed = nn.Embedding(16, emb_dim)  
        
        self.transformer = BidirectionalTransformer(self.num_tokens,
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
    
    def masked_prediction(self, transformer, class_condition, *s_in, weather_embeds):
        """
        masked prediction with classifier-free guidance
        """
        if isinstance(class_condition, type(None)):
            # unconditional 
            logits_null = transformer(*s_in, class_condition=None, weather_embeds=weather_embeds)  # (b n k)
            return logits_null
        else:
            # class-conditional
            logits = transformer(*s_in, class_condition=class_condition, weather_embeds=weather_embeds)  # (b n k)
           
            return logits
        
    def weather_to_embed(self, weather_states):
        ws = weather_states.long() # (B, T)
        # Weather embeddings
        weather_embeds = self.weather_embed(ws)  # (B, T, D)
        return weather_embeds

    def forward(self, x, y, weather_states=None):
        """
        x: (B, C, L)
        y: (B, 1)
        weather_states: if not None (B, L)
        ref [https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py]
        """
        self.encoder.eval()
        self.vq_model.eval()

        device = x.device
        _, s = self.encode_to_z_q(x, self.encoder, self.vq_model)  # (b n)

        # mask tokens
        s_M, mask = self._randomly_mask_tokens(s, self.mask_token_ids, device)  # (b n), (b n) where 0 for masking and 1 for un-masking

        if weather_states is not None:
            weather_embeds = self.weather_to_embed(weather_states=weather_states)
        else: 
            weather_embeds = None

        # prediction
        logits = self.masked_prediction(self.transformer, y, s_M, weather_embeds=weather_embeds)  # (b n k)
        
        # maksed prediction loss
        logits_on_mask = logits[~mask]  # (bm k) where m < n
        s_on_mask = s[~mask]  # (bm) where m < n
        mask_pred_loss = F.cross_entropy(logits_on_mask.float(), s_on_mask.long())

        return mask_pred_loss

    def _randomly_mask_tokens(self, s, mask_token_id, device):
        """
        s: token set
        """
        b, n = s.shape
        
        # sample masking indices
        ratio = np.random.uniform(0, 1, (b,))  # (b,)
        n_unmasks = np.floor(self.gamma(ratio) * n)  # (b,)
        n_unmasks = np.clip(n_unmasks, a_min=0, a_max=n-1).astype(int)  # ensures that there's at least one masked token
        rand = torch.rand((b, n), device=device)  # (b n)
        mask = torch.zeros((b, n), dtype=torch.bool, device=device)  # (b n)

        for i in range(b):
            ind = rand[i].topk(n_unmasks[i], dim=-1).indices
            mask[i].scatter_(dim=-1, index=ind, value=True)

        # mask the token set
        masked_indices = mask_token_id * torch.ones((b, n), device=device)  # (b n)
        s_M = mask * s + (~mask) * masked_indices  # (b n); `~` reverses bool-typed data
        return s_M.long(), mask
    
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

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0, device='cpu'):
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

        confidence = torch.log(probs + 1e-5) + temperature * gumbel_noise(probs).to(device)  # Gumbel max trick; 1e-5 for numerical stability; (b n)
        mask_len_unique = int(mask_len.unique().item())
        masking_ind = torch.topk(confidence, k=mask_len_unique, dim=-1, largest=False).indices  # (b k)
        masking = torch.zeros_like(confidence).to(device)  # (b n)
        for i in range(masking_ind.shape[0]):
            masking[i, masking_ind[i].long()] = 1.
        masking = masking.bool()
        return masking

  

    def first_pass(self,
                   s: torch.Tensor,
                   unknown_number_in_the_beginning,
                   class_condition: Union[torch.Tensor, None],
                   gamma: Callable,
                   device,
                   weather_states=None):
        
        if weather_states is not None:
            weather_embeds = self.weather_to_embed(weather_states=weather_states)
        else: 
            weather_embeds = None

        for t in range(self.T):
            logits = self.masked_prediction(self.transformer, class_condition, s, weather_embeds=weather_embeds)  # (b n k)

            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()  # (b n)
            unknown_map = (s == self.mask_token_ids)  # which tokens need to be sampled; (b n)
            sampled_ids = torch.where(unknown_map, sampled_ids, s)  # keep the previously-sampled tokens; (b n)

            # create masking according to `t`
            ratio = 1. * (t + 1) / self.T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs; (b n K)
            selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze()  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b n)
            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # assign inf probability to the previously-selected tokens; (b n)

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(mask_len, min=0.)  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio), device=device)

            # Masks tokens with lower confidence.
            s = torch.where(masking, self.mask_token_ids, sampled_ids)  # (b n)

        return s


    @torch.no_grad()
    def iterative_decoding(self, num=1, mode="cosine", class_index=None, device='cpu', weather_states=None):
        """
        It performs the iterative decoding and samples token indices
        :param num: number of samples
        :return: sampled token indices
        """
        s = self.create_input_tokens_normal(num, self.num_tokens, self.mask_token_ids, device)  # (b n)

        unknown_number_in_the_beginning = torch.sum(s == self.mask_token_ids, dim=-1)  # (b,)
        gamma = self.gamma_func(mode)
        class_condition = repeat(torch.Tensor([class_index]).int().to(device), 'i -> b i', b=num) if class_index != None else None  # (b 1)

        s = self.first_pass(s, unknown_number_in_the_beginning, class_condition, gamma, device, weather_states=weather_states)
        return s

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
            return xhat, zq
        else:
            return xhat