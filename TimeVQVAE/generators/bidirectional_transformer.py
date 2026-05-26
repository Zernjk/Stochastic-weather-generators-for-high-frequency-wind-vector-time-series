import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch import Tensor

from einops import repeat, rearrange



def FeedForward(dim, mult = 4):
    """ https://arxiv.org/abs/2110.09456 """

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.LayerNorm(dim),
        weight_norm(nn.Linear(dim, inner_dim, bias = False)),
        nn.GELU(),
        weight_norm(nn.LayerNorm(inner_dim)),
        nn.Linear(inner_dim, dim, bias = False)
    )

def exists(val):
    return val is not None

def l2norm(t):
    return F.normalize(t, dim = -1)



class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        cross_attend = False,
        # scale = 8,
    ):
        super().__init__()
        # self.scale = scale
        self.heads =  heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = nn.LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = weight_norm(nn.Linear(dim, inner_dim, bias = False))
        self.to_kv = weight_norm(nn.Linear(dim, inner_dim * 2, bias = False))

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = weight_norm(nn.Linear(inner_dim, dim, bias = False))
        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        context=None,
        mask=None
        ):

        if (exists(context) and not self.cross_attend) or (not exists(context) and self.cross_attend):
            raise AssertionError("Context and cross_attend must either both be present or both be absent.")

        n = x.shape[-2]
        h = self.heads

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        out = F.scaled_dot_product_attention(q, k, v)  # scale by 1/√d is a default setting.

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.norm_out(out)
        return out


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for dep_i in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult),
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x
    


class BidirectionalTransformer(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 codebook_sizes: int,
                 embed_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 heads: int,
                 ff_mult: int,
                 use_rmsnorm: bool,
                 p_unconditional: float,
                 n_classes: int,
                 model_dropout:float=0.3,
                 emb_dropout:float=0.3,
                 **kwargs):
        """
        :param num_tokens:
        :param codebook_sizes:
        :param embed_dim:
        :param hidden_dim:
        :param n_layers:
        :param heads:
        :param ff_mult:
        :param use_rmsnorm:
        :param p_unconditional:
        :param n_classes:
        :param num_tokens_l:
        :param kwargs:
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim 
        out_dim = embed_dim
        self.emb_dropout = emb_dropout
        self.mask_token_ind = codebook_sizes
        self.weather_embed_mask_id = 16

        # token embeddings
        self.tok_emb = nn.Embedding(codebook_sizes + 1, embed_dim)  # `+1` is for mask-token

        # transformer
        self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
        self.class_condition_emb = nn.Embedding(n_classes + 1, in_dim)  # `+1` is for no-condition
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = TransformerBlocks(dim=hidden_dim, depth=n_layers, dim_head=32, heads=heads, ff_mult=ff_mult)

        self.pred_head = nn.Sequential(*[
            weight_norm(nn.Linear(in_features=hidden_dim, out_features=hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            weight_norm(nn.Linear(in_features=hidden_dim, out_features=codebook_sizes)),
        ])

    

    def class_embedding(self, class_condition: Union[None, torch.Tensor], batch_size: int, device):
        cond_type = 'uncond' if isinstance(class_condition, type(None)) else 'class-cond'

        if cond_type == 'uncond':
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            cls_emb = self.class_condition_emb(class_uncondition)  # (b 1 dim)
            return cls_emb
        elif cond_type == 'class-cond':
            if self.training:
                ind = torch.rand(class_condition.shape).to(device) > self.p_unconditional  # to enable classifier-free guidance
            else:
                ind = torch.ones_like(class_condition, dtype=torch.bool).to(device)
            class_condition = torch.where(ind, class_condition.long(), self.n_classes)  # (b 1)
            cls_emb = self.class_condition_emb(class_condition)  # (b 1 dim)
            return cls_emb

    def _token_emb_dropout(self, 
                           s:torch.LongTensor, 
                           token_emb:torch.FloatTensor, 
                           p:float):
        mask_ind = (s == self.mask_token_ind)[:,:,None]  # (b n 1)
        token_emb_dropout = F.dropout(token_emb, p=p)  # (b n d); to make the prediction process more robust during sampling
        token_emb = torch.where(mask_ind, token_emb, token_emb_dropout)  # (b n d)
        return token_emb
    
    def forward(self, s_M, class_condition: Union[None, torch.Tensor] = None, weather_embeds=None):
        device = s_M.device

        token_embeddings = self.tok_emb(s_M)  # (b n dim)
        if self.training:
            token_embeddings = self._token_emb_dropout(s_M, token_embeddings, p=self.emb_dropout)  # (b n d)

        # Add weather embeddings if provided
        if weather_embeds is not None:
            token_embeddings = token_embeddings + weather_embeds  # (b, n, d)
            
        cls_emb = self.class_embedding(class_condition, s_M.shape[0], device)  # (b 1 dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings  # (b, n, dim)
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+n, dim)
        embed = self.proj_in(embed)
        embed = self.blocks(embed)  # (b, 1+n, dim)
        logits = self.pred_head(embed)[:, 1:, :]  # (b, n, dim)

        return logits
    


class BidirectionalTransformer_wsm(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 codebook_sizes: int,
                 embed_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 heads: int,
                 ff_mult: int,
                 use_rmsnorm: bool,
                 p_unconditional: float,
                 n_classes: int,
                 model_dropout:float=0.3,
                 emb_dropout:float=0.3,
                 **kwargs):
        """
        :param num_tokens:
        :param codebook_sizes:
        :param embed_dim:
        :param hidden_dim:
        :param n_layers:
        :param heads:
        :param ff_mult:
        :param use_rmsnorm:
        :param p_unconditional:
        :param n_classes:
        :param num_tokens_l:
        :param kwargs:
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim 
        out_dim = embed_dim
        self.emb_dropout = emb_dropout
        self.mask_token_ind = codebook_sizes
        self.weather_embed_mask_id = 16

        # token embeddings
        self.tok_emb = nn.Embedding(codebook_sizes + 1, embed_dim)  # `+1` is for mask-token

        # weather states embeddings
        # 16 possible 4-bit patterns (from weather)
        self.weather_embed = nn.Embedding(17, embed_dim)  # 0..15 = states, 16 = [MASK]

        # transformer
        self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
        self.class_condition_emb = nn.Embedding(n_classes + 1, in_dim)  # `+1` is for no-condition
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = TransformerBlocks(dim=hidden_dim, depth=n_layers, dim_head=32, heads=heads, ff_mult=ff_mult)

        self.pred_head = nn.Sequential(*[
            weight_norm(nn.Linear(in_features=hidden_dim, out_features=hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            weight_norm(nn.Linear(in_features=hidden_dim, out_features=codebook_sizes)),
        ])

        self.pred_head_ws = weight_norm(nn.Linear(in_features=codebook_sizes, out_features=16))
    

    def class_embedding(self, class_condition: Union[None, torch.Tensor], batch_size: int, device):
        cond_type = 'uncond' if isinstance(class_condition, type(None)) else 'class-cond'

        if cond_type == 'uncond':
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            cls_emb = self.class_condition_emb(class_uncondition)  # (b 1 dim)
            return cls_emb
        elif cond_type == 'class-cond':
            if self.training:
                ind = torch.rand(class_condition.shape).to(device) > self.p_unconditional  # to enable classifier-free guidance
            else:
                ind = torch.ones_like(class_condition, dtype=torch.bool).to(device)
            class_condition = torch.where(ind, class_condition.long(), self.n_classes)  # (b 1)
            cls_emb = self.class_condition_emb(class_condition)  # (b 1 dim)
            return cls_emb

    def _token_emb_dropout(self, 
                           s:torch.LongTensor, 
                           token_emb:torch.FloatTensor, 
                           p:float):
        mask_ind = (s == self.mask_token_ind)[:,:,None]  # (b n 1)
        token_emb_dropout = F.dropout(token_emb, p=p)  # (b n d); to make the prediction process more robust during sampling
        token_emb = torch.where(mask_ind, token_emb, token_emb_dropout)  # (b n d)
        return token_emb
    
    def forward(self, s_M, class_condition: Union[None, torch.Tensor] = None, ws_M=None):
        device = s_M.device

        token_embeddings = self.tok_emb(s_M)  # (b n dim)
        token_embeddings_ws = self.weather_embed(ws_M)
        
        if self.training:
            token_embeddings = self._token_emb_dropout(s_M, token_embeddings, p=self.emb_dropout)  # (b n d)
            token_embeddings_ws = self._token_emb_dropout(ws_M, token_embeddings_ws, p=self.emb_dropout)

       
        token_embeddings = token_embeddings + token_embeddings_ws  # (b, n, d)
            
        cls_emb = self.class_embedding(class_condition, s_M.shape[0], device)  # (b 1 dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings  # (b, n, dim)
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+n, dim)
        embed = self.proj_in(embed)
        embed = self.blocks(embed)  # (b, 1+n, dim)
        logits = self.pred_head(embed)[:, 1:, :]  # (b, n, dim)

        logits_ws = self.pred_head_ws(logits) # (b, n, dim)

        return logits, logits_ws
