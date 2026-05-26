"""
`python sample.py`

sample
    1) unconditional sampling
    2) class-conditional sampling
    3) consecutive sampling
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt

from generators.maskgit import MaskGIT
from generators.maskgit_consecutive import MaskGIT_consecutive
from generators.maskgit_wsm import MaskGIT_wsm
from generators.maskgit_consecutive_wsm import MaskGIT_consecutive_wsm
from preprocessing.data_pipeline import build_data_pipeline
from utils import get_root_dir, load_yaml_param_settings


@torch.no_grad()
def unconditional_sample(maskgit: MaskGIT, n_samples: int, device, class_index=None, batch_size=32, return_representations=False, weather_states=None, weather_states_mask=False):
    print('sampling...')
    n_iters = n_samples // batch_size
    is_residual_batch = False
    if n_samples % batch_size > 0:
        n_iters += 1
        is_residual_batch = True

    x_new = []
    quantize_new= []
    sample_callback = maskgit.iterative_decoding
    if weather_states is not None:
        for i in range(n_iters):
            b = batch_size
            if (i+1 == n_iters) and is_residual_batch:
                b = n_samples - ((n_iters-1) * batch_size)
            
            weather_states_batch = weather_states[i * batch_size : i * batch_size + b]
            embed_ind = sample_callback(num=b, device=device, class_index=class_index, weather_states=weather_states_batch)

            if return_representations:
                x, quantize = maskgit.decode_token_ind_to_timeseries(embed_ind, True)
                x, quantize = x.cpu(), quantize.cpu()
                quantize_new.append(quantize)
            else:
                x = maskgit.decode_token_ind_to_timeseries(embed_ind).cpu()

            x_new.append(x)  # (b c l); b=n_samples, c=1 (univariate)

        x_new = torch.cat(x_new)

        if return_representations:
            quantize_new = torch.cat(quantize_new)
            return x_new, quantize_new
        else:
            return x_new
        
    elif weather_states_mask:
        ws_new=[]

        for i in range(n_iters):
            b = batch_size
            if (i+1 == n_iters) and is_residual_batch:
                b = n_samples - ((n_iters-1) * batch_size)
            embed_ind, embed_ind_ws = sample_callback(num=b, device=device, class_index=class_index)

            if return_representations:
                x, quantize = maskgit.decode_token_ind_to_timeseries(embed_ind, True)
                x, quantize = x.cpu(), quantize.cpu()
                quantize_new.append(quantize)
            else:
                x = maskgit.decode_token_ind_to_timeseries(embed_ind).cpu()

            x_new.append(x)  # (b c l); b=n_samples, c=1 (univariate)
            ws_new.append(embed_ind_ws)

        x_new = torch.cat(x_new)
        ws_new = torch.cat(ws_new)

        if return_representations:
            quantize_new = torch.cat(quantize_new)
            return x_new, quantize_new, ws_new
        else:
            return x_new, ws_new
        
    else: 
        for i in range(n_iters):
            b = batch_size
            if (i+1 == n_iters) and is_residual_batch:
                b = n_samples - ((n_iters-1) * batch_size)
            embed_ind = sample_callback(num=b, device=device, class_index=class_index)

            if return_representations:
                x, quantize = maskgit.decode_token_ind_to_timeseries(embed_ind, True)
                x, quantize = x.cpu(), quantize.cpu()
                quantize_new.append(quantize)
            else:
                x = maskgit.decode_token_ind_to_timeseries(embed_ind).cpu()

            x_new.append(x)  # (b c l); b=n_samples, c=1 (univariate)

        x_new = torch.cat(x_new)

        if return_representations:
            quantize_new = torch.cat(quantize_new)
            return x_new, quantize_new
        else:
            return x_new



@torch.no_grad()
def conditional_sample(maskgit: MaskGIT, n_samples: int, device, class_index: int, batch_size=32, return_representations=False, weather_states=None):
    """
    class_index: starting from 0. If there are two classes, then `class_index` ∈ {0, 1}.
    """
    if return_representations:
        x_new, quantize_new = unconditional_sample(maskgit, n_samples, device, class_index, batch_size, weather_states=weather_states)
        return x_new, quantize_new
    else:
        x_new = unconditional_sample(maskgit, n_samples, device, class_index, batch_size, weather_states=weather_states)
        return x_new


@torch.no_grad()
def consecutive_sample(maskgit: MaskGIT_consecutive, n_samples: int, device, class_index=None, batch_size=32, return_representations=False, k=0, 
                       s_before=None, ws_before=None, weather_states=None, weather_states_mask=False):
    print('consecutive sampling...')
    n_iters = n_samples // batch_size
    is_residual_batch = False
    if n_samples % batch_size > 0:
        n_iters += 1
        is_residual_batch = True

    x_new = []
    quantize_new= []
    s_new = []
    sample_callback = maskgit.iterative_decoding

    if weather_states_mask:
        ws_new=[]

        for i in range(n_iters):
            b = batch_size
            if (i+1 == n_iters) and is_residual_batch:
                b = n_samples - ((n_iters-1) * batch_size)
            embed_ind, embed_ind_ws = sample_callback(num=b, device=device, class_index=class_index, k=k, 
                                                      s_before=s_before, ws_before=ws_before)

            if return_representations:
                x, quantize, s = maskgit.decode_token_ind_to_timeseries(embed_ind, True)
                x, quantize, s = x.cpu(), quantize.cpu(), s.cpu()
                quantize_new.append(quantize)
                s_new.append(s)
            else:
                x = maskgit.decode_token_ind_to_timeseries(embed_ind).cpu()

            x_new.append(x)
            ws_new.append(embed_ind_ws)

        x_new = torch.cat(x_new)
        ws_new = torch.cat(ws_new)

        if return_representations:
            quantize_new = torch.cat(quantize_new)
            s_new = torch.cat(s_new)
            return x_new, quantize_new, s_new, ws_new
        else:
            return x_new, ws_new
        
    else:

        for i in range(n_iters):
            b = batch_size
            if (i+1 == n_iters) and is_residual_batch:
                b = n_samples - ((n_iters-1) * batch_size)
            embed_ind = sample_callback(num=b, device=device, class_index=class_index, k=k, s_before=s_before, weather_states=weather_states)

            if return_representations:
                x, quantize, s = maskgit.decode_token_ind_to_timeseries(embed_ind, True)
                x, quantize, s = x.cpu(), quantize.cpu(), s.cpu()
                quantize_new.append(quantize)
                s_new.append(s)
            else:
                x = maskgit.decode_token_ind_to_timeseries(embed_ind).cpu()

            x_new.append(x)

        x_new = torch.cat(x_new)

        if return_representations:
            quantize_new = torch.cat(quantize_new)
            s_new = torch.cat(s_new)
            return x_new, quantize_new, s_new
        else:
            return x_new


def plot_generated_samples(x_new, title: str, max_len=20):
    """
    x_new: (n_samples, c, length); c=1 (univariate)
    """
    n_samples = x_new.shape[0]
    if n_samples > max_len:
        print(f"`n_samples` is too large for visualization. maximum is {max_len}")
        return None

    try:
        fig, axes = plt.subplots(1*n_samples, 1, figsize=(3.5, 1.7*n_samples))
        alpha = 0.5
        if n_samples > 1:
            for i, ax in enumerate(axes):
                ax.set_title(title)
                ax.plot(x_new[i, 0, :])
        else:
            axes.set_title(title)
            axes.plot(x_new[0, 0, :])

        plt.tight_layout()
        plt.show()
    except ValueError:
        print(f"`n_samples` is too large for visualization. maximum is {max_len}")


def save_generated_samples(x_new: np.ndarray, save: bool, fname: str = None):
    if save:
        fname = 'generated_samples.npy' if not fname else fname
        with open(get_root_dir().joinpath('generated_samples', fname), 'wb') as f:
            np.save(f, x_new)
            print("numpy matrix of the generated samples are saved as `generated_samples/generated_samples.npy`.")
